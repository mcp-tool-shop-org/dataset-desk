"""Tests for training-caption support: Qwen3-VL drafter, per-project caption rules, composition, provenance, and
blind worksheet seat-edit re-import.

All tests run offline on CPU using synthetic 16x16 images and `tests.fakes` stand-ins — never loading real weights or
touching a GPU.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from dataset_desk import backends, description_gate, desk, registry
from dataset_desk.errors import EXIT_OK, EXIT_USER, DeskError
from tests import fakes


def _write_png(path: str, rgb: tuple[int, int, int], seed: int = 1) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rng = np.random.default_rng(seed)
    base = np.array(rgb, dtype=np.int32)
    arr = np.clip(base + rng.integers(-15, 16, size=(16, 16, 3)), 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _snapshot_tree(root: str) -> dict[str, str]:
    snap: dict[str, str] = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in sorted(filenames):
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root).replace("\\", "/")
            with open(full, "rb") as f:
                snap[rel] = hashlib.sha256(f.read()).hexdigest()
    return snap


class _FakeQwenProcessor(fakes.FakeProcessor):
    def __init__(self, raw_reply: str = "<|im_start|>assistant\n  slate flagstones and a carved wooden bench  <|im_end|>"):
        super().__init__()
        self.raw_reply = raw_reply
        self.template_calls: list[tuple] = []
        self._commit_hash: str | None = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.template_calls.append((messages, tokenize, add_generation_prompt))
        text_bits = [
            part["text"]
            for msg in messages
            for part in msg.get("content", [])
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        return f"<chat>{' '.join(text_bits)}</chat>"

    def batch_decode(self, ids, skip_special_tokens=False):
        if skip_special_tokens:
            clean = self.raw_reply.replace("<|im_start|>assistant\n", "").replace("<|im_end|>", "")
            return [clean]
        return [self.raw_reply]


class DrafterSettingTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)

    def test_qwen3_vl_registry_pin_licence_and_default_drafter(self):
        self.assertEqual(registry.DEFAULT_DRAFTER, "plain-sight")
        self.assertEqual(registry.resolve_drafter(None), "plain-sight")
        self.assertEqual(registry.resolve_drafter(""), "plain-sight")
        self.assertEqual(registry.resolve_drafter("qwen3-vl"), "qwen3-vl")

        # Embedding REGISTRY stays untouched
        self.assertEqual(
            set(registry.REGISTRY),
            {"eg2", "siglip2", "bge-m3", "qwen3-emb-0.6b", "gte-modernbert", "dinov2"},
        )
        self.assertIn("plain-sight", registry.DRAFTER_REGISTRY)
        self.assertIn("qwen3-vl", registry.DRAFTER_REGISTRY)
        self.assertIn("qwen3-vl", registry.MODEL_REGISTRY)

        qspec = registry.DRAFTER_REGISTRY["qwen3-vl"]
        self.assertEqual(qspec.repo, "Qwen/Qwen3-VL-8B-Instruct")
        self.assertEqual(qspec.revision, "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b")
        self.assertEqual(qspec.licence, "Apache-2.0")
        self.assertFalse(qspec.gated)
        self.assertFalse(qspec.is_arm_model)
        self.assertEqual(qspec.loader_kind, "qwen3-vl")
        self.assertEqual(qspec.vram_budget, qspec.vram_mib)
        self.assertEqual(qspec.vram_budget_mib, qspec.vram_mib)
        self.assertEqual(qspec["licence"], "Apache-2.0")
        self.assertEqual(qspec.get("gated"), False)
        self.assertIn("revision", qspec)
        self.assertEqual(qspec.to_dict()["repo"], "Qwen/Qwen3-VL-8B-Instruct")

        # Default make_drafter() returns PlainSightDrafter
        default_d = backends.make_drafter()
        self.assertIsInstance(default_d, backends.PlainSightDrafter)
        self.assertIsNone(backends.make_drafter(None))

    def test_qwen3_vl_prompt_raw_output_and_chat_template(self):
        img_p = os.path.join(self.tmp, "plate_01.png")
        _write_png(img_p, (140, 90, 60), seed=3)

        proc = _FakeQwenProcessor()
        cfg = types.SimpleNamespace(
            _commit_hash="0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
            _name_or_path="Qwen/Qwen3-VL-8B-Instruct",
        )
        mdl = fakes.FakeModel(config=cfg)
        tf = fakes.make_transformers(processor=proc, model=mdl)
        t_mod = fakes.make_torch(bf16=False)

        with fakes.fake_modules(torch=t_mod, transformers=tf):
            d = backends.make_drafter("qwen3-vl", prompt="the visible backdrop and floor only")
            self.assertIsInstance(d, backends.Qwen3VLDrafter)
            self.assertFalse(d.is_arm_model)
            self.assertEqual(d.prompt, "the visible backdrop and floor only")
            self.assertEqual(d.draft([]), [])
            out = d.draft([img_p])

        self.assertEqual(out, ["slate flagstones and a carved wooden bench"])
        self.assertEqual(
            d.raw_outputs[img_p],
            "<|im_start|>assistant\n  slate flagstones and a carved wooden bench  <|im_end|>",
        )
        self.assertEqual(d.last_raw_outputs, [d.raw_outputs[img_p]])
        self.assertEqual(len(proc.template_calls), 1)
        sent_msgs = proc.template_calls[0][0]
        self.assertEqual(sent_msgs[0]["content"][1], {"type": "text", "text": "the visible backdrop and floor only"})

        # Verify no remote-code execution kwarg was passed
        for _kind, _repo, kw in tf.calls:
            self.assertEqual(kw["revision"], "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b")
            self.assertNotIn("trust_" + "remote_code", kw)

        with self.assertRaises(TypeError):
            d.draft("not-a-list.png")  # type: ignore[arg-type]

    def test_qwen3_vl_refuses_revision_mismatch(self):
        for bad_rev in ("main", "", "1" * 40):
            with self.assertRaises(DeskError) as cm:
                backends.Qwen3VLDrafter(revision=bad_rev)
            self.assertEqual(cm.exception.code, "MODEL_REVISION_MISMATCH")

        img_p = os.path.join(self.tmp, "plate_02.png")
        _write_png(img_p, (40, 120, 90), seed=4)
        wrong_cfg = types.SimpleNamespace(_commit_hash="a" * 40, _name_or_path="Qwen/Qwen3-VL-8B-Instruct")
        tf_bad_model = fakes.make_transformers(processor=_FakeQwenProcessor(), model=fakes.FakeModel(config=wrong_cfg))
        with fakes.fake_modules(torch=fakes.make_torch(), transformers=tf_bad_model):
            d = backends.Qwen3VLDrafter()
            with self.assertRaises(DeskError) as cm2:
                d.draft([img_p])
            self.assertEqual(cm2.exception.code, "MODEL_REVISION_MISMATCH")

        proc_bad = _FakeQwenProcessor()
        proc_bad._commit_hash = "b" * 40
        ok_cfg = types.SimpleNamespace(_commit_hash=backends.QWEN3_VL_REVISION)
        tf_bad_proc = fakes.make_transformers(processor=proc_bad, model=fakes.FakeModel(config=ok_cfg))
        with fakes.fake_modules(torch=fakes.make_torch(), transformers=tf_bad_proc):
            d2 = backends.Qwen3VLDrafter()
            with self.assertRaises(DeskError) as cm3:
                d2.draft([img_p])
            self.assertEqual(cm3.exception.code, "MODEL_REVISION_MISMATCH")

    def test_qwen3_vl_honours_hf_hub_offline_and_vram_governor(self):
        img_p = os.path.join(self.tmp, "plate_03.png")
        _write_png(img_p, (70, 70, 160), seed=5)

        os.environ["HF_HUB_OFFLINE"] = "1"
        proc = _FakeQwenProcessor()
        cfg = types.SimpleNamespace(_commit_hash=backends.QWEN3_VL_REVISION)
        tf = fakes.make_transformers(processor=proc, model=fakes.FakeModel(config=cfg))
        with fakes.fake_modules(torch=fakes.make_torch(), transformers=tf):
            d = backends.make_drafter("qwen3-vl")
            d.draft([img_p])
        self.assertTrue(all(kw.get("local_files_only") is True for _k, _r, kw in tf.calls))

        # Offline cache miss raises structured DEP_DRAFTER_OFFLINE
        tf_miss = fakes.make_transformers()
        tf_miss.AutoProcessor.from_pretrained = staticmethod(
            lambda *_a, **_kw: (_ for _ in ()).throw(OSError("snapshot missing locally"))
        )
        with fakes.fake_modules(torch=fakes.make_torch(), transformers=tf_miss):
            d_miss = backends.Qwen3VLDrafter()
            with self.assertRaises(DeskError) as cm_off:
                d_miss.draft([img_p])
            self.assertEqual(cm_off.exception.code, "DEP_DRAFTER_OFFLINE")

        # VRAM governor refuses busy GPU and permits roomy GPU with --grant
        with fakes.fake_modules(torch=fakes.make_torch(bf16=True, free_mib=4096, total_mib=32768), transformers=tf):
            with self.assertRaises(DeskError) as cm_busy:
                backends.make_drafter("qwen3-vl", device="cuda:0", grant="grant-1")
            self.assertEqual(cm_busy.exception.code, "PERM_GPU_BUSY")

        with fakes.fake_modules(torch=fakes.make_torch(bf16=True, free_mib=30720, total_mib=32768), transformers=tf):
            d_gpu = backends.make_drafter("qwen3-vl", device="cuda:0", grant="grant-1")
            self.assertTrue(d_gpu._vram_checked)

    def test_multimodal_pool_allows_non_arm_drafter_and_refuses_arm_drafter(self):
        locked_proj = os.path.join(self.tmp, "salt-road")
        os.makedirs(locked_proj, exist_ok=True)
        qwen_d = backends.Qwen3VLDrafter()
        desk.check_drafter_pool(locked_proj, qwen_d, allow_multimodal_pool=False)

        arm_drafter = types.SimpleNamespace(name="arm-s-drafter", is_arm_model=True)
        with self.assertRaises(DeskError) as cm:
            desk.check_drafter_pool(locked_proj, arm_drafter, allow_multimodal_pool=False)
        self.assertEqual(cm.exception.code, "PERM_LOCKED_POOL")


class CaptionRulesGateTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name
        self.proj = os.path.join(self.tmp, "synthetic-proj")
        self.out = os.path.join(self.tmp, "out")
        os.makedirs(os.path.join(self.proj, "records"), exist_ok=True)
        os.makedirs(os.path.join(self.proj, "outputs", "w1"), exist_ok=True)

    def _write_record(self, rid: str, asset_name: str, explanation: str, *, category: str | None = None, seed: int = 1):
        rel_asset = f"outputs/w1/{asset_name}"
        _write_png(os.path.join(self.proj, rel_asset), (100 + seed * 7, 80, 90), seed=seed)
        canon: dict = {"subject": "unrelated short tag"}
        if category is not None:
            canon["category"] = category
        rec = {
            "id": rid,
            "asset_path": rel_asset,
            "canon": canon,
            "judgment": {"status": "approved", "explanation": explanation},
            "provenance": {"prompt": "completely distinct synthetic generation prompt words"},
        }
        with open(os.path.join(self.proj, "records", f"{rid}.json"), "w", encoding="utf-8") as f:
            json.dump(rec, f)

    def test_caption_rules_flags_prefix_required_term_banned_word_and_duplicates(self):
        rules_doc = {
            "prefix": "etching plate,",
            "class_field": "canon.category",
            "class_pattern": r"(?:^|/)(?P<class>[a-z]+)_[0-9]+\.png$",
            "backdrop": {
                "text": "pale stone wall",
                "required": True,
                "banned_words": ["neon"],
            },
            "banned_words": ["photorealistic"],
            "classes": {
                "vessel": {
                    "required_terms": ["timber hull", "coiled rope"],
                    "banned_words": ["portrait", "automobile"],
                },
                "archway": {
                    "required_terms": ["masonry arch"],
                    "banned_words": ["ship"],
                },
            },
        }
        with open(os.path.join(self.proj, "caption-rules.json"), "w", encoding="utf-8") as f:
            json.dump(rules_doc, f, indent=2)

        # Clean vessel record (class from canon.category)
        self._write_record(
            "r_ok",
            "item_01.png",
            "etching plate, timber hull, coiled rope, pale stone wall, two wooden skiffs moored beside dark pilings",
            category="vessel",
            seed=1,
        )
        # Missing prefix
        self._write_record(
            "r_no_prefix",
            "item_02.png",
            "timber hull, coiled rope, pale stone wall, low barge resting along wet shingle bank",
            category="vessel",
            seed=2,
        )
        # Missing required term (class resolved from filename pattern archway_03.png)
        self._write_record(
            "r_no_term",
            "archway_03.png",
            "etching plate, pale stone wall, narrow cobbled passage beneath weathered iron lantern",
            category=None,
            seed=3,
        )
        # Banned word for class ("portrait" banned on vessel + "neon" banned on backdrop)
        self._write_record(
            "r_banned",
            "item_04.png",
            "etching plate, timber hull, coiled rope, pale stone wall, portrait beneath neon harbor sign",
            category="vessel",
            seed=4,
        )
        # Exact duplicate pair
        dup_text = "etching plate, masonry arch, pale stone wall, twin stone piers beside quiet courtyard steps"
        self._write_record("r_dup_a", "archway_05.png", dup_text, seed=5)
        self._write_record("r_dup_b", "archway_06.png", dup_text, seed=6)
        # Near duplicate pair (Jaccard >= 0.85)
        self._write_record(
            "r_near_a",
            "archway_07.png",
            "etching plate, masonry arch, pale stone wall, weathered granite bridge spanning narrow canal water",
            seed=7,
        )
        self._write_record(
            "r_near_b",
            "archway_08.png",
            "etching plate, masonry arch, pale stone wall, weathered granite bridge spanning narrow canal",
            seed=8,
        )

        audit = description_gate.audit_project(self.proj)
        by_id = {r["id"]: r for r in audit["records"]}
        self.assertTrue(by_id["r_ok"]["passed_model_free"], by_id["r_ok"]["flags"])
        self.assertIn("prefix_missing", by_id["r_no_prefix"]["flags"])
        self.assertIn("required_term_missing", by_id["r_no_term"]["flags"])
        self.assertEqual(by_id["r_no_term"]["caption_class"], "archway")
        self.assertIn("banned_word", by_id["r_banned"]["flags"])
        self.assertIn("exact_duplicate", by_id["r_dup_a"]["flags"])
        self.assertIn("exact_duplicate", by_id["r_dup_b"]["flags"])
        self.assertIn("near_duplicate", by_id["r_near_a"]["flags"])
        self.assertIn("near_duplicate", by_id["r_near_b"]["flags"])

        jsonl_path, _ws_path = description_gate.write_review_files(self.proj, self.out, audit)
        with open(jsonl_path, encoding="utf-8") as f:
            review_rows = {row["id"]: row for row in (json.loads(line) for line in f if line.strip())}
        self.assertIn("prefix_missing", review_rows["r_no_prefix"]["flags"])
        self.assertIn("required_term_missing", review_rows["r_no_term"]["flags"])
        self.assertIn("banned_word", review_rows["r_banned"]["flags"])
        self.assertIn("exact_duplicate", review_rows["r_dup_a"]["flags"])
        self.assertIn("near_duplicate", review_rows["r_near_a"]["flags"])

    def test_caption_rules_validation_errors(self):
        with self.assertRaises(DeskError) as cm_missing:
            description_gate.load_caption_rules(config_path=os.path.join(self.tmp, "missing.json"))
        self.assertEqual(cm_missing.exception.code, "INPUT_BAD_ARGUMENT")

        for bad_doc in (
            {},
            {"prefix": "   "},
            {"prefix": "etching,", "class_pattern": "[unclosed"},
            {"prefix": "etching,", "classes": {"a": {"required_terms": "not-a-list"}}},
        ):
            with self.assertRaises(DeskError) as cm_bad:
                description_gate.coerce_caption_rules(bad_doc)
            self.assertEqual(cm_bad.exception.code, "INPUT_BAD_ARGUMENT")


class CompositionAndProvenanceTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name
        self.proj = os.path.join(self.tmp, "lora-set")
        self.out = os.path.join(self.tmp, "out")
        os.makedirs(os.path.join(self.proj, "records"), exist_ok=True)
        os.makedirs(os.path.join(self.proj, "outputs", "w1"), exist_ok=True)
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)

        # Multi-line config with known 1-based line numbers
        self.config_lines = [
            "{",                                                                              # line 1
            '  "prefix": "etching plate,",',                                                  # line 2
            '  "drafter": "qwen3-vl",',                                                       # line 3
            '  "drafter_prompt": "describe visible pose, props, and floor only",',            # line 4
            '  "class_field": "canon.category",',                                             # line 5
            '  "backdrop": {',                                                                # line 6
            '    "text": "plain parchment ground",',                                          # line 7
            '    "field": "canon.backdrop"',                                                  # line 8
            "  },",                                                                           # line 9
            '  "classes": {',                                                                 # line 10
            '    "vessel": {',                                                                # line 11
            '      "required_terms": [',                                                      # line 12
            '        "timber hull",',                                                         # line 13
            '        "coiled rope"',                                                          # line 14
            "      ],",                                                                       # line 15
            '      "banned_words": ["neon"]',                                                 # line 16
            "    }",                                                                          # line 17
            "  }",                                                                            # line 18
            "}",                                                                              # line 19
        ]
        with open(os.path.join(self.proj, "caption-rules.json"), "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(self.config_lines) + "\n")

        self.sha_v1 = _write_png(os.path.join(self.proj, "outputs/w1/vessel_01.png"), (150, 60, 40), seed=11)
        self.sha_v2 = _write_png(os.path.join(self.proj, "outputs/w1/vessel_02.png"), (40, 150, 60), seed=12)

        for rid, fname, backdrop_override, explanation in (
            ("v1", "vessel_01.png", None, "short bad"),
            ("v2", "vessel_02.png", "clouded harbour sky", "short bad"),
        ):
            canon: dict = {"subject": "secret subject phrase here", "category": "vessel"}
            if backdrop_override:
                canon["backdrop"] = backdrop_override
            rec = {
                "id": rid,
                "asset_path": f"outputs/w1/{fname}",
                "canon": canon,
                "judgment": {"status": "approved", "explanation": explanation},
                "provenance": {"prompt": "secret generation prompt text never leaked"},
            }
            with open(os.path.join(self.proj, "records", f"{rid}.json"), "w", encoding="utf-8") as f:
                json.dump(rec, f)

    def test_compose_writes_lane3_manifest_and_per_part_provenance_without_touching_project(self):
        before_snap = _snapshot_tree(self.proj)

        # Refuses --out inside --project
        err_buf = io.StringIO()
        with contextlib.redirect_stderr(err_buf):
            rc_bad = desk.run(["compose", "--project", self.proj, "--out", os.path.join(self.proj, "sub")])
        self.assertEqual(rc_bad, EXIT_USER)
        self.assertIn("PERM_OUT_INSIDE_PROJECT", err_buf.getvalue())
        self.assertEqual(_snapshot_tree(self.proj), before_snap)

        # Step 1: report + draft with fake Qwen3-VL stand-in, then import a seat edit on v2 only
        replies = iter([
            "<|im_start|>assistant\n  broad wooden skiff resting beside stone pilings at low tide  <|im_end|>",
            "<|im_start|>assistant\n  twin-masted barque anchored near weathered timber quay  <|im_end|>",
        ])

        class _SeqQwenProc(_FakeQwenProcessor):
            def __init__(self):
                super().__init__()
                self._last = ""

            def batch_decode(self, ids, skip_special_tokens=False):
                if not skip_special_tokens:
                    self._last = next(replies)
                    return [self._last]
                return [self._last.replace("<|im_start|>assistant\n", "").replace("<|im_end|>", "")]

        cfg = types.SimpleNamespace(
            _commit_hash="0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
            _name_or_path="Qwen/Qwen3-VL-8B-Instruct",
        )
        tf = fakes.make_transformers(processor=_SeqQwenProc(), model=fakes.FakeModel(config=cfg))

        with fakes.fake_modules(torch=fakes.make_torch(), transformers=tf), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(desk.run(["index", "--project", self.proj, "--out", self.out, "--backend", "none"]), EXIT_OK)
            self.assertEqual(
                desk.run(["report", "--project", self.proj, "--out", self.out, "--backend", "none", "--draft", "qwen3-vl"]),
                EXIT_OK,
            )

        # Edit row v2 on the blind worksheet and import with seat:studio-1
        ws_path = os.path.join(self.out, "description-worksheet.md")
        with open(ws_path, encoding="utf-8") as f:
            ws_lines = f.read().splitlines()
        edited_lines = []
        for ln in ws_lines:
            if ln.startswith("| `v2` |"):
                cells = ln.split("|")
                # cells: ['', ' `v2` ', target, sibs, instr, author, draft, edited_text, seat_id, '']
                cells[7] = " tall rigged barque moored beside stone quay bollards "
                ln = "|".join(cells)
            edited_lines.append(ln)
        with open(ws_path, "w", encoding="utf-8") as f:
            f.write("\n".join(edited_lines) + "\n")

        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(
                desk.run(["import-edits", "--project", self.proj, "--out", self.out, "--seat", "studio-1"]),
                EXIT_OK,
            )

        # Compose using cached drafts + imported seat edit (--draft none so v1 uses its cached Qwen3-VL draft)
        out_buf = io.StringIO()
        with contextlib.redirect_stdout(out_buf):
            rc_comp = desk.run(["compose", "--project", self.proj, "--out", self.out, "--draft", "none"])
        self.assertEqual(rc_comp, EXIT_OK)

        manifest_path = out_buf.getvalue().strip()
        self.assertEqual(manifest_path, os.path.join(self.out, "captions-manifest.json"))
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)

        items = {it["id"]: it for it in manifest["items"]}
        self.assertEqual(set(items), {"v1", "v2"})

        expected_psha = description_gate.prompt_sha256("describe visible pose, props, and floor only")
        expected_drafter_writer = (
            f"drafter:Qwen/Qwen3-VL-8B-Instruct@0c351dd01ed87e9c1b53cbc748cba10e6187ff3b:{expected_psha}"
        )

        # v1: unedited drafter caption -> captioner_draft_unedited
        it_v1 = items["v1"]
        self.assertEqual(it_v1["image"], "vessel_01.png")
        self.assertEqual(it_v1["sha256"], self.sha_v1)
        self.assertEqual(
            it_v1["caption"],
            "etching plate, timber hull, coiled rope, plain parchment ground, "
            "broad wooden skiff resting beside stone pilings at low tide",
        )
        self.assertEqual(it_v1["author_kind"], "captioner_draft_unedited")
        self.assertEqual(
            it_v1["writers"],
            [
                "config:caption-rules.json:2",
                "config:caption-rules.json:13",
                "config:caption-rules.json:14",
                "config:caption-rules.json:7",
                expected_drafter_writer,
            ],
        )

        # v2: seat-edited drafter caption + record-field backdrop -> captioner_assisted
        it_v2 = items["v2"]
        self.assertEqual(it_v2["image"], "vessel_02.png")
        self.assertEqual(it_v2["sha256"], self.sha_v2)
        self.assertEqual(
            it_v2["caption"],
            "etching plate, timber hull, coiled rope, clouded harbour sky, "
            "tall rigged barque moored beside stone quay bollards",
        )
        self.assertEqual(it_v2["author_kind"], "captioner_assisted")
        self.assertEqual(
            it_v2["writers"],
            [
                "config:caption-rules.json:2",
                "config:caption-rules.json:13",
                "config:caption-rules.json:14",
                "record:records/v2.json",
                "seat:studio-1",
            ],
        )
        self.assertEqual(it_v2["drafter_writer"], expected_drafter_writer)

        # Review queue carries identical provenance and strata, and both composed captions pass the model-free gate
        with open(os.path.join(self.out, "description-review.jsonl"), encoding="utf-8") as f:
            rev_rows = {r["id"]: r for r in (json.loads(line) for line in f if line.strip())}
        self.assertEqual(rev_rows["v1"]["author_kind"], "captioner_draft_unedited")
        self.assertEqual(rev_rows["v1"]["writers"], it_v1["writers"])
        self.assertTrue(rev_rows["v1"]["passed_model_free"], rev_rows["v1"]["flags"])
        self.assertEqual(rev_rows["v2"]["author_kind"], "captioner_assisted")
        self.assertEqual(rev_rows["v2"]["writers"], it_v2["writers"])
        self.assertTrue(rev_rows["v2"]["passed_model_free"], rev_rows["v2"]["flags"])

        # Raw drafter output was saved per image to drafter-raw.jsonl
        raw_path = os.path.join(self.out, "drafter-raw.jsonl")
        self.assertTrue(os.path.isfile(raw_path))
        with open(raw_path, encoding="utf-8") as f:
            raw_rows = {r["id"]: r for r in (json.loads(line) for line in f if line.strip())}
        self.assertIn("<|im_start|>assistant", raw_rows["v1"]["raw_output"])
        self.assertEqual(raw_rows["v1"]["drafter_prompt"], "describe visible pose, props, and floor only")

        # Project folder untouched byte-for-byte
        self.assertEqual(_snapshot_tree(self.proj), before_snap)

    def test_compose_with_record_fallback_and_report_compose_flag(self):
        # Update v1 to already have a passing human free description and run compose with --draft none
        rec_v1_path = os.path.join(self.proj, "records", "v1.json")
        with open(rec_v1_path, encoding="utf-8") as f:
            v1_doc = json.load(f)
        v1_doc["judgment"]["explanation"] = "weathered fishing lugger beached beside wooden capstan"
        with open(rec_v1_path, "w", encoding="utf-8") as f:
            json.dump(v1_doc, f)

        comp = description_gate.compose_captions(self.proj, self.out, drafter=None)
        by_id = {it["id"]: it for it in comp["items"]}
        self.assertEqual(by_id["v1"]["author_kind"], "human")
        self.assertEqual(by_id["v1"]["writers"][-1], "record:records/v1.json")

        # Also exercise `report --compose --draft fake`
        out2 = os.path.join(self.tmp, "out2")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(desk.run(["index", "--project", self.proj, "--out", out2, "--backend", "none"]), EXIT_OK)
            self.assertEqual(
                desk.run([
                    "report", "--project", self.proj, "--out", out2, "--backend", "none",
                    "--draft", "fake", "--compose",
                ]),
                EXIT_OK,
            )
        self.assertTrue(os.path.isfile(os.path.join(out2, "captions-manifest.json")))


class BlindWorksheetAndReimportTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name
        # Use locked pool name `salt-road` to verify image-only sibling selection and pool lock enforcement
        self.proj = os.path.join(self.tmp, "salt-road")
        self.out = os.path.join(self.tmp, "out")
        os.makedirs(os.path.join(self.proj, "records"), exist_ok=True)
        os.makedirs(os.path.join(self.proj, "outputs", "w1"), exist_ok=True)
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)

    def test_blind_worksheet_uses_dinov2_and_dhash_siblings_hides_prompts_and_reimports_seat(self):
        items_list: list[desk.Item] = []
        vecs: list[np.ndarray] = []
        # Target image sr_0 plus 6 siblings sr_1..sr_6
        # Give sr_4, sr_5, sr_6 highest DINOv2 image-only cosine to sr_0, while sr_1..sr_3 have smaller dHash distance
        raw_dirs = [
            [1.0, 0.0, 0.0, 0.0],   # sr_0 target
            [0.0, 1.0, 0.0, 0.0],   # sr_1 orthogonal in DINOv2
            [0.0, 0.0, 1.0, 0.0],   # sr_2 orthogonal in DINOv2
            [0.0, 0.0, 0.0, 1.0],   # sr_3 orthogonal in DINOv2
            [0.99, 0.14, 0.0, 0.0], # sr_4 closest in DINOv2
            [0.96, 0.28, 0.0, 0.0], # sr_5 second in DINOv2
            [0.92, 0.39, 0.0, 0.0], # sr_6 third in DINOv2
        ]
        for idx, vec in enumerate(raw_dirs):
            rid = f"sr_{idx}"
            rel = f"outputs/w1/{rid}.png"
            sha = _write_png(os.path.join(self.proj, rel), (50 + idx * 20, 80, 110), seed=20 + idx)
            is_target = idx == 0
            explanation = (
                "old leaked bad caption phrase"
                if is_target
                else f"distinct approved human description number {idx} showing stone courtyard pillars and shadows"
            )
            rec = {
                "id": rid,
                "asset_path": rel,
                "canon": {"subject": f"secret canon subject phrase {idx}"},
                "judgment": {"status": "approved", "explanation": explanation},
                "provenance": {"prompt": f"secret generation prompt phrase {idx}"},
            }
            with open(os.path.join(self.proj, "records", f"{rid}.json"), "w", encoding="utf-8") as f:
                json.dump(rec, f)
            dh = "0" * 64 if idx <= 3 else "f" * 64
            items_list.append(desk.Item(
                sha256=sha,
                paths=[rel],
                folders=["outputs/w1"],
                dhash=dh,
                dhash_flip=dh,
                record={"id": rid, "status": "approved", "explanation": explanation, "recorded": True},
            ))
            v = np.asarray(vec, dtype=np.float32)
            vecs.append(v / np.linalg.norm(v))

        img_mat = np.vstack(vecs)
        meta = {
            "backend": "dinov2",
            "revision": registry.REGISTRY["dinov2"].revision,
            "dim": 4,
            "files": 7,
            "unique": 7,
            "records": 7,
            "created": "2026-11-11T00:00:00Z",
            "images_only": True,
        }
        desk.save_cache(self.out, "dinov2", items_list, img_mat, {}, meta)

        # Locked pool refuses multimodal text-embedding report on salt-road without --images-only
        with self.assertRaises(DeskError) as cm_lock:
            desk.check_multimodal_pool(self.proj, "eg2", images_only=False, allow_multimodal_pool=False)
        self.assertEqual(cm_lock.exception.code, "PERM_LOCKED_POOL")

        # Report with --backend dinov2 and --draft fake succeeds on locked salt-road pool
        with contextlib.redirect_stdout(io.StringIO()):
            rc = desk.run([
                "report", "--project", self.proj, "--out", self.out, "--backend", "dinov2",
                "--draft", "fake", "--drafter-prompt", "secret drafter instruction never in worksheet",
            ])
        self.assertEqual(rc, EXIT_OK)

        ws_path = os.path.join(self.out, "description-worksheet.md")
        with open(ws_path, encoding="utf-8") as f:
            ws_text = f.read()

        # Blind worksheet hides generation prompt, drafter prompt, canon.subject, and old caption
        self.assertNotIn("secret generation prompt phrase", ws_text)
        self.assertNotIn("secret drafter instruction", ws_text)
        self.assertNotIn("secret canon subject phrase", ws_text)
        self.assertNotIn("old leaked bad caption phrase", ws_text)

        # Shows draft beside 3-5 nearest DINOv2 image-only siblings (sr_4, sr_5, sr_6 ranked ahead of sr_1..sr_3)
        self.assertIn("dHash/DINOv2 image-only", ws_text)
        self.assertIn("seat_id", ws_text)
        sr0_line = next(ln for ln in ws_text.splitlines() if ln.startswith("| `sr_0` |"))
        self.assertLess(sr0_line.index("sr_4.png"), sr0_line.index("sr_1.png"))
        self.assertLess(sr0_line.index("sr_5.png"), sr0_line.index("sr_1.png"))
        self.assertLess(sr0_line.index("sr_6.png"), sr0_line.index("sr_1.png"))

        # Fill edited text (including an escaped pipe) WITHOUT seat_id and verify missing --seat is refused
        cells = sr0_line.split("|")
        cells[7] = " stone courtyard archway \\| weathered timber bench "
        edited_ws = ws_text.replace(sr0_line, "|".join(cells))
        with open(ws_path, "w", encoding="utf-8") as f:
            f.write(edited_ws)

        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(desk.run(["import-edits", "--project", self.proj, "--out", self.out]), EXIT_USER)

        # Now import with --seat seat-alpha
        out_imp = io.StringIO()
        with contextlib.redirect_stdout(out_imp):
            rc_imp = desk.run(["import-edits", "--project", self.proj, "--out", self.out, "--seat", "seat-alpha"])
        self.assertEqual(rc_imp, EXIT_OK)
        imp_summary = json.loads(out_imp.getvalue())
        self.assertEqual(imp_summary["imported"], 1)

        loaded_edits = description_gate.load_seat_edits(self.out)
        self.assertEqual(loaded_edits["sr_0"]["edited_text"], "stone courtyard archway | weathered timber bench")
        self.assertEqual(loaded_edits["sr_0"]["writer"], "seat:seat-alpha")


if __name__ == "__main__":
    unittest.main()
