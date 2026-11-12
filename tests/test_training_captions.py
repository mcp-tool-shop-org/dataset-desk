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
import re
import subprocess
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
from PIL import Image, PngImagePlugin

from dataset_desk import backends, description_gate, desk, registry
from dataset_desk.errors import EXIT_OK, EXIT_USER, DeskError

try:
    import fakes
except ImportError:
    from tests import fakes


def _write_png(
    path: str,
    rgb: tuple[int, int, int],
    seed: int = 1,
    *,
    comfy_prompt: str | None = None,
) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rng = np.random.default_rng(seed)
    base = np.array(rgb, dtype=np.int32)
    arr = np.clip(base + rng.integers(-15, 16, size=(16, 16, 3)), 0, 255).astype(np.uint8)
    pnginfo = None
    if comfy_prompt is not None:
        pnginfo = PngImagePlugin.PngInfo()
        graph = {"1": {"class_type": "CLIPTextEncode", "inputs": {"text": comfy_prompt}}}
        pnginfo.add_text("prompt", json.dumps(graph))
    Image.fromarray(arr, mode="RGB").save(path, pnginfo=pnginfo)
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


class DrafterSettingTests(fakes.IsolatedDeskTestCase):
    def setUp(self):
        super().setUp()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name

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


class CaptionRulesGateTests(fakes.IsolatedDeskTestCase):
    def setUp(self):
        super().setUp()
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


class CompositionAndProvenanceTests(fakes.IsolatedDeskTestCase):
    def setUp(self):
        super().setUp()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name
        self.proj = os.path.join(self.tmp, "lora-set")
        self.out = os.path.join(self.tmp, "out")
        os.makedirs(os.path.join(self.proj, "records"), exist_ok=True)
        os.makedirs(os.path.join(self.proj, "outputs", "w1"), exist_ok=True)

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

        # v1: flagged record with NO seat edit -> draft stays in review queue, caption keeps record text
        it_v1 = items["v1"]
        self.assertEqual(it_v1["image"], "vessel_01.png")
        self.assertEqual(it_v1["sha256"], self.sha_v1)
        self.assertEqual(
            it_v1["caption"],
            "etching plate, timber hull, coiled rope, plain parchment ground, short bad",
        )
        self.assertEqual(it_v1["author_kind"], "human")
        self.assertEqual(
            it_v1["writers"],
            [
                "config:caption-rules.json:2",
                "config:caption-rules.json:13",
                "config:caption-rules.json:14",
                "config:caption-rules.json:7",
                "record:records/v1.json",
            ],
        )
        self.assertIsNone(it_v1["drafter_writer"])

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

        # Review queue queues v1's draft for the seat (still flagged) while v2 passes after seat edit
        with open(os.path.join(self.out, "description-review.jsonl"), encoding="utf-8") as f:
            rev_rows = {r["id"]: r for r in (json.loads(line) for line in f if line.strip())}
        self.assertEqual(rev_rows["v1"]["author_kind"], "captioner_assisted")
        self.assertEqual(
            rev_rows["v1"]["draft"],
            "broad wooden skiff resting beside stone pilings at low tide",
        )
        self.assertEqual(rev_rows["v1"]["drafter_writer"], expected_drafter_writer)
        self.assertEqual(rev_rows["v1"]["writers"], [*it_v1["writers"], expected_drafter_writer])
        self.assertFalse(rev_rows["v1"]["passed_model_free"])
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
        v1_doc["judgment"]["explanation"] = "weathered fishing lugger beached beside wooden capstan post"
        with open(rec_v1_path, "w", encoding="utf-8") as f:
            json.dump(v1_doc, f)

        comp = description_gate.compose_captions(self.proj, self.out, drafter=None)
        by_id = {it["id"]: it for it in comp["items"]}
        self.assertEqual(by_id["v1"]["author_kind"], "human")
        self.assertEqual(by_id["v1"]["writers"][-1], "record:records/v1.json")

        # Also exercise `report --compose --draft fake` (unflagged v1 stays human, untouched)
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
        with open(os.path.join(out2, "captions-manifest.json"), encoding="utf-8") as f:
            mf2 = json.load(f)
        by_id2 = {it["id"]: it for it in mf2["items"]}
        self.assertEqual(by_id2["v1"]["author_kind"], "human")
        self.assertEqual(by_id2["v1"]["writers"][-1], "record:records/v1.json")


class BlindWorksheetAndReimportTests(fakes.IsolatedDeskTestCase):
    def setUp(self):
        super().setUp()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name
        # Use locked pool name `salt-road` to verify image-only sibling selection and pool lock enforcement
        self.proj = os.path.join(self.tmp, "salt-road")
        self.out = os.path.join(self.tmp, "out")
        os.makedirs(os.path.join(self.proj, "records"), exist_ok=True)
        os.makedirs(os.path.join(self.proj, "outputs", "w1"), exist_ok=True)

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


class Addendum1SidecarAndPoolLockTests(fakes.IsolatedDeskTestCase):
    """Synthetic tests for Addendum 1 (Parts A-D):
    1. Tagged copy refused (`multimodal_pool: <name>`);
    2. Renamed copy WITHOUT tag and with matching image sha256s refused fail-closed;
    3. Renamed copy without tag and with no manifest configured emits explicit 'detection off' warning;
    4. Sidecar import: pairs, missing sidecars (reported, never invented), gate rules, and `sidecar:<file>` provenance.
    """

    def setUp(self):
        super().setUp()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name

    def test_tagged_copy_is_refused(self):
        # Renamed training copy carrying explicit `multimodal_pool: "salt-road"` tag in caption-rules.json
        tagged_dir = os.path.join(self.tmp, "lora-training-copy")
        out_dir = os.path.join(self.tmp, "out-tagged")
        os.makedirs(os.path.join(tagged_dir, "outputs", "w1"), exist_ok=True)
        _write_png(os.path.join(tagged_dir, "outputs", "w1", "img_01.png"), (90, 110, 130), seed=31)
        with open(os.path.join(tagged_dir, "caption-rules.json"), "w", encoding="utf-8") as f:
            json.dump({"multimodal_pool": "salt-road", "prefix": "etching plate,"}, f)

        # Refuses multimodal description embeddings and arm-model acceptance on the tagged copy
        with self.assertRaises(DeskError) as cm_eg2:
            desk.check_multimodal_pool(tagged_dir, "eg2", images_only=False, allow_multimodal_pool=False)
        self.assertEqual(cm_eg2.exception.code, "PERM_LOCKED_POOL")
        self.assertIn("salt-road", cm_eg2.exception.message)

        with self.assertRaises(DeskError) as cm_arm_s:
            desk.ai_eyes_verify(tagged_dir, {"img_01": "synthetic description text"})
        self.assertEqual(cm_arm_s.exception.code, "PERM_LOCKED_POOL")

        with self.assertRaises(DeskError) as cm_arm_g:
            desk.eg2_fit_accept(tagged_dir, {"img_01": "synthetic description text"})
        self.assertEqual(cm_arm_g.exception.code, "PERM_LOCKED_POOL")

        err_cli = io.StringIO()
        with contextlib.redirect_stderr(err_cli):
            rc = desk.run(["index", "--project", tagged_dir, "--out", out_dir, "--backend", "siglip2"])
        self.assertEqual(rc, EXIT_USER)
        self.assertIn("PERM_LOCKED_POOL", err_cli.getvalue())
        self.assertIn("salt-road", err_cli.getvalue())

        # Tag-only project.json (`multimodal_pool` without `prefix`) is also refused
        proj_json_dir = os.path.join(self.tmp, "another-renamed-copy")
        os.makedirs(proj_json_dir, exist_ok=True)
        with open(os.path.join(proj_json_dir, "project.json"), "w", encoding="utf-8") as f:
            json.dump({"multimodal_pool": "tallow-fen"}, f)
        with self.assertRaises(DeskError) as cm_pjson:
            desk.check_multimodal_pool(proj_json_dir, "siglip2", images_only=False)
        self.assertEqual(cm_pjson.exception.code, "PERM_LOCKED_POOL")
        self.assertIn("tallow-fen", cm_pjson.exception.message)

        # Unknown pool tag is rejected as INPUT_BAD_ARGUMENT (typos never silently unlock)
        typo_dir = os.path.join(self.tmp, "typo-copy")
        os.makedirs(typo_dir, exist_ok=True)
        with open(os.path.join(typo_dir, "project.json"), "w", encoding="utf-8") as f:
            json.dump({"multimodal_pool": "not-a-valid-pool"}, f)
        with self.assertRaises(DeskError) as cm_typo:
            desk.check_multimodal_pool(typo_dir, "none")
        self.assertEqual(cm_typo.exception.code, "INPUT_BAD_ARGUMENT")

        # Image-only (`--backend none`, `dinov2`, `--images-only`) and `--allow-multimodal-pool` remain permitted
        desk.check_multimodal_pool(tagged_dir, "none")
        desk.check_multimodal_pool(tagged_dir, "dinov2")
        desk.check_multimodal_pool(tagged_dir, "eg2", images_only=True)
        desk.check_multimodal_pool(tagged_dir, "eg2", allow_multimodal_pool=True)

    def test_untagged_renamed_copy_with_matching_shas_is_refused_fail_closed(self):
        # Synthetic locked pool folder `salt-road` with 3 images
        locked_sr = os.path.join(self.tmp, "salt-road")
        sha_1 = _write_png(os.path.join(locked_sr, "outputs", "w1", "sr_01.png"), (120, 40, 60), seed=41)
        sha_2 = _write_png(os.path.join(locked_sr, "outputs", "w1", "sr_02.png"), (40, 120, 60), seed=42)
        sha_3 = _write_png(os.path.join(locked_sr, "outputs", "w1", "sr_03.png"), (60, 40, 120), seed=43)

        # Untagged renamed copy containing 2 byte-identical images from `salt-road` + 1 unrelated image
        untagged_copy = os.path.join(self.tmp, "renamed-training-copy")
        os.makedirs(untagged_copy, exist_ok=True)
        for src_name, dst_name in (("sr_01.png", "train_a.png"), ("sr_02.png", "train_b.png")):
            with open(os.path.join(locked_sr, "outputs", "w1", src_name), "rb") as fsrc:
                data = fsrc.read()
            with open(os.path.join(untagged_copy, dst_name), "wb") as fdst:
                fdst.write(data)
        _write_png(os.path.join(untagged_copy, "train_c.png"), (200, 180, 50), seed=44)

        # 1. Local manifest file via DATASET_DESK_POOL_MANIFEST env var (outside repo)
        manifest_file = os.path.join(self.tmp, "local-pool-hashes.json")
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump({"salt-road": [sha_1, sha_2, sha_3]}, f)
        os.environ["DATASET_DESK_POOL_MANIFEST"] = manifest_file

        # Refused fail-closed even when backend == "none" (lite pass)!
        with self.assertRaises(DeskError) as cm_env:
            desk.check_multimodal_pool(untagged_copy, "none")
        self.assertEqual(cm_env.exception.code, "PERM_LOCKED_POOL")
        self.assertIn("salt-road", cm_env.exception.message)
        self.assertIn("2 matching images", cm_env.exception.message)

        err_idx = io.StringIO()
        with contextlib.redirect_stderr(err_idx):
            rc_idx = desk.run([
                "index", "--project", untagged_copy, "--out", os.path.join(self.tmp, "out-u1"),
                "--backend", "none", "--sidecars",
            ])
        self.assertEqual(rc_idx, EXIT_USER)
        self.assertIn("PERM_LOCKED_POOL", err_idx.getvalue())
        self.assertIn("salt-road", err_idx.getvalue())
        self.assertIn("2 matching images", err_idx.getvalue())

        # 2. User config JSON via DATASET_DESK_USER_CONFIG
        os.environ.pop("DATASET_DESK_POOL_MANIFEST", None)
        user_cfg = os.path.join(self.tmp, "user-config.json")
        with open(user_cfg, "w", encoding="utf-8") as f:
            json.dump({"pool_manifest": "local-pool-hashes.json"}, f)
        os.environ["DATASET_DESK_USER_CONFIG"] = user_cfg

        with self.assertRaises(DeskError) as cm_ucfg:
            desk.check_multimodal_pool(untagged_copy, "dinov2", images_only=True)
        self.assertEqual(cm_ucfg.exception.code, "PERM_LOCKED_POOL")
        self.assertIn("salt-road", cm_ucfg.exception.message)
        self.assertIn("2 matching images", cm_ucfg.exception.message)

        # 3. Hashes computed directly from the locked project's own folder via --locked-pool-dir
        os.environ["DATASET_DESK_USER_CONFIG"] = ""
        err_dir = io.StringIO()
        with contextlib.redirect_stderr(err_dir):
            rc_dir = desk.run([
                "sidecar-import", "--project", untagged_copy, "--out", os.path.join(self.tmp, "out-u2"),
                "--locked-pool-dir", locked_sr,
            ])
        self.assertEqual(rc_dir, EXIT_USER)
        self.assertIn("PERM_LOCKED_POOL", err_dir.getvalue())
        self.assertIn("salt-road", err_dir.getvalue())
        self.assertIn("2 matching images", err_dir.getvalue())

    def test_untagged_renamed_copy_without_manifest_emits_detection_off_warning(self):
        untagged_dir = os.path.join(self.tmp, "untagged-without-manifest")
        out_dir = os.path.join(self.tmp, "out-nomf")
        _write_png(os.path.join(untagged_dir, "sample_01.png"), (80, 140, 100), seed=51)
        with open(os.path.join(untagged_dir, "sample_01.txt"), "w", encoding="utf-8") as f:
            f.write("carved wooden door set inside weathered limestone archway beside stone steps")

        # Ensure zero manifest/folder sources are configured
        os.environ.pop("DATASET_DESK_POOL_MANIFEST", None)
        os.environ.pop("DATASET_DESK_POOL_DIRS", None)
        os.environ["DATASET_DESK_USER_CONFIG"] = ""

        err_direct = io.StringIO()
        with contextlib.redirect_stderr(err_direct):
            desk.check_multimodal_pool(untagged_dir, "none")
        self.assertIn("renamed-copy detection is off", err_direct.getvalue())
        self.assertIn("detection off", err_direct.getvalue())

        err_cli = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err_cli):
            rc = desk.run(["index", "--project", untagged_dir, "--out", out_dir, "--backend", "none", "--sidecars"])
        self.assertEqual(rc, EXIT_OK)
        self.assertIn("renamed-copy detection is off", err_cli.getvalue())
        self.assertEqual(err_cli.getvalue().count("renamed-copy detection is off"), 1)

    def test_sidecar_import_pairs_missing_sidecars_gate_and_provenance_source(self):
        sidecar_proj = os.path.join(self.tmp, "sidecar-set")
        os.makedirs(sidecar_proj, exist_ok=True)

        rules_lines = [
            "{",                                                        # line 1
            '  "prefix": "woodcut print,",',                            # line 2
            '  "drafter": "none",',                                     # line 3
            '  "class_pattern": "^(?P<class>[a-z]+)_[0-9]+\\\\.png$",', # line 4
            '  "classes": {',                                           # line 5
            '    "harbor": {',                                          # line 6
            '      "required_terms": ["timber pier"],',                 # line 7
            '      "banned_words": ["neon"]',                           # line 8
            "    }",                                                    # line 9
            "  }",                                                      # line 10
            "}",                                                        # line 11
        ]
        with open(os.path.join(sidecar_proj, "caption-rules.json"), "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(rules_lines) + "\n")

        # 1. Clean paired sidecar
        sha_h1 = _write_png(os.path.join(sidecar_proj, "harbor_01.png"), (100, 120, 140), seed=61)
        with open(os.path.join(sidecar_proj, "harbor_01.txt"), "w", encoding="utf-8") as f:
            f.write("woodcut print, timber pier, two moored skiffs resting beside coiled rope and stone steps")

        # 2. Paired sidecar echoing embedded PNG generation prompt + missing prefix & term + banned word
        embedded_prompt = "moonlit quay wooden wharf ink wash crosshatching study"
        _write_png(
            os.path.join(sidecar_proj, "harbor_02.png"),
            (120, 90, 110),
            seed=62,
            comfy_prompt=embedded_prompt,
        )
        with open(os.path.join(sidecar_proj, "harbor_02.txt"), "w", encoding="utf-8") as f:
            f.write("moonlit quay wooden wharf ink wash crosshatching beside neon lantern")

        # 3 & 4. Duplicate paired sidecars
        dup_sidecar = "woodcut print, timber pier, narrow stone quay beside calm dark water at low tide"
        _write_png(os.path.join(sidecar_proj, "harbor_03.png"), (80, 130, 95), seed=63)
        with open(os.path.join(sidecar_proj, "harbor_03.txt"), "w", encoding="utf-8") as f:
            f.write(dup_sidecar)
        _write_png(os.path.join(sidecar_proj, "harbor_04.png"), (85, 125, 100), seed=64)
        with open(os.path.join(sidecar_proj, "harbor_04.txt"), "w", encoding="utf-8") as f:
            f.write(dup_sidecar)

        # 5. Unpaired image WITHOUT a .txt sidecar (must be reported, never invented)
        _write_png(os.path.join(sidecar_proj, "harbor_05.png"), (150, 110, 70), seed=65)

        before_snap = _snapshot_tree(sidecar_proj)

        # Refuses --out inside --project
        err_inside = io.StringIO()
        with contextlib.redirect_stderr(err_inside):
            rc_bad = desk.run([
                "sidecar-import", "--project", sidecar_proj, "--out", os.path.join(sidecar_proj, "inside"),
            ])
        self.assertEqual(rc_bad, EXIT_USER)
        self.assertIn("PERM_OUT_INSIDE_PROJECT", err_inside.getvalue())

        # Flow A: `sidecar-import` CLI command
        out_imp = os.path.join(self.tmp, "out-sidecar-import")
        stdout_imp = io.StringIO()
        with contextlib.redirect_stdout(stdout_imp), contextlib.redirect_stderr(io.StringIO()):
            rc_imp = desk.run(["sidecar-import", "--project", sidecar_proj, "--out", out_imp])
        self.assertEqual(rc_imp, EXIT_OK)
        imp_doc = json.loads(stdout_imp.getvalue())
        self.assertEqual(imp_doc["total_images"], 5)
        self.assertEqual(imp_doc["paired"], 4)
        self.assertEqual(imp_doc["images_without_sidecar"], ["harbor_05.png"])
        self.assertEqual(imp_doc["passed_model_free_count"], 1)
        self.assertEqual(imp_doc["flagged_count"], 3)

        with open(os.path.join(out_imp, "description-review.jsonl"), encoding="utf-8") as f:
            imp_rows = {r["id"]: r for r in (json.loads(line) for line in f if line.strip())}
        self.assertEqual(set(imp_rows), {"harbor_01", "harbor_02", "harbor_03", "harbor_04"})
        self.assertNotIn("harbor_05", imp_rows)
        self.assertEqual(imp_rows["harbor_01"]["source"], "sidecar:harbor_01.txt")
        self.assertEqual(imp_rows["harbor_01"]["author_kind"], "sidecar")
        self.assertEqual(imp_rows["harbor_01"]["writers"], ["sidecar:harbor_01.txt"])
        self.assertTrue(imp_rows["harbor_01"]["passed_model_free"], imp_rows["harbor_01"]["flags"])
        self.assertIn("prompt_echo", imp_rows["harbor_02"]["flags"])
        self.assertIn("prefix_missing", imp_rows["harbor_02"]["flags"])
        self.assertIn("required_term_missing", imp_rows["harbor_02"]["flags"])
        self.assertIn("banned_word", imp_rows["harbor_02"]["flags"])
        self.assertIn("exact_duplicate", imp_rows["harbor_03"]["flags"])
        self.assertIn("exact_duplicate", imp_rows["harbor_04"]["flags"])

        with open(os.path.join(out_imp, "description-worksheet.md"), encoding="utf-8") as f:
            ws_imp = f.read()
        self.assertIn("harbor_05.png", ws_imp)

        # Flow B: `index --sidecars` -> `report` -> `compose` (verifying Part 4 provenance `sidecar:<file>`)
        out_idx = os.path.join(self.tmp, "out-sidecar-index")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(
                desk.run(["index", "--project", sidecar_proj, "--out", out_idx, "--backend", "none", "--sidecars"]),
                EXIT_OK,
            )
            self.assertEqual(
                desk.run(["report", "--project", sidecar_proj, "--out", out_idx, "--backend", "none"]),
                EXIT_OK,
            )
            self.assertEqual(
                desk.run(["compose", "--project", sidecar_proj, "--out", out_idx, "--draft", "none"]),
                EXIT_OK,
            )

        with open(os.path.join(out_idx, "report-none.md"), encoding="utf-8") as f:
            rep_md = f.read()
        self.assertIn("image without a .txt sidecar: harbor_05.png", rep_md)

        with open(os.path.join(out_idx, "captions-manifest.json"), encoding="utf-8") as f:
            manifest = json.load(f)
        m_items = {it["id"]: it for it in manifest["items"]}
        self.assertEqual(set(m_items), {"harbor_01", "harbor_02", "harbor_03", "harbor_04"})
        self.assertNotIn("harbor_05", m_items)

        h1 = m_items["harbor_01"]
        self.assertEqual(h1["sha256"], sha_h1)
        self.assertEqual(h1["source"], "sidecar:harbor_01.txt")
        self.assertEqual(h1["author_kind"], "sidecar")
        self.assertEqual(
            h1["writers"],
            [
                "config:caption-rules.json:2",
                "config:caption-rules.json:7",
                "sidecar:harbor_01.txt",
            ],
        )
        self.assertEqual(h1["parts"][-1]["writer"], "sidecar:harbor_01.txt")

        with open(os.path.join(out_idx, "description-review.jsonl"), encoding="utf-8") as f:
            comp_rev = {r["id"]: r for r in (json.loads(line) for line in f if line.strip())}
        self.assertEqual(comp_rev["harbor_01"]["writers"], h1["writers"])
        self.assertEqual(comp_rev["harbor_01"]["parts"][-1]["writer"], "sidecar:harbor_01.txt")

        # Project folder untouched byte-for-byte
        self.assertEqual(_snapshot_tree(sidecar_proj), before_snap)


class Addendum2PoolDetectionStatusTests(fakes.IsolatedDeskTestCase):
    """Synthetic tests for Addendum 2 (Parts A-C):
    1. With nothing configured, each command's output file carries `status: "off"` and `stderr` carries the warning once;
    2. With a manifest or a pool dir configured, the output carries `status: "on"` with per-pool counts, and `stderr`
       has no warning;
    3. No locked-pool `sha256` value appears in any output file (and no `sha256` value appears in `pool_detection` or
       pool-detection reports).
    """

    def setUp(self):
        super().setUp()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name

    def _build_synthetic_projects(self, tag: str) -> tuple[str, str, str]:
        proj = os.path.join(self.tmp, f"proj-{tag}")
        os.makedirs(os.path.join(proj, "outputs", "w1"), exist_ok=True)
        os.makedirs(os.path.join(proj, "records"), exist_ok=True)
        _write_png(os.path.join(proj, "outputs", "w1", "arch_01.png"), (70, 105, 140), seed=71)
        _write_png(os.path.join(proj, "outputs", "w1", "arch_02.png"), (140, 95, 70), seed=72)

        with open(os.path.join(proj, "caption-rules.json"), "w", encoding="utf-8") as f:
            json.dump({"prefix": "copperplate study,", "drafter": "fake"}, f)

        rec_ok = {
            "id": "arch_01",
            "asset_path": "outputs/w1/arch_01.png",
            "judgment": {
                "status": "approved",
                "explanation": "copperplate study, weathered limestone arch above flagged stone steps",
            },
            "provenance": {"prompt": "distant ridge line"},
        }
        rec_flagged = {
            "id": "arch_02",
            "asset_path": "outputs/w1/arch_02.png",
            "judgment": {
                "status": "approved",
                "explanation": "carved wooden lintel beside narrow iron lantern",
            },
            "provenance": {"prompt": "harbor mist"},
        }
        with open(os.path.join(proj, "records", "arch_01.json"), "w", encoding="utf-8") as f:
            json.dump(rec_ok, f)
        with open(os.path.join(proj, "records", "arch_02.json"), "w", encoding="utf-8") as f:
            json.dump(rec_flagged, f)

        sidecar_proj = os.path.join(self.tmp, f"sidecar-{tag}")
        os.makedirs(sidecar_proj, exist_ok=True)
        _write_png(os.path.join(sidecar_proj, "plate_01.png"), (95, 130, 115), seed=73)
        with open(os.path.join(sidecar_proj, "plate_01.txt"), "w", encoding="utf-8") as f:
            f.write("copperplate study, carved timber gate set between mortared stone piers")
        with open(os.path.join(sidecar_proj, "caption-rules.json"), "w", encoding="utf-8") as f:
            json.dump({"prefix": "copperplate study,", "drafter": "none"}, f)

        subjects_file = os.path.join(self.tmp, f"subjects-{tag}.txt")
        with open(subjects_file, "w", encoding="utf-8") as f:
            f.write("limestone arch\ntimber gate\n")

        return proj, sidecar_proj, subjects_file

    @staticmethod
    def _fill_worksheet_edit(ws_path: str, row_id: str, edited_text: str) -> None:
        with open(ws_path, encoding="utf-8") as f:
            ws_text = f.read()
        row_line = next(ln for ln in ws_text.splitlines() if ln.startswith(f"| `{row_id}` |"))
        cells = row_line.split("|")
        cells[7] = f" {edited_text} "
        with open(ws_path, "w", encoding="utf-8") as f:
            f.write(ws_text.replace(row_line, "|".join(cells)))

    def test_nothing_configured_records_status_off_in_every_command_output_and_warns_once(self):
        proj, sidecar_proj, subjects_file = self._build_synthetic_projects("off")
        out_dir = os.path.join(self.tmp, "out-off")
        out_sc = os.path.join(self.tmp, "out-off-sc")

        def run_cmd(argv: list[str]) -> tuple[str, str]:
            out_buf, err_buf = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
                rc = desk.run(argv)
            self.assertEqual(rc, EXIT_OK, err_buf.getvalue())
            return out_buf.getvalue(), err_buf.getvalue()

        # 1. `index`: fails at 62a2f41 where `<out>/fake/meta.json` lacks `"pool_detection"`
        _o, err_idx = run_cmd(["index", "--project", proj, "--out", out_dir, "--backend", "fake"])
        self.assertEqual(err_idx.count("renamed-copy detection is off"), 1)
        with open(os.path.join(out_dir, "fake", "meta.json"), encoding="utf-8") as f:
            meta_doc = json.load(f)
        self.assertIn("pool_detection", meta_doc)
        self.assertEqual(meta_doc["pool_detection"]["status"], "off")
        self.assertIn("renamed-copy detection is off", meta_doc["pool_detection"]["warning"])

        # 2. `scan`: CLI command + `<out>/scan-report.json` + `<out>/pool-detection.json`
        _o, err_scan = run_cmd(["scan", "--project", proj, "--out", out_dir])
        self.assertEqual(err_scan.count("renamed-copy detection is off"), 1)
        with open(os.path.join(out_dir, "scan-report.json"), encoding="utf-8") as f:
            scan_doc = json.load(f)
        self.assertEqual(scan_doc["pool_detection"]["status"], "off")

        # 3. `dupes`: `<out>/dupes-fake.json`
        _o, err_dupes = run_cmd(["dupes", "--project", proj, "--out", out_dir, "--backend", "fake"])
        self.assertEqual(err_dupes.count("renamed-copy detection is off"), 1)
        with open(os.path.join(out_dir, "dupes-fake.json"), encoding="utf-8") as f:
            dupes_doc = json.load(f)
        self.assertEqual(dupes_doc["pool_detection"]["status"], "off")

        # 4. `search`: `<out>/search-fake.json`
        _o, err_search = run_cmd([
            "search", "--project", proj, "--out", out_dir, "--backend", "fake", "--text", "limestone arch",
        ])
        self.assertEqual(err_search.count("renamed-copy detection is off"), 1)
        with open(os.path.join(out_dir, "search-fake.json"), encoding="utf-8") as f:
            search_doc = json.load(f)
        self.assertEqual(search_doc["pool_detection"]["status"], "off")

        # 5. `coverage`: `<out>/coverage-fake.json`
        _o, err_cov = run_cmd([
            "coverage", "--project", proj, "--out", out_dir, "--backend", "fake", "--subjects", subjects_file,
        ])
        self.assertEqual(err_cov.count("renamed-copy detection is off"), 1)
        with open(os.path.join(out_dir, "coverage-fake.json"), encoding="utf-8") as f:
            cov_doc = json.load(f)
        self.assertEqual(cov_doc["pool_detection"]["status"], "off")

        # 6. `report --draft fake`: Markdown report note, worksheet note, review JSONL, drafter-raw JSONL
        _o, err_rep = run_cmd(["report", "--project", proj, "--out", out_dir, "--backend", "fake", "--draft", "fake"])
        self.assertEqual(err_rep.count("renamed-copy detection is off"), 1)
        with open(os.path.join(out_dir, "report-fake.md"), encoding="utf-8") as f:
            rep_md = f.read()
        self.assertIn("Pool detection (`pool_detection`): `off`", rep_md)
        self.assertIn("renamed-copy detection is off", rep_md)
        with open(os.path.join(out_dir, "description-worksheet.md"), encoding="utf-8") as f:
            ws_md = f.read()
        self.assertIn("Pool detection (`pool_detection`): `off`", ws_md)
        with open(os.path.join(out_dir, "description-review.jsonl"), encoding="utf-8") as f:
            rev_rows = [json.loads(ln) for ln in f if ln.strip()]
        self.assertTrue(all(r["pool_detection"]["status"] == "off" for r in rev_rows))
        with open(os.path.join(out_dir, "drafter-raw.jsonl"), encoding="utf-8") as f:
            raw_rows = [json.loads(ln) for ln in f if ln.strip()]
        self.assertTrue(all(r["pool_detection"]["status"] == "off" for r in raw_rows))

        # 7. `import-edits`: warns once on stderr and writes `status: "off"` into `seat-edits.jsonl`
        self._fill_worksheet_edit(
            os.path.join(out_dir, "description-worksheet.md"),
            "arch_02",
            "carved granite arch beside stone steps",
        )
        _o, err_imp = run_cmd(["import-edits", "--project", proj, "--out", out_dir, "--seat", "seat-beta"])
        self.assertEqual(err_imp.count("renamed-copy detection is off"), 1)
        with open(os.path.join(out_dir, "seat-edits.jsonl"), encoding="utf-8") as f:
            edit_rows = [json.loads(ln) for ln in f if ln.strip()]
        self.assertEqual(edit_rows[0]["pool_detection"]["status"], "off")

        # 8. `compose`: `<out>/captions-manifest.json` + `<out>/pool-detection.json`
        _o, err_comp = run_cmd(["compose", "--project", proj, "--out", out_dir, "--draft", "fake"])
        self.assertEqual(err_comp.count("renamed-copy detection is off"), 1)
        with open(os.path.join(out_dir, "captions-manifest.json"), encoding="utf-8") as f:
            comp_doc = json.load(f)
        self.assertEqual(comp_doc["pool_detection"]["status"], "off")
        with open(os.path.join(out_dir, "pool-detection.json"), encoding="utf-8") as f:
            pd_doc = json.load(f)
        self.assertEqual(pd_doc["pool_detection"]["status"], "off")

        # 9. `sidecar-import`: `<out>/sidecar-import.json`, `sidecar-records.jsonl`, `description-worksheet.md`
        _o, err_sc = run_cmd(["sidecar-import", "--project", sidecar_proj, "--out", out_sc])
        self.assertEqual(err_sc.count("renamed-copy detection is off"), 1)
        with open(os.path.join(out_sc, "sidecar-import.json"), encoding="utf-8") as f:
            sc_doc = json.load(f)
        self.assertEqual(sc_doc["pool_detection"]["status"], "off")
        with open(os.path.join(out_sc, "sidecar-records.jsonl"), encoding="utf-8") as f:
            sc_rows = [json.loads(ln) for ln in f if ln.strip()]
        self.assertTrue(all(r["pool_detection"]["status"] == "off" for r in sc_rows))
        with open(os.path.join(out_sc, "description-worksheet.md"), encoding="utf-8") as f:
            sc_ws = f.read()
        self.assertIn("Pool detection (`pool_detection`): `off`", sc_ws)

        # 10. Tagged locked-pool project (`multimodal_pool: "salt-road"`) ALSO warns once when detection is off
        tagged_proj = os.path.join(self.tmp, "tagged-off")
        out_tagged = os.path.join(self.tmp, "out-tagged-off")
        _write_png(os.path.join(tagged_proj, "outputs", "w1", "t_01.png"), (65, 95, 125), seed=74)
        with open(os.path.join(tagged_proj, "project.json"), "w", encoding="utf-8") as f:
            json.dump({"multimodal_pool": "salt-road"}, f)
        _o, err_tag = run_cmd(["index", "--project", tagged_proj, "--out", out_tagged, "--backend", "none"])
        self.assertEqual(err_tag.count("renamed-copy detection is off"), 1)
        with open(os.path.join(out_tagged, "none", "meta.json"), encoding="utf-8") as f:
            tag_meta = json.load(f)
        self.assertEqual(tag_meta["pool_detection"]["status"], "off")

    def test_manifest_and_pool_dir_configured_records_status_on_with_counts_and_no_warning(self):
        proj, sidecar_proj, subjects_file = self._build_synthetic_projects("on")
        out_dir = os.path.join(self.tmp, "out-on")
        out_sc = os.path.join(self.tmp, "out-on-sc")

        # Synthetic manifest via DATASET_DESK_POOL_MANIFEST (3 hashes in salt-road, 1 in rustline)
        manifest_path = os.path.join(self.tmp, "synthetic-pool-manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "salt-road": ["1" * 64, "2" * 64, "3" * 64],
                    "rustline": ["4" * 64],
                },
                f,
            )
        os.environ["DATASET_DESK_POOL_MANIFEST"] = manifest_path

        # Synthetic locked-pool folder for tallow-fen (2 distinct synthetic images) via --locked-pool-dir
        locked_tallow = os.path.join(self.tmp, "tallow-fen")
        _write_png(os.path.join(locked_tallow, "outputs", "w1", "tf_01.png"), (30, 80, 50), seed=81)
        _write_png(os.path.join(locked_tallow, "outputs", "w1", "tf_02.png"), (50, 30, 80), seed=82)

        expected_counts = {"hesperia": 0, "rustline": 1, "salt-road": 3, "tallow-fen": 2}

        # Direct API check: fails at 62a2f41 where `check_multimodal_pool` returned `None`
        err_direct = io.StringIO()
        with contextlib.redirect_stderr(err_direct):
            pd_direct = desk.check_multimodal_pool(proj, "fake", locked_pool_dirs=[locked_tallow])
        self.assertIsInstance(pd_direct, dict)
        self.assertEqual(err_direct.getvalue(), "")
        self.assertEqual(pd_direct["status"], "on")
        self.assertEqual(pd_direct["counts"], expected_counts)
        self.assertEqual(pd_direct["pools"]["salt-road"]["sources"][0]["via"], "DATASET_DESK_POOL_MANIFEST")
        self.assertEqual(pd_direct["pools"]["tallow-fen"]["sources"][0]["via"], "--locked-pool-dir")

        # Verify `user_config` source attribution when configured via DATASET_DESK_USER_CONFIG
        os.environ.pop("DATASET_DESK_POOL_MANIFEST", None)
        ucfg_path = os.path.join(self.tmp, "desk-user-cfg.json")
        with open(ucfg_path, "w", encoding="utf-8") as f:
            json.dump({"pool_manifest": manifest_path}, f)
        os.environ["DATASET_DESK_USER_CONFIG"] = ucfg_path
        err_ucfg = io.StringIO()
        with contextlib.redirect_stderr(err_ucfg):
            pd_ucfg = desk.check_multimodal_pool(proj, "fake", locked_pool_dirs=[locked_tallow])
        self.assertEqual(err_ucfg.getvalue(), "")
        self.assertEqual(pd_ucfg["pools"]["salt-road"]["sources"][0]["via"], "user_config")
        os.environ["DATASET_DESK_USER_CONFIG"] = ""
        os.environ["DATASET_DESK_POOL_MANIFEST"] = manifest_path

        def run_on(argv: list[str]) -> str:
            out_buf, err_buf = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
                rc = desk.run([*argv, "--locked-pool-dir", locked_tallow])
            self.assertEqual(rc, EXIT_OK, err_buf.getvalue())
            self.assertEqual(err_buf.getvalue(), "")
            return out_buf.getvalue()

        run_on(["scan", "--project", proj, "--out", out_dir])
        run_on(["index", "--project", proj, "--out", out_dir, "--backend", "fake"])
        run_on(["dupes", "--project", proj, "--out", out_dir, "--backend", "fake"])
        run_on(["search", "--project", proj, "--out", out_dir, "--backend", "fake", "--text", "limestone arch"])
        run_on(["coverage", "--project", proj, "--out", out_dir, "--backend", "fake", "--subjects", subjects_file])
        run_on(["report", "--project", proj, "--out", out_dir, "--backend", "fake", "--draft", "fake"])
        self._fill_worksheet_edit(
            os.path.join(out_dir, "description-worksheet.md"),
            "arch_02",
            "carved granite arch beside stone steps",
        )
        run_on(["import-edits", "--project", proj, "--out", out_dir, "--seat", "seat-gamma"])
        run_on(["compose", "--project", proj, "--out", out_dir, "--draft", "fake"])
        run_on(["sidecar-import", "--project", sidecar_proj, "--out", out_sc])

        for json_rel in (
            "scan-report.json",
            os.path.join("fake", "meta.json"),
            "dupes-fake.json",
            "search-fake.json",
            "coverage-fake.json",
            "captions-manifest.json",
            "pool-detection.json",
        ):
            with open(os.path.join(out_dir, json_rel), encoding="utf-8") as f:
                doc = json.load(f)
            self.assertEqual(doc["pool_detection"]["status"], "on", json_rel)
            self.assertEqual(doc["pool_detection"]["counts"], expected_counts, json_rel)

        for jsonl_rel in ("description-review.jsonl", "drafter-raw.jsonl", "seat-edits.jsonl"):
            with open(os.path.join(out_dir, jsonl_rel), encoding="utf-8") as f:
                rows = [json.loads(ln) for ln in f if ln.strip()]
            self.assertTrue(rows, jsonl_rel)
            for r in rows:
                self.assertEqual(r["pool_detection"]["status"], "on", jsonl_rel)
                self.assertEqual(r["pool_detection"]["counts"], expected_counts, jsonl_rel)

        with open(os.path.join(out_sc, "sidecar-import.json"), encoding="utf-8") as f:
            sc_doc = json.load(f)
        self.assertEqual(sc_doc["pool_detection"]["status"], "on")
        self.assertEqual(sc_doc["pool_detection"]["counts"], expected_counts)

        for md_path in (
            os.path.join(out_dir, "report-fake.md"),
            os.path.join(out_dir, "description-worksheet.md"),
            os.path.join(out_sc, "description-worksheet.md"),
        ):
            with open(md_path, encoding="utf-8") as f:
                md_txt = f.read()
            self.assertIn("Pool detection (`pool_detection`): `on`", md_txt)
            self.assertIn("salt-road: 3", md_txt)
            self.assertIn("tallow-fen: 2", md_txt)
            self.assertIn("rustline: 1", md_txt)
            self.assertIn("hesperia: 0", md_txt)

    def test_no_sha256_value_appears_in_pool_detection_or_locked_pool_outputs(self):
        proj, sidecar_proj, subjects_file = self._build_synthetic_projects("nosha")
        out_dir = os.path.join(self.tmp, "out-nosha")
        out_sc = os.path.join(self.tmp, "out-nosha-sc")

        mf_sha_1 = "a1b2c3d4" * 8
        mf_sha_2 = "b2c3d4e5" * 8
        mf_sha_3 = "c3d4e5f6" * 8
        manifest_path = os.path.join(self.tmp, "secret-pool-manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({"salt-road": [mf_sha_1, mf_sha_2], "hesperia": [mf_sha_3]}, f)

        locked_rust = os.path.join(self.tmp, "rustline")
        dir_sha_1 = _write_png(os.path.join(locked_rust, "outputs", "w1", "rl_01.png"), (15, 45, 75), seed=91)
        dir_sha_2 = _write_png(os.path.join(locked_rust, "outputs", "w1", "rl_02.png"), (75, 45, 15), seed=92)
        forbidden_shas = {mf_sha_1, mf_sha_2, mf_sha_3, dir_sha_1, dir_sha_2}

        captured_streams: list[str] = []

        def run_checked(argv: list[str]) -> None:
            out_buf, err_buf = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
                rc = desk.run([
                    *argv,
                    "--pool-manifest", manifest_path,
                    "--locked-pool-dir", locked_rust,
                ])
            self.assertEqual(rc, EXIT_OK, err_buf.getvalue())
            self.assertEqual(err_buf.getvalue(), "")
            captured_streams.append(out_buf.getvalue())

        run_checked(["scan", "--project", proj, "--out", out_dir])
        run_checked(["index", "--project", proj, "--out", out_dir, "--backend", "fake"])
        run_checked(["dupes", "--project", proj, "--out", out_dir, "--backend", "fake"])
        run_checked(["search", "--project", proj, "--out", out_dir, "--backend", "fake", "--text", "limestone arch"])
        run_checked(["coverage", "--project", proj, "--out", out_dir, "--backend", "fake", "--subjects", subjects_file])
        run_checked(["report", "--project", proj, "--out", out_dir, "--backend", "fake", "--draft", "fake"])
        self._fill_worksheet_edit(
            os.path.join(out_dir, "description-worksheet.md"),
            "arch_02",
            "carved granite arch beside stone steps",
        )
        run_checked(["import-edits", "--project", proj, "--out", out_dir, "--seat", "seat-delta"])
        run_checked(["compose", "--project", proj, "--out", out_dir, "--draft", "fake"])
        run_checked(["sidecar-import", "--project", sidecar_proj, "--out", out_sc])

        # Fails at 62a2f41 where `<out>/pool-detection.json` is never written
        self.assertTrue(os.path.isfile(os.path.join(out_dir, "pool-detection.json")))

        hex64_re = re.compile(r"\b[0-9a-f]{64}\b", flags=re.IGNORECASE)

        # 1. No locked-pool sha256 (or 12-hex prefix of one) appears in ANY output file or stdout stream
        all_stream_text = "\n".join(captured_streams).lower()
        for f_sha in forbidden_shas:
            self.assertNotIn(f_sha.lower(), all_stream_text)
            self.assertNotIn(f_sha[:12].lower(), all_stream_text)

        inspected_files = 0
        for root_out in (out_dir, out_sc):
            for dirpath, _dirnames, filenames in os.walk(root_out):
                for fn in filenames:
                    inspected_files += 1
                    full_path = os.path.join(dirpath, fn)
                    with open(full_path, "rb") as f:
                        raw_bytes = f.read()
                    lower_latin = raw_bytes.decode("latin-1").lower()
                    for f_sha in forbidden_shas:
                        self.assertNotIn(f_sha.lower(), lower_latin, full_path)
                        self.assertNotIn(f_sha[:12].lower(), lower_latin, full_path)

                    if fn.endswith((".json", ".jsonl", ".md")):
                        txt = raw_bytes.decode("utf-8")
                        if fn in (
                            "pool-detection.json",
                            "scan-report.json",
                            "dupes-fake.json",
                            "search-fake.json",
                            "coverage-fake.json",
                            "sidecar-import.json",
                            "report-fake.md",
                        ):
                            self.assertEqual(hex64_re.findall(txt), [], f"unexpected sha256 in {full_path}")
                        if fn.endswith(".json"):
                            doc = json.loads(txt)
                            if isinstance(doc, dict) and "pool_detection" in doc:
                                self.assertEqual(
                                    hex64_re.findall(json.dumps(doc["pool_detection"])),
                                    [],
                                    full_path,
                                )
                        elif fn.endswith(".jsonl"):
                            for ln in txt.splitlines():
                                if ln.strip():
                                    row = json.loads(ln)
                                    if isinstance(row, dict) and "pool_detection" in row:
                                        self.assertEqual(
                                            hex64_re.findall(json.dumps(row["pool_detection"])),
                                            [],
                                            full_path,
                                        )
                        elif fn.endswith(".md"):
                            for ln in txt.splitlines():
                                if "Pool detection (`pool_detection`)" in ln:
                                    self.assertEqual(hex64_re.findall(ln), [], full_path)
        self.assertGreaterEqual(inspected_files, 14)

        # Exercise structured manifest shapes, env/user_config pool_dirs, ComfyUI KSampler trace, and config guards
        rich_mf = os.path.join(self.tmp, "rich-manifest.json")
        with open(rich_mf, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "version": 1,
                    "locked_pool_dirs": {"rustline": "rustline"},
                    "pools": {
                        "salt-road": {"hashes": [mf_sha_1]},
                        "tallow-fen": {"sha256": [mf_sha_2]},
                        "hesperia": {"items": [{"sha256": mf_sha_3}]},
                    },
                },
                f,
            )
        pd_rich, hashes_rich = desk.resolve_pool_detection(pool_manifest=rich_mf, emit_warning=False)
        self.assertEqual(pd_rich["counts"], {"hesperia": 1, "rustline": 2, "salt-road": 1, "tallow-fen": 1})
        self.assertEqual(hex64_re.findall(json.dumps(pd_rich)), [])
        self.assertIsNotNone(hashes_rich)
        self.assertIsNotNone(desk._load_locked_pool_hashes(pool_manifest=rich_mf))
        self.assertIsNone(desk.write_pool_detection_status(None, pd_rich))

        empty_mf = os.path.join(self.tmp, "empty-manifest.json")
        with open(empty_mf, "w", encoding="utf-8") as f:
            json.dump({}, f)
        with self.assertRaises(DeskError) as cm_empty_mf:
            desk.resolve_pool_detection(pool_manifest=empty_mf, emit_warning=False)
        self.assertEqual(cm_empty_mf.exception.code, "INPUT_EMPTY_POOL_SOURCE")

        os.environ["DATASET_DESK_POOL_DIRS"] = f"rustline={locked_rust}"
        ucfg_dirs = os.path.join(self.tmp, "ucfg-dirs.json")
        with open(ucfg_dirs, "w", encoding="utf-8") as f:
            json.dump({"locked_pool_dirs": ["rustline"]}, f)
        os.environ["DATASET_DESK_USER_CONFIG"] = ucfg_dirs
        pd_dirs, _ = desk.resolve_pool_detection(emit_warning=False)
        self.assertEqual(pd_dirs["counts"]["rustline"], 2)
        os.environ.pop("DATASET_DESK_POOL_DIRS", None)
        os.environ["DATASET_DESK_USER_CONFIG"] = ""

        bad_mf = os.path.join(self.tmp, "bad-manifest.json")
        for bad_payload in (
            ["not-an-object"],
            {"unknown-pool": [mf_sha_1]},
            {"salt-road": {"other": []}},
            {"salt-road": "not-a-list"},
            {"salt-road": ["not-64-hex"]},
        ):
            with open(bad_mf, "w", encoding="utf-8") as f:
                json.dump(bad_payload, f)
            with self.assertRaises(DeskError):
                desk.resolve_pool_detection(pool_manifest=bad_mf, emit_warning=False)
        with open(bad_mf, "w", encoding="utf-8") as f:
            f.write("{broken json")
        with self.assertRaises(DeskError):
            desk.resolve_pool_detection(pool_manifest=bad_mf, emit_warning=False)
        with self.assertRaises(DeskError):
            desk.resolve_pool_detection(pool_manifest=os.path.join(self.tmp, "no-such.json"), emit_warning=False)
        for bad_dir_arg in (
            [f"badpool={locked_rust}"],
            [os.path.join(self.tmp, "missing-pool-dir")],
            [proj],
            [123],
            {"salt-road": 123},
            123,
        ):
            with self.assertRaises(DeskError):
                desk.resolve_pool_detection(locked_pool_dirs=bad_dir_arg, emit_warning=False)

        conflict_dir = os.path.join(self.tmp, "salt-road")
        os.makedirs(conflict_dir, exist_ok=True)
        with open(os.path.join(conflict_dir, "project.json"), "w", encoding="utf-8") as f:
            json.dump({"multimodal_pool": "rustline"}, f)
        with self.assertRaises(DeskError):
            desk.check_multimodal_pool(conflict_dir, "none")

        # ComfyUI KSampler -> ConditioningCombine -> CLIPTextEncode trace + wave generation.json + orphan sidecar
        ks_proj = os.path.join(self.tmp, "ks-sidecar")
        os.makedirs(os.path.join(ks_proj, "outputs", "w1"), exist_ok=True)
        _write_png(os.path.join(ks_proj, "outputs", "contact.png"), (10, 20, 30), seed=93)
        ks_img_path = os.path.join(ks_proj, "outputs", "w1", "ks_01.png")
        ks_pnginfo = PngImagePlugin.PngInfo()
        ks_graph = {
            "1": {"class_type": "CLIPTextEncode", "inputs": {"text": "traced sampler conditioning prompt words here"}},
            "2": {"class_type": "ConditioningCombine", "inputs": {"conditioning_1": ["1", 0]}},
            "3": {"class_type": "KSampler", "inputs": {"positive": ["2", 0]}},
        }
        ks_pnginfo.add_text("prompt", json.dumps(ks_graph))
        Image.new("RGB", (16, 16), (40, 60, 80)).save(ks_img_path, pnginfo=ks_pnginfo)
        with open(os.path.join(ks_proj, "outputs", "w1", "ks_01.txt"), "w", encoding="utf-8") as f:
            f.write("traced sampler conditioning prompt words here")
        _write_png(os.path.join(ks_proj, "outputs", "w1", "ks_02.png"), (80, 60, 40), seed=94)
        with open(os.path.join(ks_proj, "outputs", "w1", "ks_02.txt"), "w", encoding="utf-8") as f:
            f.write("wave generation prompt words echoed verbatim here")
        with open(os.path.join(ks_proj, "outputs", "w1", "orphan.txt"), "w", encoding="utf-8") as f:
            f.write("orphan sidecar note")
        with open(os.path.join(ks_proj, "outputs", "w1", "generation.json"), "w", encoding="utf-8") as f:
            json.dump({"items": [{"file": "ks_02.png", "prompt": "wave generation prompt words echoed verbatim here"}]}, f)
        with open(os.path.join(ks_proj, "caption-rules.json"), "w", encoding="utf-8") as f:
            json.dump({"prefix": "copperplate study,", "backdrop": "plain pale ground"}, f)
        ks_out = os.path.join(self.tmp, "out-ks")
        ks_res = description_gate.import_sidecars(
            ks_proj, ks_out, drafter=backends.make_drafter("fake"), pool_detection=pd_rich
        )
        self.assertEqual(ks_res["flagged_count"], 2)
        self.assertEqual(ks_res["orphan_sidecars"], ["outputs/w1/orphan.txt"])
        ks_comp_out = os.path.join(self.tmp, "out-ks-comp")
        ks_comp = description_gate.compose_captions(
            ks_proj, ks_comp_out, drafter=backends.make_drafter("fake"), sidecars=True, pool_detection=pd_rich
        )
        self.assertEqual(len(ks_comp["items"]), 2)

        for bad_rules in (
            "not-a-dict",
            {"multimodal_pool": 123, "prefix": "x,"},
            {"multimodal_pool": "unknown", "prefix": "x,"},
            {"prefix": "x,", "drafter": 123},
            {"prefix": "x,", "drafter_prompt": 123},
            {"prefix": "x,", "class_field": 123},
            {"prefix": "x,", "class_pattern": 123},
            {"prefix": "x,", "backdrop": 123},
            {"prefix": "x,", "backdrop": {"text": 123}},
            {"prefix": "x,", "backdrop": {"field": 123}},
            {"prefix": "x,", "banned_words": [123]},
            {"prefix": "x,", "classes": [1]},
            {"prefix": "x,", "classes": {"c": []}},
        ):
            with self.assertRaises(DeskError):
                description_gate._build_caption_rules(bad_rules)  # type: ignore[arg-type]


class Addendum3LockAndProvenanceTests(fakes.IsolatedDeskTestCase):
    """Synthetic tests for Addendum 3 (Parts A-E); every test fails at 5248711."""

    def setUp(self):
        super().setUp()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name

    def test_part_a_pools_non_object_refused(self):
        # Fails at 5248711 dataset_desk/desk.py:232-238 (fell back to doc and skipped 'pools', returning 0 hashes + on)
        mf_path = os.path.join(self.tmp, "pools-list.json")
        for bad_pools in ([], ["salt-road"], "salt-road", 42):
            with open(mf_path, "w", encoding="utf-8") as f:
                json.dump({"pools": bad_pools}, f)
            with self.assertRaises(DeskError) as cm:
                desk.resolve_pool_detection(pool_manifest=mf_path, emit_warning=False)
            self.assertEqual(cm.exception.code, "INPUT_BAD_ARGUMENT")
            self.assertIn("'pools' must be a JSON object", cm.exception.message)

    def test_part_a_empty_manifest_or_empty_pool_dir_refused_and_on_records_source_counts(self):
        # Fails at 5248711 dataset_desk/desk.py:368-372, 401-421 (returned status 'on' with 0 hashes)
        mf_path = os.path.join(self.tmp, "empty-source.json")
        for empty_doc in (
            {},
            {"version": 1},
            {"pools": {}},
            {"salt-road": []},
            {"pools": {"salt-road": []}},
            {"salt-road": {"hashes": []}},
            {"salt-road": {"items": []}},
            {"locked_pool_dirs": []},
        ):
            with open(mf_path, "w", encoding="utf-8") as f:
                json.dump(empty_doc, f)
            with self.assertRaises(DeskError) as cm_mf:
                desk.resolve_pool_detection(pool_manifest=mf_path, emit_warning=False)
            self.assertEqual(cm_mf.exception.code, "INPUT_EMPTY_POOL_SOURCE")
            self.assertIn("empty-source.json", cm_mf.exception.message)

        with self.assertRaises(DeskError) as cm_empty_arg:
            desk.resolve_pool_detection(locked_pool_dirs=[], emit_warning=False)
        self.assertEqual(cm_empty_arg.exception.code, "INPUT_EMPTY_POOL_SOURCE")

        empty_pool_dir = os.path.join(self.tmp, "salt-road")
        os.makedirs(os.path.join(empty_pool_dir, "outputs", "w1"), exist_ok=True)
        with self.assertRaises(DeskError) as cm_dir:
            desk.resolve_pool_detection(locked_pool_dirs=[empty_pool_dir], emit_warning=False)
        self.assertEqual(cm_dir.exception.code, "INPUT_EMPTY_POOL_SOURCE")
        self.assertIn("salt-road", cm_dir.exception.message)

        # Valid multi-source configuration: status == "on" guarantees >= 1 hash per configured pool + per-source counts
        rust_dir = os.path.join(self.tmp, "rustline")
        _write_png(os.path.join(rust_dir, "outputs", "w1", "rl_01.png"), (120, 60, 40), seed=101)
        sha_sr_a = hashlib.sha256(b"synthetic-sr-a").hexdigest()
        sha_sr_b = hashlib.sha256(b"synthetic-sr-b").hexdigest()
        valid_mf = os.path.join(self.tmp, "valid-pools.json")
        with open(valid_mf, "w", encoding="utf-8") as f:
            json.dump({"pools": {"salt-road": [sha_sr_a, sha_sr_b]}}, f)

        pd_on, hashes_on = desk.resolve_pool_detection(
            pool_manifest=valid_mf,
            locked_pool_dirs=[rust_dir],
            emit_warning=False,
        )
        self.assertEqual(pd_on["status"], "on")
        self.assertEqual(pd_on["configured_pools"], ["rustline", "salt-road"])
        self.assertIsNotNone(hashes_on)
        for p_cfg in pd_on["configured_pools"]:
            self.assertGreaterEqual(pd_on["counts"][p_cfg], 1)
        self.assertEqual(pd_on["pools"]["salt-road"]["sources"][0]["count"], 2)
        self.assertEqual(pd_on["pools"]["rustline"]["sources"][0]["count"], 1)
        by_pool_src = {s["pool"]: s["count"] for s in pd_on["sources"]}
        self.assertEqual(by_pool_src, {"rustline": 1, "salt-road": 2})

    def _setup_locked_pool_for_links(self) -> tuple[str, str, str]:
        locked_sr = os.path.join(self.tmp, "salt-road")
        locked_wave = os.path.join(locked_sr, "outputs", "w1")
        sha_1 = _write_png(os.path.join(locked_wave, "sr_link_01.png"), (95, 45, 135), seed=111)
        return locked_sr, locked_wave, sha_1

    def test_part_b_symlink_wave_and_realpath_folder_refused(self):
        # Fails at 5248711 dataset_desk/desk.py:145 (os.walk followlinks=False) and :471 (abspath vs realpath)
        locked_sr, locked_wave, sha_1 = self._setup_locked_pool_for_links()
        sym_proj = os.path.join(self.tmp, "untagged-sym-proj")
        os.makedirs(os.path.join(sym_proj, "outputs"), exist_ok=True)
        sym_wave = os.path.join(sym_proj, "outputs", "w1")
        alias_sr = os.path.join(self.tmp, "alias-to-locked")
        try:
            os.symlink(locked_wave, sym_wave, target_is_directory=True)
            os.symlink(locked_sr, alias_sr, target_is_directory=True)
        except (OSError, NotImplementedError) as exc:
            if os.name == "nt":
                self.skipTest(
                    f"directory symlinks require Developer Mode or SeCreateSymbolicLinkPrivilege on Windows: {exc}"
                )
            raise

        enum_rels = [rel for rel, _full, _lbl in desk.enumerate_project_images(sym_proj)]
        self.assertEqual(enum_rels, ["outputs/w1/sr_link_01.png"])
        self.assertEqual(desk._collect_folder_image_shas(sym_proj), {sha_1})

        with self.assertRaises(DeskError) as cm_sym:
            desk.check_multimodal_pool(sym_proj, "none", locked_pool_dirs=[locked_sr])
        self.assertEqual(cm_sym.exception.code, "PERM_LOCKED_POOL")
        self.assertIn("salt-road", cm_sym.exception.message)

        err_scan = io.StringIO()
        with contextlib.redirect_stderr(err_scan):
            rc_scan = desk.run([
                "scan", "--project", sym_proj, "--out", os.path.join(self.tmp, "out-sym"),
                "--locked-pool-dir", locked_sr,
            ])
        self.assertEqual(rc_scan, EXIT_USER)
        self.assertIn("PERM_LOCKED_POOL", err_scan.getvalue())

        # realpath basename check catches symlinked project folder even when detection is off
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(DeskError) as cm_alias:
            desk.check_multimodal_pool(alias_sr, "eg2", images_only=False)
        self.assertEqual(cm_alias.exception.code, "PERM_LOCKED_POOL")
        self.assertIn("salt-road", cm_alias.exception.message)

        # Symlink cycle safety
        cyc = os.path.join(self.tmp, "sym-cycle")
        os.makedirs(os.path.join(cyc, "outputs", "w1"), exist_ok=True)
        _write_png(os.path.join(cyc, "outputs", "w1", "c.png"), (30, 60, 90), seed=112)
        os.symlink(os.path.join(cyc, "outputs", "w1"), os.path.join(cyc, "outputs", "w1", "loop"), target_is_directory=True)
        self.assertEqual(len(desk.enumerate_project_images(cyc)), 1)

    def test_part_b_windows_junction_wave_and_realpath_folder_refused(self):
        # Fails at 5248711 dataset_desk/desk.py:145 (os.walk followlinks=False) and :471 (abspath vs realpath)
        if os.name != "nt":
            self.skipTest("Windows NTFS directory junctions (mklink /J) apply on Windows only")
        locked_sr, locked_wave, sha_1 = self._setup_locked_pool_for_links()
        junc_proj = os.path.join(self.tmp, "untagged-junction-proj")
        os.makedirs(os.path.join(junc_proj, "outputs"), exist_ok=True)
        junc_wave = os.path.join(junc_proj, "outputs", "w1")
        alias_junc = os.path.join(self.tmp, "junction-to-locked")

        for link_p, target_p in ((junc_wave, locked_wave), (alias_junc, locked_sr)):
            res = subprocess.run(
                ["cmd", "/c", "mklink", "/J", link_p, target_p],
                capture_output=True,
                text=True,
                check=False,
            )
            if res.returncode != 0:
                self.skipTest(f"mklink /J unavailable in this environment: {res.stderr or res.stdout}")
            self.addCleanup(lambda p=link_p: os.rmdir(p) if os.path.lexists(p) else None)

        enum_rels = [rel for rel, _full, _lbl in desk.enumerate_project_images(junc_proj)]
        self.assertEqual(enum_rels, ["outputs/w1/sr_link_01.png"])
        self.assertEqual(desk._collect_folder_image_shas(junc_proj), {sha_1})

        with self.assertRaises(DeskError) as cm_junc:
            desk.check_multimodal_pool(junc_proj, "none", locked_pool_dirs=[locked_sr])
        self.assertEqual(cm_junc.exception.code, "PERM_LOCKED_POOL")
        self.assertIn("salt-road", cm_junc.exception.message)

        err_idx = io.StringIO()
        with contextlib.redirect_stderr(err_idx):
            rc_idx = desk.run([
                "index", "--project", junc_proj, "--out", os.path.join(self.tmp, "out-junc"),
                "--backend", "none", "--locked-pool-dir", locked_sr,
            ])
        self.assertEqual(rc_idx, EXIT_USER)
        self.assertIn("PERM_LOCKED_POOL", err_idx.getvalue())

        # realpath basename check catches directory junction to salt-road even when detection is off
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(DeskError) as cm_alias:
            desk.check_multimodal_pool(alias_junc, "eg2", images_only=False)
        self.assertEqual(cm_alias.exception.code, "PERM_LOCKED_POOL")
        self.assertIn("salt-road", cm_alias.exception.message)

        # Self-referencing junction cycle terminates cleanly
        cyc_dir = os.path.join(self.tmp, "junc-cycle")
        cyc_w1 = os.path.join(cyc_dir, "outputs", "w1")
        os.makedirs(cyc_w1, exist_ok=True)
        _write_png(os.path.join(cyc_w1, "only.png"), (55, 85, 115), seed=113)
        cyc_loop = os.path.join(cyc_w1, "loop")
        res_cyc = subprocess.run(["cmd", "/c", "mklink", "/J", cyc_loop, cyc_w1], capture_output=True, text=True, check=False)
        if res_cyc.returncode == 0:
            self.addCleanup(lambda p=cyc_loop: os.rmdir(p) if os.path.lexists(p) else None)
            self.assertEqual([r for r, _, _ in desk.enumerate_project_images(cyc_dir)], ["outputs/w1/only.png"])

    def test_part_c_text_commands_locked_pool_images_sidecar_stems_and_pool_detection(self):
        # Fails at 5248711 dataset_desk/desk.py:3258-3297 (text_cmds ran no pool check and wrote no pool_detection)
        locked_sr = os.path.join(self.tmp, "salt-road")
        sha_sr = _write_png(os.path.join(locked_sr, "outputs", "w1", "sr_plate_01.png"), (110, 70, 50), seed=121)
        with open(os.path.join(locked_sr, "notes.py"), "w", encoding="utf-8") as f:
            f.write("def helper():\n    return 1\n")

        made_backends: list[str] = []

        def spy_make(bname: str, **kw):
            made_backends.append(bname)
            emb = backends.FakeEmbedder()
            emb.name = bname
            return emb

        # 1. Folder named salt-road (including --task code defaulting to eg2) refused BEFORE model load
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(DeskError) as cm_named:
            desk.main(
                ["text-index", "--source", locked_sr, "--out", os.path.join(self.tmp, "out-t1"), "--task", "code"],
                make=spy_make,
            )
        self.assertEqual(cm_named.exception.code, "PERM_LOCKED_POOL")
        self.assertEqual(made_backends, [])

        # 2. Tagged source folder refused on text-index and text-search
        tagged_src = os.path.join(self.tmp, "tagged-text-src")
        os.makedirs(tagged_src, exist_ok=True)
        with open(os.path.join(tagged_src, "project.json"), "w", encoding="utf-8") as f:
            json.dump({"multimodal_pool": "salt-road"}, f)
        with open(os.path.join(tagged_src, "doc.txt"), "w", encoding="utf-8") as f:
            f.write("harbour lantern beside stone steps")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(DeskError) as cm_tsearch:
            desk.main(
                [
                    "text-search", "--source", tagged_src, "--out", os.path.join(self.tmp, "out-t2"),
                    "--backend", "fake", "--text", "lantern",
                ],
                make=spy_make,
            )
        self.assertEqual(cm_tsearch.exception.code, "PERM_LOCKED_POOL")
        self.assertEqual(made_backends, [])

        # 3. Untagged --source containing a locked-pool image by sha256 is refused
        img_copy_src = os.path.join(self.tmp, "untagged-img-text-src")
        os.makedirs(img_copy_src, exist_ok=True)
        with open(os.path.join(locked_sr, "outputs", "w1", "sr_plate_01.png"), "rb") as fsrc, \
                open(os.path.join(img_copy_src, "renamed_pic.png"), "wb") as fdst:
            fdst.write(fsrc.read())
        with open(os.path.join(img_copy_src, "readme.md"), "w", encoding="utf-8") as f:
            f.write("plain notes")
        with self.assertRaises(DeskError) as cm_img_src:
            desk.main([
                "text-index", "--source", img_copy_src, "--out", os.path.join(self.tmp, "out-t3"),
                "--backend", "fake", "--locked-pool-dir", locked_sr,
            ])
        self.assertEqual(cm_img_src.exception.code, "PERM_LOCKED_POOL")
        self.assertIn("salt-road", cm_img_src.exception.message)

        # 4. Untagged --source containing ONLY a .txt sidecar whose stem matches a locked-pool image is refused
        sidecar_only_src = os.path.join(self.tmp, "sidecar-only-text-src")
        os.makedirs(sidecar_only_src, exist_ok=True)
        with open(os.path.join(sidecar_only_src, "sr_plate_01.txt"), "w", encoding="utf-8") as f:
            f.write("carved timber skiff beside stone quay steps")

        with self.assertRaises(DeskError) as cm_txt_dir:
            desk.main([
                "text-index", "--source", sidecar_only_src, "--out", os.path.join(self.tmp, "out-t4"),
                "--backend", "fake", "--locked-pool-dir", locked_sr,
            ])
        self.assertEqual(cm_txt_dir.exception.code, "PERM_LOCKED_POOL")
        self.assertIn(".txt caption sidecar", cm_txt_dir.exception.message)
        self.assertIn("salt-road", cm_txt_dir.exception.message)

        # Also refused via --pool-manifest item image filename / stems on both text-index and text-search
        stem_mf = os.path.join(self.tmp, "stem-manifest.json")
        with open(stem_mf, "w", encoding="utf-8") as f:
            json.dump({"pools": {"salt-road": {"items": [{"image": "sr_plate_01.png", "sha256": sha_sr}]}}}, f)
        with self.assertRaises(DeskError) as cm_txt_mf:
            desk.main([
                "text-search", "--source", sidecar_only_src, "--out", os.path.join(self.tmp, "out-t5"),
                "--backend", "fake", "--text", "skiff", "--pool-manifest", stem_mf,
            ])
        self.assertEqual(cm_txt_mf.exception.code, "PERM_LOCKED_POOL")

        # 5. Clean text commands record pool_detection ('off' warns once; 'on' records counts with empty stderr)
        clean_src = os.path.join(self.tmp, "clean-code-src")
        os.makedirs(clean_src, exist_ok=True)
        with open(os.path.join(clean_src, "mod_a.py"), "w", encoding="utf-8") as f:
            f.write("def alpha():\n    return 'red'\n")
        with open(os.path.join(clean_src, "caption_ok.txt"), "w", encoding="utf-8") as f:
            f.write("unrelated studio notes on wooden joinery")

        out_off = os.path.join(self.tmp, "out-text-off")
        err_off = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err_off):
            self.assertEqual(
                desk.run(["text-index", "--source", clean_src, "--out", out_off, "--backend", "fake"]),
                EXIT_OK,
            )
        self.assertEqual(err_off.getvalue().count("renamed-copy detection is off"), 1)
        with open(os.path.join(out_off, "fake", "text", "meta.json"), encoding="utf-8") as f:
            t_meta_off = json.load(f)
        self.assertEqual(t_meta_off["pool_detection"]["status"], "off")

        out_on = os.path.join(self.tmp, "out-text-on")
        for argv in (
            ["text-index", "--source", clean_src, "--out", out_on, "--backend", "fake", "--locked-pool-dir", locked_sr],
            [
                "text-dupes", "--source", clean_src, "--out", out_on, "--backend", "fake",
                "--semantic", "0.9", "--locked-pool-dir", locked_sr,
            ],
            [
                "text-search", "--source", clean_src, "--out", out_on, "--backend", "fake",
                "--text", "red", "--locked-pool-dir", locked_sr,
            ],
            ["lineage", "--source", clean_src, "--out", out_on, "--backend", "fake", "--locked-pool-dir", locked_sr],
        ):
            err_on = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err_on):
                self.assertEqual(desk.run(argv), EXIT_OK)
            self.assertEqual(err_on.getvalue(), "")

        for rel_json in (
            os.path.join("fake", "text", "meta.json"),
            "text-dupes-fake.json",
            "text-search-fake.json",
            "lineage-fake.json",
            "pool-detection.json",
        ):
            with open(os.path.join(out_on, rel_json), encoding="utf-8") as f:
                doc = json.load(f)
            self.assertEqual(doc["pool_detection"]["status"], "on", rel_json)
            self.assertEqual(doc["pool_detection"]["counts"]["salt-road"], 1, rel_json)

    def test_part_d1_sidecar_author_kind_is_sidecar_never_human_and_in_strata_counts(self):
        # Fails at 5248711 dataset_desk/description_gate.py:1031-1035, :2221-2225 (gave sidecars author_kind 'human')
        self.assertIn("sidecar", description_gate.KNOWN_AUTHOR_STRATA)
        sc_proj = os.path.join(self.tmp, "sc-honest")
        os.makedirs(sc_proj, exist_ok=True)
        with open(os.path.join(sc_proj, "caption-rules.json"), "w", encoding="utf-8") as f:
            json.dump({"prefix": "ink plate,", "drafter": "none"}, f)
        _write_png(os.path.join(sc_proj, "plate_01.png"), (85, 115, 145), seed=131)
        with open(os.path.join(sc_proj, "plate_01.txt"), "w", encoding="utf-8") as f:
            f.write("ink plate, weathered stone bridge spanning narrow millstream beside reeds")

        out_sc = os.path.join(self.tmp, "out-sc-honest")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(desk.run(["sidecar-import", "--project", sc_proj, "--out", out_sc]), EXIT_OK)
            self.assertEqual(
                desk.run(["compose", "--project", sc_proj, "--out", out_sc, "--sidecars", "--draft", "none"]),
                EXIT_OK,
            )

        with open(os.path.join(out_sc, "sidecar-import.json"), encoding="utf-8") as f:
            sc_imp = json.load(f)
        self.assertEqual(sc_imp["strata_counts"]["sidecar"], 1)
        self.assertEqual(sc_imp["strata_counts"]["human"], 0)

        with open(os.path.join(out_sc, "captions-manifest.json"), encoding="utf-8") as f:
            mf = json.load(f)
        self.assertEqual(mf["items"][0]["author_kind"], "sidecar")
        self.assertEqual(mf["strata_counts"]["sidecar"], 1)
        self.assertEqual(mf["strata_counts"]["human"], 0)

        # Even if a synthetic sidecar record claims author_kind='human', audit_records never reports it as 'human'
        forged = sc_imp
        self.assertIsNotNone(forged)
        audit_forged = description_gate.audit_records([{
            "id": "plate_01",
            "asset_path": "plate_01.png",
            "sidecar_file": "plate_01.txt",
            "author_kind": "human",
            "judgment": {
                "status": "approved",
                "explanation": "ink plate, weathered stone bridge spanning narrow millstream beside reeds",
            },
        }])
        self.assertEqual(audit_forged["records"][0]["author_kind"], "sidecar")

    def test_part_d2_seat_edits_record_draft_and_rerun_report_preserves_captioner_assisted(self):
        # Fails at 5248711 dataset_desk/description_gate.py:1924-1931 and :2202-2203 (re-running report turned draft edit into 'human')
        proj = os.path.join(self.tmp, "rerun-proj")
        out_dir = os.path.join(self.tmp, "out-rerun")
        os.makedirs(os.path.join(proj, "records"), exist_ok=True)
        _write_png(os.path.join(proj, "outputs", "w1", "item_01.png"), (130, 80, 70), seed=141)
        with open(os.path.join(proj, "caption-rules.json"), "w", encoding="utf-8") as f:
            json.dump({"prefix": "woodblock print,"}, f)
        with open(os.path.join(proj, "records", "item_01.json"), "w", encoding="utf-8") as f:
            json.dump({
                "id": "item_01",
                "asset_path": "outputs/w1/item_01.png",
                "judgment": {"status": "approved", "explanation": "bad short"},
                "provenance": {"prompt": "unrelated prompt words"},
            }, f)

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(desk.run(["index", "--project", proj, "--out", out_dir, "--backend", "none"]), EXIT_OK)
            self.assertEqual(
                desk.run(["report", "--project", proj, "--out", out_dir, "--backend", "none", "--draft", "fake"]),
                EXIT_OK,
            )

        ws_path = os.path.join(out_dir, "description-worksheet.md")
        with open(ws_path, encoding="utf-8") as f:
            ws_lines = f.read().splitlines()
        new_lines = []
        for ln in ws_lines:
            if ln.startswith("| `item_01` |"):
                cells = ln.split("|")
                cells[7] = " stone wellhead beside stacked cedar buckets in courtyard "
                ln = "|".join(cells)
            new_lines.append(ln)
        with open(ws_path, "w", encoding="utf-8") as f:
            f.write("\n".join(new_lines) + "\n")

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(
                desk.run(["import-edits", "--project", proj, "--out", out_dir, "--seat", "seat-r1"]),
                EXIT_OK,
            )

        edits = description_gate.load_seat_edits(out_dir)
        expected_d_writer = description_gate.drafter_writer_id(
            backends.FakeDrafter.drafter_id,
            backends.FakeDrafter.revision,
            backends.DEFAULT_DRAFTER_PROMPT,
        )
        self.assertTrue(edits["item_01"].get("draft_text"))
        self.assertEqual(edits["item_01"].get("draft_writer"), expected_d_writer)

        # Re-run `report` WITHOUT `--draft` (overwriting description-review.jsonl without live drafts), then `compose --draft none`
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(desk.run(["report", "--project", proj, "--out", out_dir, "--backend", "none"]), EXIT_OK)
            self.assertEqual(desk.run(["compose", "--project", proj, "--out", out_dir, "--draft", "none"]), EXIT_OK)

        with open(os.path.join(out_dir, "captions-manifest.json"), encoding="utf-8") as f:
            mf = json.load(f)
        self.assertEqual(mf["items"][0]["author_kind"], "captioner_assisted")
        self.assertEqual(mf["items"][0]["drafter_writer"], expected_d_writer)

        with open(os.path.join(out_dir, "description-review.jsonl"), encoding="utf-8") as f:
            rev_row = json.loads(f.readline())
        self.assertEqual(rev_row["author_kind"], "captioner_assisted")

    def test_part_d3_drafted_review_rows_carry_writers_and_import_edits_appends(self):
        # Fails at 5248711 dataset_desk/description_gate.py:1038, :1519-1529, :1959-1962 (writers None, then overwritten)
        proj = os.path.join(self.tmp, "writers-proj")
        out_dir = os.path.join(self.tmp, "out-writers")
        os.makedirs(os.path.join(proj, "records"), exist_ok=True)
        _write_png(os.path.join(proj, "outputs", "w1", "w_01.png"), (65, 125, 95), seed=151)
        with open(os.path.join(proj, "records", "w_01.json"), "w", encoding="utf-8") as f:
            json.dump({
                "id": "w_01",
                "asset_path": "outputs/w1/w_01.png",
                "judgment": {"status": "approved", "explanation": "too short"},
                "provenance": {"prompt": "prompt words"},
            }, f)

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(desk.run(["index", "--project", proj, "--out", out_dir, "--backend", "none"]), EXIT_OK)
            self.assertEqual(
                desk.run(["report", "--project", proj, "--out", out_dir, "--backend", "none", "--draft", "fake"]),
                EXIT_OK,
            )

        rev_path = os.path.join(out_dir, "description-review.jsonl")
        with open(rev_path, encoding="utf-8") as f:
            before_row = json.loads(f.readline())
        expected_d_writer = description_gate.drafter_writer_id(
            backends.FakeDrafter.drafter_id,
            backends.FakeDrafter.revision,
            backends.DEFAULT_DRAFTER_PROMPT,
        )
        self.assertEqual(before_row["writers"], ["record:records/w_01.json", expected_d_writer])

        ws_path = os.path.join(out_dir, "description-worksheet.md")
        with open(ws_path, encoding="utf-8") as f:
            ws_text = f.read()
        edited_lines = []
        for ln in ws_text.splitlines():
            if ln.startswith("| `w_01` |"):
                cells = ln.split("|")
                cells[7] = " stone parapet overlooking calm harbour water at dusk "
                ln = "|".join(cells)
            edited_lines.append(ln)
        with open(ws_path, "w", encoding="utf-8") as f:
            f.write("\n".join(edited_lines) + "\n")

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(
                desk.run(["import-edits", "--project", proj, "--out", out_dir, "--seat", "seat-append"]),
                EXIT_OK,
            )

        with open(rev_path, encoding="utf-8") as f:
            after_row = json.loads(f.readline())
        self.assertEqual(
            after_row["writers"],
            ["record:records/w_01.json", expected_d_writer, "seat:seat-append"],
        )

    def test_part_d4_unknown_drafter_defaults_is_arm_model_true_on_locked_pool(self):
        # Fails at 5248711 dataset_desk/desk.py:627 (getattr(drafter, 'is_arm_model', False) defaulted to False)
        locked_sr = os.path.join(self.tmp, "salt-road")
        os.makedirs(locked_sr, exist_ok=True)
        unknown_drafter = types.SimpleNamespace(name="mystery-vlm-drafter")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(DeskError) as cm:
            desk.check_drafter_pool(locked_sr, unknown_drafter, allow_multimodal_pool=False)
        self.assertEqual(cm.exception.code, "PERM_LOCKED_POOL")

    def test_part_e_all_testcases_inherit_isolated_env_and_ignore_real_user_config(self):
        # Fails at 5248711 tests/fakes.py & tests/test_*.py (no IsolatedDeskTestCase; real user config read from HOME/APPDATA)
        import importlib

        fake_home = os.path.join(self.tmp, "fake-home")
        fake_appdata = os.path.join(self.tmp, "fake-appdata")
        for bad_cfg in (
            os.path.join(fake_home, ".config", "dataset-desk", "config.json"),
            os.path.join(fake_appdata, "dataset-desk", "config.json"),
        ):
            os.makedirs(os.path.dirname(bad_cfg), exist_ok=True)
            with open(bad_cfg, "w", encoding="utf-8") as f:
                f.write("{malformed real developer user config")

        proj = os.path.join(self.tmp, "plain-proj")
        os.makedirs(proj, exist_ok=True)
        with mock.patch.dict(os.environ, {"HOME": fake_home, "USERPROFILE": fake_home, "APPDATA": fake_appdata}):
            # IsolatedDeskTestCase set DATASET_DESK_USER_CONFIG="", so the broken HOME/APPDATA config is ignored
            self.assertEqual(os.environ.get("DATASET_DESK_USER_CONFIG"), "")
            with contextlib.redirect_stderr(io.StringIO()):
                pd = desk.check_multimodal_pool(proj, "none")
            self.assertEqual(pd["status"], "off")

            # Confirm that without DATASET_DESK_USER_CONFIG, the broken developer config WOULD fail every command
            with mock.patch.dict(os.environ):
                os.environ.pop("DATASET_DESK_USER_CONFIG", None)
                with self.assertRaises(DeskError) as cm_unisolated:
                    desk.check_multimodal_pool(proj, "none")
                self.assertEqual(cm_unisolated.exception.code, "INPUT_BAD_ARGUMENT")

        tests_dir = os.path.dirname(os.path.abspath(__file__))
        checked_classes = 0
        for fn in sorted(os.listdir(tests_dir)):
            if not (fn.startswith("test_") and fn.endswith(".py")):
                continue
            mod_base = fn[:-3]
            mod = None
            for mod_name in (mod_base, f"tests.{mod_base}"):
                try:
                    mod = importlib.import_module(mod_name)
                    break
                except ImportError:
                    continue
            self.assertIsNotNone(mod, fn)
            for attr_val in vars(mod).values():
                if (
                    isinstance(attr_val, type)
                    and issubclass(attr_val, unittest.TestCase)
                    and attr_val is not unittest.TestCase
                ):
                    checked_classes += 1
                    self.assertTrue(
                        any(base.__name__ == "IsolatedDeskTestCase" for base in attr_val.__mro__),
                        f"{fn}:{attr_val.__name__} does not inherit IsolatedDeskTestCase",
                    )
        self.assertGreaterEqual(checked_classes, 30)


class Addendum3bComposeDraftingTests(fakes.IsolatedDeskTestCase):
    """Synthetic tests for Addendum 3b (Parts 1-4); every test fails at 1c0f656."""

    def setUp(self):
        super().setUp()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name

    @staticmethod
    def _set_worksheet_edit(ws_path: str, row_id: str, edited_text: str) -> None:
        with open(ws_path, encoding="utf-8") as f:
            lines = f.read().splitlines()
        edit_col = -1
        out_lines: list[str] = []
        for ln in lines:
            s = ln.strip()
            if not s.startswith("|"):
                edit_col = -1
                out_lines.append(ln)
                continue
            cells = ln.split("|")
            lower = [c.strip().lower() for c in cells]
            if "id" in lower and "edited text" in lower:
                edit_col = lower.index("edited text")
                out_lines.append(ln)
                continue
            if edit_col >= 0 and s.startswith(f"| `{row_id}` |"):
                cells[edit_col] = f" {edited_text} "
                ln = "|".join(cells)
            out_lines.append(ln)
        with open(ws_path, "w", encoding="utf-8") as f:
            f.write("\n".join(out_lines) + "\n")

    def test_1_unflagged_human_record_untouched_by_default(self):
        # Fails at 1c0f656 dataset_desk/description_gate.py:2220 and :2341-2348 (drafted every unedited record and
        # replaced human text/writers/author_kind with unedited drafter output).
        proj = os.path.join(self.tmp, "proj-human")
        os.makedirs(os.path.join(proj, "records"), exist_ok=True)
        _write_png(os.path.join(proj, "outputs", "w1", "h_01.png"), (65, 110, 145), seed=201)
        with open(os.path.join(proj, "caption-rules.json"), "w", encoding="utf-8") as f:
            json.dump({"prefix": "copperplate study,", "drafter": "fake"}, f, indent=2)
        human_text = "copperplate study, weathered limestone arch above flagged stone steps"
        with open(os.path.join(proj, "records", "h_01.json"), "w", encoding="utf-8") as f:
            json.dump({
                "id": "h_01",
                "asset_path": "outputs/w1/h_01.png",
                "judgment": {"status": "approved", "explanation": human_text, "author_kind": "human"},
                "provenance": {"prompt": "distant coastal fog"},
            }, f)

        # Spy drafter must not even be invoked when every record passes the model-free gate
        class _NeverCalledDrafter:
            drafter_id = "spy-drafter"
            revision = "spy-v1"
            is_arm_model = False

            def draft(self, paths):
                raise AssertionError(f"drafter must not be called on unflagged records, got {paths!r}")

        out_spy = os.path.join(self.tmp, "out-spy")
        with contextlib.redirect_stderr(io.StringIO()):
            spy_comp = description_gate.compose_captions(proj, out_spy, drafter=_NeverCalledDrafter())
        self.assertEqual(spy_comp["items"][0]["caption"], human_text)
        self.assertEqual(spy_comp["items"][0]["author_kind"], "human")
        self.assertEqual(
            spy_comp["items"][0]["writers"],
            ["config:caption-rules.json:2", "record:records/h_01.json"],
        )

        # CLI `compose` (default drafting) and `report --compose` both leave h_01 untouched
        out_comp = os.path.join(self.tmp, "out-comp")
        out_rep = os.path.join(self.tmp, "out-rep")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(desk.run(["compose", "--project", proj, "--out", out_comp]), EXIT_OK)
            self.assertEqual(desk.run(["index", "--project", proj, "--out", out_rep, "--backend", "none"]), EXIT_OK)
            self.assertEqual(
                desk.run(["report", "--project", proj, "--out", out_rep, "--backend", "none", "--compose"]),
                EXIT_OK,
            )

        for check_out in (out_comp, out_rep):
            with open(os.path.join(check_out, "captions-manifest.json"), encoding="utf-8") as f:
                mf = json.load(f)
            it = mf["items"][0]
            self.assertEqual(it["caption"], human_text)
            self.assertEqual(it["author_kind"], "human")
            self.assertEqual(it["writers"], ["config:caption-rules.json:2", "record:records/h_01.json"])
            self.assertIsNone(it["drafter_writer"])

            with open(os.path.join(check_out, "description-review.jsonl"), encoding="utf-8") as f:
                rev = json.loads(f.readline())
            self.assertEqual(rev["author_kind"], "human")
            self.assertEqual(rev["writers"], ["config:caption-rules.json:2", "record:records/h_01.json"])
            self.assertEqual(rev["draft"], "")
            self.assertIsNone(rev["draft_offer"])
            self.assertIsNone(rev["draft_offer_writer"])

    def test_2_unflagged_sidecar_record_untouched_by_default(self):
        # Fails at 1c0f656 dataset_desk/description_gate.py:2220 and :2341-2348 (drafted unflagged sidecar records
        # and replaced sidecar text/writers/author_kind with unedited drafter output).
        sc_proj = os.path.join(self.tmp, "proj-sidecar")
        os.makedirs(sc_proj, exist_ok=True)
        _write_png(os.path.join(sc_proj, "plate_01.png"), (90, 125, 110), seed=202)
        sidecar_text = "copperplate study, carved timber gate set between mortared stone piers"
        with open(os.path.join(sc_proj, "plate_01.txt"), "w", encoding="utf-8") as f:
            f.write(sidecar_text)
        with open(os.path.join(sc_proj, "caption-rules.json"), "w", encoding="utf-8") as f:
            json.dump({"prefix": "copperplate study,", "drafter": "fake"}, f, indent=2)

        out_comp = os.path.join(self.tmp, "out-sc-comp")
        out_rep = os.path.join(self.tmp, "out-sc-rep")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(desk.run(["compose", "--project", sc_proj, "--out", out_comp]), EXIT_OK)
            self.assertEqual(
                desk.run(["index", "--project", sc_proj, "--out", out_rep, "--backend", "none", "--sidecars"]),
                EXIT_OK,
            )
            self.assertEqual(
                desk.run(["report", "--project", sc_proj, "--out", out_rep, "--backend", "none", "--compose"]),
                EXIT_OK,
            )

        for check_out in (out_comp, out_rep):
            with open(os.path.join(check_out, "captions-manifest.json"), encoding="utf-8") as f:
                mf = json.load(f)
            it = mf["items"][0]
            self.assertEqual(it["caption"], sidecar_text)
            self.assertEqual(it["author_kind"], "sidecar")
            self.assertEqual(it["writers"], ["config:caption-rules.json:2", "sidecar:plate_01.txt"])
            self.assertIsNone(it["drafter_writer"])

            with open(os.path.join(check_out, "description-review.jsonl"), encoding="utf-8") as f:
                rev = json.loads(f.readline())
            self.assertEqual(rev["author_kind"], "sidecar")
            self.assertEqual(rev["writers"], ["config:caption-rules.json:2", "sidecar:plate_01.txt"])
            self.assertEqual(rev["draft"], "")
            self.assertIsNone(rev["draft_offer"])
            self.assertIsNone(rev["draft_offer_writer"])

    def test_3_all_flag_keeps_unflagged_human_text_and_writes_draft_only_as_offer(self):
        # Fails at 1c0f656 dataset_desk/desk.py:3506-3524 (no `--all` CLI flag), description_gate.py:1755-1795
        # (no `draft_offer` / `draft_offer_writer` fields), and description_gate.py:2341-2348 (replaced human text).
        proj = os.path.join(self.tmp, "proj-all")
        os.makedirs(os.path.join(proj, "records"), exist_ok=True)
        _write_png(os.path.join(proj, "outputs", "w1", "h_01.png"), (80, 115, 150), seed=203)
        with open(os.path.join(proj, "caption-rules.json"), "w", encoding="utf-8") as f:
            json.dump({"prefix": "copperplate study,", "drafter": "fake"}, f, indent=2)
        human_text = "copperplate study, weathered limestone arch above flagged stone steps"
        with open(os.path.join(proj, "records", "h_01.json"), "w", encoding="utf-8") as f:
            json.dump({
                "id": "h_01",
                "asset_path": "outputs/w1/h_01.png",
                "judgment": {"status": "approved", "explanation": human_text, "author_kind": "human"},
                "provenance": {"prompt": "distant coastal fog"},
            }, f)

        out_all = os.path.join(self.tmp, "out-all")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(desk.run(["compose", "--project", proj, "--out", out_all, "--all"]), EXIT_OK)

        with open(os.path.join(out_all, "captions-manifest.json"), encoding="utf-8") as f:
            mf = json.load(f)
        it = mf["items"][0]
        # Human text, writers, and author_kind are NEVER replaced by --all
        self.assertEqual(it["caption"], human_text)
        self.assertEqual(it["author_kind"], "human")
        self.assertEqual(it["writers"], ["config:caption-rules.json:2", "record:records/h_01.json"])
        self.assertIsNone(it["drafter_writer"])

        # Draft appears ONLY as the offer field (with its writer) in description-review.jsonl and worksheet
        expected_offer_writer = description_gate.drafter_writer_id(
            backends.FakeDrafter.drafter_id,
            backends.FakeDrafter.revision,
            backends.DEFAULT_DRAFTER_PROMPT,
        )
        with open(os.path.join(out_all, "description-review.jsonl"), encoding="utf-8") as f:
            rev = json.loads(f.readline())
        self.assertEqual(rev["author_kind"], "human")
        self.assertEqual(rev["writers"], ["config:caption-rules.json:2", "record:records/h_01.json"])
        self.assertEqual(rev["draft"], "")
        self.assertEqual(rev["florence2_draft"], "")
        self.assertIsNone(rev["drafter_id"])
        self.assertIsNone(rev["drafter_writer"])
        self.assertTrue(
            rev["draft_offer"].startswith("a weathered ")
            and rev["draft_offer"].endswith("beside carved stone pillars in a dim courtyard (h_01)"),
            rev["draft_offer"],
        )
        self.assertEqual(rev["draft_offer_writer"], expected_offer_writer)

        ws_path = os.path.join(out_all, "description-worksheet.md")
        with open(ws_path, encoding="utf-8") as f:
            ws_text = f.read()
        self.assertIn("draft_offer", ws_text)
        self.assertIn(expected_offer_writer, ws_text)
        self.assertIn(human_text, ws_text)

        # Only when a seat edit accepts/edits the offer does the caption change
        self._set_worksheet_edit(ws_path, "h_01", "carved limestone archway above worn flagstones")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(
                desk.run(["import-edits", "--project", proj, "--out", out_all, "--seat", "seat-offer"]),
                EXIT_OK,
            )
            self.assertEqual(
                desk.run(["compose", "--project", proj, "--out", out_all, "--draft", "none"]),
                EXIT_OK,
            )
        with open(os.path.join(out_all, "captions-manifest.json"), encoding="utf-8") as f:
            mf_after = json.load(f)
        it_after = mf_after["items"][0]
        self.assertEqual(
            it_after["caption"],
            "copperplate study, carved limestone archway above worn flagstones",
        )
        self.assertEqual(it_after["author_kind"], "captioner_assisted")
        self.assertEqual(it_after["writers"], ["config:caption-rules.json:2", "seat:seat-offer"])
        self.assertEqual(it_after["drafter_writer"], expected_offer_writer)

    def test_4_flagged_record_gets_draft_in_review_queue_and_caption_unchanged_until_seat_edit(self):
        # Fails at 1c0f656 dataset_desk/description_gate.py:2341-2348 (wrote unedited draft straight into the
        # training caption in captions-manifest.json with author_kind 'captioner_draft_unedited' before any seat edit).
        proj = os.path.join(self.tmp, "proj-flagged")
        os.makedirs(os.path.join(proj, "records"), exist_ok=True)
        _write_png(os.path.join(proj, "outputs", "w1", "f_01.png"), (145, 95, 75), seed=204)
        with open(os.path.join(proj, "caption-rules.json"), "w", encoding="utf-8") as f:
            json.dump({"prefix": "copperplate study,", "drafter": "fake"}, f, indent=2)
        with open(os.path.join(proj, "records", "f_01.json"), "w", encoding="utf-8") as f:
            json.dump({
                "id": "f_01",
                "asset_path": "outputs/w1/f_01.png",
                "judgment": {"status": "approved", "explanation": "short bad"},
                "provenance": {"prompt": "misty quay"},
            }, f)

        out_nodraft = os.path.join(self.tmp, "out-nodraft")
        out_drafted = os.path.join(self.tmp, "out-drafted")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(
                desk.run(["compose", "--project", proj, "--out", out_nodraft, "--draft", "none"]),
                EXIT_OK,
            )
            self.assertEqual(
                desk.run(["compose", "--project", proj, "--out", out_drafted]),
                EXIT_OK,
            )

        with open(os.path.join(out_nodraft, "captions-manifest.json"), encoding="utf-8") as f:
            mf_nodraft = json.load(f)
        with open(os.path.join(out_drafted, "captions-manifest.json"), encoding="utf-8") as f:
            mf_drafted = json.load(f)

        # No change to its training caption, writers, or author_kind before a seat edit is imported
        self.assertEqual(mf_drafted["items"][0]["caption"], mf_nodraft["items"][0]["caption"])
        self.assertEqual(mf_drafted["items"][0]["caption"], "copperplate study, short bad")
        self.assertEqual(mf_drafted["items"][0]["author_kind"], "human")
        self.assertEqual(
            mf_drafted["items"][0]["writers"],
            ["config:caption-rules.json:2", "record:records/f_01.json"],
        )
        self.assertIsNone(mf_drafted["items"][0]["drafter_writer"])

        # Draft IS in the review queue and worksheet for the seat
        expected_d_writer = description_gate.drafter_writer_id(
            backends.FakeDrafter.drafter_id,
            backends.FakeDrafter.revision,
            backends.DEFAULT_DRAFTER_PROMPT,
        )
        with open(os.path.join(out_drafted, "description-review.jsonl"), encoding="utf-8") as f:
            rev_before = json.loads(f.readline())
        self.assertFalse(rev_before["passed_model_free"])
        self.assertTrue(
            rev_before["draft"].startswith("a weathered ")
            and rev_before["draft"].endswith("beside carved stone pillars in a dim courtyard (f_01)"),
            rev_before["draft"],
        )
        self.assertEqual(rev_before["drafter_writer"], expected_d_writer)

        ws_path = os.path.join(out_drafted, "description-worksheet.md")
        with open(ws_path, encoding="utf-8") as f:
            ws_before = f.read()
        self.assertIn("| `f_01` |", ws_before)
        self.assertIn("beside carved stone pillars in a dim courtyard (f_01)", ws_before)

        # Import a seat edit -> NOW compose updates the training caption
        self._set_worksheet_edit(ws_path, "f_01", "weathered stone arch spanning narrow cobbled lane")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(
                desk.run(["import-edits", "--project", proj, "--out", out_drafted, "--seat", "seat-fix"]),
                EXIT_OK,
            )
            self.assertEqual(
                desk.run(["compose", "--project", proj, "--out", out_drafted, "--draft", "none"]),
                EXIT_OK,
            )

        with open(os.path.join(out_drafted, "captions-manifest.json"), encoding="utf-8") as f:
            mf_after = json.load(f)
        it_after = mf_after["items"][0]
        self.assertEqual(
            it_after["caption"],
            "copperplate study, weathered stone arch spanning narrow cobbled lane",
        )
        self.assertEqual(it_after["author_kind"], "captioner_assisted")
        self.assertEqual(it_after["writers"], ["config:caption-rules.json:2", "seat:seat-fix"])
        self.assertEqual(it_after["drafter_writer"], expected_d_writer)


if __name__ == "__main__":
    unittest.main()
