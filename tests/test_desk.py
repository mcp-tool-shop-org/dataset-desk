"""Offline tests for the dataset desk: a tiny project of patterned images, and the fake embedder (no model, no GPU).
Run from the repository root: python -W error -m unittest discover -s tests   (needs numpy and Pillow)"""

import contextlib
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

import fakes
from dataset_desk import backends, description_gate, desk
from dataset_desk.errors import DeskError

TINT = {"red": (175, 0, 0), "green": (0, 175, 0), "blue": (0, 0, 175)}


def pattern(path, seed, colour, poke=0):
    """A noise texture tinted by a colour; poke changes a few pixels for a near-duplicate."""
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 80, (32, 32, 3)).astype(np.int16) + np.array(TINT[colour], dtype=np.int16)
    for i in range(poke):
        a[i, i] = a[i, i] + 30
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).save(path)


def record(project, rid, asset, status, explanation, *, subject="", failure_modes=None, prompt=None):
    os.makedirs(os.path.join(project, "records"), exist_ok=True)
    rec = {
        "id": rid,
        "asset_path": asset,
        "judgment": {"status": status, "explanation": explanation},
        "provenance": {"prompt": prompt if prompt is not None else f"prompt for {rid}"},
    }
    if subject:
        rec["canon"] = {"subject": subject}
    if failure_modes is not None:
        rec["judgment"]["failure_modes"] = list(failure_modes)
    with open(os.path.join(project, "records", f"{rid}.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f)


class Project(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        P = self.P = os.path.join(self.tmp, "proj")
        self.out = os.path.join(self.tmp, "out")
        o = os.path.join(P, "outputs")
        pattern(f"{o}/approved/red1.png", 1, "red")
        pattern(f"{o}/approved/red4.png", 4, "red")
        pattern(f"{o}/approved/blue1.png", 2, "blue")
        pattern(f"{o}/approved/green1.png", 3, "green")
        os.makedirs(f"{o}/wave1")
        shutil.copyfile(f"{o}/approved/red1.png", f"{o}/wave1/red1-copy.png")
        pattern(f"{o}/wave1/red2.png", 1, "red", poke=2)      # near-duplicate of red1
        pattern(f"{o}/wave1/red3.png", 9, "red")              # same look, different picture
        pattern(f"{o}/sheet.png", 5, "blue")                  # top of outputs/: a contact sheet, left out
        record(P, "red1", "outputs/approved/red1.png", "approved", "a red mound creature")
        record(P, "red4", "outputs/approved/red4.png", "approved", "a red beast")
        record(P, "blue1", "outputs/approved/blue1.png", "approved", "a blue lantern")
        record(P, "green1", "outputs/approved/green1.png", "approved", "a red thing")  # disagrees with its image
        record(P, "red2", "outputs/wave1/red2.png", "rejected", "a red mound, worse",
               failure_modes=["style_residue_prop", "anatomy_defect"])

    def index(self, limit=None, images_only=False, embedder=None):
        return desk.index(
            self.P,
            self.out,
            embedder or backends.FakeEmbedder(),
            limit=limit,
            images_only=images_only,
            log=lambda *a: None,
        )

    def loaded(self):
        self.index()
        items, img, txt, meta = desk.load_cache(self.out, "fake")
        return items, img, txt, meta, {os.path.basename(it.paths[0]): i for i, it in enumerate(items)}


class ThreadCaps(unittest.TestCase):
    def test_import_desk_with_no_env_set_caps_blas_and_omp_threads(self):
        desk_dir = os.path.dirname(os.path.abspath(desk.__file__))
        env = {
            k: v for k, v in os.environ.items()
            if k not in (
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
                "OMP_WAIT_POLICY", "TOKENIZERS_PARALLELISM", "RAYON_NUM_THREADS",
            )
        }
        code = (
            "import json, os; from dataset_desk import desk; "
            "keys = ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS',"
            "'NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS',"
            "'OMP_WAIT_POLICY','TOKENIZERS_PARALLELISM','RAYON_NUM_THREADS']; "
            "print(json.dumps({k: os.environ.get(k) for k in keys}))"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=os.path.dirname(desk_dir),
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        got = json.loads(proc.stdout.strip())
        self.assertEqual(got["OPENBLAS_NUM_THREADS"], "4")
        self.assertEqual(got["MKL_NUM_THREADS"], "4")
        self.assertEqual(got["OMP_NUM_THREADS"], "4")
        self.assertEqual(got["NUMEXPR_NUM_THREADS"], "4")
        self.assertEqual(got["OMP_WAIT_POLICY"], "PASSIVE")
        self.assertEqual(got["TOKENIZERS_PARALLELISM"], "false")
        self.assertEqual(got["RAYON_NUM_THREADS"], "1")

    def test_enter_torch_region_forces_blas_to_one_and_guards_interop(self):
        with mock.patch.dict(sys.modules, {"torch": fakes.make_torch()}):
            backends.enter_torch_region(3)
            self.assertEqual(os.environ.get("OPENBLAS_NUM_THREADS"), "1")
            self.assertEqual(os.environ.get("MKL_NUM_THREADS"), "1")
            # Calling a second time must not raise RuntimeError from torch.set_num_interop_threads
            backends.enter_torch_region(3)
            backends.cap_threads(4)


class Scan(Project):
    def test_scan_merges_copies_and_leaves_out_the_top_of_outputs(self):
        items, stats = desk.scan(self.P)
        self.assertEqual((stats["files"], stats["unique"], stats["records"]), (7, 6, 5))
        red1 = next(it for it in items if "outputs/approved/red1.png" in it.paths)
        self.assertEqual(sorted(red1.paths), ["outputs/approved/red1.png", "outputs/wave1/red1-copy.png"])
        self.assertEqual((red1.status, red1.record["explanation"]), ("approved", "a red mound creature"))
        self.assertFalse(any("sheet.png" in p for it in items for p in it.paths))
        self.assertEqual(next(it for it in items if it.paths == ["outputs/wave1/red3.png"]).status, "unrecorded")

    def test_bad_and_orphaned_records_are_reported(self):
        with open(os.path.join(self.P, "records", "broken.json"), "w", encoding="utf-8") as f:
            f.write("{not json")
        record(self.P, "ghost", "outputs/approved/ghost.png", "approved", "gone")
        _, stats = desk.scan(self.P)
        self.assertTrue(any(p.startswith("broken.json") for p in stats["problems"]))
        self.assertEqual(stats["records_without_image"], ["outputs/approved/ghost.png"])

    def test_dhash_finds_the_near_duplicate_and_only_it(self):
        o = os.path.join(self.P, "outputs")
        h = {n: desk.dhash(f"{o}/{n}") for n in ("approved/red1.png", "wave1/red2.png", "wave1/red3.png")}
        self.assertLessEqual(desk.hamming(h["approved/red1.png"], h["wave1/red2.png"]), desk.NEAR_BITS)
        self.assertGreater(desk.hamming(h["approved/red1.png"], h["wave1/red3.png"]), desk.NEAR_BITS)
        self.assertEqual(len(h["approved/red1.png"]), 64)

    def test_canon_subject_preferred_over_director_ruling(self):
        with open(os.path.join(self.P, "records", "red1.json"), "w", encoding="utf-8") as f:
            json.dump({
                "id": "red1",
                "asset_path": "outputs/approved/red1.png",
                "canon": {"subject": "sunlit salt-crust road beside an ochre storehouse"},
                "judgment": {"status": "approved", "explanation": "Director ruling 2026-07-30: pool look approved"},
                "provenance": {"prompt": "salt-road style, sunlit road"},
            }, f)
        items, _ = desk.scan(self.P)
        red1 = next(it for it in items if "outputs/approved/red1.png" in it.paths)
        self.assertEqual(red1.record["explanation"], "sunlit salt-crust road beside an ochre storehouse")
        self.assertEqual(red1.record["judgment_note"], "Director ruling 2026-07-30: pool look approved")

    def test_failure_modes_loaded_and_rejection_notes_not_used_as_descriptions(self):
        items, _ = desk.scan(self.P)
        red2 = next(it for it in items if "outputs/wave1/red2.png" in it.paths)
        self.assertEqual(red2.status, "rejected")
        self.assertIsNone(red2.record["explanation"])
        self.assertEqual(red2.record["judgment_note"], "a red mound, worse")
        self.assertEqual(red2.record["failure_modes"], ["style_residue_prop", "anatomy_defect"])
        self.assertIn("style_residue_prop", desk._status_label(red2))

    def test_byte_identical_files_with_different_records_prefer_approved_deterministically(self):
        # a_early/copy.png is walked BEFORE outputs/approved/red1.png alphabetically
        early_rel = "outputs/a_early/red1-early.png"
        early_abs = os.path.join(self.P, early_rel)
        os.makedirs(os.path.dirname(early_abs), exist_ok=True)
        shutil.copyfile(os.path.join(self.P, "outputs/approved/red1.png"), early_abs)
        record(self.P, "red1_early", early_rel, "rejected", "rejected early copy",
               failure_modes=["duplicate_copy"])
        items, stats = desk.scan(self.P)
        red1 = next(it for it in items if "outputs/approved/red1.png" in it.paths)
        self.assertEqual(red1.status, "approved")
        self.assertEqual(red1.record["id"], "red1")
        self.assertEqual(red1.record["explanation"], "a red mound creature")
        self.assertTrue(any("record collision" in p for p in stats["problems"]))

    def test_wave_generation_curation_and_png_sampler_positive_vs_negative(self):
        wdir = os.path.join(self.P, "outputs", "wave1")
        with open(os.path.join(wdir, "generation.json"), "w", encoding="utf-8") as f:
            json.dump({
                "wave": "wave1-test",
                "base": "qwen-image",
                "unet": "qwen_image_fp8_e4m3fn.safetensors",
                "style_prefix": "tallow-fen style",
                "loras": [{"name": "tallow_fen_style_v3.safetensors", "weight": 1.5}],
                "items": [{"id": "red3_subj", "variation": 0, "seed": 4242, "file": "red3.png",
                           "prompt": "tallow-fen style, a bog creature", "status": "ok"}],
            }, f)
        with open(os.path.join(wdir, "_curation.json"), "w", encoding="utf-8") as f:
            json.dump({"passed": [], "failed": [{"file": "red3.png", "votes": 0, "decisions": 3,
                                                 "issues": ["leaked lantern prop"]}]}, f)
        # Write an image with ComfyUI PNG tEXt prompt where the NEGATIVE prompt is much longer than the positive prompt
        from PIL.PngImagePlugin import PngInfo
        png_path = os.path.join(wdir, "blue_pngmeta.png")
        rng = np.random.default_rng(77)
        arr = np.clip(rng.integers(0, 60, (32, 32, 3)) + np.array(TINT["blue"]), 0, 255).astype(np.uint8)
        long_negative = (
            "photorealistic, 3d render, smooth plastic skin, neon cyberpunk glow, watermark, signature, "
            "blurry, low quality, deformed hands, extra limbs, oversaturated, modern signage, lens flare"
        )
        info = PngInfo()
        info.add_text("prompt", json.dumps({
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": 9001,
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                },
            },
            "37": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen_image_fp8_e4m3fn.safetensors"}},
            "40": {"class_type": "LoraLoaderModelOnly",
                   "inputs": {"lora_name": "rustline_v3.safetensors", "strength_model": 1.25}},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "rustline style, a blue crane"}},
            "7": {"class_type": "CLIPTextEncode", "inputs": {"text": long_negative}},
        }))
        Image.fromarray(arr).save(png_path, pnginfo=info)

        items, _ = desk.scan(self.P)
        red3 = next(it for it in items if it.paths == ["outputs/wave1/red3.png"])
        self.assertEqual(red3.status, "unrecorded")
        self.assertEqual(red3.record["seed"], 4242)
        self.assertEqual(red3.record["loras"], [{"name": "tallow_fen_style_v3.safetensors", "weight": 1.5}])
        self.assertEqual(red3.record["curation_status"], "failed")
        self.assertEqual(red3.record["curation_issues"], ["leaked lantern prop"])

        blue_m = next(it for it in items if it.paths == ["outputs/wave1/blue_pngmeta.png"])
        self.assertEqual(blue_m.status, "unrecorded")
        self.assertEqual(blue_m.record["seed"], 9001)
        self.assertEqual(blue_m.record["loras"], [{"name": "rustline_v3.safetensors", "weight": 1.25}])
        self.assertEqual(blue_m.record["prompt"], "rustline style, a blue crane")
        self.assertEqual(blue_m.record["negative_prompt"], long_negative)

    def test_corrupt_image_logged_or_strict_halted(self):
        bad = os.path.join(self.P, "outputs", "wave1", "corrupt.png")
        with open(bad, "wb") as f:
            f.write(b"not a valid png bitstream")
        items, stats = desk.scan(self.P, strict=False)
        self.assertTrue(any("corrupt.png: unreadable image" in p for p in stats["problems"]))
        self.assertFalse(any("corrupt.png" in p for it in items for p in it.paths))
        with self.assertRaises((OSError, ValueError, SyntaxError)):
            desk.scan(self.P, strict=True)


class Index(Project):
    def test_a_rerun_embeds_only_what_is_new_and_shares_scan_cache(self):
        meta = self.index()
        # 6 unique images; 4 approved descriptions (rejected red2's note is not embedded)
        self.assertEqual((meta["embedded_now"], meta["texts_now"]), (6, 4))
        self.assertEqual(meta["look_computed_now"], 6)
        self.assertIn("look_seconds_per_image", meta)
        self.assertTrue(os.path.exists(os.path.join(self.out, "scan.jsonl")))
        meta2 = self.index()
        self.assertEqual(meta2["embedded_now"], 0)
        self.assertEqual(meta2["look_computed_now"], 0)
        # Running a second backend ('none') reuses scan.jsonl without recomputing look features
        meta_none = self.index(embedder=backends.NoneEmbedder())
        self.assertEqual(meta_none["look_computed_now"], 0)
        pattern(os.path.join(self.P, "outputs", "wave1", "blue9.png"), 11, "blue")
        meta3 = self.index()
        self.assertEqual(meta3["embedded_now"], 1)
        self.assertEqual(meta3["look_computed_now"], 1)

    def test_limit_and_cache_shape(self):
        self.index(limit=3)
        items, img, txt, meta = desk.load_cache(self.out, "fake")
        self.assertEqual((len(items), img.shape), (3, (3, 4)))
        self.assertTrue(np.allclose(np.linalg.norm(img, axis=1), 1, atol=1e-5))
        self.assertEqual(meta["revision"], "fake-1")
        self.assertEqual(meta["vision_tokens"], 280)

    def test_a_missing_cache_says_how_to_make_one(self):
        with self.assertRaises(SystemExit) as cm:
            desk.load_cache(self.out, "eg2")
        self.assertIn("dataset-desk index --backend eg2", str(cm.exception))


class Analyses(Project):
    def test_dupes_in_three_tiers(self):
        items, img, _, _, ix = self.loaded()
        d = desk.dupes(items, img, semantic=0.99)
        self.assertEqual([len(it.paths) for it in d["exact"]], [2])
        self.assertEqual({frozenset((a, b)) for a, b, _ in d["near"]}, {frozenset((ix["red1.png"], ix["red2.png"]))})
        sem = {frozenset((a, b)) for a, b, _ in d["semantic"]}
        self.assertIn(frozenset((ix["red1.png"], ix["red3.png"])), sem)
        self.assertNotIn(frozenset((ix["red1.png"], ix["red2.png"])), sem)  # already a near pair
        self.assertTrue({"p50", "p99", "max"} <= set(d["top1_percentiles"]))

    def test_outliers_and_nearest_canon(self):
        items, img, _, _, ix = self.loaded()
        o = desk.outliers(items, img, k=1)
        self.assertEqual({i for i, _ in o["least_typical"][:2]}, {ix["blue1.png"], ix["green1.png"]})
        self.assertIn(o["nearest_canon"][0][0], {ix["red2.png"], ix["red3.png"]})

    def test_a_description_that_disagrees_with_its_image_ranks_worst(self):
        items, img, txt, _, ix = self.loaded()
        c = desk.captions(items, img, txt)
        self.assertEqual(c["described"], 4)  # only the 4 approved items have embedded descriptions
        self.assertEqual(c["worst"][0]["item"], ix["green1.png"])
        self.assertGreater(c["worst"][0]["rank"], 1)

    def test_group_rank_and_text_disjoint_swap_check_with_shared_descriptions(self):
        # Give red1, red4, and red2 the exact same approved description (simulating rustline's shared subject prompts)
        for rid, asset in (("red1", "outputs/approved/red1.png"),
                           ("red4", "outputs/approved/red4.png"),
                           ("red2", "outputs/wave1/red2.png")):
            record(self.P, rid, asset, "approved", "shared red mech description")
        record(self.P, "green1", "outputs/approved/green1.png", "approved", "a green moss stone")
        items, img, txt, _, ix = self.loaded()
        c = desk.captions(items, img, txt)
        by_item = {r["item"]: r for r in c["worst"]}
        # red1 and red4 tie with their siblings in raw rank, but group_rank masks siblings so they rank 1.0
        self.assertEqual(by_item[ix["red1.png"]]["siblings"], 3)
        self.assertEqual(by_item[ix["red1.png"]]["group_rank"], 1.0)
        # Even though 3 of 5 items (>50%) share one description, swap_check deduplicates to unique descriptions
        # and completes without an infinite loop
        sw = desk.swap_check(items, img, txt, seed=0)
        self.assertIsNotNone(sw)
        self.assertEqual(sw["pairs"], 3)
        self.assertGreater(sw["auc"], 0.8)

    def test_mid_rank_splits_ties(self):
        self.assertEqual(desk.mid_rank(np.array([0.5, 0.5, 0.1]), 0), 1.5)
        self.assertEqual(desk.mid_rank(np.array([0.5, 0.5 + 5e-7, 0.1]), 0), 1.5)
        self.assertEqual(desk.mid_rank(np.array([0.5, 0.5 + 1e-4, 0.1]), 0), 2.0)
        self.assertEqual(desk.mid_rank(np.array([0.9, 0.5, 0.1]), 1), 2.0)

    def test_search_and_coverage(self):
        items, img, _, _, ix = self.loaded()
        emb = backends.FakeEmbedder()
        self.assertEqual(desk.search(img, emb.texts(["blue"], "query")[0], 1)[0][0], ix["blue1.png"])
        subjects = ["a red creature", "a purple bird", "a blue light"]
        cov = desk.coverage(items, img, subjects, emb.texts(subjects, "query"))
        self.assertEqual(cov[0]["subject"], "a purple bird")  # nothing like it: the gap comes first
        approved = set(desk.approved_idx(items))
        self.assertTrue(all(i in approved for row in cov for i, _ in row["top"]))
        self.assertTrue(all("path" in ex and "look_rank" in ex for row in cov for ex in row["exemplars"]))

    def test_vendi_and_dual_covariance(self):
        self.assertAlmostEqual(desk.vendi(np.eye(4, dtype=np.float32)), 4.0, places=4)
        self.assertAlmostEqual(desk.vendi(np.tile([[1.0, 0, 0]], (5, 1)).astype(np.float32)), 1.0, places=4)
        # Verify n > d dual covariance matches full n x n Gram eigensolve to machine precision
        rng = np.random.default_rng(123)
        vecs = backends.unit_rows(rng.standard_normal((25, 4)))
        lam_full = np.linalg.eigvalsh((vecs @ vecs.T) / len(vecs))
        lam_full = lam_full[lam_full > 1e-12]
        expected = float(np.exp(-(lam_full * np.log(lam_full)).sum()))
        self.assertAlmostEqual(desk.vendi(vecs), expected, places=5)
        items, img, *_ = self.loaded()
        rows = desk.diversity(items, img)
        self.assertEqual(rows[0][:2], ("approved", 4))

    def test_look_channel_has_30_bounded_features_and_safe_spread_floor(self):
        items, *_ = self.loaded()
        for it in items:
            self.assertEqual(len(it.palette), 30)
            self.assertTrue(all(0.0 <= x <= 1.0 for x in it.palette))
        fit = desk.look_fit(items, k=2)
        self.assertEqual(len(fit), 4)
        self.assertTrue(all(np.isfinite(v) and v < 50.0 for v in fit.values()))

    def test_eg2_guards_vision_tokens_and_multimodal_embedder(self):
        self.assertEqual(
            set(backends.EG2_PREFIX),
            {"query", "code_query", "document", "symmetric", "clustering", "classification"},
        )
        self.assertEqual(backends.EG2_VISION_TOKENS, (70, 140, 280, 560, 1120))
        with self.assertRaises(ValueError):
            backends.EG2Embedder(dim=128)
        with self.assertRaises(ValueError):
            backends.EG2Embedder(dtype="float16")
        with self.assertRaises(ValueError):
            backends.EG2Embedder(vision_tokens=16)
        with self.assertRaises(ValueError):
            backends.EG2Embedder(vision_tokens=200)  # must be one of (70, 140, 280, 560, 1120)
        eg = backends.EG2Embedder(vision_tokens=140)
        self.assertEqual(eg.vision_tokens, 140)
        self.assertEqual(eg.processor_kwargs, {"image_seq_length": 140})
        self.assertEqual(eg.processing_kwargs, {"image": {"max_soft_tokens": 140}})
        with self.assertRaises(ValueError):
            backends.unit_rows([1.0, float("nan"), 0.0])
        emb = backends.FakeEmbedder()
        mm = emb.multimodal([{"text": "blue <|image|>", "images": [os.path.join(self.P, "outputs/approved/red1.png")]}])
        self.assertEqual(mm.shape, (1, 4))
        self.assertAlmostEqual(float(np.linalg.norm(mm[0])), 1.0, places=5)


class DescriptionGateTests(unittest.TestCase):
    @staticmethod
    def _rec(rid, explanation, *, prompt="unrelated studio prompt words", subject="", status="approved", asset=None):
        return {
            "id": rid,
            "asset_path": asset or f"outputs/approved/{rid}.png",
            "canon": {"subject": subject} if subject else {},
            "judgment": {"status": status, "explanation": explanation},
            "provenance": {"prompt": prompt},
        }

    def test_stopword_file_hash_exposed_and_stable(self):
        self.assertTrue(os.path.isfile(description_gate.STOPWORDS_PATH))
        with open(description_gate.STOPWORDS_PATH, "rb") as f:
            expected = hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()
        self.assertEqual(len(expected), 64)
        self.assertEqual(description_gate.stopwords_sha256(), expected)
        self.assertEqual(description_gate.STOPWORDS_SHA256, expected)
        audit = description_gate.audit_records([
            self._rec("r1", "tall weathered lighthouse stands beside calm harbour water"),
        ])
        self.assertEqual(audit["stopwords_sha256"], expected)

    def test_boundary_length_words_and_content_words(self):
        # 8 words total, 5 content words ("tall weathered stone tower stands") + 3 stopwords ("in", "by", "the") -> PASS
        pass_8w_5c = self._rec("pass_len", "tall weathered stone tower stands in by the")
        # 7 words total, 5 distinct content words + 2 stopwords -> FAIL (fewer than 8 words)
        fail_7w_5c = self._rec("fail_7w", "rusted iron harbor crane tilts in the")
        # 8 words total, 4 distinct content words + 4 stopwords ("is in by the") -> FAIL (< 5 content)
        fail_8w_4c = self._rec("fail_4c", "mossy wooden barge anchor is in by the")

        audit = description_gate.audit_records(
            [pass_8w_5c, fail_7w_5c, fail_8w_4c],
            token_counter=lambda texts: [20 for _ in texts],
        )
        by_id = {r["id"]: r for r in audit["records"]}
        self.assertTrue(by_id["pass_len"]["passed"])
        self.assertNotIn("too_short", by_id["pass_len"]["flags"])

        self.assertFalse(by_id["fail_7w"]["passed"])
        self.assertIn("too_short", by_id["fail_7w"]["flags"])

        self.assertFalse(by_id["fail_4c"]["passed"])
        self.assertIn("too_short", by_id["fail_4c"]["flags"])

    def test_boundary_contiguous_run_5_passes_6_fails(self):
        prompt = "alpha bravo charlie delta echo foxtrot golf hotel india juliet"
        # Shares 5 contiguous words ("alpha bravo charlie delta echo"), 5 of 12 content words (0.417), 5 of 12 all words
        pass_run5 = self._rec(
            "run5",
            "alpha bravo charlie delta echo lantern harbour cobble timber barge anchor",
            prompt=prompt,
        )
        # Shares 6 contiguous words ("alpha bravo charlie delta echo foxtrot"), 6 of 12 content words (0.50 < 0.80)
        fail_run6 = self._rec(
            "run6",
            "alpha bravo charlie delta echo foxtrot granite masonry gull mast rope",
            prompt=prompt,
        )
        audit = description_gate.audit_records(
            [pass_run5, fail_run6],
            token_counter=lambda texts: [20 for _ in texts],
        )
        by_id = {r["id"]: r for r in audit["records"]}
        self.assertEqual(by_id["run5"]["prompt_contiguous_run"], 5)
        self.assertTrue(by_id["run5"]["passed"])
        self.assertEqual(by_id["run6"]["prompt_contiguous_run"], 6)
        self.assertFalse(by_id["run6"]["passed"])
        self.assertIn("prompt_echo", by_id["run6"]["flags"])

    def test_boundary_all_word_share_090_and_content_word_share_080(self):
        # Isolate all-word share >= 0.90 while content-word share < 0.80 and contiguous run < 6:
        # 20 words total: 5 content words (3 in prompt = 0.60 < 0.80) + 15 stopwords.
        # Reverse the stopword order relative to prompt_all so max contiguous run is 1 (< 6).
        prompt_all = "through along across near over under to from with by on in an a the alpha bravo charlie Sierra Tango Uniform"
        # fail_all90: 3 of 5 content words in prompt (0.60 < 0.80) + all 15 stopwords in prompt -> 18/20 = 0.90 (FAIL)
        fail_all90 = self._rec(
            "fail_all90",
            "alpha the bravo a charlie delta echo an in on by with from to under over near across along through",
            prompt=prompt_all,
        )
        # pass_all85: 3 of 5 distinct content words in prompt (0.60 < 0.80) + 14 of 15 stopwords in prompt -> 17/20 = 0.85 (PASS)
        pass_all85 = self._rec(
            "pass_all85",
            "sierra the tango a uniform foxtrot golf an in on by with from to under over near across along without",
            prompt=prompt_all,
        )

        # Isolate content-word share >= 0.80 while all-word share < 0.90 and contiguous run < 6:
        # 10 words total: 5 content words + 5 stopwords NOT in prompt.
        # fail_cw80: 4 of 5 content words in prompt (4/5 = 0.80 >= 0.80), all-word share = 4/10 = 0.40 < 0.90 (FAIL)
        # pass_cw60: 3 of 5 content words in prompt (3/5 = 0.60 < 0.80), all-word share = 3/10 = 0.30 < 0.90 (PASS)
        prompt_cw = "bridge cobblestone archway lantern copper"
        fail_cw80 = self._rec(
            "fail_cw80",
            "copper the lantern a archway an cobblestone in granite on",
            prompt=prompt_cw,
        )
        pass_cw60 = self._rec(
            "pass_cw60",
            "copper the lantern a archway an timber in barge on",
            prompt=prompt_cw,
        )

        audit = description_gate.audit_records(
            [fail_all90, pass_all85, fail_cw80, pass_cw60],
            token_counter=lambda texts: [25 for _ in texts],
        )
        by_id = {r["id"]: r for r in audit["records"]}
        self.assertAlmostEqual(by_id["fail_all90"]["prompt_word_share"], 0.90, places=4)
        self.assertLess(by_id["fail_all90"]["prompt_content_word_share"], 0.80)
        self.assertLess(by_id["fail_all90"]["prompt_contiguous_run"], 6)
        self.assertFalse(by_id["fail_all90"]["passed"])
        self.assertIn("prompt_echo", by_id["fail_all90"]["flags"])

        self.assertAlmostEqual(by_id["pass_all85"]["prompt_word_share"], 0.85, places=4)
        self.assertLess(by_id["pass_all85"]["prompt_content_word_share"], 0.80)
        self.assertLess(by_id["pass_all85"]["prompt_contiguous_run"], 6)
        self.assertTrue(by_id["pass_all85"]["passed"])

        self.assertAlmostEqual(by_id["fail_cw80"]["prompt_content_word_share"], 0.80, places=4)
        self.assertLess(by_id["fail_cw80"]["prompt_word_share"], 0.90)
        self.assertLess(by_id["fail_cw80"]["prompt_contiguous_run"], 6)
        self.assertFalse(by_id["fail_cw80"]["passed"])
        self.assertIn("prompt_echo", by_id["fail_cw80"]["flags"])

        self.assertAlmostEqual(by_id["pass_cw60"]["prompt_content_word_share"], 0.60, places=4)
        self.assertTrue(by_id["pass_cw60"]["passed"])

    def test_boundary_wraps_canon_subject_4_words_vs_3_words(self):
        # 4-word canon.subject appearing as a contiguous slice of words(desc) -> FAIL
        fail_wrap4 = self._rec(
            "wrap4",
            "weathered dockside scene showing a rusted iron crane gantry beside mossy steps",
            subject="rusted iron crane gantry",
        )
        # 3-word canon.subject appearing as a contiguous slice of words(desc) -> PASS (< 4 words)
        pass_wrap3 = self._rec(
            "wrap3",
            "weathered dockside scene showing a rusted iron crane beside mossy granite steps",
            subject="rusted iron crane",
        )
        # 4-word canon.subject where 4th word is only a character prefix ("gantry" vs "gantrywork"),
        # NOT a contiguous word slice in words(desc) -> PASS
        pass_prefix_only = self._rec(
            "prefix_only",
            "weathered dockside scene showing a rusted iron crane gantrywork beside granite steps",
            subject="rusted iron crane gantry",
        )
        audit = description_gate.audit_records(
            [fail_wrap4, pass_wrap3, pass_prefix_only],
            token_counter=lambda texts: [20 for _ in texts],
        )
        by_id = {r["id"]: r for r in audit["records"]}
        self.assertFalse(by_id["wrap4"]["passed"])
        self.assertIn("wraps_canon_subject", by_id["wrap4"]["flags"])
        self.assertIn(
            "contains canon.subject as a contiguous run of words (4+ words)",
            by_id["wrap4"]["reasons"],
        )
        self.assertTrue(by_id["wrap3"]["passed"])
        self.assertNotIn("wraps_canon_subject", by_id["wrap3"]["flags"])
        self.assertTrue(by_id["prefix_only"]["passed"])
        self.assertNotIn("wraps_canon_subject", by_id["prefix_only"]["flags"])

    def test_boundary_pool_duplicates_punctuation_and_content_jaccard_085(self):
        # 1. "".join(words(a)) == "".join(words(b)) boundary:
        # join_a (10 words) and join_b (8 words) have content-word Jaccard = 6/12 = 0.50 (< 0.85),
        # yet "".join(words(join_a)) == "".join(words(join_b)) -> FAIL (exact_duplicate)
        join_a = self._rec("join_a", "Cobble-stone, arch-way: dock side sea wall lamp post!")
        join_b = self._rec("join_b", "cobblestone archway dock side sea wall lamp post")
        # join_pass differs by 1 character ("posts" vs "post") and has content-word Jaccard <= 5/11 = 0.4545 -> PASS
        join_pass = self._rec("join_pass", "granite archway dock side sea wall lamp posts")

        # 2. Content-word Jaccard boundary:
        # 17 shared / 20 union = 0.8500 -> FAIL (near_duplicate)
        shared_17 = [f"cword{i:02d}" for i in range(17)]
        jac_85_a = self._rec("jac_85_a", " ".join([*shared_17, "uniquea1", "uniquea2"]))
        jac_85_b = self._rec("jac_85_b", " ".join([*shared_17, "uniqueb1"]))

        # 16 shared / 19 union = 0.8421 (< 0.85) -> PASS
        shared_16 = [f"dword{i:02d}" for i in range(16)]
        jac_84_a = self._rec("jac_84_a", " ".join([*shared_16, "onlya1", "onlya2"]))
        jac_84_b = self._rec("jac_84_b", " ".join([*shared_16, "onlyb1"]))

        audit = description_gate.audit_records(
            [join_a, join_b, join_pass, jac_85_a, jac_85_b, jac_84_a, jac_84_b],
            token_counter=lambda texts: [25 for _ in texts],
        )
        by_id = {r["id"]: r for r in audit["records"]}
        self.assertEqual(
            "".join(description_gate.words(join_a["judgment"]["explanation"])),
            "".join(description_gate.words(join_b["judgment"]["explanation"])),
        )
        self.assertIsNone(by_id["join_a"]["max_content_jaccard"])
        self.assertIn("exact_duplicate", by_id["join_a"]["flags"])
        self.assertIn("exact_duplicate", by_id["join_b"]["flags"])
        self.assertTrue(by_id["join_pass"]["passed"])

        self.assertAlmostEqual(by_id["jac_85_a"]["max_content_jaccard"], 0.85, places=4)
        self.assertAlmostEqual(by_id["jac_85_b"]["max_content_jaccard"], 0.85, places=4)
        self.assertIn("near_duplicate", by_id["jac_85_a"]["flags"])
        self.assertIn("near_duplicate", by_id["jac_85_b"]["flags"])

        self.assertIsNone(by_id["jac_84_a"]["max_content_jaccard"])
        self.assertTrue(by_id["jac_84_a"]["passed"])
        self.assertTrue(by_id["jac_84_b"]["passed"])

    def test_boundary_siglip2_tokens_64_65_and_unchecked(self):
        r64 = self._rec("tok64", "Weathered Limestone Watchtower Overlooking A Misty Tidal Marsh At Dawn")
        r65 = self._rec("tok65", "Carved Obsidian Monolith Standing Inside A Flooded Subterranean Cavern Chamber")

        # Without token_counter, records are "unchecked" and CANNOT be marked as passing
        unchecked_audit = description_gate.audit_records([r64, r65])
        self.assertFalse(unchecked_audit["passed_gate"])
        self.assertEqual(unchecked_audit["unchecked_count"], 2)
        for row in unchecked_audit["records"]:
            self.assertEqual(row["gate_status"], "unchecked")
            self.assertFalse(row["passed"])

        # Verify siglip2_token_counts calls tokenizer(s.lower(), padding=False, truncation=False)["input_ids"]
        calls = []

        def fake_siglip2_tokenizer(s: str, *, padding: bool = True, truncation: bool = True):
            self.assertEqual(s, s.lower(), "tokenizer must receive s.lower()")
            self.assertFalse(padding, "padding must be False so counts are not padded to 64")
            self.assertFalse(truncation, "truncation must be False so counts above 64 are not truncated to 64")
            calls.append(s)
            n_toks = 64 if "watchtower" in s else 65
            return {"input_ids": list(range(n_toks))}

        checked_audit = description_gate.audit_records(
            [r64, r65],
            token_counter=lambda texts: description_gate.siglip2_token_counts(texts, tokenizer=fake_siglip2_tokenizer),
        )
        self.assertEqual(len(calls), 2)
        by_id = {r["id"]: r for r in checked_audit["records"]}
        self.assertEqual(by_id["tok64"]["siglip2_tokens"], 64)
        self.assertEqual(by_id["tok64"]["gate_status"], "passed")
        self.assertTrue(by_id["tok64"]["passed"])
        self.assertEqual(by_id["tok65"]["siglip2_tokens"], 65)
        self.assertEqual(by_id["tok65"]["gate_status"], "failed")
        self.assertFalse(by_id["tok65"]["passed"])
        self.assertIn("too_long_siglip2", by_id["tok65"]["flags"])

    def test_54_copy_ruling_inside_prompt_blind_worksheet_and_fake_drafter(self):
        ruling = (
            "Director ruling 2026-07-30: pool look approved as the target style; "
            "palette, brushwork, and surface grain are canon."
        )
        recs = []
        for i in range(54):
            recs.append({
                "id": f"sr_{i:02d}",
                "asset_path": f"outputs/approved/sr_{i:02d}.png",
                "canon": {
                    "subject": f"sunlit salt-crust road beside ochre storehouse {i}",
                },
                "judgment": {
                    "status": "approved",
                    "explanation": ruling,
                },
                "provenance": {
                    "prompt": f"salt-road style, sunlit salt-crust road beside ochre storehouse {i}",
                },
            })
        # Add one record whose description sits verbatim inside its generation prompt
        recs.append({
            "id": "sr_inside",
            "asset_path": "outputs/approved/sr_inside.png",
            "canon": {"subject": "hidden canon subject string for inside test"},
            "judgment": {
                "status": "approved",
                "explanation": "weathered timber sluice gate spanning a narrow canal",
            },
            "provenance": {
                "prompt": "salt-road style, weathered timber sluice gate spanning a narrow canal, warm gouache",
            },
        })
        # Add one clean human record that passes model-free checks
        recs.append({
            "id": "sr_clean_human",
            "asset_path": "outputs/approved/sr_clean_human.png",
            "canon": {"subject": "secret canon subject that is not wrapped"},
            "judgment": {
                "status": "approved",
                "explanation": "two merchants unloading banded oak casks beside a stone mooring bollard at dusk",
            },
            "provenance": {
                "prompt": "salt-road style, harbour quay in evening light",
            },
        })

        audit = description_gate.audit_records(recs)
        self.assertFalse(audit["passed_gate"])
        self.assertEqual(audit["checked"], 56)
        self.assertEqual(audit["flagged_count"], 55)
        self.assertEqual(audit["passed_model_free_count"], 1)
        self.assertEqual(audit["max_exact_group"], 54)
        by_id = {r["id"]: r for r in audit["records"]}
        self.assertTrue(by_id["sr_inside"]["inside_prompt"])
        self.assertIn("prompt_echo", by_id["sr_inside"]["flags"])
        self.assertEqual(by_id["sr_clean_human"]["author_kind"], "human")

        # Run blind drafting stage with FakeDrafter
        drafter = backends.FakeDrafter()
        with tempfile.TemporaryDirectory() as tmp:
            description_gate.draft_flagged_records(tmp, audit, drafter)
            # Drafter must have been called ONLY with image paths, never with prompts, canon.subject, or current_text
            self.assertEqual(len(drafter.seen_paths), 55)
            for p in drafter.seen_paths:
                self.assertTrue(p.endswith(".png"))
                self.assertNotIn("Director ruling", p)
                self.assertNotIn("salt-road style", p)
                self.assertNotIn("storehouse", p)

            for row in audit["flagged"]:
                self.assertEqual(row["author_kind"], "captioner_assisted")
                self.assertEqual(row["drafter_id"], "plain-sight/florence-2-fake")
                self.assertEqual(row["drafter_revision"], "fake-florence2-1")
                self.assertTrue(row["florence2_draft"].startswith("a weathered"))

            rev_jsonl, ws_path = description_gate.write_review_files(tmp, tmp, audit)
            with open(ws_path, encoding="utf-8") as f:
                ws = f.read()

            # Worksheet MUST NOT contain the current text, canon.subject, or generation prompt
            self.assertNotIn("Director ruling 2026-07-30", ws)
            self.assertNotIn("salt-road style", ws)
            self.assertNotIn("sunlit salt-crust road beside ochre storehouse", ws)
            self.assertNotIn("secret canon subject", ws)
            self.assertNotIn("weathered timber sluice gate", ws)

            # Worksheet MUST contain the required instruction, sibling links, Florence-2 draft, edited text, and author_kind
            self.assertIn(description_gate.WORKSHEET_INSTRUCTION, ws)
            self.assertIn("| Florence-2 draft | edited text |", ws)
            self.assertIn("`captioner_assisted`", ws)
            self.assertIn("`human`", ws)
            self.assertIn("a weathered neutral figure beside carved stone pillars", ws)

            with open(rev_jsonl, encoding="utf-8") as f:
                rows_jsonl = [json.loads(line) for line in f if line.strip()]
            self.assertEqual(len(rows_jsonl), 56)
            first_flagged = next(r for r in rows_jsonl if r["id"] == "sr_00")
            self.assertEqual(first_flagged["duplicate_group_size"], 54)
            self.assertEqual(first_flagged["author_kind"], "captioner_assisted")
            self.assertEqual(first_flagged["drafter_id"], "plain-sight/florence-2-fake")
            self.assertEqual(first_flagged["drafter_revision"], "fake-florence2-1")
            self.assertGreaterEqual(len(first_flagged["siblings"]), 3)
            self.assertLessEqual(len(first_flagged["siblings"]), 5)
            clean_row = next(r for r in rows_jsonl if r["id"] == "sr_clean_human")
            self.assertEqual(clean_row["author_kind"], "human")

    def test_plain_sight_drafter_pin_offline_no_remote_code_and_missing_snapshot(self):
        import types

        with mock.patch.dict(sys.modules, {"torch": fakes.make_torch(), "transformers": fakes.make_transformers()}):
            self._plain_sight_pin_checks(types)

    def _plain_sight_pin_checks(self, types):
        # 1. Verify the remote-code flag appears nowhere in the folder's Python code
        here = os.path.dirname(os.path.abspath(backends.__file__))
        forbidden_flag = "_".join(("trust", "remote", "code"))
        for fname in sorted(os.listdir(here)):
            if fname.endswith(".py"):
                with open(os.path.join(here, fname), encoding="utf-8") as f:
                    src = f.read()
                self.assertNotIn(forbidden_flag, src, f"{forbidden_flag} must not appear in {fname}")

        # 2. Verify PlainSightDrafter pins plain-sight's exact repo and revision
        self.assertEqual(backends.PLAIN_SIGHT_REPO, "florence-community/Florence-2-large")
        self.assertEqual(backends.PLAIN_SIGHT_REVISION, "4271c66b88cdbc05735372ec13b2360108de5317")

        # 3. Missing snapshot raises a named error (naming repo, revision, and install plain-sight) without downloading
        with mock.patch("transformers.AutoProcessor.from_pretrained", side_effect=OSError("snapshot not in cache")):
            drafter_missing = backends.PlainSightDrafter()
            with self.assertRaises(RuntimeError) as cm_miss:
                drafter_missing.draft(["dummy.png"])
            err_msg = str(cm_miss.exception)
            self.assertIn("florence-community/Florence-2-large", err_msg)
            self.assertIn("4271c66b88cdbc05735372ec13b2360108de5317", err_msg)
            self.assertIn("install plain-sight", err_msg)

        # 4. Monkeypatched plain_sight.engine.Florence2Engine: requests plain-sight's repo & revision offline,
        # runs detail="medium" (<DETAILED_CAPTION>), and records actual loaded model_id & revision_resolved
        engine_calls = []

        class StubFlorence2Engine:
            loaded = True

            def __init__(self, **kwargs):
                engine_calls.append({
                    "kwargs": kwargs,
                    "hf_offline": os.environ.get("HF_HUB_OFFLINE"),
                })
                self.model_id = kwargs.get("model_id")
                self.revision = kwargs.get("revision")

            def describe(self, image_path: str, detail: str = "high") -> str:
                engine_calls[-1]["describe"] = (image_path, detail)
                return "a weathered stone lighthouse beside dark harbour water"

            def output_provenance(self) -> dict:
                return {
                    "model_id": "florence-community/Florence-2-large",
                    "revision_resolved": "4271c66b88cdbc05735372ec13b2360108de5317",
                }

        fake_ps_pkg = types.ModuleType("plain_sight")
        fake_ps_eng = types.ModuleType("plain_sight.engine")
        fake_ps_eng.Florence2Engine = StubFlorence2Engine
        with mock.patch.dict(sys.modules, {"plain_sight": fake_ps_pkg, "plain_sight.engine": fake_ps_eng}):
            ps_drafter = backends.PlainSightDrafter()
            audit = description_gate.audit_records([self._rec("r_short", "too short")])
            description_gate.draft_flagged_records("/proj", audit, ps_drafter)

        self.assertEqual(len(engine_calls), 1)
        self.assertEqual(engine_calls[0]["hf_offline"], "1")
        self.assertEqual(engine_calls[0]["kwargs"]["model_id"], "florence-community/Florence-2-large")
        self.assertEqual(engine_calls[0]["kwargs"]["revision"], "4271c66b88cdbc05735372ec13b2360108de5317")
        self.assertEqual(engine_calls[0]["describe"][1], "medium")
        self.assertEqual(audit["flagged"][0]["drafter_id"], "florence-community/Florence-2-large")
        self.assertEqual(audit["flagged"][0]["drafter_revision"], "4271c66b88cdbc05735372ec13b2360108de5317")

        # 5. Monkeypatched native transformers loader (when plain_sight is not installed): requests local_files_only=True
        loader_calls = []

        def fake_proc_load(repo, **kwargs):
            loader_calls.append(("proc", repo, kwargs, os.environ.get("HF_HUB_OFFLINE")))
            raise OSError("missing local snapshot")

        with mock.patch.dict(sys.modules, {"plain_sight": None, "plain_sight.engine": None}):
            with mock.patch("transformers.AutoProcessor.from_pretrained", side_effect=fake_proc_load):
                native_drafter = backends.PlainSightDrafter()
                with self.assertRaises(RuntimeError) as cm_nat:
                    native_drafter.draft(["dummy.png"])
                self.assertIn("florence-community/Florence-2-large", str(cm_nat.exception))
                self.assertIn("4271c66b88cdbc05735372ec13b2360108de5317", str(cm_nat.exception))

        self.assertEqual(len(loader_calls), 1)
        _, req_repo, req_kwargs, req_offline = loader_calls[0]
        self.assertEqual(req_repo, "florence-community/Florence-2-large")
        self.assertEqual(req_kwargs["revision"], "4271c66b88cdbc05735372ec13b2360108de5317")
        self.assertTrue(req_kwargs["local_files_only"])
        self.assertNotIn(forbidden_flag, req_kwargs)
        self.assertEqual(req_offline, "1")

    def test_real_salt_road_and_rustline_flagged_when_present(self):
        root = os.environ.get("DATASET_DESK_REAL_PROJECTS", "")
        if not root or not os.path.isdir(root):
            self.skipTest("set DATASET_DESK_REAL_PROJECTS to a style-dataset-lab projects folder to run this check")
        sr_path = os.path.join(root, "salt-road")
        sr = description_gate.audit_project(sr_path)
        self.assertFalse(sr["passed_gate"])
        self.assertFalse(sr["passed_model_free_gate"])
        self.assertEqual(sr["checked"], 73)
        self.assertEqual(sr["max_exact_group"], 54)
        self.assertGreaterEqual(sr["flagged_count"], 70)
        # Also verify that if salt-road's canon.subject is tested as queries, all 73 fail wraps_canon_subject
        sr_recs = description_gate.load_project_records(sr_path)
        sr_canon_queries = {
            r["id"]: (r.get("canon") or {}).get("subject", "")
            for r in sr_recs
            if (r.get("judgment") or {}).get("status") == "approved"
        }
        sr_canon_audit = description_gate.audit_records(sr_recs, queries=sr_canon_queries)
        self.assertEqual(sr_canon_audit["flagged_count"], 73)
        self.assertTrue(all("wraps_canon_subject" in r["flags"] for r in sr_canon_audit["records"]))

        rl = description_gate.audit_project(os.path.join(root, "rustline"))
        self.assertFalse(rl["passed_gate"])
        self.assertFalse(rl["passed_model_free_gate"])
        self.assertEqual(rl["checked"], 172)
        self.assertEqual(rl["max_exact_group"], 14)
        self.assertGreater(rl["flagged_count"], 165)


class CommandLine(Project):
    def run_main(self, *args, project=None, backend="fake"):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = desk.main([
                args[0],
                "--project", project or self.P,
                "--backend", backend,
                "--out", self.out,
                *args[1:],
            ])
        return rc, buf.getvalue()

    def test_index_report_search_coverage_dupes_and_drafting(self):
        self.assertEqual(self.run_main("index")[0], 0)
        subj = os.path.join(self.tmp, "subjects.txt")
        with open(subj, "w", encoding="utf-8") as f:
            f.write("# wanted\na red creature\na purple bird\n")
        rc, out = self.run_main("report", "--subjects", subj, "--draft", "fake")
        with open(out.strip(), encoding="utf-8") as f:
            text = f.read()
        for heading in (
            "## Duplicates",
            "## Diversity",
            "## Least typical",
            "## Description audit (model-free gate)",
            "## Do the descriptions find",
            "## Coverage of the subject list",
        ):
            self.assertIn(heading, text)
        self.assertIn("](../proj/outputs/", text)  # links resolve from the report's folder
        self.assertIn("style_residue_prop", text)  # failure_modes shown in report
        self.assertTrue(os.path.exists(os.path.join(self.out, "description-review.jsonl")))
        ws_file = os.path.join(self.out, "description-worksheet.md")
        self.assertTrue(os.path.exists(ws_file))
        with open(ws_file, encoding="utf-8") as f:
            ws_text = f.read()
        self.assertIn(description_gate.WORKSHEET_INSTRUCTION, ws_text)
        self.assertNotIn("a red mound creature", ws_text)  # current text never shown in worksheet
        self.assertIn("plain-sight/florence-2-fake", ws_text)
        self.assertIn("blue1.png", self.run_main("search", "--text", "blue", "-k", "1")[1])
        # Joint multimodal search (--text + --image)
        red_img = os.path.join(self.P, "outputs", "approved", "red1.png")
        mm_out = self.run_main("search", "--text", "blue", "--image", red_img, "-k", "2")[1]
        self.assertEqual(len(mm_out.strip().splitlines()), 2)
        self.assertTrue(self.run_main("coverage", "--subjects", subj)[1].startswith("a purple bird"))
        d = json.loads(self.run_main("dupes")[1])
        self.assertEqual(len(d["exact"]), 1)

    def test_peeking_and_arm_model_guards_refuse_salt_road_without_flag(self):
        sr_dir = os.path.join(self.tmp, "salt-road")
        shutil.copytree(self.P, sr_dir)
        with self.assertRaises(SystemExit) as cm:
            self.run_main("index", project=sr_dir)
        self.assertIn("salt-road", str(cm.exception))
        self.assertIn("--images-only", str(cm.exception))
        self.assertIn("--allow-multimodal-pool", str(cm.exception))

        # Stub ai-eyes (SigLIP2 / arm S) and EG2 (arm G) acceptance steps must refuse on salt-road
        with self.assertRaises(SystemExit) as cm_ai:
            desk.ai_eyes_verify(sr_dir, {"red1": "a red mound creature"})
        self.assertIn("salt-road", str(cm_ai.exception))
        self.assertIn("ai-eyes", str(cm_ai.exception))
        self.assertIn("--allow-multimodal-pool", str(cm_ai.exception))

        with self.assertRaises(SystemExit) as cm_eg:
            desk.eg2_fit_accept(sr_dir, {"red1": "a red mound creature"})
        self.assertIn("salt-road", str(cm_eg.exception))
        self.assertIn("eg2", str(cm_eg.exception))

        # Allowed when allow_multimodal_pool=True or on a non-pool project
        self.assertEqual(desk.ai_eyes_verify(self.P, {"red1": "ok"})["status"], "stub")
        self.assertEqual(
            desk.ai_eyes_verify(sr_dir, {"red1": "ok"}, allow_multimodal_pool=True)["status"],
            "stub",
        )

        # Succeeds with --images-only
        self.assertEqual(self.run_main("index", "--images-only", project=sr_dir)[0], 0)
        # Report without flag still refuses unless --images-only or --allow-multimodal-pool
        with self.assertRaises(SystemExit):
            self.run_main("report", project=sr_dir)
        rc, out = self.run_main("report", "--images-only", project=sr_dir)
        self.assertEqual(rc, 0)
        with open(out.strip(), encoding="utf-8") as f:
            rep = f.read()
        self.assertIn("skipped (`--images-only`)", rep)
        # Also succeeds with --allow-multimodal-pool
        self.assertEqual(self.run_main("index", "--allow-multimodal-pool", project=sr_dir)[0], 0)

        # Image-only backend (dinov2) succeeds on a locked pool without --images-only from both Python and CLI
        desk.check_multimodal_pool(sr_dir, "dinov2")
        for text_capable in ("eg2", "siglip2"):
            with self.assertRaises(DeskError) as cm_pool:
                desk.check_multimodal_pool(sr_dir, text_capable)
            self.assertEqual(cm_pool.exception.code, "PERM_LOCKED_POOL")

        class _DinoStub(backends.FakeEmbedder):
            modalities = ("image",)

            def __init__(self):
                super().__init__(dim=4)
                self.name = "dinov2"
                self.revision = desk.registry.REGISTRY["dinov2"].revision
                self.vision_tokens = 256
                self.max_tokens = 0

        dino_meta = desk.index(sr_dir, self.out, _DinoStub(), log=lambda *a: None)
        self.assertTrue(dino_meta["images_only"])
        d_items, d_img, d_txt, d_m = desk.load_cache(self.out, "dinov2", dim=4, vision_tokens=256)
        rep_path = desk.report(sr_dir, self.out, d_items, d_img, d_txt, d_m)
        self.assertTrue(os.path.isfile(rep_path))

    def test_lite_backend_none_runs_without_embeddings(self):
        self.assertEqual(self.run_main("index", backend="none")[0], 0)
        rc, out = self.run_main("report", backend="none")
        self.assertEqual(rc, 0)
        with open(out.strip(), encoding="utf-8") as f:
            text = f.read()
        self.assertIn("## Duplicates", text)
        self.assertIn("## Description audit (model-free gate)", text)
        self.assertIn("skipped (`--backend none`", text)
        d = json.loads(self.run_main("dupes", backend="none")[1])
        self.assertEqual(len(d["exact"]), 1)
        self.assertEqual(d["semantic"], [])

    def test_refusals(self):
        with self.assertRaises(SystemExit) as cm:
            desk.main(["index", "--project", self.P, "--backend", "fake", "--out", os.path.join(self.P, "x")])
        self.assertIn("never writes into a project", str(cm.exception))
        with self.assertRaises(SystemExit) as cm:
            desk.main(["index", "--project", self.P, "--backend", "fake", "--out", self.out, "--device", "cuda"])
        self.assertIn("needs a grant id", str(cm.exception))
        with self.assertRaises(SystemExit):
            self.run_main("search")  # no index yet
        self.run_main("index")
        with self.assertRaises(SystemExit) as cm:
            self.run_main("search")
        self.assertIn("--text or --image", str(cm.exception))


class TextAndCodeInputTests(unittest.TestCase):
    SENTINEL_RED = "SECRET_BODY_ALPHA_99182_RED " + ("red " * 20)
    SENTINEL_BLUE = "SECRET_BODY_BETA_77412_BLUE " + ("blue " * 20)

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.out = os.path.join(self.tmp, "out")

    def _write(self, base: str, rel: str, content: bytes | str) -> str:
        full = os.path.join(base, *rel.split("/"))
        os.makedirs(os.path.dirname(full), exist_ok=True)
        mode = "wb" if isinstance(content, bytes) else "w"
        kw = {} if isinstance(content, bytes) else {"encoding": "utf-8"}
        with open(full, mode, **kw) as f:
            f.write(content)
        return full

    def test_task_defaults_and_backend_override(self):
        self.assertEqual(desk.command_backend("index"), "siglip2")
        self.assertEqual(desk.command_backend("index", task="image-variety"), "dinov2")
        self.assertEqual(desk.command_backend("text-index"), "bge-m3")
        self.assertEqual(desk.command_backend("text-dupes"), "bge-m3")
        self.assertEqual(desk.command_backend("text-search"), "bge-m3")
        self.assertEqual(desk.command_backend("lineage"), "eg2")
        self.assertEqual(desk.command_backend("text-index", backend="qwen3-emb-0.6b"), "qwen3-emb-0.6b")
        self.assertEqual(desk.command_backend("lineage", backend="gte-modernbert"), "gte-modernbert")

        src = os.path.join(self.tmp, "src")
        self._write(src, "doc.md", "red text sample")
        seen = []

        def fake_make(backend, **kw):
            seen.append((backend, kw))
            emb = backends.FakeEmbedder()
            emb.name = backend
            return emb

        with mock.patch.object(desk, "lower_priority"), contextlib.redirect_stdout(io.StringIO()):
            desk.main(["text-index", "--source", src, "--out", self.out], make=fake_make)
            desk.main(["text-index", "--source", src, "--out", self.out, "--task", "code"], make=fake_make)
            desk.main(["text-index", "--source", src, "--out", self.out, "--backend", "gte-modernbert"], make=fake_make)
        self.assertEqual([b for b, _ in seen], ["bge-m3", "eg2", "gte-modernbert"])

    def test_suffix_filtering_nul_skipping_and_sha256_dedupe(self):
        src = os.path.join(self.tmp, "src")
        for idx, ext in enumerate((".md", ".txt", ".json", ".jsonl", ".log", ".out", ".py", ".sh", ".toml", ".yml", ".yaml", "")):
            fname = f"file{idx}{ext}" if ext else "Makefile"
            self._write(src, fname, f"unique content {idx} red")
        # Byte-identical duplicate of file0.md
        self._write(src, "sub/copy.md", "unique content 0 red")
        # Disallowed suffixes
        self._write(src, "skip.bin", "text inside bin")
        self._write(src, "skip.png", "not a text suffix")
        self._write(src, "skip.bak", "backup file")
        # File with NUL in first 4 KB
        self._write(src, "nul_file.txt", b"hello\x00world")

        items, stats = desk.scan_text_source(src)
        self.assertEqual(stats["files"], 13)
        self.assertEqual(stats["unique"], 12)
        self.assertEqual(stats["nul_skipped"], 1)
        dup_item = next(it for it in items if len(it.paths) > 1)
        self.assertEqual(sorted(dup_item.paths), ["file0.md", "sub/copy.md"])

    def test_chunk_limits_and_mean_pooling_with_fake_embedder(self):
        self.assertEqual(desk.chunk_text(""), [""])
        # Token-counted chunking: default FakeEmbedder max_tokens=8192 -> budget 8188 tokens per chunk
        long_text = "word " * 80000  # 80,000 tokens -> capped at 8 chunks of 8188 tokens
        chunks = desk.chunk_text(long_text)
        self.assertEqual(len(chunks), 8)
        self.assertTrue(all(len(c.split()) == 8188 for c in chunks))

        # EG2 max_tokens=8000 vs BGE-M3/Qwen3/GTE max_tokens=8192
        eg2_chunks = desk.chunk_text(long_text, max_tokens=8000, prefix="title: gate.py | text: ")
        self.assertEqual(len(eg2_chunks), 8)
        # "title: gate.py | text: " is 5 tokens -> budget = 8000 - 5 - 4 = 7991 tokens
        self.assertTrue(all(len(c.split()) == 7991 for c in eg2_chunks))
        bge_chunks = desk.chunk_text(long_text, max_tokens=8192)
        self.assertEqual(len(bge_chunks), 8)
        self.assertTrue(all(len(c.split()) == 8188 for c in bge_chunks))

        # Tokenizer with encode/decode (e.g. HF tokenizer / SentenceTransformer)
        class _FakeTok:
            def encode(self, s, add_special_tokens=False):
                return s.split()
            def decode(self, ids, skip_special_tokens=True):
                return " ".join(ids)

        class _EmbWithTok:
            max_tokens = 20
            tokenizer = _FakeTok()

        tok_chunks = desk.chunk_text(" ".join(f"t{i}" for i in range(50)), emb=_EmbWithTok(), prefix="p1 p2")
        # budget = 20 - 2 - 4 = 14 tokens -> 50 tokens splits into 4 chunks (14, 14, 14, 8)
        self.assertEqual([len(c.split()) for c in tok_chunks], [14, 14, 14, 8])

        # Callable tokenizer without decode fallback + offset_mapping fallback
        class _CallableOnlyTok:
            def __call__(self, s, **kw):
                words = s.split()
                if kw.get("return_offsets_mapping"):
                    pos = 0
                    offsets = []
                    for w in words:
                        idx = s.find(w, pos)
                        offsets.append((idx, idx + len(w)))
                        pos = idx + len(w)
                    return {"input_ids": list(range(len(words))), "offset_mapping": offsets}
                return {"input_ids": list(range(len(words)))}

        class _ProcHolder:
            tokenizer = _CallableOnlyTok()

        class _EmbWithProc:
            max_tokens = 12
            processor = _ProcHolder()

        proc_chunks = desk.chunk_text(" ".join(f"w{i}" for i in range(20)), emb=_EmbWithProc())
        self.assertEqual(len(proc_chunks), 3)

        # Fake embedder whose model is None at the first chunk_text call, and whose tokenizer counts 2 tokens per span:
        # every chunk must come out within max_tokens - prefix - 4 by that fake's count.
        class _TwoPerSpanTok:
            def count(self, s: str) -> int:
                return 2 * len(desk._FALLBACK_TOKENIZER.spans(s))

            def encode(self, s: str, add_special_tokens: bool = False) -> list[int]:
                return list(range(self.count(s)))

        class _UnloadedModelEmb(backends.FakeEmbedder):
            def __init__(self):
                super().__init__(dim=768)
                self.name = "gte-modernbert"
                self.model = None
                self._tok = _TwoPerSpanTok()

            @property
            def tokenizer(self):
                return self._tok

        unloaded_emb = _UnloadedModelEmb()
        unloaded_emb.max_tokens = 24
        self.assertIsNone(unloaded_emb.model)
        prefix_str = "p1 p2"  # 2 spans -> 4 tokens by _TwoPerSpanTok; budget = 24 - 4 - 4 = 16 tokens (8 spans)
        two_span_chunks = desk.chunk_text(
            " ".join(f"tok{i}" for i in range(30)),
            emb=unloaded_emb,
            prefix=prefix_str,
        )
        self.assertIsNone(unloaded_emb.model)
        budget = unloaded_emb.max_tokens - unloaded_emb.tokenizer.count(prefix_str) - 4
        self.assertEqual(budget, 16)
        self.assertEqual(len(two_span_chunks), 4)
        for ch in two_span_chunks:
            self.assertLessEqual(unloaded_emb.tokenizer.count(ch), budget)

        # Verify text_index also chunks the very first file using emb.tokenizer while emb.model is None
        src_unloaded = os.path.join(self.tmp, "src_unloaded")
        self._write(src_unloaded, "first.py", " ".join(f"tok{i}" for i in range(30)))
        out_unloaded = os.path.join(self.tmp, "out_unloaded")
        unloaded_for_idx = _UnloadedModelEmb()
        unloaded_for_idx.has_symmetric_vectors = False
        unloaded_for_idx.max_tokens = 20  # budget = 20 - 0 - 4 = 16 tokens (8 spans per chunk -> 4 chunks for 30 spans)
        desk.text_index(out_unloaded, unloaded_for_idx, source=src_unloaded, log=lambda *a: None)
        idx_items, _, _ = desk.load_text_cache(out_unloaded, "gte-modernbert")
        self.assertEqual(idx_items[0].chunks, 4)

        # Real backend with no tokenizer is refused with INPUT_BAD_ARGUMENT rather than chunked by _SpanTokenizer
        with self.assertRaises(DeskError) as cm_no_tok:
            desk.chunk_text("some text", emb=backends.DINOv2Embedder())
        self.assertEqual(cm_no_tok.exception.code, "INPUT_BAD_ARGUMENT")

        # embed_file_chunks does not swallow TypeError from inside emb.texts()
        class _BrokenTextsEmb:
            def texts(self, chunks, role="document", titles=None):
                raise TypeError("internal type bug inside texts()")

        with self.assertRaises(TypeError):
            desk.embed_file_chunks(_BrokenTextsEmb(), ["chunk"], role="document", title="doc.py")

        emb = backends.FakeEmbedder()
        c1 = "red " * 5000
        c2 = "blue " * 5000
        two_chunk_text = c1 + c2
        split = desk.chunk_text(two_chunk_text, emb=emb)
        self.assertEqual(len(split), 2)
        expected = backends.unit_rows(emb.texts(split, "document", titles=["doc.py", "doc.py"]).mean(axis=0))[0]
        got = desk.embed_file_chunks(emb, split, role="document", title="sub/doc.py")
        np.testing.assert_allclose(got, expected, atol=1e-6)
        self.assertAlmostEqual(float(np.linalg.norm(got)), 1.0, places=5)

    def test_rnd_data_reads_only_rnd_data_copies_outcomes_and_never_leaks_contents(self):
        chk = os.path.join(self.tmp, "rnd-data")
        # Run 1: outcome at top level; has rnd-data copy, 'd' blob copy, and unlisted file
        self._write(chk, "runs/r1/files/outcome.json", json.dumps({"outcome": "pass"}))
        f1 = self._write(chk, "runs/r1/files/gate.py", self.SENTINEL_RED)
        self._write(chk, "runs/r1/files/unlisted.py", "green " * 20)
        self._write(chk, "blobs/aa/blobfile.py", "green " * 20)
        self._write(
            chk,
            "runs/r1/manifest.json",
            json.dumps({
                "id": "r1",
                "copies": [
                    {"sha256": desk.sha256_file(f1), "store": "rnd-data", "location": "runs/r1/files/gate.py"},
                    {"sha256": "deadbeef" * 8, "store": "d", "location": "blobs/aa/blobfile.py"},
                ],
            }),
        )
        # Run 2: outcome under run.outcome; same-name gate.py with near-identical red content + blue helper.sh
        self._write(chk, "runs/r2/files/outcome.json", json.dumps({"run": {"outcome": "gate-reject"}}))
        f2 = self._write(chk, "runs/r2/files/gate.py", self.SENTINEL_RED + " red")
        f3 = self._write(chk, "runs/r2/files/helper.sh", self.SENTINEL_BLUE)
        self._write(
            chk,
            "runs/r2/manifest.json",
            json.dumps({
                "id": "r2",
                "copies": [
                    {"sha256": desk.sha256_file(f2), "store": "rnd-data", "location": "runs/r2/files/gate.py"},
                    {"sha256": desk.sha256_file(f3), "store": "rnd-data", "location": "runs/r2/files/helper.sh"},
                ],
            }),
        )

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = desk.main(["text-index", "--rnd-data", chk, "--out", self.out, "--backend", "fake"])
        self.assertEqual(rc, 0)
        self.assertTrue(os.path.isdir(os.path.join(self.out, "fake", "text")))

        # Second index run reuses cached vectors by sha256 and revision
        meta2 = desk.text_index(self.out, backends.FakeEmbedder(), rnd_data=chk, log=lambda *a: None)
        self.assertEqual(meta2["embedded_now"], 0)

        items, _vecs, _meta = desk.load_text_cache(self.out, "fake")
        all_paths = [p for it in items for p in it.paths]
        self.assertIn("runs/r1/files/gate.py", all_paths)
        self.assertIn("runs/r2/files/gate.py", all_paths)
        self.assertIn("runs/r2/files/helper.sh", all_paths)
        self.assertNotIn("runs/r1/files/unlisted.py", all_paths)
        self.assertFalse(any("blobfile" in p for p in all_paths))

        by_path = {it.paths[0]: it for it in items}
        self.assertEqual(by_path["runs/r1/files/gate.py"].outcome, "pass")
        self.assertEqual(by_path["runs/r2/files/gate.py"].outcome, "gate-reject")

        # text-dupes
        dupes_buf = io.StringIO()
        with contextlib.redirect_stdout(dupes_buf):
            desk.main(["text-dupes", "--rnd-data", chk, "--out", self.out, "--backend", "fake", "--semantic", "0.90"])
        dupes_out = dupes_buf.getvalue()
        dupes_json = json.loads(dupes_out)
        self.assertEqual(len(dupes_json["pairs"]), 1)
        self.assertEqual(dupes_json["pairs"][0]["outcome_a"], "pass")
        self.assertEqual(dupes_json["pairs"][0]["outcome_b"], "gate-reject")

        # text-search
        search_buf = io.StringIO()
        with contextlib.redirect_stdout(search_buf):
            desk.main(["text-search", "--out", self.out, "--backend", "fake", "--text", "blue", "-k", "1"])
        search_out = search_buf.getvalue()
        self.assertIn("runs/r2/files/helper.sh", search_out)
        self.assertIn("gate-reject", search_out)

        # lineage
        lin_buf = io.StringIO()
        with contextlib.redirect_stdout(lin_buf):
            desk.main(["lineage", "--out", self.out, "--backend", "fake"])
        lin_out = lin_buf.getvalue()
        lin_json = json.loads(lin_out)
        self.assertIn("gate.py", lin_json["files"])
        self.assertNotIn("helper.sh", lin_json["files"])  # only in one run
        pair = lin_json["files"]["gate.py"][0]
        self.assertEqual((pair["run_a"], pair["run_b"]), ("r1", "r2"))
        self.assertEqual((pair["outcome_a"], pair["outcome_b"]), ("pass", "gate-reject"))
        self.assertGreater(pair["cosine"], 0.99)

        # Verify NO file contents ever appear in CLI outputs or cache files
        with open(os.path.join(self.out, "fake", "text", "items.jsonl"), encoding="utf-8") as f:
            cache_items_raw = f.read()
        for text_blob in (buf.getvalue(), dupes_out, search_out, lin_out, cache_items_raw):
            self.assertNotIn("SECRET_BODY_ALPHA_99182_RED", text_blob)
            self.assertNotIn("SECRET_BODY_BETA_77412_BLUE", text_blob)

    def test_rnd_data_refuses_private_store_and_paths_outside_checkout(self):
        # 1. Manifest naming rnd-data-private
        chk_priv = os.path.join(self.tmp, "chk1")
        self._write(
            chk_priv,
            "runs/r1/manifest.json",
            json.dumps({
                "id": "r1",
                "copies": [{"sha256": "00" * 32, "store": "rnd-data-private", "location": "runs/r1/files/secret.py"}],
            }),
        )
        with self.assertRaises(DeskError) as cm1:
            desk.scan_rnd_data(chk_priv)
        self.assertTrue(cm1.exception.code.startswith("PERM_"))

        # 2. Manifest with .. traversal outside checkout
        chk_trav = os.path.join(self.tmp, "chk2")
        self._write(chk_trav, "outside.py", "print(1)")
        self._write(
            chk_trav,
            "runs/r1/manifest.json",
            json.dumps({
                "id": "r1",
                "copies": [{"sha256": "00" * 32, "store": "rnd-data", "location": "../outside.py"}],
            }),
        )
        with self.assertRaises(DeskError) as cm2:
            desk.scan_rnd_data(chk_trav)
        self.assertTrue(cm2.exception.code.startswith("PERM_"))

        # 3. Manifest with assembled absolute path outside checkout (no literal machine path)
        chk_abs = os.path.join(self.tmp, "chk3")
        abs_like = ("C" + ":/outside/file.py") if os.name == "nt" else ("/ho" + "me/user/outside.py")
        self._write(
            chk_abs,
            "runs/r1/manifest.json",
            json.dumps({
                "id": "r1",
                "copies": [{"sha256": "00" * 32, "store": "rnd-data", "location": abs_like}],
            }),
        )
        with self.assertRaises(DeskError) as cm3:
            desk.scan_rnd_data(chk_abs)
        self.assertTrue(cm3.exception.code.startswith("PERM_"))

        # 4. --out inside --source is refused
        src = os.path.join(self.tmp, "src_ro")
        self._write(src, "a.py", "red")
        with self.assertRaises(DeskError) as cm4:
            desk.main(["text-index", "--source", src, "--out", os.path.join(src, "inside"), "--backend", "fake"])
        self.assertEqual(cm4.exception.code, "PERM_OUT_INSIDE_PROJECT")

    def test_text_and_rnd_data_edge_branches_and_cli_validations(self):
        # Empty chunks and NoneEmbedder chunk embedding
        self.assertEqual(desk.embed_file_chunks(backends.NoneEmbedder(), []).shape, (0,))
        self.assertEqual(desk.embed_file_chunks(backends.FakeEmbedder(), []).shape, (4,))

        # Missing source directory
        with self.assertRaises(DeskError) as cm_miss:
            desk.scan_text_source(os.path.join(self.tmp, "does_not_exist"))
        self.assertEqual(cm_miss.exception.code, "INPUT_BAD_ARGUMENT")

        # Private or blobs folder path refused directly
        priv_dir = os.path.join(self.tmp, "rnd-data-private")
        os.makedirs(priv_dir, exist_ok=True)
        with self.assertRaises(DeskError) as cm_pdir:
            desk.scan_text_source(priv_dir)
        self.assertEqual(cm_pdir.exception.code, "PERM_PRIVATE_SOURCE")

        # Missing runs/ directory
        empty_chk = os.path.join(self.tmp, "empty_chk")
        os.makedirs(empty_chk, exist_ok=True)
        with self.assertRaises(DeskError) as cm_noruns:
            desk.scan_rnd_data(empty_chk)
        self.assertEqual(cm_noruns.exception.code, "PERM_PRIVATE_SOURCE")

        # Manifest with source naming rnd-data-private
        chk_src_priv = os.path.join(self.tmp, "chk_src_priv")
        self._write(
            chk_src_priv,
            "runs/r1/manifest.json",
            json.dumps({"id": "r1", "source": "rnd-data-private/secret", "copies": []}),
        )
        with self.assertRaises(DeskError):
            desk.scan_rnd_data(chk_src_priv)

        # Manifest with rnd-data store pointing into blobs/ (outside runs/)
        chk_blob_loc = os.path.join(self.tmp, "chk_blob_loc")
        self._write(
            chk_blob_loc,
            "runs/r1/manifest.json",
            json.dumps({
                "id": "r1",
                "copies": [{"sha256": "00" * 32, "store": "rnd-data", "location": "blobs/ab/file.py"}],
            }),
        )
        with self.assertRaises(DeskError):
            desk.scan_rnd_data(chk_blob_loc)

        # Manifest with missing file on disk
        chk_missing_file = os.path.join(self.tmp, "chk_missing_file")
        self._write(
            chk_missing_file,
            "runs/r1/manifest.json",
            json.dumps({
                "id": "r1",
                "copies": [{"sha256": "00" * 32, "store": "rnd-data", "location": "runs/r1/files/gone.py"}],
            }),
        )
        with self.assertRaises(DeskError):
            desk.scan_rnd_data(chk_missing_file)

        # Identical code files across two runs in lineage (cos == 1.0) + NUL skip + non-text skip in rnd-data
        chk_lin = os.path.join(self.tmp, "chk_lin")
        f_r1 = self._write(chk_lin, "runs/r1/files/same.py", "red green")
        f_r2 = self._write(chk_lin, "runs/r2/files/same.py", "red green")
        f_nul = self._write(chk_lin, "runs/r2/files/bin.py", b"a\x00b")
        f_png = self._write(chk_lin, "runs/r2/files/pic.png", "ignored")
        self._write(
            chk_lin,
            "runs/r1/manifest.json",
            json.dumps({
                "id": "r1",
                "copies": [{"sha256": desk.sha256_file(f_r1), "store": "rnd-data", "location": "runs/r1/files/same.py"}],
            }),
        )
        self._write(
            chk_lin,
            "runs/r2/manifest.json",
            json.dumps({
                "id": "r2",
                "copies": [
                    {"sha256": desk.sha256_file(f_r2), "store": "rnd-data", "location": "runs/r2/files/same.py"},
                    {"sha256": desk.sha256_file(f_nul), "store": "rnd-data", "location": "runs/r2/files/bin.py"},
                    {"sha256": desk.sha256_file(f_png), "store": "rnd-data", "location": "runs/r2/files/pic.png"},
                ],
            }),
        )
        out_lin = os.path.join(self.tmp, "out_lin")
        desk.text_index(out_lin, backends.FakeEmbedder(), rnd_data=chk_lin, limit=5, log=lambda *a: None)
        t_items, t_vecs, _ = desk.load_text_cache(out_lin, "fake")
        lin = desk.lineage(t_items, t_vecs)
        self.assertEqual(lin["files"]["same.py"][0]["cosine"], 1.0)

        # Missing text cache raises STATE_NO_INDEX
        with self.assertRaises(DeskError) as cm_no_idx:
            desk.load_text_cache(os.path.join(self.tmp, "no_cache"), "fake")
        self.assertEqual(cm_no_idx.exception.code, "STATE_NO_INDEX")

        # CLI argument errors on text commands
        with self.assertRaises(DeskError):
            desk.main(["text-index", "--out", out_lin, "--backend", "fake"])
        with self.assertRaises(DeskError):
            desk.main(["text-index", "--source", chk_lin, "--rnd-data", chk_lin, "--out", out_lin, "--backend", "fake"])
        with self.assertRaises(DeskError):
            desk.main(["text-dupes", "--backend", "fake"])
        with self.assertRaises(DeskError):
            desk.main(["text-search", "--out", out_lin, "--backend", "none", "--text", "red"])
        with self.assertRaises(DeskError):
            desk.main(["text-search", "--out", out_lin, "--backend", "fake"])
        # --vram-budget and --wait forwarded to make()
        seen_kw = []
        with contextlib.redirect_stdout(io.StringIO()):
            desk.main(
                ["text-search", "--out", out_lin, "--backend", "fake", "--text", "red",
                 "--vram-budget", "20000", "--wait", "10"],
                make=lambda b, **kw: (seen_kw.append(kw), backends.FakeEmbedder())[1],
            )
        self.assertEqual(seen_kw[0]["vram_budget"], 20000)
        self.assertEqual(seen_kw[0]["wait"], 10.0)

    def test_dual_role_vectors_code_query_cache_paths_and_meta_invalidation(self):
        src = os.path.join(self.tmp, "src_dual")
        self._write(src, "dir/alpha.py", "red green")
        self._write(src, "dir/beta.py", "blue")
        out_dual = os.path.join(self.tmp, "out_dual")

        emb = backends.FakeEmbedder()
        desk.text_index(out_dual, emb, source=src, log=lambda *a: None)
        tdir = desk.text_cache_dir(out_dual, "fake")
        self.assertTrue(os.path.isfile(os.path.join(tdir, "texts.npy")))
        self.assertTrue(os.path.isfile(os.path.join(tdir, "texts_symmetric.npy")))

        # Verify texts.npy used document role with base name title, and texts_symmetric.npy used symmetric role without title
        roles_and_prefixes = [
            (c["role"], backends.format_eg2_prefix(c["role"], c["titles"][0] if c["titles"] else None))
            for c in emb.seen_calls
        ]
        self.assertIn(("document", "title: alpha.py | text: "), roles_and_prefixes)
        self.assertIn(("document", "title: beta.py | text: "), roles_and_prefixes)
        self.assertIn(("symmetric", "task: sentence similarity | query: "), roles_and_prefixes)
        items_doc, vecs_doc, _meta_doc = desk.load_text_cache(out_dual, "fake", role="document")
        _items_sym, vecs_sym, _ = desk.load_text_cache(out_dual, "fake", role="symmetric")
        self.assertEqual(len(items_doc), 2)
        alpha_idx = next(i for i, it in enumerate(items_doc) if it.name == "alpha.py")
        expected_doc = desk.embed_file_chunks(emb, ["red green"], role="document", title="alpha.py")
        expected_sym = desk.embed_file_chunks(emb, ["red green"], role="symmetric", title=None)
        np.testing.assert_allclose(vecs_doc[alpha_idx], expected_doc, atol=1e-6)
        np.testing.assert_allclose(vecs_sym[alpha_idx], expected_sym, atol=1e-6)
        self.assertFalse(np.allclose(vecs_doc[alpha_idx], vecs_sym[alpha_idx]))

        # File longer than 49,024 chars (old char read limit) is read past that point by text_index
        src_long = os.path.join(self.tmp, "src_long_read")
        long_body = ("green " * 9000) + ("blue " * 2000)  # 54,000 chars of green, then blue past char 49,024
        self.assertGreater(len(long_body), 49024)
        self._write(src_long, "big.py", long_body)
        out_long = os.path.join(self.tmp, "out_long_read")
        desk.text_index(out_long, backends.FakeEmbedder(), source=src_long, log=lambda *a: None)
        _, long_vecs, _ = desk.load_text_cache(out_long, "fake", role="document")
        # Channel 2 is 'blue' in FakeEmbedder; only present if text_index read past 54,000 chars into chunk 2
        self.assertGreater(float(long_vecs[0][2]), 0.1)

        # Unprefixed backend (e.g. gte-modernbert) writes only texts.npy and load_text_cache(role="symmetric") keeps reading texts.npy
        class _UnprefixedStub(backends.FakeEmbedder):
            has_symmetric_vectors = False

            def __init__(self):
                super().__init__(dim=768)
                self.name = "gte-modernbert"
                self.revision = "gte-rev"
                self.max_tokens = 8192
                self.tokenizer = desk._FALLBACK_TOKENIZER

        out_bge = os.path.join(self.tmp, "out_bge")
        desk.text_index(out_bge, _UnprefixedStub(), source=src, log=lambda *a: None)
        bge_tdir = desk.text_cache_dir(out_bge, "gte-modernbert")
        self.assertTrue(os.path.isfile(os.path.join(bge_tdir, "texts.npy")))
        self.assertFalse(os.path.isfile(os.path.join(bge_tdir, "texts_symmetric.npy")))
        _, bge_sym_vecs, _ = desk.load_text_cache(out_bge, "gte-modernbert", role="symmetric")
        self.assertEqual(bge_sym_vecs.shape[0], 2)

        # Prefix-sensitive backend (fake/eg2) refuses load_text_cache(role="symmetric") when texts_symmetric.npy is missing
        sym_file_dual = os.path.join(tdir, "texts_symmetric.npy")
        os.remove(sym_file_dual)
        with self.assertRaises(DeskError) as cm_miss_sym:
            desk.load_text_cache(out_dual, "fake", role="symmetric")
        self.assertEqual(cm_miss_sym.exception.code, "STATE_NO_INDEX")
        self.assertIn("text-index", cm_miss_sym.exception.message)
        with self.assertRaises(DeskError) as cm_lin_sym:
            desk.main(["lineage", "--out", out_dual, "--backend", "fake"])
        self.assertEqual(cm_lin_sym.exception.code, "STATE_NO_INDEX")
        # Restore texts_symmetric.npy
        desk.text_index(out_dual, emb, source=src, log=lambda *a: None)

        # text-search uses "code_query" when --task code and "query" otherwise
        emb_spy = backends.FakeEmbedder()
        with contextlib.redirect_stdout(io.StringIO()):
            desk.main(
                ["text-search", "--out", out_dual, "--backend", "fake", "--task", "code", "--text", "red"],
                make=lambda b, **kw: emb_spy,
            )
        self.assertEqual(emb_spy.seen_calls[-1]["role"], "code_query")
        self.assertEqual(backends.format_eg2_prefix(emb_spy.seen_calls[-1]["role"]), "task: code retrieval | query: ")
        with contextlib.redirect_stdout(io.StringIO()):
            desk.main(
                ["text-search", "--out", out_dual, "--backend", "fake", "--task", "text", "--text", "red"],
                make=lambda b, **kw: emb_spy,
            )
        self.assertEqual(emb_spy.seen_calls[-1]["role"], "query")
        self.assertEqual(backends.format_eg2_prefix(emb_spy.seen_calls[-1]["role"]), "task: search result | query: ")

        # Non-default dim and vision_tokens in cache_dir and text_cache_dir
        self.assertEqual(desk.cache_dir(out_dual, "eg2", dim=768, vision_tokens=280), os.path.join(out_dual, "eg2"))
        self.assertEqual(desk.cache_dir(out_dual, "eg2", dim=256, vision_tokens=280), os.path.join(out_dual, "eg2-d256"))
        self.assertEqual(desk.cache_dir(out_dual, "eg2", dim=512, vision_tokens=560), os.path.join(out_dual, "eg2-d512-vt560"))
        self.assertEqual(desk.text_cache_dir(out_dual, "eg2", dim=768), os.path.join(out_dual, "eg2", "text"))
        self.assertEqual(desk.text_cache_dir(out_dual, "eg2", dim=256), os.path.join(out_dual, "eg2-d256", "text"))

        # CLI text-index and text-search with --dim 256 use the dim-scoped cache path
        with contextlib.redirect_stdout(io.StringIO()):
            desk.main(["text-index", "--source", src, "--out", out_dual, "--backend", "fake", "--dim", "256"])
        _, vecs_256, meta_256 = desk.load_text_cache(out_dual, "fake", dim=256)
        self.assertEqual(vecs_256.shape, (2, 256))
        self.assertEqual(meta_256["dim"], 256)

        # 7-field meta.json invalidation check for text_index
        for field, mutated in (
            ("revision", "other-rev"),
            ("dim", 999),
            ("vision_tokens", 1120),
            ("max_tokens", 4096),
            ("max_chunks", 4),
            ("role", "other"),
            ("dtype", "float16"),
        ):
            # Restore clean cache first
            desk.text_index(out_dual, emb, source=src, log=lambda *a: None)
            meta_path = os.path.join(tdir, "meta.json")
            with open(meta_path, encoding="utf-8") as f:
                m = json.load(f)
            m[field] = mutated
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(m, f)
            re_meta = desk.text_index(out_dual, emb, source=src, log=lambda *a: None)
            self.assertEqual(re_meta["embedded_now"], 2, f"Expected recompute when text meta.{field} changed")

        # 7-field meta.json invalidation check for image index
        img_src = os.path.join(self.tmp, "img_src_meta")
        os.makedirs(os.path.join(img_src, "outputs", "w1"), exist_ok=True)
        from PIL import Image
        Image.new("RGB", (16, 16), (220, 20, 20)).save(os.path.join(img_src, "outputs", "w1", "a.png"))
        out_img = os.path.join(self.tmp, "out_img_meta")
        desk.index(img_src, out_img, emb, log=lambda *a: None)
        img_meta_path = os.path.join(desk.cache_dir(out_img, "fake"), "meta.json")
        with open(img_meta_path, encoding="utf-8") as f:
            im = json.load(f)
        for k in ("revision", "dim", "vision_tokens", "max_tokens", "max_chunks", "role", "dtype"):
            self.assertIn(k, im)
        im["vision_tokens"] = 1120
        with open(img_meta_path, "w", encoding="utf-8") as f:
            json.dump(im, f)
        re_img_meta = desk.index(img_src, out_img, emb, log=lambda *a: None)
        self.assertEqual(re_img_meta["embedded_now"], 1)


def _bit_reverse_invert_dhash(h: str, size: int = 16) -> str:
    """Bit-reverse each `size`-bit row and invert (~) as a naive shortcut for dhash_flip."""
    val = int(h, 16)
    mask = (1 << size) - 1
    out = 0
    for row in range(size):
        row_bits = (val >> ((size - 1 - row) * size)) & mask
        rev = 0
        for b in range(size):
            if not (row_bits & (1 << b)):
                rev |= 1 << (size - 1 - b)
        out = (out << size) | rev
    return f"{out:0{size * size // 4}x}"


class D2DupesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="dd_d2_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_asymmetric_fixture(self, path: str, flip: bool = False):
        from PIL import Image
        im = Image.new("RGB", (67, 53))
        px = im.load()
        for y in range(53):
            for x in range(67):
                # Asymmetric horizontal + vertical gradient with off-centre L-shape and diagonal band
                r = int((x * 230) / 66)
                g = int((y * 180) / 52)
                b = int(((x * 3 + y * 5) % 256))
                if 9 <= x <= 24 and 7 <= y <= 38:
                    r, g, b = 250, 30, 15
                elif 24 < x <= 41 and 28 <= y <= 38:
                    r, g, b = 20, 240, 60
                elif (x - 48) ** 2 + (y - 16) ** 2 <= 36:
                    r, g, b = 15, 35, 245
                px[x, y] = (r, g, b)
        if flip:
            im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        im.save(path)

    def test_mirror_dhash_and_scan_v2_invalidation(self):
        orig_path = os.path.join(self.tmp, "orig.png")
        flip_path = os.path.join(self.tmp, "flipped.png")
        self._make_asymmetric_fixture(orig_path, flip=False)
        self._make_asymmetric_fixture(flip_path, flip=True)

        dh_orig = desk.dhash(orig_path)
        dh_flip_orig = desk.dhash_flip(orig_path)
        dh_flipped_file = desk.dhash(flip_path)
        dh_flip_flipped_file = desk.dhash_flip(flip_path)

        # Plain hamming between orig and flipped copy is large (> NEAR_BITS), while flip_hamming is within NEAR_BITS (in fact 0)
        self.assertGreater(desk.hamming(dh_orig, dh_flipped_file), desk.NEAR_BITS)
        it_orig = desk.Item("orig", [orig_path], ["w1"], dh_orig, dhash_flip=dh_flip_orig)
        it_flip = desk.Item("flip", [flip_path], ["w1"], dh_flipped_file, dhash_flip=dh_flip_flipped_file)
        self.assertEqual(desk.flip_hamming(it_orig, it_flip), 0)
        self.assertLessEqual(desk.flip_hamming(it_orig, it_flip), desk.NEAR_BITS)

        # Fallback when dhash_flip is empty on in-memory test items
        it_plain_a = desk.Item("a", [orig_path], ["w1"], dh_orig)
        it_plain_b = desk.Item("b", [flip_path], ["w1"], dh_flipped_file)
        self.assertEqual(desk.flip_hamming(it_plain_a, it_plain_b), desk.hamming(dh_orig, dh_flipped_file))

        # The fixture's dhash_flip differs from bit-reversed, inverted dhash (because LANCZOS 9x8 is asymmetric on 67px width)
        self.assertNotEqual(dh_flip_orig, _bit_reverse_invert_dhash(dh_orig))

        # Scan cache stamps scan_v == 2 and recomputes rows with missing scan_v or scan_v == 1
        proj = os.path.join(self.tmp, "proj")
        out = os.path.join(self.tmp, "out")
        os.makedirs(out, exist_ok=True)
        wdir = os.path.join(proj, "outputs", "w1")
        os.makedirs(wdir, exist_ok=True)
        shutil.copyfile(orig_path, os.path.join(wdir, "orig.png"))
        shutil.copyfile(flip_path, os.path.join(wdir, "flipped.png"))

        scan_file = os.path.join(out, "scan.jsonl")
        items1, _ = desk.scan(proj, out=out)
        self.assertEqual(len(items1), 2)
        for it in items1:
            self.assertEqual(len(it.dhash_flip), 64)

        with open(scan_file, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        self.assertEqual(len(rows), 2)
        for r in rows:
            self.assertEqual(r["scan_v"], desk.SCAN_V)
            self.assertEqual(len(r["dhash_flip"]), 64)

        # Mutate row 0 to have no scan_v (and bogus dhash_flip) and row 1 to have scan_v: 1 (and bogus dhash_flip)
        rows[0].pop("scan_v", None)
        rows[0]["dhash_flip"] = "0" * 64
        rows[1]["scan_v"] = 1
        rows[1]["dhash_flip"] = "f" * 64
        with open(scan_file, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

        # _load_scan_cache rejects stale rows and scan() recomputes them
        self.assertEqual(desk._load_scan_cache(out), {})
        items2, _ = desk.scan(proj, out=out)
        by_path = {os.path.basename(it.paths[0]): it for it in items2}
        self.assertEqual(by_path["orig.png"].dhash_flip, dh_flip_orig)
        self.assertEqual(by_path["flipped.png"].dhash_flip, dh_flip_flipped_file)

        # Second scan hits fresh v2 cache
        cached_map = desk._load_scan_cache(out)
        self.assertEqual(len(cached_map), 2)
        items3, _ = desk.scan(proj, out=out)
        self.assertEqual(len(items3), 2)

    def test_dinov2_two_signal_rule_and_buckets_and_missing_index(self):
        # 4 items to test both branches and rejection:
        # Pair (0, 1): cos = 0.95 (>= 0.94, < 0.985) and flip_hamming = 12 (<= 44) -> ACCEPTED by branch 1
        # Pair (0, 2): cos = 0.99 (>= 0.985) and flip_hamming = 60 (> 44) -> ACCEPTED by branch 2
        # Pair (0, 3): cos = 0.96 (>= 0.94, < 0.985) and flip_hamming = 52 (> 44) -> REJECTED
        dh0 = "0" * 64
        dh1 = "0" * 61 + "fff"
        dh2 = "f" * 15 + "0" * 49
        dh3 = "0" * 15 + "f" * 13 + "0" * 36
        it0 = desk.Item("h0", ["a.png"], ["w"], dh0, dhash_flip=dh0)
        it1 = desk.Item("h1", ["b.png"], ["w"], dh1, dhash_flip="f" * 64)  # 12 bits from it0
        it2 = desk.Item("h2", ["c.png"], ["w"], dh2, dhash_flip=dh2)       # 60 bits from it0
        it3 = desk.Item("h3", ["d.png"], ["w"], dh3, dhash_flip=dh3)       # 52 bits from it0, 112 bits from it2
        items = [it0, it1, it2, it3]

        # Construct 4D orthonormal basis vectors so we can set exact cosines with it0 = [1, 0, 0, 0]
        v0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        v1 = np.array([0.95, np.sqrt(1.0 - 0.95**2), 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.99, 0.0, np.sqrt(1.0 - 0.99**2), 0.0], dtype=np.float32)
        v3 = np.array([0.96, 0.0, 0.0, np.sqrt(1.0 - 0.96**2)], dtype=np.float32)
        img = np.stack([v0, v1, v2, v3], axis=0)

        d_dino = desk.dupes(items, img, semantic=None, backend="dinov2")
        self.assertEqual(d_dino["rule"], "dinov2-two-signal")
        self.assertEqual(d_dino["near"], [])
        sem_pairs = {(i, j) for i, j, _ in d_dino["semantic"]}
        self.assertIn((0, 1), sem_pairs)
        self.assertIn((0, 2), sem_pairs)
        self.assertNotIn((0, 3), sem_pairs)
        self.assertEqual(
            d_dino["thresholds"],
            {
                "near_bits": 44,
                "semantic_cosine": 0.94,
                "dinov2_cos_low": 0.94,
                "dinov2_cos_high": 0.985,
                "two_signal_bits": 44,
            },
        )

        # Passing explicit semantic threshold on dinov2 uses near+cosine rule
        d_dino_override = desk.dupes(items, img, semantic=0.955, backend="dinov2")
        self.assertEqual(d_dino_override["rule"], "near+cosine")
        self.assertEqual([(i, j) for i, j, _ in d_dino_override["near"]], [(0, 1)])
        sem_over_pairs = {(i, j) for i, j, _ in d_dino_override["semantic"]}
        self.assertIn((0, 2), sem_over_pairs)
        self.assertIn((0, 3), sem_over_pairs)
        self.assertNotIn((0, 1), sem_over_pairs)  # already in near

        # A siglip2 cache gets the near+cosine rule
        d_sig = desk.dupes(items, img, semantic=None, backend="siglip2")
        self.assertEqual(d_sig["rule"], "near+cosine")
        self.assertEqual([(i, j) for i, j, _ in d_sig["near"]], [(0, 1)])
        self.assertEqual(d_sig["thresholds"], {"near_bits": 26, "semantic_cosine": 0.95})

        # CLI dupes defaults to dinov2 and missing dinov2 index names dataset-desk index --project <p> --backend dinov2
        self.assertEqual(desk.command_task("dupes"), "dupes")
        proj = os.path.join(self.tmp, "myproj")
        out = os.path.join(self.tmp, "myout")
        os.makedirs(os.path.join(proj, "outputs"), exist_ok=True)
        with self.assertRaises(DeskError) as cm:
            desk.main(["dupes", "--project", proj, "--out", out])
        self.assertEqual(cm.exception.code, "STATE_NO_INDEX")
        self.assertIn(f"dataset-desk index --project {proj} --backend dinov2", cm.exception.message)

    def test_view_check_record_and_stem_fallback(self):
        # Two records of one asset with different render_id are never paired in near or semantic;
        # two records with the same render_id are still paired.
        rec_v1 = {"id": "r1", "asset_id": "hero_ship", "render_id": "y0_e15"}
        rec_v2 = {"id": "r2", "asset_id": "hero_ship", "render_id": "y90_e15"}
        rec_v1_dup = {"id": "r3", "asset_id": "hero_ship", "render_id": "y0_e15"}

        it_v1 = desk.Item("h1", ["/p/a.png"], ["w"], "0000000000000000", dhash_flip="0000000000000000", record=rec_v1)
        it_v2 = desk.Item("h2", ["/p/b.png"], ["w"], "0000000000000001", dhash_flip="0000000000000001", record=rec_v2)
        it_v1_dup = desk.Item("h3", ["/p/c.png"], ["w"], "0000000000000003", dhash_flip="0000000000000003", record=rec_v1_dup)

        # All 3 have identical vectors (cos = 1.0) and dhash within 2 bits (would hit both near and semantic)
        vecs3 = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32)

        # Under near+cosine (e.g. siglip2): (0, 1) and (1, 2) are different views of hero_ship;
        # each is dropped from both near and semantic and counted ONCE in view_pairs_skipped (= 2).
        # Pair (0, 2) has the same asset_id and same render_id, so it is still paired!
        d_sig = desk.dupes([it_v1, it_v2, it_v1_dup], vecs3, semantic=None, backend="siglip2")
        self.assertEqual([(i, j) for i, j, _ in d_sig["near"]], [(0, 2)])
        self.assertEqual(d_sig["semantic"], [])
        self.assertEqual(d_sig["view_pairs_skipped"], 2)

        # Under dinov2-two-signal: (0, 1) and (1, 2) are also skipped and (0, 2) is kept
        d_dino = desk.dupes([it_v1, it_v2, it_v1_dup], vecs3, semantic=None, backend="dinov2")
        self.assertEqual([(i, j) for i, j, _ in d_dino["semantic"]], [(0, 2)])
        self.assertEqual(d_dino["view_pairs_skipped"], 2)

        # File-stem fallback when no record exists: ^(.+)__(y-?\d+_e-?\d+(?:_.*)?)$
        it_stem1 = desk.Item("s1", ["/p/turret_mk2__y-45_e30_front.png"], ["w"], "0000000000000000")
        it_stem2 = desk.Item("s2", ["/p/turret_mk2__y45_e-15.png"], ["w"], "0000000000000001")
        self.assertEqual(desk._asset_view(it_stem1), ("turret_mk2", "y-45_e30_front"))
        self.assertEqual(desk._asset_view(it_stem2), ("turret_mk2", "y45_e-15"))
        d_stem = desk.dupes([it_stem1, it_stem2], vecs3[:2], semantic=None, backend="siglip2")
        self.assertEqual(d_stem["near"], [])
        self.assertEqual(d_stem["semantic"], [])
        self.assertEqual(d_stem["view_pairs_skipped"], 1)

        # Record with asset_id but no render_id takes the view from the stem
        rec_partial = {"id": "p1", "asset_id": "canonical_turret", "render_id": None}
        it_part1 = desk.Item("p1", ["/p/ignored_stem__y0_e0.png"], ["w"], "0000000000000000", record=rec_partial)
        it_part2 = desk.Item("p2", ["/p/ignored_stem__y180_e0.png"], ["w"], "0000000000000001", record=rec_partial)
        self.assertEqual(desk._asset_view(it_part1), ("canonical_turret", "y0_e0"))
        self.assertEqual(desk._asset_view(it_part2), ("canonical_turret", "y180_e0"))
        d_part = desk.dupes([it_part1, it_part2], vecs3[:2], semantic=None, backend="dinov2")
        self.assertEqual(d_part["semantic"], [])
        self.assertEqual(d_part["view_pairs_skipped"], 1)

        # load_records populates asset_id and render_id from provenance (or None when absent)
        proj = os.path.join(self.tmp, "proj_rec")
        rdir = os.path.join(proj, "records")
        os.makedirs(rdir, exist_ok=True)
        with open(os.path.join(rdir, "one.json"), "w", encoding="utf-8") as f:
            json.dump({
                "id": "one",
                "asset_path": "outputs/w/one.png",
                "provenance": {"asset_id": "crate_01", "render_id": "y30_e20"},
            }, f)
        with open(os.path.join(rdir, "two.json"), "w", encoding="utf-8") as f:
            json.dump({
                "id": "two",
                "asset_path": "outputs/w/two.png",
                "provenance": {},
            }, f)
        recs_by_path, _ = desk.load_records(proj)
        self.assertEqual(recs_by_path["outputs/w/one.png"]["asset_id"], "crate_01")
        self.assertEqual(recs_by_path["outputs/w/one.png"]["render_id"], "y30_e20")
        self.assertIsNone(recs_by_path["outputs/w/two.png"]["asset_id"])
        self.assertIsNone(recs_by_path["outputs/w/two.png"]["render_id"])

    def test_text_dupes_code_filter_thresholds_backend_guard_and_percentiles(self):
        t_items = [
            desk.TextItem("c1", ["src/a.py"], "a.py", ".py"),
            desk.TextItem("c2", ["src/b.sh"], "b.sh", ".sh"),
            desk.TextItem("j1", ["data/c.json"], "c.json", ".json"),
            desk.TextItem("l1", ["logs/d.log"], "d.log", ".log"),
        ]
        # Make c1 & c2 cosine = 0.96 (>= 0.95 for code, < 0.98 for text)
        # Make j1 & l1 cosine = 0.99 (identical or near-identical, should be dropped under task="code")
        vc1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        vc2 = np.array([0.96, np.sqrt(1.0 - 0.96**2), 0.0], dtype=np.float32)
        vj1 = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        vl1 = np.array([0.0, np.sqrt(1.0 - 0.99**2), 0.99], dtype=np.float32)
        t_vecs = np.stack([vc1, vc2, vj1, vl1], axis=0)

        # --task code drops .json and .log items and defaults to 0.95 on eg2
        res_code = desk.text_dupes(t_items, t_vecs, semantic=None, task="code", backend="eg2")
        self.assertEqual(res_code["threshold"], 0.95)
        self.assertEqual([p[:2] for p in res_code["semantic"]], [("src/a.py", "src/b.sh")])
        self.assertIn("top1_percentiles", res_code)
        self.assertIn("p50", res_code["top1_percentiles"])

        # task="text" defaults to 0.98 on bge-m3 (so 0.96 pair is below 0.98, 0.99 pair is included)
        res_text = desk.text_dupes(t_items, t_vecs, semantic=None, task="text", backend="bge-m3")
        self.assertEqual(res_text["threshold"], 0.98)
        self.assertEqual([p[:2] for p in res_text["semantic"]], [("data/c.json", "logs/d.log")])
        self.assertIn("top1_percentiles", res_text)

        # --semantic overrides both (and bypasses the backend guard)
        res_override_code = desk.text_dupes(t_items, t_vecs, semantic=0.97, task="code", backend="qwen3-emb-0.6b")
        self.assertEqual(res_override_code["threshold"], 0.97)
        self.assertEqual(res_override_code["semantic"], [])

        res_override_text = desk.text_dupes(t_items, t_vecs, semantic=0.95, task="text", backend="gte-modernbert")
        self.assertEqual(res_override_text["threshold"], 0.95)
        self.assertEqual(len(res_override_text["semantic"]), 2)

        # Backend guard: qwen3-emb-0.6b or gte-modernbert (or wrong task-backend combo) with no --semantic refuses with INPUT_BAD_ARGUMENT
        for bad_backend, task in (
            ("qwen3-emb-0.6b", "text"),
            ("gte-modernbert", "text"),
            ("qwen3-emb-0.6b", "code"),
            ("gte-modernbert", "code"),
            ("bge-m3", "code"),
            ("eg2", "text"),
            ("fake", "text"),
        ):
            with self.assertRaises(DeskError) as cm:
                desk.text_dupes(t_items, t_vecs, semantic=None, task=task, backend=bad_backend)
            self.assertEqual(cm.exception.code, "INPUT_BAD_ARGUMENT")
            self.assertIn("--semantic", cm.exception.message)

        # CLI text-dupes wiring test for --task code and refusal on uncalibrated backend
        src = os.path.join(self.tmp, "tsrc")
        os.makedirs(src, exist_ok=True)
        with open(os.path.join(src, "m1.py"), "w", encoding="utf-8") as f:
            f.write("def alpha():\n    return 42\n")
        with open(os.path.join(src, "m2.py"), "w", encoding="utf-8") as f:
            f.write("def alpha():\n    return 42\n")
        with open(os.path.join(src, "ignore.json"), "w", encoding="utf-8") as f:
            f.write('{"same": true}\n')
        with open(os.path.join(src, "ignore.log"), "w", encoding="utf-8") as f:
            f.write('{"same": true}\n')

        class _EG2Stub(backends.FakeEmbedder):
            def __init__(self):
                super().__init__(dim=768)
                self.name = "eg2"
                self.revision = desk.registry.REGISTRY["eg2"].revision
                self.tokenizer = desk._FALLBACK_TOKENIZER

        out_eg2 = os.path.join(self.tmp, "out_eg2")
        desk.text_index(out_eg2, _EG2Stub(), source=src, log=lambda *a: None)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            desk.main(["text-dupes", "--out", out_eg2, "--task", "code"])
        cli_code = json.loads(buf.getvalue())
        self.assertEqual(cli_code["threshold"], 0.95)
        self.assertIn("top1_percentiles", cli_code)
        # Only .py files should appear in exact/semantic under --task code
        for grp in cli_code["exact"]:
            for p in grp:
                self.assertTrue(p.endswith(".py"))

        # Refusal from CLI when running text-dupes without --semantic on fake cache
        out_fake = os.path.join(self.tmp, "out_fake")
        desk.text_index(out_fake, backends.FakeEmbedder(), source=src, log=lambda *a: None)
        with self.assertRaises(DeskError) as cm_cli:
            desk.main(["text-dupes", "--out", out_fake, "--backend", "fake"])
        self.assertEqual(cm_cli.exception.code, "INPUT_BAD_ARGUMENT")


if __name__ == "__main__":
    unittest.main()
