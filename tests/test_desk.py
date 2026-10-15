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

import numpy as np
from PIL import Image

from dataset_desk import backends, description_gate, desk

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
        self.assertEqual(set(backends.EG2_PREFIX), {"query", "document", "symmetric", "clustering", "classification"})
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
        self.assertLess(by_id["join_a"]["max_content_jaccard"], 0.85)
        self.assertIn("exact_duplicate", by_id["join_a"]["flags"])
        self.assertIn("exact_duplicate", by_id["join_b"]["flags"])
        self.assertTrue(by_id["join_pass"]["passed"])

        self.assertAlmostEqual(by_id["jac_85_a"]["max_content_jaccard"], 0.85, places=4)
        self.assertIn("near_duplicate", by_id["jac_85_a"]["flags"])
        self.assertIn("near_duplicate", by_id["jac_85_b"]["flags"])

        self.assertAlmostEqual(by_id["jac_84_a"]["max_content_jaccard"], 16 / 19, places=4)
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
        from unittest import mock

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


if __name__ == "__main__":
    unittest.main()
