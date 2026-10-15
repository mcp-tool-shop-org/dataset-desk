"""Backend tests: every model-backed path runs against the stand-ins in fakes.py (no torch, no weights, no network)."""

import os
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
from PIL import Image

import fakes
from dataset_desk import backends
from dataset_desk.errors import DeskError


def png(folder, name, colour=(200, 10, 10), size=(16, 16)):
    path = os.path.join(folder, name)
    Image.new("RGB", size, colour).save(path)
    return path


class TmpCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name


class ArgvThreads(unittest.TestCase):
    def test_flag_forms_and_fallbacks(self):
        f = backends._argv_threads
        self.assertEqual(f(["--threads", "6"]), 6)
        self.assertEqual(f(["--threads=7"]), 7)
        self.assertEqual(f(["--threads", "0"]), 1)
        self.assertEqual(f(["--threads", "abc"], default=3), 3)
        self.assertEqual(f(["--threads=abc"], default=5), 5)
        self.assertEqual(f(["--threads"]), backends.DEFAULT_THREADS)
        self.assertEqual(f([]), backends.DEFAULT_THREADS)

    def test_reads_sys_argv_when_none(self):
        with mock.patch.object(sys, "argv", ["prog", "index", "--threads", "2"]):
            self.assertEqual(backends._argv_threads(), 2)


class EnterTorchRegion(unittest.TestCase):
    def test_sets_env_threads_and_interop_once(self):
        t = fakes.make_torch()
        with mock.patch.dict(sys.modules, {"torch": t}), mock.patch.object(backends, "_TORCH_INTEROP_SET", False):
            backends.enter_torch_region(3)
            backends.enter_torch_region(2)
        self.assertEqual(t.calls.count(("interop", 1)), 1)
        self.assertEqual([c for c in t.calls if c[0] == "threads"], [("threads", 3), ("threads", 2)])
        self.assertEqual(os.environ["OMP_NUM_THREADS"], "2")
        self.assertEqual(os.environ["OPENBLAS_NUM_THREADS"], "1")

    def test_interop_runtime_error_is_swallowed_and_not_retried(self):
        t = fakes.make_torch()
        t.interop_raises = True
        with mock.patch.dict(sys.modules, {"torch": t}), mock.patch.object(backends, "_TORCH_INTEROP_SET", False):
            backends.enter_torch_region(1)
            backends.enter_torch_region(1)
            self.assertTrue(backends._TORCH_INTEROP_SET)
        self.assertEqual(t.calls.count(("interop", 1)), 1)


class UnitRowsAndGuards(unittest.TestCase):
    def test_unit_rows_normalises_and_lifts_1d(self):
        out = backends.unit_rows([3.0, 4.0])
        self.assertEqual(out.shape, (1, 2))
        np.testing.assert_allclose(out[0], [0.6, 0.8], atol=1e-6)
        self.assertEqual(out.dtype, np.float32)

    def test_unit_rows_zero_row_stays_finite_and_empty_ok(self):
        self.assertTrue(np.isfinite(backends.unit_rows([[0.0, 0.0]])).all())
        self.assertEqual(backends.unit_rows(np.zeros((0, 3))).shape, (0, 3))

    def test_unit_rows_rejects_nan_and_inf(self):
        for bad in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                backends.unit_rows([[1.0, bad]])

    def test_check_device(self):
        backends.check_device("cpu", None)
        backends.check_device("cuda", "grant-1")
        with self.assertRaises(DeskError) as cm:
            backends.check_device("cuda", None)
        self.assertEqual(cm.exception.code, "PERM_GPU_NEEDS_GRANT")


class SimpleBackends(TmpCase):
    def test_none_embedder_shapes(self):
        e = backends.NoneEmbedder()
        self.assertEqual(e.images(["a", "b"]).shape, (2, 0))
        self.assertEqual(e.texts(["a"]).shape, (1, 0))

    def test_fake_embedder_images_texts(self):
        e = backends.FakeEmbedder()
        v = e.images([png(self.tmp, "r.png", (255, 0, 0))])
        self.assertEqual(int(np.argmax(v[0])), 0)
        self.assertEqual(e.images([]).shape, (0, 4))
        t = e.texts(["red red blue"])
        self.assertGreater(t[0][0], t[0][2])
        self.assertEqual(e.texts([]).shape, (0, 4))
        with self.assertRaises(ValueError):
            e.texts(["x"], role="bogus")

    def test_fake_multimodal_variants(self):
        e = backends.FakeEmbedder()
        g = png(self.tmp, "g.png", (0, 255, 0))
        both = e.multimodal([{"text": "green", "images": [g]}])
        self.assertEqual(int(np.argmax(both[0])), 1)
        single = e.multimodal([{"image": g}])  # no text: neutral text vector; 'image' key form
        self.assertEqual(int(np.argmax(single[0])), 1)
        text_only = e.multimodal([{"text": "blue"}], role=None)
        self.assertEqual(int(np.argmax(text_only[0])), 2)
        self.assertEqual(e.multimodal([]).shape, (0, 4))
        with self.assertRaises(ValueError):
            e.multimodal([{"text": "x"}], role="bogus")

    def test_make_dispatch(self):
        self.assertIsInstance(backends.make("none"), backends.NoneEmbedder)
        self.assertIsInstance(backends.make("fake"), backends.FakeEmbedder)
        eg2 = backends.make("eg2", dim=512)
        self.assertIsInstance(eg2, backends.EG2Embedder)
        self.assertEqual(eg2.dim, 512)
        self.assertIsInstance(backends.make("siglip2", batch=2), backends.SigLIP2Embedder)
        with self.assertRaises(DeskError) as cm:
            backends.make("nope")
        self.assertEqual(cm.exception.code, "INPUT_UNKNOWN_BACKEND")
        with self.assertRaises(DeskError) as cm:
            backends.make("fake", device="cuda")
        self.assertEqual(cm.exception.code, "PERM_GPU_NEEDS_GRANT")

    def test_make_caps_threads_in_environment(self):
        with mock.patch.dict(os.environ):
            backends.make("none", threads=3)
            self.assertEqual(os.environ["OMP_NUM_THREADS"], "3")
            self.assertEqual(os.environ["RAYON_NUM_THREADS"], "1")


class FakeDrafterTests(TmpCase):
    def test_tones_and_recording(self):
        d = backends.FakeDrafter()
        r = png(self.tmp, "r.png", (255, 0, 0))
        g = png(self.tmp, "g.png", (0, 255, 0))
        b = png(self.tmp, "b.png", (0, 0, 255))
        bad = os.path.join(self.tmp, "bad.png")
        with open(bad, "wb") as f:
            f.write(b"not an image")
        out = d.draft([r, g, b, bad, os.path.join(self.tmp, "missing.png")])
        self.assertIn("crimson", out[0])
        self.assertIn("emerald", out[1])
        self.assertIn("cobalt", out[2])
        self.assertIn("neutral", out[3])
        self.assertIn("(missing)", out[4])
        self.assertEqual(d.seen_paths[0], r)
        self.assertEqual(len(d.seen_paths), 5)

    def test_rejects_non_path_input(self):
        d = backends.FakeDrafter()
        for bad in ("a.png", [1], [None]):
            with self.assertRaises(TypeError):
                d.draft(bad)
        self.assertEqual(d.seen_paths, [])

    def test_make_drafter(self):
        with mock.patch.dict(os.environ):
            self.assertIsNone(backends.make_drafter("none"))
            self.assertIsNone(backends.make_drafter(""))
            self.assertIsInstance(backends.make_drafter("fake"), backends.FakeDrafter)
            ps = backends.make_drafter("florence2", threads=2)
            self.assertIsInstance(ps, backends.PlainSightDrafter)
            self.assertEqual(ps.threads, 2)
            self.assertIsInstance(backends.make_drafter("plain-sight"), backends.PlainSightDrafter)
            with self.assertRaises(DeskError) as cm:
                backends.make_drafter("zzz")
            self.assertEqual(cm.exception.code, "INPUT_UNKNOWN_DRAFTER")
            with self.assertRaises(DeskError):
                backends.make_drafter("fake", device="cuda:0")


class EG2Tests(TmpCase):
    def mods(self, bf16=False):
        return fakes.fake_modules(torch=fakes.make_torch(bf16), sentence_transformers=fakes.make_sentence_transformers())

    def test_constructor_refusals(self):
        for kw in ({"dim": 128}, {"vision_tokens": 100}, {"dtype": "float16"}, {"dtype": "fp16"}, {"dtype": "half"}):
            with self.assertRaises(ValueError) as cm:
                backends.EG2Embedder(**kw)
            self.assertIn("REFUSED", str(cm.exception))

    def test_constructor_state(self):
        e = backends.EG2Embedder(vision_tokens=560, dtype="bfloat16")
        self.assertEqual(e.dtype_name, "bfloat16")
        self.assertEqual(e.processor_kwargs, {"image_seq_length": 560})
        self.assertEqual(e.processing_kwargs, {"image": {"max_soft_tokens": 560}})
        self.assertEqual(backends.EG2Embedder().dtype_name, "float32")

    def test_texts_use_role_prefix_pinned_revision_and_truncate(self):
        with self.mods():
            e = backends.EG2Embedder(dim=256, batch=2, vision_tokens=140)
            out = e.texts(["alpha", "beta"], role="document")
            st = fakes.FakeSentenceTransformer.instances[0]
        self.assertEqual(st.repo, backends.EG2_REPO)
        self.assertEqual(st.kw["revision"], backends.EG2_REVISION)
        self.assertEqual(st.kw["processor_kwargs"], {"image_seq_length": 140})
        self.assertEqual(st.kw["config_kwargs"], {"audio_config": None})
        self.assertEqual(st.kw["model_kwargs"], {"dtype": "float32"})
        sent, kw = st.encoded[0]
        self.assertEqual(sent, [backends.EG2_PREFIX["document"] + "alpha", backends.EG2_PREFIX["document"] + "beta"])
        self.assertEqual((kw["batch_size"], kw["truncate_dim"], kw["normalize_embeddings"]), (2, 256, True))
        self.assertEqual(out.shape, (2, 256))
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)
        # module wiring and parameter count
        self.assertEqual(st.mod0.processing_kwargs, {"image": {"max_soft_tokens": 140}})
        self.assertEqual(st.mod0.processor.image_seq_length, 140)
        self.assertEqual(st.mod0.processor.image_processor.max_soft_tokens, 140)
        self.assertEqual(e.params, 10)

    def test_model_loaded_once_and_empty_inputs_skip_loading(self):
        with self.mods():
            e = backends.EG2Embedder()
            self.assertEqual(e.texts([]).shape, (0, 768))
            self.assertEqual(e.images([]).shape, (0, 768))
            self.assertEqual(e.multimodal([]).shape, (0, 768))
            self.assertEqual(len(fakes.FakeSentenceTransformer.instances), 0)
            e.texts(["a"])
            e.texts(["b"])
            self.assertEqual(len(fakes.FakeSentenceTransformer.instances), 1)

    def test_bad_role(self):
        e = backends.EG2Embedder()
        with self.assertRaises(ValueError):
            e.texts(["x"], role="nope")
        with self.assertRaises(ValueError):
            e.multimodal([{"text": "x"}], role="nope")

    def test_dtype_selection(self):
        with self.mods(bf16=True):
            e = backends.EG2Embedder(device="cuda:0", dtype="auto")
            e.texts(["x"])
            self.assertEqual(e.dtype_name, "bfloat16")
            self.assertEqual(fakes.FakeSentenceTransformer.instances[0].kw["model_kwargs"]["dtype"], "bfloat16")
        with self.mods(bf16=False):
            e = backends.EG2Embedder(device="cuda:0", dtype="auto")
            e.texts(["x"])
            self.assertEqual(e.dtype_name, "float32")
        with self.mods():
            e = backends.EG2Embedder(dtype="bfloat16")
            e.texts(["x"])
            self.assertEqual(e.dtype_name, "bfloat16")

    def test_model_without_modules_skips_wiring(self):
        st_mod = fakes.make_sentence_transformers()

        class Empty(fakes.FakeSentenceTransformer):
            instances = []

            def __len__(self):
                return 0

        st_mod.SentenceTransformer = Empty
        with fakes.fake_modules(torch=fakes.make_torch(), sentence_transformers=st_mod):
            backends.EG2Embedder().texts(["x"])
        self.assertIsNone(Empty.instances[0].mod0.processing_kwargs)

    def test_processor_without_image_processor(self):
        class NoImgProc(fakes.FakeSentenceTransformer):
            instances = []

            def __init__(self, repo, **kw):
                super().__init__(repo, **kw)
                self.mod0.processor.image_processor = None

        st_mod = fakes.make_sentence_transformers()
        st_mod.SentenceTransformer = NoImgProc
        with fakes.fake_modules(torch=fakes.make_torch(), sentence_transformers=st_mod):
            backends.EG2Embedder(vision_tokens=70).texts(["x"])
        self.assertEqual(NoImgProc.instances[0].mod0.processor.image_seq_length, 70)

    def test_processor_absent_is_tolerated(self):
        class NoProc(fakes.FakeSentenceTransformer):
            instances = []

            def __init__(self, repo, **kw):
                super().__init__(repo, **kw)
                self.mod0 = types.SimpleNamespace(processing_kwargs=None)

        st_mod = fakes.make_sentence_transformers()
        st_mod.SentenceTransformer = NoProc
        with fakes.fake_modules(torch=fakes.make_torch(), sentence_transformers=st_mod):
            backends.EG2Embedder(vision_tokens=70).texts(["x"])
        self.assertEqual(NoProc.instances[0].mod0.processing_kwargs, {"image": {"max_soft_tokens": 70}})

    def test_images_batched_as_rgb_pil(self):
        paths = [png(self.tmp, f"{i}.png") for i in range(3)]
        with self.mods():
            e = backends.EG2Embedder(batch=2, dim=512)
            out = e.images(paths)
            st = fakes.FakeSentenceTransformer.instances[0]
        self.assertEqual(out.shape, (3, 512))
        self.assertEqual([len(s) for s, _ in st.encoded], [2, 1])
        self.assertTrue(all(isinstance(im, Image.Image) and im.mode == "RGB" for s, _ in st.encoded for im in s))

    def test_multimodal_placeholders_prefix_and_payload_shape(self):
        a, b = png(self.tmp, "a.png"), png(self.tmp, "b.png")
        items = [
            {"text": "a lamp", "images": [a]},                         # one image: bare, one placeholder prepended
            {"text": "<|image|> one <|image|> two", "images": [a, b]},  # enough placeholders already
            {"text": "x", "images": [a, b]},                           # two images, none in text: padded
            {"image": b},                                              # 'image' key, no text
            {"text": "only text"},                                     # no image: empty list
        ]
        with self.mods():
            e = backends.EG2Embedder()
            out = e.multimodal(items, role="document")
            sent = fakes.FakeSentenceTransformer.instances[0].encoded[0][0]
        pre = backends.EG2_PREFIX["document"]
        self.assertEqual(out.shape, (5, 768))
        self.assertEqual(sent[0]["text"], pre + "<|image|> a lamp")
        self.assertIsInstance(sent[0]["image"], Image.Image)
        self.assertEqual(sent[1]["text"], pre + "<|image|> one <|image|> two")
        self.assertEqual(len(sent[1]["image"]), 2)
        self.assertEqual(sent[2]["text"], pre + "<|image|> <|image|> x")
        self.assertEqual(sent[3]["text"], pre + "<|image|>")
        self.assertEqual(sent[4], {"text": pre + "only text", "image": []})

    def test_multimodal_role_none_has_no_prefix(self):
        with self.mods():
            e = backends.EG2Embedder()
            e.multimodal([{"text": "plain"}], role=None)
            sent = fakes.FakeSentenceTransformer.instances[0].encoded[0][0]
        self.assertEqual(sent[0]["text"], "plain")


class SigLIP2Tests(TmpCase):
    def mods(self, pooled_as_object=False):
        tf = fakes.make_transformers(model=fakes.FakeModel(pooled_as_object=pooled_as_object))
        return tf, fakes.fake_modules(torch=fakes.make_torch(), transformers=tf)

    def test_empty_inputs_do_not_load(self):
        e = backends.SigLIP2Embedder()
        self.assertEqual(e.images([]).shape, (0, 1152))
        self.assertEqual(e.texts([]).shape, (0, 1152))
        self.assertIsNone(e.model)

    def test_images_pinned_load_batches_and_unit_norm(self):
        paths = [png(self.tmp, f"{i}.png") for i in range(3)]
        tf, ctx = self.mods()
        with ctx:
            e = backends.SigLIP2Embedder(batch=2, device="cpu")
            out = e.images(paths)
            e.images(paths[:1])
        kinds = [c[0] for c in tf.calls]
        self.assertEqual(kinds, ["processor", "model"])  # loaded once
        self.assertEqual(tf.calls[0][1:], (backends.SIGLIP2_REPO, {"revision": backends.SIGLIP2_REVISION}))
        self.assertEqual(tf.calls[1][2]["revision"], backends.SIGLIP2_REVISION)
        self.assertEqual(tf.calls[1][2]["dtype"], "float32")
        self.assertEqual(out.shape, (3, 4))
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)
        self.assertEqual([len(c["images"]) for c in tf.processor.calls[:2]], [2, 1])
        self.assertEqual(e.params, 15)
        self.assertEqual(tf.model.evals, 1)
        self.assertEqual(tf.model.devices, ["cpu"])

    def test_pooler_output_object_branch(self):
        tf, ctx = self.mods(pooled_as_object=True)
        with ctx:
            out = backends.SigLIP2Embedder().images([png(self.tmp, "a.png")])
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)

    def test_texts_pad_to_64_and_chunk_by_16(self):
        tf, ctx = self.mods()
        with ctx:
            out = backends.SigLIP2Embedder().texts([f"t{i}" for i in range(20)])
        self.assertEqual(out.shape, (20, 4))
        calls = tf.processor.calls
        self.assertEqual([len(c["text"]) for c in calls], [16, 4])
        for c in calls:
            self.assertEqual((c["padding"], c["max_length"], c["truncation"]), ("max_length", 64, True))


class TransformersWarningsOff(TmpCase):
    def _logging(self):
        log = types.ModuleType("transformers.utils.logging")
        log.level, log.history = 30, []
        log.get_verbosity = lambda: log.level

        def set_verbosity(v):
            log.level = v
            log.history.append(v)

        log.set_verbosity = set_verbosity
        log.set_verbosity_error = lambda: set_verbosity(40)
        utils = types.ModuleType("transformers.utils")
        utils.logging = log
        return log, utils

    def test_holds_error_level_during_the_block_and_restores_it(self):
        log, utils = self._logging()
        tf = fakes.make_transformers()
        tf.utils = utils
        with fakes.fake_modules(transformers=tf, **{"transformers.utils": utils,
                                                    "transformers.utils.logging": log}):
            with backends.transformers_warnings_off():
                self.assertEqual(log.level, 40)
            self.assertEqual(log.level, 30)
            with self.assertRaises(RuntimeError):
                with backends.transformers_warnings_off():
                    raise RuntimeError("load failed")
            self.assertEqual(log.level, 30)  # restored even when the load raises

    def test_no_transformers_logging_is_a_no_op(self):
        with mock.patch.dict(sys.modules, {"transformers.utils": None}):
            with backends.transformers_warnings_off():
                pass

    def test_siglip2_loads_inside_it(self):
        log, utils = self._logging()
        levels = []
        tf = fakes.make_transformers()
        tf.utils = utils
        real = tf.AutoModel.from_pretrained

        def watched(repo, **kw):
            levels.append(log.level)
            return real(repo, **kw)

        tf.AutoModel.from_pretrained = staticmethod(watched)
        with fakes.fake_modules(torch=fakes.make_torch(), transformers=tf,
                                **{"transformers.utils": utils, "transformers.utils.logging": log}):
            backends.SigLIP2Embedder().images([png(self.tmp, "a.png")])
        self.assertEqual(levels, [40])
        self.assertEqual(log.level, 30)


class PlainSightNativePath(TmpCase):
    """The transformers-only fallback (plain_sight not installed) with local_files_only loading."""

    def setUp(self):
        super().setUp()
        self.img = png(self.tmp, "a.png", size=(20, 10))

    def mods(self, tf):
        return fakes.fake_modules(torch=fakes.make_torch(), transformers=tf, plain_sight=None,
                                  **{"plain_sight.engine": None})

    def test_draft_runs_generate_and_parses(self):
        model = fakes.FakeModel(config=types.SimpleNamespace(_name_or_path="repo/x", _commit_hash="abc123"))
        tf = fakes.make_transformers(model=model)
        with mock.patch.dict(os.environ), self.mods(tf):
            os.environ.pop("HF_HUB_OFFLINE", None)
            d = backends.PlainSightDrafter(device="cpu", detail="medium")
            out = d.draft([self.img, self.img])
            self.assertNotIn("HF_HUB_OFFLINE", os.environ)  # restored
        self.assertEqual(out, ["a lantern on a wall"] * 2)
        self.assertEqual((d.drafter_id, d.revision), ("repo/x", "abc123"))
        self.assertEqual(tf.calls[0][2], {"revision": backends.PLAIN_SIGHT_REVISION, "local_files_only": True})
        self.assertEqual(tf.calls[1][0], "model")
        self.assertEqual(len([c for c in tf.calls if c[0] == "model"]), 1)  # loaded once
        self.assertEqual(tf.processor.post, ("<DETAILED_CAPTION>raw text", "<DETAILED_CAPTION>", (20, 10)))
        gk = model.generate_kwargs[0]
        self.assertEqual((gk["max_new_tokens"], gk["num_beams"], gk["do_sample"]), (1024, 3, False))
        self.assertEqual(tf.processor.calls[0]["text"], "<DETAILED_CAPTION>")

    def test_config_without_provenance_keeps_requested_pin(self):
        tf = fakes.make_transformers(model=fakes.FakeModel(config=None))
        with self.mods(tf):
            d = backends.PlainSightDrafter()
            d.draft([self.img])
        self.assertEqual((d.drafter_id, d.revision), (backends.PLAIN_SIGHT_REPO, backends.PLAIN_SIGHT_REVISION))

    def test_falls_back_to_florence2_class_when_auto_model_refuses(self):
        tf = fakes.make_transformers()

        def refuse(repo, **kw):
            raise ValueError("unrecognised config")

        tf.AutoModelForImageTextToText.from_pretrained = staticmethod(refuse)
        with mock.patch.dict(os.environ, {"HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0"}), self.mods(tf):
            d = backends.PlainSightDrafter()
            d.draft([self.img])
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "0")  # previous values restored
            self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "0")
        self.assertIs(d.model, tf.model)

    def test_missing_snapshot_error_names_repo_revision_and_hint(self):
        tf = fakes.make_transformers()

        def boom(repo, **kw):
            raise OSError("not cached")

        tf.AutoProcessor.from_pretrained = staticmethod(boom)
        with self.mods(tf):
            with self.assertRaises(RuntimeError) as cm:
                backends.PlainSightDrafter(model_id="r/m", revision="deadbeef").draft([self.img])
        msg = str(cm.exception)
        self.assertIn("'r/m'", msg)
        self.assertIn("deadbeef", msg)
        self.assertIn("install plain-sight", msg)
        self.assertIsInstance(cm.exception.__cause__, OSError)

    def test_draft_validates_and_short_circuits(self):
        d = backends.PlainSightDrafter()
        self.assertEqual(d.draft([]), [])
        with self.assertRaises(TypeError):
            d.draft("x.png")
        with self.assertRaises(TypeError):
            d.draft([3])
        self.assertIsNone(d.model)

    def test_ensure_model_is_idempotent(self):
        d = backends.PlainSightDrafter()
        sentinel = object()
        d.model = sentinel
        d._ensure_model()  # returns immediately, imports nothing
        self.assertIs(d.model, sentinel)
        d.model, d.engine = None, sentinel
        d._ensure_model()
        self.assertIs(d.engine, sentinel)


class PlainSightEnginePath(TmpCase):
    def engine_mods(self, engine_cls):
        pkg = types.ModuleType("plain_sight")
        eng = types.ModuleType("plain_sight.engine")
        eng.Florence2Engine = engine_cls
        return fakes.fake_modules(torch=fakes.make_torch(), plain_sight=pkg, **{"plain_sight.engine": eng})

    def test_engine_without_provenance_and_lazy_load(self):
        calls = {}

        class Eng:
            loaded = False

            def __init__(self, **kw):
                calls["init"] = kw
                self.model_id = "loaded/id"
                self.revision = "rev9"

            def _ensure_loaded(self):
                calls["ensure"] = True

            def describe(self, p, detail="high"):
                return f"  desc {os.path.basename(p)} {detail} "

        with self.engine_mods(Eng):
            d = backends.PlainSightDrafter(device="cpu", detail="high")
            out = d.draft(["/x/a.png"])
        self.assertEqual(out, ["desc a.png high"])
        self.assertTrue(calls["ensure"])
        self.assertEqual(calls["init"]["dtype"], "float32")
        self.assertTrue(calls["init"]["eager_load"])
        self.assertEqual((d.drafter_id, d.revision), ("loaded/id", "rev9"))

    def test_engine_resolved_revision_method_and_gpu_dtype(self):
        seen = {}

        class Eng:
            loaded = True

            def __init__(self, **kw):
                seen.update(kw)

            def resolved_revision(self):
                return "resolved-1"

            def output_provenance(self):
                return {}

            def describe(self, p, detail="high"):
                return "x"

        with self.engine_mods(Eng):
            d = backends.PlainSightDrafter(device="cuda:0")
            d.draft(["a.png"])
        self.assertIsNone(seen["dtype"])
        self.assertEqual(d.revision, "resolved-1")
        self.assertEqual(d.drafter_id, backends.PLAIN_SIGHT_REPO)

    def test_engine_without_optional_methods_falls_back_to_pin(self):
        class Eng:
            loaded = True

            def __init__(self, **kw):
                pass

            def describe(self, p, detail="high"):
                return "x"

        with self.engine_mods(Eng):
            d = backends.PlainSightDrafter()
            d.draft(["a.png"])
        self.assertEqual((d.drafter_id, d.revision), (backends.PLAIN_SIGHT_REPO, backends.PLAIN_SIGHT_REVISION))

    def test_engine_load_failure_is_wrapped_with_offline_context(self):
        class Eng:
            def __init__(self, **kw):
                raise OSError("no snapshot")

        with self.engine_mods(Eng):
            with self.assertRaises(RuntimeError) as cm:
                backends.PlainSightDrafter().draft(["a.png"])
        self.assertIn("offline load failed", str(cm.exception))
        self.assertIn("no downloads permitted", str(cm.exception))
        self.assertIn("no snapshot", str(cm.exception))

    def test_engine_module_without_engine_class_uses_native_path(self):
        # plain_sight imports fine, but Florence2Engine cannot be imported -> ImportError -> native path.
        pkg = types.ModuleType("plain_sight")
        eng = types.ModuleType("plain_sight.engine")  # no Florence2Engine attribute: ImportError on from-import
        tf = fakes.make_transformers()

        def boom(repo, **kw):
            raise OSError("gone")

        tf.AutoProcessor.from_pretrained = staticmethod(boom)
        with fakes.fake_modules(torch=fakes.make_torch(), transformers=tf, plain_sight=pkg,
                                **{"plain_sight.engine": eng}):
            with self.assertRaises(RuntimeError) as cm:
                backends.PlainSightDrafter().draft(["a.png"])
        # plain_sight_installed only turns true when the engine class imported, so this still gets the install hint
        self.assertIn("install plain-sight (https://github.com/mcp-tool-shop-org/plain-sight)", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
