"""The CLI boundary: DeskError rendering, desk.run exit codes, main() wiring, and python -m dataset_desk."""

import contextlib
import io
import json
import os
import runpy
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from dataset_desk import backends, desk
from dataset_desk.errors import CODES, EXIT_OK, EXIT_RUNTIME, EXIT_USER, DeskError

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_project(root):
    """A two-image project with one record, enough for index/search/dupes through the fake backend."""
    proj = os.path.join(root, "proj")
    os.makedirs(os.path.join(proj, "outputs", "approved"))
    os.makedirs(os.path.join(proj, "records"))
    for name, colour in (("red1", (180, 10, 10)), ("blue1", (10, 10, 180))):
        rng = np.random.default_rng(len(name) + colour[0])
        a = np.clip(rng.integers(0, 40, (16, 16, 3)) + np.array(colour), 0, 255).astype(np.uint8)
        Image.fromarray(a).save(os.path.join(proj, "outputs", "approved", f"{name}.png"))
        rec = {"id": name, "asset_path": f"outputs/approved/{name}.png",
               "judgment": {"status": "approved", "explanation": f"a {name[:-1]} creature"},
               "provenance": {"prompt": f"prompt {name}"}}
        with open(os.path.join(proj, "records", f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump(rec, f)
    return proj


class DeskErrorTests(unittest.TestCase):
    def test_registry_codes_have_hints(self):
        self.assertTrue(CODES)
        self.assertTrue(all(h for h in CODES.values()))
        self.assertEqual((EXIT_OK, EXIT_USER, EXIT_RUNTIME), (0, 1, 2))

    def test_default_hint_comes_from_registry_and_str_is_message(self):
        e = DeskError("STATE_NO_INDEX", "no cache here")
        self.assertEqual(str(e), "no cache here")
        self.assertEqual(e.hint, CODES["STATE_NO_INDEX"])
        self.assertEqual((e.exit_code, e.retryable, e.cause), (EXIT_USER, False, None))
        self.assertIsInstance(e, SystemExit)

    def test_explicit_hint_and_unknown_code(self):
        self.assertEqual(DeskError("STATE_NO_INDEX", "m", "do this").hint, "do this")
        self.assertEqual(DeskError("NOT_A_CODE", "m").hint, "")

    def test_to_dict_without_and_with_cause(self):
        plain = DeskError("INPUT_BAD_ARGUMENT", "bad").to_dict()
        self.assertEqual(plain, {"code": "INPUT_BAD_ARGUMENT", "message": "bad",
                                 "hint": CODES["INPUT_BAD_ARGUMENT"], "retryable": False})
        withc = DeskError("RUNTIME_UNEXPECTED", "boom", cause=KeyError("k"), retryable=True).to_dict()
        self.assertEqual(withc["cause"], "KeyError: 'k'")
        self.assertTrue(withc["retryable"])

    def test_render_lines(self):
        self.assertEqual(DeskError("STATE_NO_INDEX", "no cache").render().splitlines(),
                         ["error STATE_NO_INDEX: no cache", f"hint: {CODES['STATE_NO_INDEX']}"])
        self.assertEqual(DeskError("NOT_A_CODE", "just this").render(), "error NOT_A_CODE: just this")
        lines = DeskError("RUNTIME_UNEXPECTED", "x", cause=ValueError("why")).render().splitlines()
        self.assertEqual(lines[-1], "cause: ValueError: why")


class RunBoundary(unittest.TestCase):
    def call(self, side_effect=None, argv=None, return_value=0):
        err = io.StringIO()
        with mock.patch.object(desk, "main", side_effect=side_effect, return_value=return_value) as m, \
                contextlib.redirect_stderr(err):
            rc = desk.run(argv if argv is not None else ["index"])
        return rc, err.getvalue(), m

    def test_success_passes_main_result_and_argv_through(self):
        rc, err, m = self.call(argv=["dupes", "--project", "p"], return_value=0)
        self.assertEqual((rc, err), (0, ""))
        m.assert_called_once_with(["dupes", "--project", "p"])

    def test_defaults_to_sys_argv(self):
        with mock.patch.object(sys, "argv", ["prog", "search", "--text", "x"]),                 mock.patch.object(desk, "main", return_value=0) as mm:
            self.assertEqual(desk.run(), 0)
        mm.assert_called_once_with(["search", "--text", "x"])

    def test_desk_error_prints_code_hint_and_returns_its_exit_code(self):
        rc, err, _ = self.call(DeskError("STATE_NO_INDEX", "no cache"))
        self.assertEqual(rc, EXIT_USER)
        self.assertIn("error STATE_NO_INDEX: no cache", err)
        self.assertIn("hint: run `dataset-desk index`", err)
        rc, err, _ = self.call(DeskError("RUNTIME_UNEXPECTED", "x", exit_code=7))
        self.assertEqual(rc, 7)

    def test_unexpected_exception_is_structured_with_no_traceback(self):
        rc, err, _ = self.call(RuntimeError("kaboom"))
        self.assertEqual(rc, EXIT_RUNTIME)
        self.assertIn("error RUNTIME_UNEXPECTED: the desk stopped on an unexpected error", err)
        self.assertIn("cause: RuntimeError: kaboom", err)
        self.assertIn("rerun with --debug", err)
        self.assertNotIn("Traceback", err)

    def test_debug_reraises_unexpected_exception(self):
        err = io.StringIO()
        with mock.patch.object(desk, "main", side_effect=RuntimeError("kaboom")), contextlib.redirect_stderr(err):
            with self.assertRaises(RuntimeError):
                desk.run(["index", "--debug"])
        self.assertEqual(err.getvalue(), "")

    def test_debug_does_not_swallow_desk_errors(self):
        rc, err, _ = self.call(DeskError("STATE_NO_INDEX", "no cache"), argv=["index", "--debug"])
        self.assertEqual(rc, EXIT_USER)
        self.assertIn("error STATE_NO_INDEX", err)

    def test_keyboard_interrupt_returns_130(self):
        rc, err, _ = self.call(KeyboardInterrupt())
        self.assertEqual(rc, 130)
        self.assertIn("RUNTIME_INTERRUPTED", err)
        self.assertNotIn("Traceback", err)


class MainWiring(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name
        self.proj = make_project(self.tmp)
        self.out = os.path.join(self.tmp, "out")
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)

    def run_cli(self, *args, **kw):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = desk.run([args[0], "--project", self.proj, "--out", self.out, *args[1:]], **kw) \
                if not kw else desk.main([args[0], "--project", self.proj, "--out", self.out, *args[1:]], **kw)
        return rc, out.getvalue(), err.getvalue()

    def test_run_end_to_end_index_then_search_and_dupes(self):
        self.assertEqual(self.run_cli("index", "--backend", "fake")[0], 0)
        rc, out, _ = self.run_cli("search", "--backend", "fake", "--text", "red", "-k", "1")
        self.assertEqual(rc, 0)
        self.assertIn("red1.png", out)
        rc, out, _ = self.run_cli("search", "--backend", "fake", "--image",
                                  os.path.join(self.proj, "outputs", "approved", "blue1.png"), "-k", "1")
        self.assertIn("blue1.png", out)
        rc, out, _ = self.run_cli("dupes", "--backend", "fake")
        self.assertEqual(sorted(json.loads(out)), ["exact", "near", "semantic", "thresholds", "top1_percentiles"])

    def test_missing_index_is_a_structured_user_error(self):
        rc, _, err = self.run_cli("dupes", "--backend", "fake")
        self.assertEqual(rc, EXIT_USER)
        self.assertIn("error STATE_NO_INDEX", err)
        self.assertIn("hint:", err)

    def test_out_inside_project_refused(self):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = desk.run(["index", "--project", self.proj, "--backend", "fake",
                           "--out", os.path.join(self.proj, "inside")])
        self.assertEqual(rc, EXIT_USER)
        self.assertIn("error PERM_OUT_INSIDE_PROJECT", err.getvalue())
        self.assertFalse(os.path.exists(os.path.join(self.proj, "inside")))

    def test_out_on_another_drive_is_not_inside_the_project(self):
        # On Windows, commonpath raises ValueError for paths on different drives; that means "not inside".
        with mock.patch("os.path.commonpath", side_effect=ValueError("Paths don't have the same drive")):
            rc, _, err = self.run_cli("index", "--backend", "fake")
        self.assertEqual(rc, EXIT_OK, err)
        self.assertTrue(os.path.isdir(self.out))

    def test_gpu_without_grant_refused_with_hint(self):
        rc, _, err = self.run_cli("index", "--backend", "fake", "--device", "cuda")
        self.assertEqual(rc, EXIT_USER)
        self.assertIn("error PERM_GPU_NEEDS_GRANT", err)
        self.assertIn("hint: pass --grant", err)

    def test_none_backend_refuses_search_and_coverage(self):
        for cmd in ("search", "coverage"):
            rc, _, err = self.run_cli(cmd, "--backend", "none", "--text", "x")
            self.assertEqual(rc, EXIT_USER)
            self.assertIn("error INPUT_BAD_ARGUMENT", err)
            self.assertIn("requires an embedding backend", err)

    def test_search_argument_errors(self):
        self.run_cli("index", "--backend", "fake")
        rc, _, err = self.run_cli("search", "--backend", "fake")
        self.assertEqual(rc, EXIT_USER)
        self.assertIn("--text or --image", err)

    def test_joint_search_without_multimodal_raises_desk_error(self):
        self.run_cli("index", "--backend", "fake")
        plain = types.SimpleNamespace()
        img = os.path.join(self.proj, "outputs", "approved", "red1.png")
        with self.assertRaises(DeskError) as cm:
            desk.main(["search", "--project", self.proj, "--out", self.out, "--backend", "fake",
                       "--text", "red", "--image", img], make=lambda *a, **k: plain)
        self.assertEqual(cm.exception.code, "INPUT_BAD_ARGUMENT")
        self.assertIn("does not support joint", cm.exception.message)

    def test_coverage_needs_subjects(self):
        self.run_cli("index", "--backend", "fake")
        rc, _, err = self.run_cli("coverage", "--backend", "fake")
        self.assertEqual(rc, EXIT_USER)
        self.assertIn("coverage needs --subjects", err)

    def test_report_with_subjects_none_backend_skips_coverage(self):
        subj = os.path.join(self.tmp, "s.txt")
        with open(subj, "w", encoding="utf-8") as f:
            f.write("a red creature\n")
        self.run_cli("index", "--backend", "none")
        make = mock.Mock(side_effect=lambda *a, **k: backends.NoneEmbedder())
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = desk.main(["report", "--project", self.proj, "--out", self.out, "--backend", "none",
                            "--subjects", subj], make=make)
        self.assertEqual(rc, 0)
        make.assert_called_once()  # report with subjects builds the embedder
        self.assertTrue(os.path.isfile(out.getvalue().strip()))

    def test_real_backend_lowers_priority_before_make_but_fake_does_not(self):
        made = []

        def make(backend, **kw):
            made.append((backend, kw))
            return backends.FakeEmbedder()

        with mock.patch.object(desk, "lower_priority") as lp, contextlib.redirect_stderr(io.StringIO()),                 contextlib.redirect_stdout(io.StringIO()):
            desk.main(["index", "--project", self.proj, "--out", self.out, "--backend", "fake", "--threads", "2"],
                      make=make)
            lp.assert_not_called()
            desk.main(["index", "--project", self.proj, "--out", self.out, "--backend", "eg2", "--threads", "2",
                       "--images-only"], make=make)
            lp.assert_called_once_with()
        self.assertEqual([m[0] for m in made], ["fake", "eg2"])
        self.assertEqual(made[1][1], {"device": "cpu", "threads": 2, "grant": None})
        self.assertEqual(os.environ["OMP_NUM_THREADS"], "2")

    def test_default_out_is_desk_out_under_cwd(self):
        cwd = os.getcwd()
        os.chdir(self.tmp)
        self.addCleanup(os.chdir, cwd)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(desk.main(["index", "--project", self.proj, "--backend", "fake"]), 0)
        self.assertTrue(os.path.isdir(os.path.join(self.tmp, "desk-out", "proj")))


class LowerPriority(unittest.TestCase):
    def test_windows_sets_below_normal_class(self):
        k32 = mock.Mock()
        k32.GetCurrentProcess.return_value = 99
        fake_ctypes_windll = types.SimpleNamespace(kernel32=k32)
        with mock.patch.object(desk.os, "name", "nt"), mock.patch("ctypes.windll", fake_ctypes_windll, create=True):
            desk.lower_priority()
        k32.SetPriorityClass.assert_called_once_with(99, 0x00004000)

    def test_posix_nices(self):
        with mock.patch.object(desk.os, "name", "posix"), mock.patch.object(desk.os, "nice", create=True) as nice:
            desk.lower_priority()
        nice.assert_called_once_with(10)


class ModuleEntryPoint(unittest.TestCase):
    def test_runpy_exit_code_comes_from_run(self):
        err = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            proj = make_project(tmp)
            argv = ["dataset_desk", "dupes", "--project", proj, "--backend", "fake", "--out", os.path.join(tmp, "out")]
            with mock.patch.object(sys, "argv", argv), contextlib.redirect_stderr(err),                     self.assertRaises(SystemExit) as cm:
                runpy.run_module("dataset_desk", run_name="__main__")
        self.assertEqual(cm.exception.code, EXIT_USER)
        self.assertIn("error STATE_NO_INDEX", err.getvalue())

    def test_subprocess_unknown_flag_is_argparse_exit_2_and_user_error_is_1(self):
        env = dict(os.environ, PYTHONPATH=ROOT)
        bad = subprocess.run([sys.executable, "-m", "dataset_desk", "index"], cwd=ROOT, env=env,
                             capture_output=True, text=True, timeout=60)
        self.assertEqual(bad.returncode, 2)  # argparse: --project required
        self.assertIn("--project", bad.stderr)
        with tempfile.TemporaryDirectory() as tmp:
            r = subprocess.run([sys.executable, "-m", "dataset_desk", "dupes", "--project", tmp, "--backend",
                                "fake", "--out", os.path.join(tmp, "..", "dd-out-" + os.path.basename(tmp))],
                               cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, EXIT_USER)
        self.assertIn("error STATE_NO_INDEX", r.stderr)
        self.assertNotIn("Traceback", r.stderr)

    def test_subprocess_full_index_run_succeeds(self):
        env = dict(os.environ, PYTHONPATH=ROOT)
        with tempfile.TemporaryDirectory() as tmp:
            proj = make_project(tmp)
            r = subprocess.run([sys.executable, "-m", "dataset_desk", "index", "--project", proj, "--backend",
                                "fake", "--out", os.path.join(tmp, "out")], cwd=ROOT, env=env,
                               capture_output=True, text=True, timeout=60)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertTrue(os.path.isdir(os.path.join(tmp, "out")))


if __name__ == "__main__":
    unittest.main()
