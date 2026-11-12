"""Hermetic CPU-only tests for the dataset-desk Colab caption-drafting lane (`scripts/colab/`).

Every test runs against `fake_colab_desk.FakeColabEnv` (`#SDLAB_FAKE_COLAB_CLI=1`,
`SDLAB_TEST_FAKE_COLAB=1`) with `HOME`, `USERPROFILE`, and `XDG_*` pointed at empty temporary
directories so real `colab` CLI state or live Colab sessions are never touched.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import pathlib
import shutil
import tarfile
import tempfile
import unittest
from unittest import mock

import fakes
import fake_colab_desk
from dataset_desk import backends, registry
from dataset_desk.errors import CODES, DeskError

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
COLAB_DIR = REPO_ROOT / "scripts" / "colab"
LAUNCH_SH = COLAB_DIR / "launch_desk_draft.sh"
COLAB_WINDOW_SH = COLAB_DIR / "colab_window.sh"
LAUNCH_GUARDS_PY = COLAB_DIR / "launch_guards.py"
REMOTE_RUNNER_PY = COLAB_DIR / "remote_runner.py"
DESK_BUNDLE_NOW_PY = COLAB_DIR / "desk_bundle_now.py"
DESK_DRAFT_HELPER_PY = COLAB_DIR / "desk_draft_helper.py"
DESK_LOCK_FILE = COLAB_DIR / "locks" / "desk_draft.lock"

EXPECTED_SDLAB_COMMIT = "df9a1b3be2f8729a57776253e9b0cb32b7f5299c"
EXPECTED_COLAB_WINDOW_SHA = "db761fb414da915c8835bad8ef1544992e39aa5b3d636a1d81d17d231f26e974"
EXPECTED_LAUNCH_GUARDS_SHA = "4c0c0ab6410eaa511a808d00f09a09d42d5d5b7d0504d85d58cfe3c65baeb085"
EXPECTED_REMOTE_RUNNER_SHA = "393929ed704ccbacaee2b590010e348a5d635cbdd70361209d3e0103d1fba6eb"


def _load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ColabTempTestCase(fakes.IsolatedDeskTestCase):
    """Base test case providing a cleaned temporary directory per test."""

    def setUp(self) -> None:
        super().setUp()
        self.tmp_path = pathlib.Path(tempfile.mkdtemp(prefix="desk_colab_test_"))
        self.addCleanup(shutil.rmtree, self.tmp_path, True)

    def make_env(self, name: str = "env", config: dict | None = None) -> fake_colab_desk.FakeColabEnv:
        return fake_colab_desk.FakeColabEnv(self.tmp_path / name, config=config)

    def make_project(
        self,
        name: str = "src",
        *,
        pool_tag: str | None = "rustline",
        flagged_count: int = 2,
        unflagged_count: int = 2,
        extra_pad_bytes: int = 0,
    ) -> tuple[pathlib.Path, pathlib.Path]:
        return fake_colab_desk.make_synthetic_project(
            self.tmp_path / name,
            pool_tag=pool_tag,
            flagged_count=flagged_count,
            unflagged_count=unflagged_count,
            extra_pad_bytes=extra_pad_bytes,
        )


class VendoredKitAndBackendGuardsTests(ColabTempTestCase):
    """Verify vendored sdlab hashes, self-pin, embedder prohibition, and draft heartbeat hooks."""

    def test_vendored_window_library_hashes_commit_and_self_pin(self) -> None:
        self.assertEqual(_sha256_file(COLAB_WINDOW_SH), EXPECTED_COLAB_WINDOW_SHA)
        self.assertEqual(_sha256_file(LAUNCH_GUARDS_PY), EXPECTED_LAUNCH_GUARDS_SHA)
        self.assertEqual(_sha256_file(REMOTE_RUNNER_PY), EXPECTED_REMOTE_RUNNER_SHA)

        guards = _load_module("launch_guards_test", LAUNCH_GUARDS_PY)
        embedded_pin, norm_sha = guards.sha256_self_normalized(LAUNCH_SH)
        self.assertIsNotNone(embedded_pin)
        self.assertEqual(embedded_pin, norm_sha)

        launch_txt = LAUNCH_SH.read_text(encoding="utf-8")
        self.assertIn(f'SDLAB_VENDORED_COMMIT="{EXPECTED_SDLAB_COMMIT}"', launch_txt)
        helper_mod = _load_module("desk_draft_helper_test", DESK_DRAFT_HELPER_PY)
        self.assertEqual(helper_mod.SDLAB_VENDORED_COMMIT, EXPECTED_SDLAB_COMMIT)

        verified = guards.verify_input_shas(
            {
                str(COLAB_WINDOW_SH): EXPECTED_COLAB_WINDOW_SHA,
                str(LAUNCH_GUARDS_PY): EXPECTED_LAUNCH_GUARDS_SHA,
                str(REMOTE_RUNNER_PY): EXPECTED_REMOTE_RUNNER_SHA,
                str(DESK_BUNDLE_NOW_PY): _sha256_file(DESK_BUNDLE_NOW_PY),
                str(DESK_DRAFT_HELPER_PY): _sha256_file(DESK_DRAFT_HELPER_PY),
                str(DESK_LOCK_FILE): _sha256_file(DESK_LOCK_FILE),
                str(LAUNCH_SH): embedded_pin,
            }
        )
        self.assertEqual(len(verified), 7)

        # Fake CLI marker must always be present on our test shim and HOME isolated.
        env = self.make_env()
        colab_shim = (env.rig_bin / "colab").read_text(encoding="utf-8")
        self.assertIn("#SDLAB_FAKE_COLAB_CLI=1", colab_shim)
        built_env = env.build_env()
        self.assertEqual(built_env["HOME"], fake_colab_desk.to_posix(env.home_dir))
        self.assertEqual(list(env.home_dir.iterdir()), [])

    def test_forbid_embedders_env_blocks_every_embedding_backend_and_allows_none(self) -> None:
        self.assertIn("EMBEDDING_BACKEND_FORBIDDEN", CODES)
        log_file = self.tmp_path / "logs" / "embedder_events.jsonl"

        with mock.patch.dict(
            os.environ,
            {
                backends.FORBID_EMBEDDERS_ENV: "1",
                backends.EMBEDDER_LOG_ENV: str(log_file),
            },
        ):
            none_emb = backends.make("none")
            self.assertEqual(none_emb.name, "none")

            all_non_none = [*registry.REGISTRY.keys(), "fake"]
            for bname in all_non_none:
                with self.assertRaises(DeskError) as ctx:
                    backends.make(bname)
                self.assertEqual(ctx.exception.code, "EMBEDDING_BACKEND_FORBIDDEN")

            # Direct class construction is also refused before any model library can be touched.
            for direct_call in (
                lambda: backends.FakeEmbedder(),
                lambda: backends.EG2Embedder(),
                lambda: backends.SigLIP2Embedder(),
                lambda: backends.SentenceTransformerTextEmbedder(registry.REGISTRY["bge-m3"]),
                lambda: backends.DINOv2Embedder(),
            ):
                with self.assertRaises(DeskError) as ctx:
                    direct_call()
                self.assertEqual(ctx.exception.code, "EMBEDDING_BACKEND_FORBIDDEN")

        events = [json.loads(ln) for ln in log_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
        self.assertEqual(events[0]["backend"], "none")
        self.assertTrue(events[0]["allowed"])
        self.assertTrue(all(not ev["allowed"] for ev in events[1:]))

        # Without FORBID_EMBEDDERS_ENV, FakeEmbedder is allowed and recorded as allowed=True.
        with mock.patch.dict(
            os.environ,
            {
                backends.FORBID_EMBEDDERS_ENV: "0",
                backends.EMBEDDER_LOG_ENV: str(log_file),
            },
        ):
            fe = backends.FakeEmbedder()
            self.assertEqual(fe.name, "fake")

    def test_drafter_heartbeat_and_stall_hook_in_process(self) -> None:
        from PIL import Image

        img1 = self.tmp_path / "p1.png"
        img2 = self.tmp_path / "p2.png"
        Image.new("RGB", (24, 24), (200, 30, 30)).save(img1)
        Image.new("RGB", (24, 24), (30, 200, 60)).save(img2)
        hb_file = self.tmp_path / "hb" / "draft_hb.json"

        with mock.patch.dict(
            os.environ,
            {
                backends.DRAFT_HEARTBEAT_ENV: str(hb_file),
                "DATASET_DESK_TEST_STALL_AT_RECORD": "1",
                "DATASET_DESK_TEST_STALL_SECONDS": "0.01",
            },
        ):
            drafter = backends.FakeDrafter()
            out = drafter.draft([str(img1), str(img2)])
            self.assertEqual(len(out), 2)

            # Exercise PermissionError retry path in _record_draft_heartbeat on Windows/POSIX.
            real_replace = os.replace
            calls = {"n": 0}

            def flaky_replace(src, dst):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise PermissionError("simulated transient reader lock")
                return real_replace(src, dst)

            with mock.patch("os.replace", side_effect=flaky_replace):
                backends._record_draft_heartbeat("fake", phase="done", drafted=2, total=2, last_path=str(img2))
            self.assertGreaterEqual(calls["n"], 2)

        hb_doc = json.loads(hb_file.read_text(encoding="utf-8"))
        self.assertEqual(hb_doc["phase"], "done")
        self.assertEqual(hb_doc["drafted"], 2)
        self.assertEqual(hb_doc["total"], 2)
        hb_lines = [
            json.loads(ln)
            for ln in pathlib.Path(f"{hb_file}.jsonl").read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        phases = [row["phase"] for row in hb_lines]
        self.assertIn("loading_model", phases)
        self.assertIn("drafting", phases)
        self.assertEqual(phases[-1], "done")


class RigPreflightAndDryRunTests(ColabTempTestCase):
    """Test --dry, deterministic tar packing, and every rig-side refusal before any `colab` call."""

    def _base_gpu_args(self, proj: pathlib.Path, cap_cfg: pathlib.Path, out_dir: pathlib.Path) -> list[str]:
        return [
            "--project",
            str(proj),
            "--caption-config",
            str(cap_cfg),
            "--out",
            str(out_dir),
            "--card",
            "L4",
            "--cap",
            "5.0",
            "--wall-s",
            "600",
            "--margin",
            "0.5",
        ]

    def _assert_rig_refused_before_colab(
        self,
        args: list[str],
        expected_code: str,
        *,
        extra_env: dict[str, str | None] | None = None,
        allow_any_out_drive: bool = True,
        test_mode: bool = True,
    ) -> None:
        env = self.make_env(f"rig_{expected_code.lower()}_{abs(hash(tuple(args))) % 10000}")
        res = env.run_launch(
            args,
            extra_env=extra_env,
            allow_any_out_drive=allow_any_out_drive,
            test_mode=test_mode,
        )
        self.assertNotEqual(res.returncode, 0, msg=f"expected {expected_code} refusal, got rc=0")
        combined = f"{res.stderr}\n{res.stdout}"
        self.assertIn(expected_code, combined, msg=f"expected {expected_code} in {combined!r}")
        self.assertEqual(env.colab_calls(), [], msg=f"rig refusal {expected_code} must never invoke colab CLI")

    def test_dry_run_gpu_and_stub_build_deterministic_tar_and_never_call_colab(self) -> None:
        proj, cap_cfg = self.make_project("dry_src", flagged_count=2, unflagged_count=1)
        env1 = self.make_env("dry1")
        out1 = self.tmp_path / "out_dry1"
        res1 = env1.run_launch([*self._base_gpu_args(proj, cap_cfg, out1), "--dry"])
        self.assertEqual(res1.returncode, 0, msg=res1.stderr)
        self.assertEqual(env1.colab_calls(), [])

        env2 = self.make_env("dry2")
        out2 = self.tmp_path / "out_dry2"
        res2 = env2.run_launch([*self._base_gpu_args(proj, cap_cfg, out2), "--dry"])
        self.assertEqual(res2.returncode, 0, msg=res2.stderr)
        self.assertEqual(env2.colab_calls(), [])

        m1 = json.loads((out1 / "project_manifest.json").read_text(encoding="utf-8"))
        m2 = json.loads((out2 / "project_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(m1["archive_sha256"], m2["archive_sha256"])
        self.assertEqual(m1["archive_bytes"], m2["archive_bytes"])
        self.assertEqual(m1["files"], m2["files"])
        self.assertEqual(m1["delivery_method"], "colab_upload_checked")
        self.assertEqual(m1["multimodal_pool"], "rustline")
        self.assertTrue(m1["multimodal_pool_locked"])
        for fentry in m1["files"]:
            self.assertIn("path", fentry)
            self.assertIn("bytes", fentry)
            self.assertIn("sha256", fentry)

        plan = json.loads((out1 / "dry_run_plan.json").read_text(encoding="utf-8"))
        self.assertTrue(plan["dry_run"])
        self.assertEqual(plan["embedding_backend"], "forbidden")
        self.assertEqual(plan["sdlab_vendored_commit"], EXPECTED_SDLAB_COMMIT)
        self.assertEqual(plan["qwen3_vl_revision"], fake_colab_desk.QWEN3_VL_REVISION)

        # --stub --dry also never touches colab
        env_stub = self.make_env("dry_stub")
        out_stub = self.tmp_path / "out_dry_stub"
        res_stub = env_stub.run_launch(
            [
                "--stub",
                "--dry",
                "--out",
                str(out_stub),
                "--card",
                "CPU-high-mem",
                "--cap",
                "1.5",
                "--wall-s",
                "300",
                "--margin",
                "0.2",
            ]
        )
        self.assertEqual(res_stub.returncode, 0, msg=res_stub.stderr)
        self.assertEqual(env_stub.colab_calls(), [])

    def test_rig_refusals_cli_forbidden_flags_and_missing_required_args(self) -> None:
        proj, cap_cfg = self.make_project("cli_src")
        base = self._base_gpu_args(proj, cap_cfg, self.tmp_path / "out_unused")

        # --backend forbidden in every spelling
        self._assert_rig_refused_before_colab([*base, "--backend", "eg2"], "BACKEND_FORBIDDEN")
        self._assert_rig_refused_before_colab([*base, "--backend=none"], "BACKEND_FORBIDDEN")

        # Pool unlock and --all forbidden
        self._assert_rig_refused_before_colab([*base, "--allow-multimodal-pool"], "MULTIMODAL_POOL_UNLOCK_FORBIDDEN")
        self._assert_rig_refused_before_colab([*base, "--all"], "DRAFT_ALL_FORBIDDEN")
        self._assert_rig_refused_before_colab([*base, "--draft-all"], "DRAFT_ALL_FORBIDDEN")

        # Skip-install flags and env vars forbidden in stub and GPU modes alike
        self._assert_rig_refused_before_colab(
            [
                "--stub",
                "--skip-install",
                "--out",
                str(self.tmp_path / "o_skip1"),
                "--card",
                "CPU-high-mem",
                "--cap",
                "1.5",
                "--wall-s",
                "300",
                "--margin",
                "0.2",
            ],
            "STUB_SKIP_INSTALL_FORBIDDEN",
        )
        self._assert_rig_refused_before_colab(
            [
                "--stub",
                "--out",
                str(self.tmp_path / "o_skip2"),
                "--card",
                "CPU-high-mem",
                "--cap",
                "1.5",
                "--wall-s",
                "300",
                "--margin",
                "0.2",
            ],
            "STUB_SKIP_INSTALL_FORBIDDEN",
            extra_env={"COLAB_STUB_SKIP_INSTALL": "1"},
        )

        # Unrecognized CLI option
        self._assert_rig_refused_before_colab([*base, "--not-a-valid-flag"], "UNKNOWN_ARGUMENT")

        # Required flags: --out, --card, --cap, --wall-s, --margin, --project, --caption-config
        self._assert_rig_refused_before_colab(
            ["--project", str(proj), "--caption-config", str(cap_cfg), "--card", "L4", "--cap", "5", "--wall-s", "300", "--margin", "0.5"],
            "OUT_DIR_REQUIRED",
        )
        self._assert_rig_refused_before_colab(
            ["--project", str(proj), "--caption-config", str(cap_cfg), "--out", str(self.tmp_path / "o_nocard"), "--cap", "5", "--wall-s", "300", "--margin", "0.5"],
            "CARD_REQUIRED",
        )
        self._assert_rig_refused_before_colab(
            ["--project", str(proj), "--caption-config", str(cap_cfg), "--out", str(self.tmp_path / "o_nocap"), "--card", "L4", "--cap", "0", "--wall-s", "300", "--margin", "0.5"],
            "CAP_REQUIRED",
        )
        self._assert_rig_refused_before_colab(
            ["--project", str(proj), "--caption-config", str(cap_cfg), "--out", str(self.tmp_path / "o_nowall"), "--card", "L4", "--cap", "5", "--wall-s", "0", "--margin", "0.5"],
            "WALL_S_REQUIRED",
        )
        self._assert_rig_refused_before_colab(
            ["--project", str(proj), "--caption-config", str(cap_cfg), "--out", str(self.tmp_path / "o_nomarg"), "--card", "L4", "--cap", "5", "--wall-s", "300"],
            "MARGIN_REQUIRED",
        )
        self._assert_rig_refused_before_colab(
            ["--caption-config", str(cap_cfg), "--out", str(self.tmp_path / "o_noproj"), "--card", "L4", "--cap", "5", "--wall-s", "300", "--margin", "0.5"],
            "PROJECT_REQUIRED",
        )
        self._assert_rig_refused_before_colab(
            ["--project", str(proj), "--out", str(self.tmp_path / "o_nocfg"), "--card", "L4", "--cap", "5", "--wall-s", "300", "--margin", "0.5"],
            "CAPTION_CONFIG_REQUIRED",
        )

    def test_rig_refusals_card_shape_out_dir_pins_and_cap_budget(self) -> None:
        proj, cap_cfg = self.make_project("card_src")

        # Card & shape gates
        self._assert_rig_refused_before_colab(
            [*self._base_gpu_args(proj, cap_cfg, self.tmp_path / "o_t4"), "--card", "T4"],
            "GPU_CARD_TOO_SMALL",
        )
        self._assert_rig_refused_before_colab(
            [*self._base_gpu_args(proj, cap_cfg, self.tmp_path / "o_badcard"), "--card", "RTX4090"],
            "CARD_INVALID",
        )
        self._assert_rig_refused_before_colab(
            ["--stub", "--out", str(self.tmp_path / "o_stubcard"), "--card", "L4", "--cap", "5", "--wall-s", "300", "--margin", "0.5"],
            "STUB_CARD_MISMATCH",
        )
        self._assert_rig_refused_before_colab(
            [*self._base_gpu_args(proj, cap_cfg, self.tmp_path / "o_l4std"), "--shape", "standard"],
            "SHAPE_STANDARD_UNAVAILABLE",
        )

        # --out must be on E:, empty, and outside --project
        self._assert_rig_refused_before_colab(
            [
                "--project",
                str(proj),
                "--caption-config",
                str(cap_cfg),
                "--out",
                "C:/non_e_drive_output/desk_draft_test",
                "--card",
                "L4",
                "--cap",
                "5",
                "--wall-s",
                "300",
                "--margin",
                "0.5",
            ],
            "OUT_DIR_NOT_ON_E",
            allow_any_out_drive=False,
        )
        non_empty = self.tmp_path / "o_nonempty"
        non_empty.mkdir(parents=True)
        (non_empty / "existing.txt").write_text("occupied\n", encoding="utf-8")
        self._assert_rig_refused_before_colab(
            self._base_gpu_args(proj, cap_cfg, non_empty),
            "OUT_DIR_NOT_EMPTY",
        )
        self._assert_rig_refused_before_colab(
            self._base_gpu_args(proj, cap_cfg, proj / "inside_out"),
            "OUT_INSIDE_PROJECT",
        )

        # Pin mismatch and cap budget reserve check before colab call
        self._assert_rig_refused_before_colab(
            self._base_gpu_args(proj, cap_cfg, self.tmp_path / "o_badsha"),
            "INPUT_SHA_MISMATCH",
            extra_env={"EXPECTED_DESK_LOCK_SHA": "0" * 64},
        )
        self._assert_rig_refused_before_colab(
            [
                "--project",
                str(proj),
                "--caption-config",
                str(cap_cfg),
                "--out",
                str(self.tmp_path / "o_smallcap"),
                "--card",
                "L4",
                "--cap",
                "0.5",
                "--wall-s",
                "300",
                "--margin",
                "0.6",
            ],
            "CAP_TOO_SMALL_FOR_CLEANUP",
        )

    def test_rig_refusals_project_pool_tag_and_private_image_routes(self) -> None:
        proj, cap_cfg = self.make_project("priv_src")

        # Missing project / missing caption-config / invalid JSON / empty project
        self._assert_rig_refused_before_colab(
            self._base_gpu_args(self.tmp_path / "does_not_exist", cap_cfg, self.tmp_path / "o_pmiss"),
            "PROJECT_MISSING",
        )
        self._assert_rig_refused_before_colab(
            self._base_gpu_args(proj, self.tmp_path / "missing_cfg.json", self.tmp_path / "o_cmiss"),
            "CAPTION_CONFIG_MISSING",
        )
        bad_json = self.tmp_path / "bad_cfg.json"
        bad_json.write_text("{not-json\n", encoding="utf-8")
        self._assert_rig_refused_before_colab(
            self._base_gpu_args(proj, bad_json, self.tmp_path / "o_cbad"),
            "CAPTION_CONFIG_INVALID",
        )
        empty_proj = self.tmp_path / "empty_proj"
        empty_proj.mkdir(parents=True)
        self._assert_rig_refused_before_colab(
            self._base_gpu_args(empty_proj, cap_cfg, self.tmp_path / "o_pempty"),
            "PROJECT_EMPTY",
        )

        # Missing or invalid multimodal_pool tag in --caption-config
        proj_notag, cfg_notag = self.make_project("notag_src", pool_tag=None)
        self._assert_rig_refused_before_colab(
            self._base_gpu_args(proj_notag, cfg_notag, self.tmp_path / "o_notag"),
            "MULTIMODAL_POOL_TAG_MISSING",
        )
        proj_badtag, cfg_badtag = self.make_project("badtag_src", pool_tag="not_a_known_pool")
        self._assert_rig_refused_before_colab(
            self._base_gpu_args(proj_badtag, cfg_badtag, self.tmp_path / "o_badtag"),
            "MULTIMODAL_POOL_TAG_INVALID",
        )

        # Private image delivery rule: CLI flags, URLs in records, and escaping paths are refused on rig
        for bad_flag in ("--drive=/content/drive/MyDrive", "--image-url=https://example.com/a.png", "--bucket=gs://pub/x", "--hf-dataset=org/ds"):
            self._assert_rig_refused_before_colab(
                [*self._base_gpu_args(proj, cap_cfg, self.tmp_path / f"o_priv_{abs(hash(bad_flag)) % 9999}"), bad_flag],
                "PRIVATE_IMAGE_ROUTE_FORBIDDEN",
            )

        proj_url, cfg_url = self.make_project("url_src", flagged_count=1, unflagged_count=0)
        rec_file = next((proj_url / "records").glob("*.json"))
        rec_doc = json.loads(rec_file.read_text(encoding="utf-8"))
        rec_doc["source_url"] = "https://storage.googleapis.com/public-bucket/plate.png"
        rec_file.write_text(json.dumps(rec_doc), encoding="utf-8")
        self._assert_rig_refused_before_colab(
            self._base_gpu_args(proj_url, cfg_url, self.tmp_path / "o_rec_url"),
            "PRIVATE_IMAGE_ROUTE_FORBIDDEN",
        )

        proj_esc, cfg_esc = self.make_project("esc_src", flagged_count=1, unflagged_count=0)
        esc_rec = next((proj_esc / "records").glob("*.json"))
        esc_doc = json.loads(esc_rec.read_text(encoding="utf-8"))
        esc_doc["asset_path"] = "../outside.png"
        esc_rec.write_text(json.dumps(esc_doc), encoding="utf-8")
        self._assert_rig_refused_before_colab(
            self._base_gpu_args(proj_esc, cfg_esc, self.tmp_path / "o_rec_esc"),
            "PRIVATE_IMAGE_ROUTE_FORBIDDEN",
        )


class SessionOpenAndVmRefusalTests(ColabTempTestCase):
    """Test HTTP 412 vs 503 session-open classification, VM step refusals, and cleanup invariants."""

    def _run_gpu(
        self,
        name: str,
        *,
        config: dict | None = None,
        extra_args: list[str] | None = None,
        extra_env: dict[str, str | None] | None = None,
    ) -> tuple[fake_colab_desk.FakeColabEnv, pathlib.Path, int, dict]:
        proj, cap_cfg = self.make_project(f"src_{name}", flagged_count=2, unflagged_count=1)
        env = self.make_env(f"env_{name}", config=config)
        out_dir = self.tmp_path / f"out_{name}"
        args = [
            "--project",
            str(proj),
            "--caption-config",
            str(cap_cfg),
            "--out",
            str(out_dir),
            "--card",
            "L4",
            "--cap",
            "5.0",
            "--wall-s",
            "300",
            "--margin",
            "0.5",
            *(extra_args or []),
        ]
        res = env.run_launch(args, extra_env=extra_env)
        win = json.loads((out_dir / "window.json").read_text(encoding="utf-8"))
        return env, out_dir, res.returncode, win

    def _assert_vm_failure_record_and_cleanup(
        self,
        env: fake_colab_desk.FakeColabEnv,
        out_dir: pathlib.Path,
        rc: int,
        win: dict,
        expected_code: str,
        *,
        expect_cleanup_ok: bool = True,
    ) -> None:
        self.assertNotEqual(rc, 0)
        self.assertEqual(win["refusal_code"], expected_code)
        self.assertIsNotNone(win["why"])
        self.assertTrue(str(win["why"]).strip())
        self.assertEqual(win["lane"], "draft")
        self.assertFalse(win["embedding_backend_loaded"])
        self.assertEqual(win["private_images"]["delivery_method"], "colab_upload_checked")
        self.assertFalse(win["private_images"]["external_urls_allowed"])
        self.assertTrue(win["vm_workdir_cleanup"]["attempted"])
        if expect_cleanup_ok:
            self.assertEqual(win["vm_workdir_cleanup"]["exit_status"], 0)
            self.assertTrue(win["vm_workdir_cleanup"]["ok"])
            self.assertTrue(win["private_images"]["vm_workdir_deleted"])
            self._assert_vm_workdir_clean(env)
        else:
            self.assertNotEqual(win["vm_workdir_cleanup"]["exit_status"], 0)
            self.assertFalse(win["vm_workdir_cleanup"]["ok"])
            self.assertFalse(win["private_images"]["vm_workdir_deleted"])

        outcome = json.loads((out_dir / "outcome.draft.json").read_text(encoding="utf-8"))
        self.assertEqual(outcome["refusal_code"], expected_code)
        self.assertIsNotNone(outcome["why"])
        self.assertFalse((out_dir / "outcome.train.json").exists())
        self.assertFalse(env.session_state().get("active", False))
        self.assertGreaterEqual(env.session_state().get("stop_count", 0), 1)

    def _assert_vm_workdir_clean(self, env: fake_colab_desk.FakeColabEnv) -> None:
        for rel in (
            "project_input.tar",
            "project_manifest.json",
            "caption_config.json",
            "desk_work",
            "desk_out",
            "desk_state",
            "desk_draft_bundle.tar.gz",
            "desk_hf_token",
        ):
            self.assertFalse((env.vm_content / rel).exists(), msg=f"VM workdir survivor: {rel}")
        self.assertEqual(list(env.vm_content.glob(".sdlab_up_*")), [])
        self.assertEqual(list(env.vm_content.glob(".sdlab_dl_*")), [])

    def test_session_open_412_never_retry_vs_503_retry_later_and_shape_mismatch(self) -> None:
        # HTTP 412 -> COLAB_ACCOUNT_CAP_412, retryable=False, never_retry
        env_412, out_412, rc_412, win_412 = self._run_gpu("open_412", config={"new_http_status": 412})
        self.assertNotEqual(rc_412, 0)
        self.assertEqual(win_412["refusal_code"], "COLAB_ACCOUNT_CAP_412")
        self.assertIsNotNone(win_412["why"])
        self.assertFalse(win_412["retryable"])
        self.assertEqual(win_412["retry_disposition"], "never_retry")
        outc_412 = json.loads((out_412 / "outcome.draft.json").read_text(encoding="utf-8"))
        self.assertEqual(outc_412["retry_disposition"], "never_retry")
        self.assertEqual([c["subcmd"] for c in env_412.colab_calls()].count("new"), 1)

        # HTTP 503 -> COLAB_POOL_EMPTY_503, retryable=True, retry_later
        env_503, out_503, rc_503, win_503 = self._run_gpu("open_503", config={"new_http_status": 503})
        self.assertNotEqual(rc_503, 0)
        self.assertEqual(win_503["refusal_code"], "COLAB_POOL_EMPTY_503")
        self.assertIsNotNone(win_503["why"])
        self.assertTrue(win_503["retryable"])
        self.assertEqual(win_503["retry_disposition"], "retry_later")
        outc_503 = json.loads((out_503 / "outcome.draft.json").read_text(encoding="utf-8"))
        self.assertEqual(outc_503["retry_disposition"], "retry_later")
        self.assertEqual([c["subcmd"] for c in env_503.colab_calls()].count("new"), 1)

        # Post-open shape mismatch -> SHAPE_MISMATCH
        env_shp, out_shp, rc_shp, win_shp = self._run_gpu("shape_mm", config={"session_shape_override": "Standard"})
        self._assert_vm_failure_record_and_cleanup(env_shp, out_shp, rc_shp, win_shp, "SHAPE_MISMATCH")

    def test_vm_pre_download_guards_gpu_vram_disk_ram_and_drive_mount(self) -> None:
        # Google Drive mount on VM -> PRIVATE_IMAGE_ROUTE_FORBIDDEN before any download
        env_drv, out_drv, rc_drv, win_drv = self._run_gpu("vm_drive", config={"mount_drive": True})
        self._assert_vm_failure_record_and_cleanup(env_drv, out_drv, rc_drv, win_drv, "PRIVATE_IMAGE_ROUTE_FORBIDDEN")
        self.assertEqual(env_drv.hf_calls(), [])

        # Missing GPU on VM -> GPU_MISSING before any download
        env_nogpu, out_nogpu, rc_nogpu, win_nogpu = self._run_gpu("vm_nogpu", config={"gpu_missing": True})
        self._assert_vm_failure_record_and_cleanup(env_nogpu, out_nogpu, rc_nogpu, win_nogpu, "GPU_MISSING")
        self.assertEqual(env_nogpu.hf_calls(), [])

        # VM reports Tesla T4 -> GPU_CARD_TOO_SMALL before any download
        env_t4, out_t4, rc_t4, win_t4 = self._run_gpu(
            "vm_t4",
            config={"gpu_name": "Tesla T4", "gpu_vram_mib": 15360},
        )
        self._assert_vm_failure_record_and_cleanup(env_t4, out_t4, rc_t4, win_t4, "GPU_CARD_TOO_SMALL")
        self.assertEqual(env_t4.hf_calls(), [])

        # VM reports L4 with < 22,000 MiB VRAM -> GPU_VRAM_TOO_LOW before any download
        env_vram, out_vram, rc_vram, win_vram = self._run_gpu(
            "vm_vram",
            config={"gpu_name": "NVIDIA L4", "gpu_vram_mib": 21500},
        )
        self._assert_vm_failure_record_and_cleanup(env_vram, out_vram, rc_vram, win_vram, "GPU_VRAM_TOO_LOW")
        self.assertEqual(env_vram.hf_calls(), [])

        # Low free disk -> DISK_FREE_LOW before any download
        env_dsk, out_dsk, rc_dsk, win_dsk = self._run_gpu("vm_disk", config={"disk_gib": 20.0})
        self._assert_vm_failure_record_and_cleanup(env_dsk, out_dsk, rc_dsk, win_dsk, "DISK_FREE_LOW")
        self.assertEqual(env_dsk.hf_calls(), [])

        # Low available host RAM -> HOST_RAM_LOW before any download
        env_ram, out_ram, rc_ram, win_ram = self._run_gpu("vm_ram", config={"ram_gib": 8.0})
        self._assert_vm_failure_record_and_cleanup(env_ram, out_ram, rc_ram, win_ram, "HOST_RAM_LOW")
        self.assertEqual(env_ram.hf_calls(), [])

    def test_vm_install_and_weight_download_refusals(self) -> None:
        # Uploaded wheel tampered on VM -> WHEEL_SHA_MISMATCH
        env_whl, out_whl, rc_whl, win_whl = self._run_gpu("vm_whl", config={"corrupt_wheel_on_vm": True})
        self._assert_vm_failure_record_and_cleanup(env_whl, out_whl, rc_whl, win_whl, "WHEEL_SHA_MISMATCH")

        # Uploaded lockfile tampered on VM -> LOCK_HASH_MISMATCH
        env_lck, out_lck, rc_lck, win_lck = self._run_gpu("vm_lock", config={"corrupt_lock_on_vm": True})
        self._assert_vm_failure_record_and_cleanup(env_lck, out_lck, rc_lck, win_lck, "LOCK_HASH_MISMATCH")

        # Weights directory on a FUSE/network mount -> WEIGHT_MOUNT_FORBIDDEN
        env_mnt, out_mnt, rc_mnt, win_mnt = self._run_gpu("vm_mount", config={"hf_mount_fstype": "fuse.gcsfuse"})
        self._assert_vm_failure_record_and_cleanup(env_mnt, out_mnt, rc_mnt, win_mnt, "WEIGHT_MOUNT_FORBIDDEN")

        # Token shred failure -> TOKEN_SHRED_FAILED
        env_shred, out_shred, rc_shred, win_shred = self._run_gpu(
            "vm_shred",
            config={"token_shred_fail": True},
            extra_env={"HF_TOKEN": "hf_" + ("SyntheticShredTok" * 2)},
        )
        self._assert_vm_failure_record_and_cleanup(env_shred, out_shred, rc_shred, win_shred, "TOKEN_SHRED_FAILED")

        # HTTP 429 exceeding stated download bound -> DOWNLOAD_429_BOUND_EXCEEDED
        env_429, out_429, rc_429, win_429 = self._run_gpu(
            "vm_429_bound",
            config={"weight_429_count": 8},
            extra_args=["--download-bound-s", "0.05"],
        )
        self._assert_vm_failure_record_and_cleanup(env_429, out_429, rc_429, win_429, "DOWNLOAD_429_BOUND_EXCEEDED")

        # Resolved revision mismatch -> WEIGHT_REVISION_MISMATCH
        env_rev, out_rev, rc_rev, win_rev = self._run_gpu("vm_rev", config={"weight_wrong_revision": True})
        self._assert_vm_failure_record_and_cleanup(env_rev, out_rev, rc_rev, win_rev, "WEIGHT_REVISION_MISMATCH")

        # Downloaded safetensors sha256 mismatch -> WEIGHT_SHA_MISMATCH
        env_wsha, out_wsha, rc_wsha, win_wsha = self._run_gpu("vm_wsha", config={"weight_corrupt_sha": True})
        self._assert_vm_failure_record_and_cleanup(env_wsha, out_wsha, rc_wsha, win_wsha, "WEIGHT_SHA_MISMATCH")

        # Offline AutoConfig _commit_hash mismatch -> WEIGHT_REVISION_MISMATCH
        env_cfg, out_cfg, rc_cfg, win_cfg = self._run_gpu("vm_cfg_rev", config={"config_wrong_commit_hash": True})
        self._assert_vm_failure_record_and_cleanup(env_cfg, out_cfg, rc_cfg, win_cfg, "WEIGHT_REVISION_MISMATCH")

    def test_vm_heartbeat_stall_output_manifest_cleanup_failure_and_abrupt_crash(self) -> None:
        # Stalled drafter -> DRAFT_HEARTBEAT_STALLED stops GPU and closes window
        env_hb, out_hb, rc_hb, win_hb = self._run_gpu(
            "vm_hb_stall",
            extra_args=["--heartbeat-stall-s", "0.15"],
            extra_env={
                "DESK_TEST_GPU_DRAFTER": "fake",
                "DATASET_DESK_TEST_STALL_AT_RECORD": "1",
                "DATASET_DESK_TEST_STALL_SECONDS": "0.75",
            },
        )
        self._assert_vm_failure_record_and_cleanup(env_hb, out_hb, rc_hb, win_hb, "DRAFT_HEARTBEAT_STALLED")

        # Missing required output file in fetched bundle -> OUTPUT_FILE_MISSING
        env_miss, out_miss, rc_miss, win_miss = self._run_gpu(
            "vm_out_miss",
            config={"delete_output_file": "desk_artifacts/drafter-raw.jsonl"},
            extra_env={"DESK_TEST_GPU_DRAFTER": "fake"},
        )
        self._assert_vm_failure_record_and_cleanup(env_miss, out_miss, rc_miss, win_miss, "OUTPUT_FILE_MISSING")

        # Corrupted output file vs VM manifest -> OUTPUT_MANIFEST_SHA_MISMATCH
        env_cor, out_cor, rc_cor, win_cor = self._run_gpu(
            "vm_out_corrupt",
            config={"corrupt_output_file": "desk_artifacts/captions-manifest.json"},
            extra_env={"DESK_TEST_GPU_DRAFTER": "fake"},
        )
        self._assert_vm_failure_record_and_cleanup(env_cor, out_cor, rc_cor, win_cor, "OUTPUT_MANIFEST_SHA_MISMATCH")

        # Failed VM workdir deletion -> VM_WORKDIR_CLEANUP_FAILED named in window.json
        env_cln, out_cln, rc_cln, win_cln = self._run_gpu(
            "vm_clean_fail",
            config={"cleanup_fail": True},
            extra_env={"DESK_TEST_GPU_DRAFTER": "fake"},
        )
        self._assert_vm_failure_record_and_cleanup(
            env_cln,
            out_cln,
            rc_cln,
            win_cln,
            "VM_WORKDIR_CLEANUP_FAILED",
            expect_cleanup_ok=False,
        )

        # Abrupt non-RefusalError crash inside remote step -> REMOTE_STEP_FAILED with non-null code & why
        env_cr, out_cr, rc_cr, win_cr = self._run_gpu(
            "vm_abrupt_crash",
            config={"abrupt_crash_at_step": "desk_install"},
        )
        self._assert_vm_failure_record_and_cleanup(env_cr, out_cr, rc_cr, win_cr, "REMOTE_STEP_FAILED")
        self.assertEqual(win_cr["last_remote_step"], "desk_install")


class EndToEndStubAndGpuDraftingTests(ColabTempTestCase):
    """End-to-end tests for --stub, GPU drafting (fake + qwen3-vl), 429 retry + token file route, and zero embedders."""

    def test_stub_e2e_runs_real_install_order_config_load_fake_compose_cleanup_and_stop(self) -> None:
        env = self.make_env("e2e_stub")
        out_dir = self.tmp_path / "out_e2e_stub"
        res = env.run_launch(
            [
                "--stub",
                "--out",
                str(out_dir),
                "--card",
                "CPU-high-mem",
                "--cap",
                "2.0",
                "--wall-s",
                "300",
                "--margin",
                "0.2",
            ]
        )
        self.assertEqual(res.returncode, 0, msg=res.stderr)

        # Assert exact install command order on CPU-high-mem:
        #   1. pip install --require-hashes ... -r /content/uv-bootstrap.txt
        #   2. uv python install 3.12.15
        #   3. uv venv --python 3.12.15 /content/venv-desk
        #   4. uv pip install --python .../venv-desk/bin/python --require-hashes ... -r .../desk_draft.lock
        #   5. uv pip install --python .../venv-desk/bin/python --no-deps --only-binary :all: .../dataset_desk-*.whl
        inst = env.install_commands()
        self.assertEqual(len(inst), 5, msg=f"unexpected install command count: {inst}")
        self.assertEqual(inst[0]["tool"], "pip")
        self.assertIn("--require-hashes", inst[0]["argv"])
        self.assertTrue(any("uv-bootstrap.txt" in tok for tok in inst[0]["argv"]))

        self.assertEqual(inst[1]["tool"], "uv")
        self.assertEqual(inst[1]["argv"][:3], ["python", "install", "3.12.15"])

        self.assertEqual(inst[2]["tool"], "uv")
        self.assertEqual(inst[2]["argv"][:3], ["venv", "--python", "3.12.15"])
        self.assertTrue(inst[2]["argv"][-1].endswith("venv-desk"))

        self.assertEqual(inst[3]["tool"], "uv")
        self.assertEqual(inst[3]["argv"][:2], ["pip", "install"])
        self.assertIn("--require-hashes", inst[3]["argv"])
        self.assertTrue(any(tok.endswith("desk_draft.lock") for tok in inst[3]["argv"]))

        self.assertEqual(inst[4]["tool"], "uv")
        self.assertEqual(inst[4]["argv"][:2], ["pip", "install"])
        self.assertIn("--no-deps", inst[4]["argv"])
        self.assertTrue(inst[4]["argv"][-1].endswith(".whl"))

        # Assert weight download + offline AutoConfig load happened before compose
        hf = env.hf_calls()
        ops = [c["op"] for c in hf]
        self.assertEqual(ops, ["model_info", "snapshot_download", "AutoConfig.from_pretrained"])
        self.assertEqual(hf[2]["hf_hub_offline"], "1")
        self.assertTrue(hf[2]["local_files_only"])
        self.assertEqual(hf[2]["revision"], fake_colab_desk.QWEN3_VL_REVISION)

        win = json.loads((out_dir / "window.json").read_text(encoding="utf-8"))
        self.assertEqual(win["status"], "completed")
        self.assertIsNone(win["refusal_code"])
        self.assertEqual(win["mode"], "stub")
        self.assertEqual(win["drafter"], "fake")
        self.assertEqual(win["colab_runtime_version"], "release-colab-2026-11-05")
        self.assertTrue(len(win["usr_local_cuda"]) >= 1)
        self.assertTrue(win["config_only_load"]["ok"])
        self.assertTrue(win["output_manifest_verified"])
        self.assertFalse(win["embedding_backend_loaded"])
        self.assertEqual(win["vm_workdir_cleanup"]["exit_status"], 0)
        self.assertTrue(win["private_images"]["vm_workdir_deleted"])

        raw_doc = json.loads((out_dir / "outputs" / "desk_artifacts" / "raw_outputs.json").read_text(encoding="utf-8"))
        self.assertEqual(raw_doc["flagged_drafted_count"], 1)
        self.assertEqual(raw_doc["unflagged_untouched_count"], 1)
        self.assertFalse(env.session_state().get("active", False))

    def test_gpu_e2e_fake_drafter_drafts_only_flagged_records_and_never_loads_embedders(self) -> None:
        proj, cap_cfg = self.make_project("gpu_fake_src", flagged_count=3, unflagged_count=2)
        env = self.make_env("e2e_gpu_fake")
        out_dir = self.tmp_path / "out_e2e_gpu_fake"

        res = env.run_launch(
            [
                "--project",
                str(proj),
                "--caption-config",
                str(cap_cfg),
                "--out",
                str(out_dir),
                "--card",
                "L4",
                "--cap",
                "5.0",
                "--wall-s",
                "300",
                "--margin",
                "0.5",
            ],
            extra_env={"DESK_TEST_GPU_DRAFTER": "fake"},
        )
        self.assertEqual(res.returncode, 0, msg=res.stderr)

        artifacts = out_dir / "outputs" / "desk_artifacts"
        for req_name in (
            "drafter-raw.jsonl",
            "raw_outputs.json",
            "description-review.jsonl",
            "description-worksheet.md",
            "captions-manifest.json",
            "pool-detection.json",
            "report-none.md",
        ):
            self.assertTrue((artifacts / req_name).is_file(), msg=f"missing output {req_name}")

        review_rows = [
            json.loads(ln)
            for ln in (artifacts / "description-review.jsonl").read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        self.assertEqual(len(review_rows), 5)
        flagged_rows = [r for r in review_rows if not r["passed_model_free"]]
        unflagged_rows = [r for r in review_rows if r["passed_model_free"]]
        self.assertEqual(len(flagged_rows), 3)
        self.assertEqual(len(unflagged_rows), 2)
        for r in flagged_rows:
            self.assertTrue(r.get("draft"), msg=f"expected draft on flagged row {r['id']}")
            # Unedited drafts stay in review queue and never overwrite final training captions.
            self.assertNotIn(r["draft"], r.get("final_caption") or "")
        for r in unflagged_rows:
            self.assertFalse(r.get("draft"), msg=f"unflagged row {r['id']} must never be drafted")
            self.assertFalse(r.get("draft_offer"), msg=f"unflagged row {r['id']} must have no draft_offer")

        raw_rows = [
            json.loads(ln)
            for ln in (artifacts / "drafter-raw.jsonl").read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        self.assertEqual(len(raw_rows), 3)

        # Assert no embedding backend other than 'none' was ever constructed on the VM.
        vm_win = json.loads((out_dir / "vm_state" / "vm_window.json").read_text(encoding="utf-8"))
        self.assertFalse(vm_win["embedding_backend_loaded"])
        self.assertTrue(len(vm_win["embedder_events"]) >= 1)
        self.assertTrue(all(ev["backend"] == "none" and ev["allowed"] for ev in vm_win["embedder_events"]))

        # Assert private images were NOT bundled into the output archive and VM workdir was wiped.
        with tarfile.open(out_dir / "desk_draft_bundle.tar.gz", "r:gz") as tf:
            bundled_names = [m.name for m in tf.getmembers()]
        self.assertTrue(all(not n.endswith(".png") for n in bundled_names))

    def test_gpu_e2e_qwen3_vl_chunked_transfers_429_retry_and_token_file_shred(self) -> None:
        secret_tok = "hf_" + ("SyntheticTestTok" * 2)
        proj, cap_cfg = self.make_project(
            "gpu_qwen_src",
            flagged_count=2,
            unflagged_count=2,
            extra_pad_bytes=90000,
        )
        env = self.make_env(
            "e2e_gpu_qwen",
            config={
                "weight_429_count": 2,
                "upload_fail_500_once": True,
                "download_fail_500_once": True,
            },
        )
        out_dir = self.tmp_path / "out_e2e_gpu_qwen"

        res = env.run_launch(
            [
                "--project",
                str(proj),
                "--caption-config",
                str(cap_cfg),
                "--out",
                str(out_dir),
                "--card",
                "L4",
                "--cap",
                "5.0",
                "--wall-s",
                "300",
                "--margin",
                "0.5",
                "--download-bound-s",
                "30",
            ],
            extra_env={
                "HF_TOKEN": secret_tok,
                "SDLAB_TEST_UPLOAD_CHUNK_BYTES": "65536",
                "SDLAB_TEST_DOWNLOAD_CHUNK_BYTES": "2048",
            },
        )
        self.assertEqual(res.returncode, 0, msg=res.stderr)

        # Verify HF_TOKEN never appeared in any colab CLI argv or --env flag, and was shredded on VM.
        calls = env.colab_calls()
        self.assertTrue(all(not c["token_leaked_in_cli"] for c in calls))
        for c in calls:
            joined = " ".join(c["argv"])
            self.assertNotIn(secret_tok, joined)
            self.assertNotIn("HF_TOKEN=", joined)
        self.assertFalse((env.vm_content / "desk_hf_token").exists())
        for log_file in out_dir.rglob("*"):
            if log_file.is_file() and log_file.suffix in (".log", ".json", ".jsonl", ".md", ".txt"):
                self.assertNotIn(secret_tok, log_file.read_text(encoding="utf-8", errors="replace"))

        # Verify chunked upload AND chunked download actually split into .part.0000/.part.0001 transfers.
        upload_targets = [c["argv"][-1] for c in calls if c["subcmd"] == "upload"]
        download_sources = [c["argv"][-2] for c in calls if c["subcmd"] == "download"]
        self.assertTrue(any(".part.0000" in u for u in upload_targets), msg=f"expected chunked upload in {upload_targets}")
        self.assertTrue(any(".part.0001" in u for u in upload_targets), msg=f"expected >=2 upload parts in {upload_targets}")
        self.assertTrue(any(".part.0000" in d for d in download_sources), msg=f"expected chunked download in {download_sources}")

        # Verify 2 HTTP 429 failures were retried with backoff before succeeding, then HF_HUB_OFFLINE=1 locked.
        hf = env.hf_calls()
        statuses_429 = [c for c in hf if c.get("status") == 429]
        self.assertEqual(len(statuses_429), 2)
        dl_success = [c for c in hf if c.get("op") == "snapshot_download"]
        self.assertEqual(len(dl_success), 1)
        self.assertTrue(dl_success[0]["token_passed"])
        self.assertFalse(dl_success[0]["hf_env_token_set"])

        qwen_loads = [
            c
            for c in hf
            if c.get("op") in (
                "AutoConfig.from_pretrained",
                "AutoProcessor.from_pretrained",
                "AutoModelForImageTextToText.from_pretrained",
            )
        ]
        self.assertEqual(len(qwen_loads), 3)
        for ld in qwen_loads:
            self.assertEqual(ld["hf_hub_offline"], "1")
            self.assertTrue(ld["local_files_only"])
            self.assertEqual(ld["revision"], fake_colab_desk.QWEN3_VL_REVISION)

        win = json.loads((out_dir / "window.json").read_text(encoding="utf-8"))
        self.assertEqual(win["status"], "completed")
        self.assertEqual(win["qwen3_vl_weights"]["retried_429_count"], 2)
        self.assertEqual(win["qwen3_vl_weights"]["attempts"], 3)
        self.assertTrue(win["multimodal_pool_locked"])
        self.assertEqual(win["multimodal_pool"], "rustline")
        self.assertEqual(win["vm_workdir_cleanup"]["exit_status"], 0)
        self.assertTrue(win["private_images"]["vm_workdir_deleted"])


if __name__ == "__main__":
    unittest.main()
