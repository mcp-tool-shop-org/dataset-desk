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
import re
import shutil
import sys
import tarfile
import tempfile
import types
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
EXPECTED_COLAB_WINDOW_SHA = "415e5f541cb6bfed3eac98c6a0cd926d61c3b1fbeb9d69618d08725b987132e7"
EXPECTED_LAUNCH_GUARDS_SHA = "2998b4f8e3aea3aea2ecc9f779c743b4b44824b30e7ff55fb1da782d7a416c69"
EXPECTED_REMOTE_RUNNER_SHA = "393929ed704ccbacaee2b590010e348a5d635cbdd70361209d3e0103d1fba6eb"


def _load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _detect_image_magic_in_test(data: bytes) -> str | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith((b"II*\x00", b"MM\x00*")):
        return "image/tiff"
    if len(data) >= 14 and data[:2] == b"BM" and data[6:10] == b"\x00\x00\x00\x00":
        return "image/bmp"
    return None


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
            [*self._base_gpu_args(proj, cap_cfg, self.tmp_path / "o_l4hm"), "--shape", "high-mem"],
            "SHAPE_HIGH_MEM_UNAVAILABLE",
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
        env_shp, out_shp, rc_shp, win_shp = self._run_gpu("shape_mm", config={"session_shape_override": "High-RAM"})
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

        # Assert private images were NOT bundled into the output archive (by name or magic bytes) and VM workdir was wiped.
        with tarfile.open(out_dir / "desk_draft_bundle.tar.gz", "r:gz") as tf:
            bundled_names: list[str] = []
            bundled_magic_hits: list[tuple[str, str]] = []
            for member in tf.getmembers():
                bundled_names.append(member.name)
                if member.isfile():
                    ext_fh = tf.extractfile(member)
                    assert ext_fh is not None
                    hit = _detect_image_magic_in_test(ext_fh.read(64))
                    if hit is not None:
                        bundled_magic_hits.append((member.name, hit))
        self.assertTrue(all(not n.endswith(".png") for n in bundled_names))
        self.assertEqual(bundled_magic_hits, [])

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


def _iter_json_strings(obj: object, prefix: str = "$") -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    if isinstance(obj, str):
        found.append((prefix, obj))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str):
                found.append((f"{prefix}.<key>", k))
            found.extend(_iter_json_strings(v, f"{prefix}.{k}"))
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            found.extend(_iter_json_strings(item, f"{prefix}[{idx}]"))
    return found


def _is_abs_host_path_string(val: str, tmp_marker: str) -> bool:
    s = val.strip()
    if not s:
        return False
    if s.startswith("/") or s.startswith("\\\\"):
        return True
    if len(s) >= 3 and s[0].isalpha() and s[1] == ":" and s[2] in ("/", "\\"):
        return True
    norm = s.replace("\\", "/").lower()
    if tmp_marker and tmp_marker.lower() in norm:
        return True
    return False


class FixTurn1PrivacyAndBoundsTests(ColabTempTestCase):
    """Fix turn 1 tests (N1, N2, N3, N4, N6, and magic-byte bundle verification)."""

    def test_n1_upload_allowlist_excludes_rejected_contact_sheet_and_dotenv_and_records_counts_only(self) -> None:
        from PIL import Image

        proj, cap_cfg = self.make_project("n1_src", flagged_count=2, unflagged_count=1)
        rejected_dir = proj / "rejected"
        rejected_dir.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 32), (90, 10, 10)).save(rejected_dir / "discarded_plate.png")
        Image.new("RGB", (64, 64), (20, 20, 20)).save(proj / "outputs" / "contact_sheet.png")
        Image.new("RGB", (32, 32), (40, 40, 90)).save(proj / "outputs" / "wave-1" / "unreferenced_extra.png")
        Image.new("RGB", (32, 32), (70, 70, 10)).save(proj / "outputs" / "wave-1" / "rejected_asset.png")
        (proj / "records" / "rec_rejected_00.json").write_text(
            json.dumps(
                {
                    "id": "rec_rejected_00",
                    "asset_path": "outputs/wave-1/rejected_asset.png",
                    "canon": {"subject": "discarded stone"},
                    "judgment": {"status": "rejected", "explanation": "blurry edge"},
                    "provenance": {"prompt": "discarded stone"},
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (proj / "outputs" / "wave-1" / "generation.json").write_text(
            json.dumps({"wave": "wave-1", "seed": 42}) + "\n",
            encoding="utf-8",
        )
        unused_wave = proj / "outputs" / "wave-unused"
        unused_wave.mkdir(parents=True, exist_ok=True)
        (unused_wave / "generation.json").write_text(json.dumps({"wave": "unused"}) + "\n", encoding="utf-8")
        (proj / ".env").write_text("PRIVATE_API_KEY=never_upload_this\n", encoding="utf-8")

        env = self.make_env("n1_env")
        out_dir = self.tmp_path / "out_n1"
        res = env.run_launch(
            [
                "--project",
                str(proj),
                "--caption-config",
                str(cap_cfg),
                "--out",
                str(out_dir),
                "--stub",
                "--card",
                "CPU-high-mem",
                "--cap",
                "5.0",
                "--wall-s",
                "300",
                "--margin",
                "0.5",
            ]
        )
        self.assertEqual(res.returncode, 0, msg=res.stderr)

        uploaded = env.uploaded_tar_members()
        self.assertNotIn(".env", uploaded)
        self.assertFalse(any(m.startswith("rejected/") for m in uploaded), msg=f"rejected leaked: {uploaded}")
        self.assertNotIn("outputs/contact_sheet.png", uploaded)
        self.assertNotIn("outputs/wave-1/unreferenced_extra.png", uploaded)
        self.assertNotIn("outputs/wave-1/rejected_asset.png", uploaded)
        self.assertNotIn("records/rec_rejected_00.json", uploaded)
        self.assertNotIn("outputs/wave-unused/generation.json", uploaded)
        self.assertIn("outputs/wave-1/flagged_00.png", uploaded)
        self.assertIn("outputs/wave-1/clean_00.png", uploaded)
        self.assertIn("outputs/wave-1/generation.json", uploaded)

        win = json.loads((out_dir / "window.json").read_text(encoding="utf-8"))
        priv = win["private_images"]
        self.assertEqual(priv.get("uploaded_file_count"), len(uploaded))
        self.assertIsInstance(priv.get("uploaded_total_bytes"), int)
        self.assertGreater(priv["uploaded_total_bytes"], 0)
        for jpath, sval in _iter_json_strings(win):
            for forbidden_name in (
                "flagged_00",
                "clean_00",
                "discarded_plate",
                "contact_sheet",
                "rec_flagged_00",
                ".env",
            ):
                self.assertNotIn(forbidden_name, sval, msg=f"filename leaked in window.json at {jpath}: {sval!r}")

    def test_n1_link_cycle_refuses_on_rig_with_named_code(self) -> None:
        import subprocess

        proj, cap_cfg = self.make_project("n1_cycle_src", flagged_count=1, unflagged_count=1)
        sub_dir = proj / "records" / "nested"
        sub_dir.mkdir(parents=True, exist_ok=True)
        loop_link = sub_dir / "cycle_back"
        used_junction = False
        try:
            os.symlink(str(proj), str(loop_link), target_is_directory=True)
        except OSError:
            subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(loop_link), str(proj)],
                check=True,
                capture_output=True,
                text=True,
            )
            used_junction = True

        def _remove_loop_link() -> None:
            if used_junction:
                subprocess.run(["cmd", "/c", "rmdir", str(loop_link)], check=False, capture_output=True)
            else:
                try:
                    loop_link.unlink(missing_ok=True)
                except OSError:
                    pass

        self.addCleanup(_remove_loop_link)

        env = self.make_env("n1_cycle_env")
        out_dir = self.tmp_path / "out_n1_cycle"
        res = env.run_launch(
            [
                "--project",
                str(proj),
                "--caption-config",
                str(cap_cfg),
                "--out",
                str(out_dir),
                "--stub",
                "--card",
                "CPU-high-mem",
                "--cap",
                "5.0",
                "--wall-s",
                "120",
                "--margin",
                "0.5",
            ],
            timeout=25.0,
        )
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("PROJECT_LINK_CYCLE", res.stderr)
        self.assertEqual(env.colab_calls(), [])

    def test_n2_project_manifest_and_kept_artifacts_contain_no_absolute_paths(self) -> None:
        from PIL import Image

        proj, cap_cfg = self.make_project("n2_src", flagged_count=1, unflagged_count=1)
        pool_root = self.tmp_path / "studio_pools_secret_host_dir"
        pool_rust = pool_root / "rustline"
        pool_tallow = pool_root / "tallow-fen"
        pool_rust.mkdir(parents=True, exist_ok=True)
        pool_tallow.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (24, 24), (120, 60, 30)).save(pool_rust / "ref_00.png")
        Image.new("RGB", (24, 24), (160, 140, 40)).save(pool_tallow / "ref_01.png")

        studio_cfg = self.tmp_path / "studio_config.json"
        studio_cfg.write_text(
            json.dumps(
                {"locked_pool_dirs": {"rustline": str(pool_rust), "tallow-fen": str(pool_tallow)}},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        env = self.make_env("n2_env")
        out_dir = self.tmp_path / "out_n2"
        res = env.run_launch(
            [
                "--project",
                str(proj),
                "--caption-config",
                str(cap_cfg),
                "--out",
                str(out_dir),
                "--stub",
                "--card",
                "CPU-high-mem",
                "--cap",
                "5.0",
                "--wall-s",
                "300",
                "--margin",
                "0.5",
            ],
            extra_env={
                "DATASET_DESK_USER_CONFIG": str(studio_cfg),
                "DESK_TEST_ISOLATE_USER_CONFIG": "0",
            },
        )
        self.assertEqual(res.returncode, 0, msg=res.stderr)

        tmp_marker = self.tmp_path.name
        checked_json_files = [
            out_dir / "project_manifest.json",
            out_dir / "wheel_manifest.json",
            out_dir / "outputs" / "desk_artifacts" / "pool-detection.json",
            out_dir / "outputs" / "desk_artifacts" / "none" / "meta.json",
        ]
        for jf in checked_json_files:
            self.assertTrue(jf.is_file(), msg=f"expected JSON file {jf}")
            doc = json.loads(jf.read_text(encoding="utf-8"))
            for jpath, sval in _iter_json_strings(doc):
                self.assertFalse(
                    _is_abs_host_path_string(sval, tmp_marker),
                    msg=f"absolute path leaked in {jf.name} at {jpath}: {sval!r}",
                )

        manifest = json.loads((out_dir / "project_manifest.json").read_text(encoding="utf-8"))
        pd = manifest["pool_detection"]
        self.assertNotIn("sources", pd)
        self.assertIn("counts", pd)
        self.assertEqual(pd["counts"].get("rustline"), 1)
        self.assertEqual(pd["counts"].get("tallow-fen"), 1)

        win = json.loads((out_dir / "window.json").read_text(encoding="utf-8"))
        for jpath, sval in _iter_json_strings(win):
            self.assertNotIn(
                tmp_marker.lower(),
                sval.replace("\\", "/").lower(),
                msg=f"host path leaked in window.json at {jpath}: {sval!r}",
            )

        for jsonl_file in (
            out_dir / "outputs" / "desk_artifacts" / "drafter-raw.jsonl",
            out_dir / "outputs" / "desk_artifacts" / "description-review.jsonl",
        ):
            for idx, ln in enumerate(jsonl_file.read_text(encoding="utf-8").splitlines()):
                if not ln.strip():
                    continue
                for jpath, sval in _iter_json_strings(json.loads(ln), f"line[{idx}]"):
                    self.assertFalse(
                        _is_abs_host_path_string(sval, tmp_marker),
                        msg=f"absolute path leaked in {jsonl_file.name} {jpath}: {sval!r}",
                    )

    def _find_leftover_staging_entries(self, out_dir: pathlib.Path) -> list[str]:
        if not out_dir.is_dir():
            return []
        bad: list[str] = []
        for entry in out_dir.iterdir():
            nm = entry.name
            if (
                nm in ("_rig_stage", "_stub_input", "_bundle_extract", "_vm_cleanup_exec.py")
                or nm.startswith(".sdlab_up_tmp_")
                or nm.startswith(".sdlab_dl_tmp_")
                or nm.endswith(".sdlab_reassembling")
            ):
                bad.append(nm)
        return sorted(bad)

    def test_n3_staging_dirs_removed_on_trip_signal_and_refusal(self) -> None:
        import subprocess
        import time

        proj, cap_cfg = self.make_project(
            "n3_src",
            flagged_count=2,
            unflagged_count=1,
            extra_pad_bytes=95000,
        )

        # 1) Pre-session refusal (CAP_TOO_SMALL_FOR_CLEANUP) must not leave _rig_stage behind.
        env_ref = self.make_env("n3_ref")
        out_ref = self.tmp_path / "out_n3_ref"
        res_ref = env_ref.run_launch(
            [
                "--project",
                str(proj),
                "--caption-config",
                str(cap_cfg),
                "--out",
                str(out_ref),
                "--card",
                "L4",
                "--cap",
                "0.1",
                "--wall-s",
                "300",
                "--margin",
                "0.05",
            ]
        )
        self.assertNotEqual(res_ref.returncode, 0)
        self.assertIn("CAP_TOO_SMALL_FOR_CLEANUP", res_ref.stderr)
        self.assertEqual(self._find_leftover_staging_entries(out_ref), [])

        # 2) Wall trip during chunked upload (.sdlab_up_tmp_* active on rig) must wipe all staging dirs.
        env_trip = self.make_env("n3_trip", config={"upload_part_sleep_s": 12.0})
        out_trip = self.tmp_path / "out_n3_trip"
        res_trip = env_trip.run_launch(
            [
                "--project",
                str(proj),
                "--caption-config",
                str(cap_cfg),
                "--out",
                str(out_trip),
                "--card",
                "L4",
                "--cap",
                "5.0",
                "--wall-s",
                "8",
                "--grace-s",
                "1",
                "--margin",
                "0.5",
            ],
            extra_env={"SDLAB_TEST_UPLOAD_CHUNK_BYTES": "65536"},
            timeout=90.0,
        )
        self.assertNotEqual(res_trip.returncode, 0)
        self.assertTrue((env_trip.state_dir / "upload_part_in_flight.txt").is_file())
        self.assertEqual(self._find_leftover_staging_entries(out_trip), [])
        self.assertFalse(env_trip.session_state().get("active", False))

        # 3) POSIX SIGTERM while chunked upload is in flight must wipe .sdlab_up_tmp_* and stop session.
        env_sig = self.make_env("n3_sig", config={"upload_part_sleep_s": 8.0})
        out_sig = self.tmp_path / "out_n3_sig"
        in_flight_marker = fake_colab_desk.to_posix(env_sig.state_dir / "upload_part_in_flight.txt")
        launch_posix = fake_colab_desk.to_posix(LAUNCH_SH)
        proj_posix = fake_colab_desk.to_posix(proj)
        cap_posix = fake_colab_desk.to_posix(cap_cfg)
        out_posix = fake_colab_desk.to_posix(out_sig)
        bash_bin = fake_colab_desk.resolve_git_bash()
        sig_script = (
            f'bash "{launch_posix}" --project "{proj_posix}" --caption-config "{cap_posix}" '
            f'--out "{out_posix}" --card L4 --cap 5.0 --wall-s 300 --margin 0.5 &\n'
            "lpid=$!\n"
            "for _i in $(seq 1 200); do\n"
            f'  if [ -f "{in_flight_marker}" ]; then break; fi\n'
            "  sleep 0.05\n"
            "done\n"
            'kill -TERM "${lpid}" 2>/dev/null || true\n'
            'wait "${lpid}"\n'
        )
        t0 = time.monotonic()
        res_sig = subprocess.run(
            [bash_bin, "-c", sig_script],
            env=env_sig.build_env(extra_env={"SDLAB_TEST_UPLOAD_CHUNK_BYTES": "65536"}),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdin=subprocess.DEVNULL,
            timeout=60.0,
            check=False,
        )
        _ = t0
        self.assertNotEqual(res_sig.returncode, 0)
        self.assertTrue((env_sig.state_dir / "upload_part_in_flight.txt").is_file())
        self.assertEqual(self._find_leftover_staging_entries(out_sig), [])
        self.assertFalse(env_sig.session_state().get("active", False))

    def test_n4_vm_cleanup_step_counted_in_cleanup_bounds_and_cap_reserve(self) -> None:
        proj, cap_cfg = self.make_project("n4_src", flagged_count=1, unflagged_count=1)
        prod_timing_env: dict[str, str | None] = {
            "SDLAB_TEST_KILL_GRACE_S": None,
            "SDLAB_TEST_STOP_TIMEOUT_S": None,
            "SDLAB_TEST_SESSIONS_TIMEOUT_S": None,
            "SDLAB_TEST_UPLOAD_TIMEOUT_S": None,
            "SDLAB_TEST_DOWNLOAD_TIMEOUT_S": None,
            "SDLAB_TEST_SALVAGE_RESTART_TIMEOUT_S": None,
            "SDLAB_TEST_SALVAGE_EXEC_TIMEOUT_S": None,
        }

        # Default bounds include +70 s (60 s stop_timeout + 10 s kill_grace) for desk_run_vm_workdir_cleanup:
        # 1150 -> 1220 s stop_attempted_by_s, 1290 -> 1360 s cleanup_bound_s.
        env_dry = self.make_env("n4_dry")
        out_dry = self.tmp_path / "out_n4_dry"
        res_dry = env_dry.run_launch(
            [
                "--project",
                str(proj),
                "--caption-config",
                str(cap_cfg),
                "--out",
                str(out_dry),
                "--card",
                "L4",
                "--cap",
                "5.0",
                "--max-rate",
                "6.0",
                "--wall-s",
                "300",
                "--margin",
                "0.5",
                "--dry",
            ],
            extra_env=prod_timing_env,
        )
        self.assertEqual(res_dry.returncode, 0, msg=res_dry.stderr)
        plan = json.loads((out_dry / "dry_run_plan.json").read_text(encoding="utf-8"))
        self.assertEqual(plan["cap_budget"]["cleanup_bound_s"], 1360)
        self.assertEqual(plan.get("cleanup_stop_attempted_by_s"), 1220)

        # At max_rate=6.0 CCU/hr, margin=0.0, min_runtime_s=60:
        # 1290 s needed 2.25 CCU (so --cap 2.30 passed at 393a6e4), whereas 1360 s needs 2.3667 CCU -> refuses!
        env_tight = self.make_env("n4_tight")
        out_tight = self.tmp_path / "out_n4_tight"
        res_tight = env_tight.run_launch(
            [
                "--project",
                str(proj),
                "--caption-config",
                str(cap_cfg),
                "--out",
                str(out_tight),
                "--card",
                "L4",
                "--cap",
                "2.30",
                "--max-rate",
                "6.0",
                "--wall-s",
                "300",
                "--margin",
                "0.0",
                "--min-runtime-s",
                "60",
                "--dry",
            ],
            extra_env=prod_timing_env,
        )
        self.assertNotEqual(res_tight.returncode, 0)
        self.assertIn("CAP_TOO_SMALL_FOR_CLEANUP", res_tight.stderr)

        # Live window.json also records 1220 / 1360 at production timing defaults.
        env_win = self.make_env("n4_win")
        out_win = self.tmp_path / "out_n4_win"
        res_win = env_win.run_launch(
            [
                "--project",
                str(proj),
                "--caption-config",
                str(cap_cfg),
                "--out",
                str(out_win),
                "--stub",
                "--card",
                "CPU-high-mem",
                "--cap",
                "5.0",
                "--wall-s",
                "300",
                "--margin",
                "0.5",
            ],
            extra_env=prod_timing_env,
        )
        self.assertEqual(res_win.returncode, 0, msg=res_win.stderr)
        win = json.loads((out_win / "window.json").read_text(encoding="utf-8"))
        self.assertEqual(win["cleanup_stop_attempted_by_s"], 1220)
        self.assertEqual(win["cleanup_bound_s"], 1360)

    def test_n6_model_load_stall_allowance_and_desk_index_report_timeout(self) -> None:
        proj, cap_cfg = self.make_project("n6_src", flagged_count=1, unflagged_count=1)

        # 1) Separate loading_model allowance lets a slow weight load pass even with a tight 0.15 s draft heartbeat.
        env_ok = self.make_env("n6_load_ok", config={"model_load_sleep_s": 0.45})
        out_ok = self.tmp_path / "out_n6_load_ok"
        res_ok = env_ok.run_launch(
            [
                "--project",
                str(proj),
                "--caption-config",
                str(cap_cfg),
                "--out",
                str(out_ok),
                "--card",
                "L4",
                "--cap",
                "5.0",
                "--wall-s",
                "300",
                "--margin",
                "0.5",
                "--heartbeat-stall-s",
                "0.15",
                "--model-load-stall-s",
                "3.0",
            ]
        )
        self.assertEqual(res_ok.returncode, 0, msg=res_ok.stderr)

        # 2) Exceeding --model-load-stall-s during loading_model refuses MODEL_LOAD_STALLED.
        env_stall = self.make_env("n6_load_stall", config={"model_load_sleep_s": 0.6})
        out_stall = self.tmp_path / "out_n6_load_stall"
        res_stall = env_stall.run_launch(
            [
                "--project",
                str(proj),
                "--caption-config",
                str(cap_cfg),
                "--out",
                str(out_stall),
                "--card",
                "L4",
                "--cap",
                "5.0",
                "--wall-s",
                "300",
                "--margin",
                "0.5",
                "--heartbeat-stall-s",
                "10",
                "--model-load-stall-s",
                "0.15",
            ]
        )
        self.assertNotEqual(res_stall.returncode, 0)
        win_stall = json.loads((out_stall / "window.json").read_text(encoding="utf-8"))
        self.assertEqual(win_stall["refusal_code"], "MODEL_LOAD_STALLED")

        # 3) Hung `index` and `report` subcommands time out cleanly with DESK_COMMAND_TIMEOUT.
        for subcmd in ("index", "report"):
            env_to = self.make_env(
                f"n6_to_{subcmd}",
                config={"hang_desk_subcmd": subcmd, "hang_desk_sleep_s": 5.0},
            )
            out_to = self.tmp_path / f"out_n6_to_{subcmd}"
            res_to = env_to.run_launch(
                [
                    "--project",
                    str(proj),
                    "--caption-config",
                    str(cap_cfg),
                    "--out",
                    str(out_to),
                    "--stub",
                    "--card",
                    "CPU-high-mem",
                    "--cap",
                    "5.0",
                    "--wall-s",
                    "300",
                    "--margin",
                    "0.5",
                    "--desk-cmd-timeout-s",
                    "0.2",
                ],
                timeout=30.0,
            )
            self.assertNotEqual(res_to.returncode, 0)
            win_to = json.loads((out_to / "window.json").read_text(encoding="utf-8"))
            self.assertEqual(win_to["refusal_code"], "DESK_COMMAND_TIMEOUT")

    def test_bundle_content_type_magic_bytes_guard_rejects_disguised_images(self) -> None:
        import io

        proj, cap_cfg = self.make_project("magic_src", flagged_count=1, unflagged_count=1)
        env = self.make_env("magic_env", config={"inject_disguised_image_in_desk_out": True})
        out_dir = self.tmp_path / "out_magic"
        res = env.run_launch(
            [
                "--project",
                str(proj),
                "--caption-config",
                str(cap_cfg),
                "--out",
                str(out_dir),
                "--stub",
                "--card",
                "CPU-high-mem",
                "--cap",
                "5.0",
                "--wall-s",
                "300",
                "--margin",
                "0.5",
            ]
        )
        self.assertNotEqual(res.returncode, 0)
        win = json.loads((out_dir / "window.json").read_text(encoding="utf-8"))
        self.assertEqual(win["refusal_code"], "PRIVATE_IMAGE_IN_BUNDLE")

        # Direct unit check across PNG, JPEG, WebP, GIF, BMP, and TIFF magic headers disguised as .dat files.
        helper = _load_module("desk_draft_helper_magic_test", DESK_DRAFT_HELPER_PY)
        samples = {
            "png": b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR",
            "jpeg": b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01",
            "gif": b"GIF89a\x01\x00\x01\x00\x80\x00\x00",
            "webp": b"RIFF\x24\x00\x00\x00WEBP",
            "bmp": b"BM\x36\x00\x00\x00\x00\x00\x00\x00\x36\x00\x00\x00",
            "tiff": b"II*\x00\x08\x00\x00\x00",
        }
        for fmt_name, header in samples.items():
            case_out = self.tmp_path / f"magic_unit_{fmt_name}"
            case_out.mkdir(parents=True, exist_ok=True)
            bundle_p = case_out / "disguised.tar.gz"
            with tarfile.open(bundle_p, "w:gz") as tf:
                ti = tarfile.TarInfo(name=f"desk_out/logs/disguised_{fmt_name}.dat")
                ti.size = len(header)
                tf.addfile(ti, io.BytesIO(header))
            ns = mock.MagicMock(out_dir=str(case_out), bundle_path=str(bundle_p), salvage_only=True)
            with self.assertRaises(helper.RefusalError) as ctx:
                helper.cmd_extract_and_verify_outputs(ns)
            self.assertEqual(ctx.exception.code, "PRIVATE_IMAGE_IN_BUNDLE")


class PreL4HardeningTests(ColabTempTestCase):
    """Five fixes made before the first real L4 window (private training images go to a Colab VM)."""

    FORBIDDEN_RIG_PATH_PATTERNS = (
        re.compile(r"/home/[A-Za-z0-9_.-]+"),
        re.compile(r"/mnt/[A-Za-z]/Users"),
        re.compile(r"[A-Za-z]:[\\/]+Users", re.IGNORECASE),
        re.compile(r"/Users/[A-Za-z0-9_.-]+"),
        re.compile(r"\.config/dataset-desk"),
    )

    def _gpu_args(self, proj: pathlib.Path, cap_cfg: pathlib.Path, out_dir: pathlib.Path, *extra: str) -> list[str]:
        return [
            "--project", str(proj), "--caption-config", str(cap_cfg), "--out", str(out_dir),
            "--card", "L4", "--cap", "5.0", "--wall-s", "300", "--margin", "0.5", *extra,
        ]

    def _stub_args(self, proj: pathlib.Path, cap_cfg: pathlib.Path, out_dir: pathlib.Path) -> list[str]:
        return [
            "--project", str(proj), "--caption-config", str(cap_cfg), "--out", str(out_dir), "--stub",
            "--card", "CPU-high-mem", "--cap", "5.0", "--wall-s", "300", "--margin", "0.5",
        ]

    def _assert_no_rig_paths(self, out_dir: pathlib.Path, label: str) -> int:
        scanned = 0
        for f in sorted(out_dir.rglob("*")):
            if not f.is_file() or f.name.endswith((".tar", ".gz", ".whl", ".safetensors", ".png")):
                continue
            raw = f.read_bytes()
            if b"\x00" in raw:
                continue
            text = raw.decode("utf-8", errors="replace")
            for pat in self.FORBIDDEN_RIG_PATH_PATTERNS:
                m = pat.search(text)
                self.assertIsNone(m, msg=f"{label}: {f.relative_to(out_dir)} names a rig path {m and m.group(0)!r}")
            scanned += 1
        return scanned

    # ---- fix 1: kill only our own descendants (launch_desk_draft.sh desk_kill_tracked_children) ----

    @unittest.skipUnless(os.name == "posix", "needs /proc")
    def test_tracked_pid_kill_spares_unrelated_process_and_kills_own_descendants(self) -> None:
        import subprocess
        import time

        text = LAUNCH_SH.read_text(encoding="utf-8")
        starts = [text.index(m) for m in ("desk_pid_stat_fields() {", "desk_kill_tracked_children() {") if m in text]
        funcs = text[min(starts) : text.index("cleanup_on_exit() {")]
        outd = self.tmp_path / "kill_out"
        outd.mkdir()
        driver = (
            "set -u\n"
            + funcs
            + f'OUT_DIR="{outd}"\n'
            + "sleep 300 & CHILD=$!\n"
            + "bash -c 'sleep 300 & echo $! > \"$0\"; wait' \"${OUT_DIR}/grandchild.pid\" & MID=$!\n"
            + 'for _ in $(seq 1 100); do [ -s "${OUT_DIR}/grandchild.pid" ] && break; sleep 0.1; done\n'
            + 'GC="$(cat "${OUT_DIR}/grandchild.pid")"\n'
            + 'echo ready > "${OUT_DIR}/ready"\n'
            + 'for _ in $(seq 1 200); do [ -s "${OUT_DIR}/unrelated.pid" ] && break; sleep 0.1; done\n'
            + 'printf "%s\\n%s\\n%s\\n999999\\n" "${CHILD}" "${MID}" "$(cat "${OUT_DIR}/unrelated.pid")" > "${OUT_DIR}/.watcher_tracked_pids.txt"\n'
            + 'printf "%s\\n" "${GC}" >> "${OUT_DIR}/.watcher_tracked_pids.txt"\n'
            + "desk_kill_tracked_children\n"
            + "sleep 0.5\n"
            + 'for p in "${CHILD}" "${MID}" "${GC}"; do kill -0 "${p}" 2>/dev/null && echo "alive:${p}" || echo "dead:${p}"; done\n'
        )
        proc = subprocess.Popen(["bash", "-c", driver], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        unrelated = None
        try:
            for _ in range(200):
                if (outd / "ready").exists():
                    break
                time.sleep(0.1)
            # Started after the launcher, so only the ancestry walk (not the start-time check) can spare it.
            unrelated = subprocess.Popen(["sleep", "300"], start_new_session=True)
            (outd / "unrelated.pid").write_text(f"{unrelated.pid}\n", encoding="utf-8")
            stdout, stderr = proc.communicate(timeout=60)
            self.assertEqual(proc.returncode, 0, msg=stderr)
            self.assertIsNone(unrelated.poll(), msg="an unrelated process holding a tracked PID was killed")
            self.assertEqual(sum(1 for ln in stdout.splitlines() if ln.startswith("dead:")), 3, msg=stdout)
            guards = (outd / "guards.log").read_text(encoding="utf-8")
            self.assertIn(f"TRACKED_PID_SKIPPED pid={unrelated.pid} ", guards)
        finally:
            if unrelated is not None:
                unrelated.kill()
                unrelated.wait()
            if proc.poll() is None:
                proc.kill()
                proc.wait()

    @unittest.skipUnless(os.name == "posix", "needs /proc")
    def test_vendored_watcher_guard_signals_only_verified_descendants(self) -> None:
        import subprocess
        import time

        text = COLAB_WINDOW_SH.read_text(encoding="utf-8")
        start = text.index("# --- desk-local patch (dataset-desk")
        end = text.index("# --- end desk-local patch")
        guard_src = text[start:end]
        self.assertEqual(text.count("def is_our_descendant(pid):"), 3, msg="all three watcher snippets carry the guard")
        outd = self.tmp_path / "watch_out"
        outd.mkdir()
        mid = subprocess.Popen(["bash", "-c", "sleep 300 & echo $! > \"$0\"; wait", str(outd / "gc.pid")])
        unrelated = subprocess.Popen(["sleep", "300"], start_new_session=True)
        try:
            for _ in range(100):
                if (outd / "gc.pid").exists() and (outd / "gc.pid").read_text().strip():
                    break
                time.sleep(0.1)
            grandchild = int((outd / "gc.pid").read_text().strip())
            from pathlib import Path

            ns = {"Path": Path, "main_pid": mid.pid, "tracked_p": outd / ".watcher_tracked_pids.txt"}
            exec(compile(guard_src, "guard", "exec"), ns)
            self.assertTrue(ns["is_our_descendant"](grandchild))
            self.assertFalse(ns["is_our_descendant"](unrelated.pid))
            self.assertIn(f"pid={unrelated.pid} ", (outd / "guards.log").read_text(encoding="utf-8"))
            # Identity is remembered: a verified child that is later orphaned still counts, a recycled PID does not.
            self.assertIn(str(grandchild), (outd / ".watcher_verified_pids.txt").read_text(encoding="utf-8"))
        finally:
            for pr in (unrelated, mid):
                pr.kill()
                pr.wait()
            try:
                os.kill(int((outd / "gc.pid").read_text().strip()), 9)
            except (OSError, ValueError):
                pass

    # ---- fix 2: no absolute rig paths in any kept file ----

    def test_no_rig_paths_in_any_kept_file_on_every_exit_path(self) -> None:
        home_root = self.tmp_path / "home" / "mikey"
        studio_cfg = "/home/mikey/.config/dataset-desk/config.json"
        proj, cap_cfg = fake_colab_desk.make_synthetic_project(home_root / "src", flagged_count=2, unflagged_count=1)
        total = 0

        # (a) pool-config failure at preflight (guards.log + helper refusal)
        env = self.make_env("rp_pool")
        out_a = home_root / "out_pool"
        res = env.run_launch(
            self._gpu_args(proj, cap_cfg, out_a),
            extra_env={"DATASET_DESK_USER_CONFIG": studio_cfg, "DESK_TEST_ISOLATE_USER_CONFIG": "0"},
        )
        self.assertEqual(res.returncode, 2, msg=res.stderr)
        self.assertIn("user config file", (out_a / "guards.log").read_text(encoding="utf-8"))
        total += self._assert_no_rig_paths(out_a, "pool-config failure")

        # (b) upload failure (colab_window.sh UPLOAD_FAILED -> guards.log + window.json refusal why)
        env = self.make_env("rp_up", config={"upload_fail_always": True})
        out_b = home_root / "out_upload"
        res = env.run_launch(self._gpu_args(proj, cap_cfg, out_b))
        self.assertNotEqual(res.returncode, 0)
        self.assertEqual(json.loads((out_b / "window.json").read_text(encoding="utf-8"))["refusal_code"], "UPLOAD_FAILED")
        total += self._assert_no_rig_paths(out_b, "upload failure")

        # (c) dry run and (d) full stub run
        env = self.make_env("rp_dry")
        out_c = home_root / "out_dry"
        res = env.run_launch(self._gpu_args(proj, cap_cfg, out_c, "--dry"))
        self.assertEqual(res.returncode, 0, msg=res.stderr)
        total += self._assert_no_rig_paths(out_c, "dry run")

        env = self.make_env("rp_stub")
        out_d = home_root / "out_stub"
        res = env.run_launch(self._stub_args(proj, cap_cfg, out_d))
        self.assertEqual(res.returncode, 0, msg=res.stderr)
        total += self._assert_no_rig_paths(out_d, "stub run")
        self.assertGreater(total, 20)

    def test_scrub_covers_case_spaces_wsl_unc_url_encoding_and_suffixless_files(self) -> None:
        helper = _load_module("desk_draft_helper_scrub2_test", DESK_DRAFT_HELPER_PY)
        cases = [
            "C:\\users\\mikey\\x",
            "/mnt/c/users/mikey/x",
            "C:\\Users\\John Smith\\x",
            "/Users/John Smith/x",
            "\\\\wsl.localhost\\Ubuntu\\home\\mikey\\x",
            "\\\\wsl$\\Ubuntu\\home\\mikey",
            "C%3A%5CUsers%5Cmikey%5Cx",
            "%2Fhome%2Fmikey%2Fx",
            "file:///C:/Users/mikey/x",
        ]
        for raw in cases:
            cleaned = helper.scrub_rig_paths(f"see {raw} here")
            self.assertNotRegex(cleaned, r"(?i)mikey|john|smith|ubuntu", msg=f"{raw!r} -> {cleaned!r}")
        out_dir = self.tmp_path / "scrub_sfx"
        out_dir.mkdir()
        (out_dir / "noext").write_text("error at /home/mikey/x\n", encoding="utf-8")
        for name in ("a.err", "a.out", "a.html", "a.ini", "a.cfg"):
            (out_dir / name).write_text("path C:\\Users\\mikey\\z\n", encoding="utf-8")
        (out_dir / "blob.bin").write_bytes(b"\x00\x01/home/mikey/x")
        ns = mock.MagicMock(out_dir=str(out_dir), root=[])
        self.assertEqual(helper.cmd_scrub_kept_paths(ns), 0)
        for name in ("noext", "a.err", "a.out", "a.html", "a.ini", "a.cfg"):
            self.assertNotIn("mikey", (out_dir / name).read_text(encoding="utf-8"), msg=name)
        report = json.loads((out_dir / "scrub_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["skipped"], ["blob.bin (binary)"])

    def test_watcher_exit_path_scrubs_last_and_startup_sweeps_earlier_runs(self) -> None:
        import subprocess

        text = LAUNCH_SH.read_text(encoding="utf-8")
        watcher = text[text.index("colab_run_wall_watcher \\") : text.index("WALL_WATCHER_PID=\"$!\"", text.index("colab_run_wall_watcher \\"))]
        self.assertLess(watcher.index("desk_clean_rig_staging"), watcher.index("desk_scrub_kept_paths"))
        self.assertIn("desk_sweep_earlier_runs", text)

        # What the watcher writes after the launcher's own scrub: window.json, outcome, guards.log, stop.log.
        out_dir = self.tmp_path / "watch_scrub" / "run_a"
        out_dir.mkdir(parents=True)
        for name, body in (
            ("window.json", '{"refusal_why": "upload /home/mikey/p/x.png failed"}'),
            ("outcome.draft.json", '{"why": "C:\\\\Users\\\\mikey\\\\p"}'),
            ("guards.log", "WALL_WATCHER_ESCALATED /mnt/c/Users/mikey/p\n"),
            ("stop.log", "stopped from /home/mikey\n"),
        ):
            (out_dir / name).write_text(body, encoding="utf-8")
        funcs = text[text.index("desk_scrub_kept_paths() {") : text.index("# desk_sweep_earlier_runs")]
        driver = (
            f'OUT_DIR="{out_dir}"; PY_BIN="{sys.executable}"; HELPER_PY="{DESK_DRAFT_HELPER_PY}"\n'
            + 'PROJECT_DIR=""; REPO_ROOT="/nonexistent-repo"\n'
            + funcs
            + "desk_scrub_kept_paths\n"
        )
        res = subprocess.run(["bash", "-c", driver], capture_output=True, text=True, timeout=60, check=False)
        self.assertEqual(res.returncode, 0, msg=res.stderr)
        self._assert_no_rig_paths(out_dir, "watcher exit scrub")

        # An earlier desk run folder left unscrubbed (SIGKILL) is swept by the next run's startup.
        old_run = self.tmp_path / "watch_scrub" / "run_old"
        old_run.mkdir()
        (old_run / "window.json").write_text('{"why": "/home/mikey/x"}', encoding="utf-8")
        (old_run / "remote_desk_draft.sh").write_text("echo /home/mikey\n", encoding="utf-8")
        other = self.tmp_path / "watch_scrub" / "not_a_run"
        other.mkdir()
        (other / "notes.txt").write_text("/home/mikey/keep\n", encoding="utf-8")
        helper = _load_module("desk_draft_helper_sweep_test", DESK_DRAFT_HELPER_PY)
        ns = types.SimpleNamespace(parent=str(out_dir.parent), exclude=str(out_dir))
        self.assertEqual(helper.cmd_sweep_unscrubbed(ns), 0)
        self._assert_no_rig_paths(old_run, "startup sweep")
        self.assertIn("/home/mikey/keep", (other / "notes.txt").read_text(encoding="utf-8"))

    def test_scrub_rig_paths_unit(self) -> None:
        helper = _load_module("desk_draft_helper_scrub_test", DESK_DRAFT_HELPER_PY)
        cases = {
            "/home/mikey/x/y": "<rig>/x/y",
            "C:\\Users\\mikey\\a": "<rig>\\a",
            "C:\\\\Users\\\\mikey\\\\a": "<rig>\\\\a",
            "C:/Users/mikey/a": "<rig>/a",
            "/mnt/c/Users/mikey/z": "<rig>/z",
            "/mnt/e/AI/x": "<rig>/AI/x",
            "https://example.test/a": "https://example.test/a",
        }
        for raw, want in cases.items():
            self.assertEqual(helper.scrub_rig_paths(raw), want)
        self.assertEqual(helper.scrub_rig_paths("/tmp/o/guards.log", [("/tmp/o", "<out>")]), "<out>/guards.log")

    # ---- fix 3: rejected/ never uploads; every uploaded asset is an image ----

    def _sidecar_project(self, name: str) -> pathlib.Path:
        from PIL import Image

        proj = self.tmp_path / name
        (proj / "rejected").mkdir(parents=True)
        (proj / "Rejected" / "deep").mkdir(parents=True)
        (proj / "keep").mkdir(parents=True)
        for rel in ("rejected/x", "Rejected/deep/z", "keep/y"):
            Image.new("RGB", (16, 16), (10, 20, 30)).save(proj / f"{rel}.png")
            (proj / f"{rel}.txt").write_text("a caption", encoding="utf-8")
        return proj

    def test_rejected_directory_never_uploads_in_sidecar_and_record_modes(self) -> None:
        from dataset_desk import description_gate, desk

        helper = _load_module("desk_draft_helper_rejected_test", DESK_DRAFT_HELPER_PY)
        proj = self._sidecar_project("sc_rejected")
        rel_files, count = helper._collect_compose_upload_files(proj, desk, description_gate)
        rels = [r for r, _ in rel_files]
        self.assertEqual(count, 1, msg=rels)
        self.assertEqual(sorted(rels), ["keep/y.png", "keep/y.txt"])
        for r in rels:
            self.assertFalse(helper.has_rejected_component(r))

        # Record mode: an approved record whose asset sits under rejected/ is excluded.
        proj2, _cfg = self.make_project("rec_rejected", flagged_count=1, unflagged_count=1)
        (proj2 / "outputs" / "rejected").mkdir(parents=True)
        shutil.copy2(proj2 / "outputs" / "wave-1" / "flagged_00.png", proj2 / "outputs" / "rejected" / "bad.png")
        (proj2 / "records" / "bad.json").write_text(
            json.dumps(
                {"id": "bad", "asset_path": "outputs/rejected/bad.png", "judgment": {"status": "approved", "explanation": "x"}}
            ),
            encoding="utf-8",
        )
        rel2, _n = helper._collect_compose_upload_files(proj2, desk, description_gate)
        self.assertFalse(any("rejected" in r.lower() for r, _ in rel2), msg=[r for r, _ in rel2])

    @unittest.skipUnless(hasattr(os, "symlink"), "needs symlinks")
    def test_symlink_into_rejected_never_uploads_and_aliases_are_caught(self) -> None:
        from dataset_desk import description_gate, desk

        helper = _load_module("desk_draft_helper_symlink_test", DESK_DRAFT_HELPER_PY)
        for alias in ("rejected.", "REJECTED", "Rejected ", "REJECT~1"):
            self.assertTrue(helper.has_rejected_component(f"a/{alias}/b.png"), msg=alias)
        self.assertFalse(helper.has_rejected_component("a/rejection/b.png"))

        proj, _cfg = self.make_project("sym_src", flagged_count=1, unflagged_count=1)
        (proj / "outputs" / "rejected").mkdir(parents=True)
        shutil.copy2(proj / "outputs" / "wave-1" / "flagged_00.png", proj / "outputs" / "rejected" / "bad.png")
        try:
            os.symlink(proj / "outputs" / "rejected" / "bad.png", proj / "outputs" / "wave-1" / "linked.png")
            os.symlink(proj / "outputs" / "rejected", proj / "outputs" / "wave-1" / "linkdir", target_is_directory=True)
        except OSError:
            self.skipTest("symlinks not permitted here")
        for rec_id, asset in (("lnk_file", "outputs/wave-1/linked.png"), ("lnk_dir", "outputs/wave-1/linkdir/bad.png")):
            (proj / "records" / f"{rec_id}.json").write_text(
                json.dumps({"id": rec_id, "asset_path": asset, "judgment": {"status": "approved", "explanation": "x"}}),
                encoding="utf-8",
            )
        rel_files, _n = helper._collect_compose_upload_files(proj, desk, description_gate)
        rels = [r for r, _ in rel_files]
        self.assertFalse(any("linked" in r or "linkdir" in r or "rejected" in r for r in rels), msg=rels)
        for _r, real in rel_files:
            self.assertNotIn("rejected", str(real.relative_to(proj.resolve())).lower())

    def test_non_image_asset_is_refused_and_avif_heic_are_detected(self) -> None:
        from dataset_desk import description_gate, desk

        helper = _load_module("desk_draft_helper_magic2_test", DESK_DRAFT_HELPER_PY)
        proj, _cfg = self.make_project("notimg", flagged_count=1, unflagged_count=1)
        (proj / "outputs" / "wave-1" / "flagged_00.png").write_bytes(b"this is a text file, not a png\n")
        with self.assertRaises(helper.RefusalError) as ctx:
            helper._collect_compose_upload_files(proj, desk, description_gate)
        self.assertEqual(ctx.exception.code, "PROJECT_ASSET_NOT_IMAGE")
        self.assertEqual(helper.detect_image_magic_bytes(b"\x00\x00\x00\x1cftypavif\x00\x00\x00\x00avifmif1"), "image/avif")
        self.assertEqual(helper.detect_image_magic_bytes(b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00mif1heic"), "image/heic")
        self.assertIsNone(helper.detect_image_magic_bytes(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00isom"))

    def test_upload_tar_carries_scrubbed_copies_of_records_and_wave_metadata(self) -> None:
        proj, cap_cfg = self.make_project("gen_src", flagged_count=1, unflagged_count=1)
        gen = proj / "outputs" / "wave-1" / "generation.json"
        gen.write_text(json.dumps({"output_dir": "/home/mikey/work/wave-1", "src": "C:\\Users\\mikey\\w"}), encoding="utf-8")
        helper = _load_module("desk_draft_helper_gen_test", DESK_DRAFT_HELPER_PY)
        out_dir = self.tmp_path / "gen_out"
        out_dir.mkdir()
        ns = mock.MagicMock(
            repo_root=str(REPO_ROOT), project=str(proj), caption_config=str(cap_cfg), out_dir=str(out_dir),
            isolate_user_config=True,
        )
        self.assertEqual(helper.cmd_preflight_project(ns), 0)
        with tarfile.open(out_dir / "_rig_stage" / "project_input.tar") as tf:
            member = tf.extractfile("outputs/wave-1/generation.json")
            self.assertIsNotNone(member)
            body = member.read().decode("utf-8")
            png_member = tf.extractfile("outputs/wave-1/flagged_00.png")
            png_head = png_member.read(8)
        self.assertNotIn("mikey", body)
        self.assertEqual(png_head, bytes([0x89]) + b"PNG" + bytes([0x0D, 0x0A, 0x1A, 0x0A]))
        manifest = json.loads((out_dir / "project_manifest.json").read_text(encoding="utf-8"))
        entry = next(e for e in manifest["files"] if e["path"] == "outputs/wave-1/generation.json")
        self.assertEqual(entry["sha256"], hashlib.sha256(body.encode("utf-8")).hexdigest())

    # ---- fix 4: no image bundle left behind when the rig-side PRIVATE_IMAGE_IN_BUNDLE check fires ----

    def test_private_image_bundle_is_deleted_and_recorded_on_a_rig_trip(self) -> None:
        import io
        import subprocess

        helper = _load_module("desk_draft_helper_trip_test", DESK_DRAFT_HELPER_PY)
        out_dir = self.tmp_path / "out_trip"
        out_dir.mkdir()
        png = bytes([0x89]) + b"PNG" + bytes([0x0D, 0x0A, 0x1A, 0x0A]) + bytes(16)
        bundle = out_dir / "desk_draft_bundle.tar.gz"
        with tarfile.open(bundle, "w:gz") as tf:
            ti = tarfile.TarInfo(name="desk_out/logs/late_plate.dat")
            ti.size = len(png)
            tf.addfile(ti, io.BytesIO(png))
        want_sha, want_bytes = _sha256_file(bundle), bundle.stat().st_size
        (out_dir / ".sdlab_dl_tmp_1_2").mkdir()
        (out_dir / ".sdlab_dl_tmp_1_2" / "part.0000").write_bytes(png)

        ns = mock.MagicMock(out_dir=str(out_dir), bundle_path=str(bundle), salvage_only=True)
        with self.assertRaises(helper.RefusalError) as ctx:
            helper.cmd_extract_and_verify_outputs(ns)
        self.assertEqual(ctx.exception.code, "PRIVATE_IMAGE_IN_BUNDLE")
        rec = json.loads((out_dir / "private_image_bundle_refusal.json").read_text(encoding="utf-8"))
        self.assertEqual(rec["code"], "PRIVATE_IMAGE_IN_BUNDLE")
        self.assertEqual(rec["bundle_sha256"], want_sha)
        self.assertEqual(rec["bundle_bytes"], want_bytes)

        # The launcher's staging clean (every exit path calls it) then deletes the bundle and partial downloads.
        text = LAUNCH_SH.read_text(encoding="utf-8")
        funcs = text[text.index("desk_clean_rig_staging() {") : text.index("desk_pre_session_exit() {")]
        driver = f'OUT_DIR="{out_dir}"\n' + funcs.split("# desk_scrub_kept_paths")[0] + "\ndesk_clean_rig_staging\n"
        res = subprocess.run(["bash", "-c", driver], capture_output=True, text=True, timeout=60, check=False)
        self.assertEqual(res.returncode, 0, msg=res.stderr)
        self.assertFalse(bundle.exists(), msg="image bundle left on the rig")
        self.assertEqual(list(out_dir.glob(".sdlab_dl_tmp_*")), [])
        self.assertTrue((out_dir / "private_image_bundle_refusal.json").is_file())

        # Without the refusal record (an ordinary failure) the bundle is kept for salvage.
        keep_dir = self.tmp_path / "out_keep"
        keep_dir.mkdir()
        (keep_dir / "desk_draft_bundle.tar.gz").write_bytes(b"keep")
        driver2 = f'OUT_DIR="{keep_dir}"\n' + funcs.split("# desk_scrub_kept_paths")[0] + "\ndesk_clean_rig_staging\n"
        subprocess.run(["bash", "-c", driver2], capture_output=True, text=True, timeout=60, check=False)
        self.assertTrue((keep_dir / "desk_draft_bundle.tar.gz").exists())

    # ---- fix 6: every non-LFS weight file is verified against its git blob id ----

    def test_weight_download_verifies_four_lfs_and_twelve_git_blob_files(self) -> None:
        proj, cap_cfg = self.make_project("w6_src", flagged_count=2, unflagged_count=1)
        env = self.make_env("w6_ok")
        out_dir = self.tmp_path / "out_w6_ok"
        res = env.run_launch(self._gpu_args(proj, cap_cfg, out_dir))
        self.assertEqual(res.returncode, 0, msg=res.stderr[-800:])
        rec = json.loads((out_dir / "vm_state" / "weight_download.json").read_text(encoding="utf-8"))
        self.assertEqual(rec["sha_verified_count"], 4)
        self.assertEqual(rec["blob_verified_count"], 12)
        self.assertEqual(rec["file_count"], 16)

    def test_tampered_tokenizer_json_is_refused_by_blob_id(self) -> None:
        proj, cap_cfg = self.make_project("w6t_src", flagged_count=2, unflagged_count=1)
        env = self.make_env("w6_tamper", config={"weight_tamper_tokenizer": True})
        out_dir = self.tmp_path / "out_w6_tamper"
        res = env.run_launch(self._gpu_args(proj, cap_cfg, out_dir))
        self.assertNotEqual(res.returncode, 0)
        win = json.loads((out_dir / "window.json").read_text(encoding="utf-8"))
        self.assertEqual(win["refusal_code"], "WEIGHT_BLOB_MISMATCH")
        self.assertIn("tokenizer.json", str(win["why"]))

    def test_missing_pinned_weight_file_refuses(self) -> None:
        proj, cap_cfg = self.make_project("w6m_src", flagged_count=2, unflagged_count=1)
        env = self.make_env("w6_missing", config={"weight_omit_vocab": True})
        out_dir = self.tmp_path / "out_w6_missing"
        res = env.run_launch(self._gpu_args(proj, cap_cfg, out_dir))
        self.assertNotEqual(res.returncode, 0)
        win = json.loads((out_dir / "window.json").read_text(encoding="utf-8"))
        self.assertEqual(win["refusal_code"], "WEIGHT_FILE_MISSING")
        self.assertIn("vocab.json", str(win["why"]))

    def test_extra_file_beyond_the_sixteen_pinned_refuses(self) -> None:
        proj, cap_cfg = self.make_project("w6x_src", flagged_count=2, unflagged_count=1)
        env = self.make_env("w6_extra", config={"weight_extra_file": True})
        out_dir = self.tmp_path / "out_w6_extra"
        res = env.run_launch(self._gpu_args(proj, cap_cfg, out_dir))
        self.assertNotEqual(res.returncode, 0)
        win = json.loads((out_dir / "window.json").read_text(encoding="utf-8"))
        self.assertEqual(win["refusal_code"], "WEIGHT_UNEXPECTED_FILE")
        self.assertIn("extra.bin", str(win["why"]))

    def test_missing_pinned_weight_file_unit_checks(self) -> None:
        from types import SimpleNamespace

        helper = _load_module("desk_draft_helper_blob_test", DESK_DRAFT_HELPER_PY)
        self.assertEqual(len(helper.PINNED_QWEN3_VL_FILES), 16)
        self.assertEqual(sum(n.endswith(".safetensors") for n in helper.PINNED_QWEN3_VL_FILES), 4)
        f = self.tmp_path / "blob.txt"
        f.write_bytes(b"hello\n")
        self.assertEqual(helper.git_blob_sha1(f), "ce013625030ba8dba906f756967f9e9ca394464a")
        self.assertIsNone(helper._extract_expected_sibling_blob_id(SimpleNamespace(rfilename="x")))
        self.assertEqual(
            helper._extract_expected_sibling_blob_id(SimpleNamespace(blob_id="CE013625030BA8DBA906F756967F9E9CA394464A")),
            "ce013625030ba8dba906f756967f9e9ca394464a",
        )

    # ---- fix 5: an L4 opens Standard only ----

    def test_l4_opens_standard_and_high_mem_refuses_before_any_colab_call(self) -> None:
        proj, cap_cfg = self.make_project("l4_src", flagged_count=2, unflagged_count=1)
        for label, extra in (("default", []), ("explicit", ["--shape", "standard"])):
            env = self.make_env(f"l4_{label}")
            out_dir = self.tmp_path / f"out_l4_{label}"
            res = env.run_launch(self._gpu_args(proj, cap_cfg, out_dir, *extra))
            self.assertEqual(res.returncode, 0, msg=f"{label}: {res.stderr[-800:]}")
            news = [c for c in env.colab_calls() if c["subcmd"] == "new"]
            self.assertEqual(len(news), 1)
            self.assertNotIn("--high-mem", news[0]["argv"])
            self.assertEqual(env.session_state().get("shape_label"), "Standard")
            self.assertEqual((out_dir / "shape_requested.txt").read_text(encoding="utf-8").strip(), "standard")

        env = self.make_env("l4_hm")
        out_hm = self.tmp_path / "out_l4_hm"
        res = env.run_launch(self._gpu_args(proj, cap_cfg, out_hm, "--shape", "high-mem"))
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("SHAPE_HIGH_MEM_UNAVAILABLE", f"{res.stderr}\n{res.stdout}")
        self.assertEqual(env.colab_calls(), [])
        self.assertIn("SHAPE_HIGH_MEM_UNAVAILABLE", COLAB_WINDOW_SH.read_text(encoding="utf-8").split("REFUSAL_CODES = (")[1])

    # ---- parallel windows (polyglot-mcp PR #80, ported via Ripple): price by the window's own rate ----

    def _par_args(self, proj, cap_cfg, out_dir):
        return self._gpu_args(proj, cap_cfg, out_dir, "--max-rate", "2.25")

    def test_parallel_1_neighbours_opened_while_waiting_for_the_open_lock_are_in_the_pre_open_baseline(self) -> None:
        """A holder owns the open lock while the launcher waits, turns two neighbours on only after the launcher's
        first `colab usage` read, then releases. The old code priced from the stale launch-time read and refused
        RATE_ABOVE_MAX at the account's 5.88/h; the fix re-reads under the lock and prices 1.96/h."""
        import subprocess
        import time

        proj, cap_cfg = self.make_project("par1_src", flagged_count=2, unflagged_count=1)
        env = self.make_env("par1", config={"per_session_rate": 1.96, "neighbours": 2, "neighbours_when": "external"})
        out_dir = self.tmp_path / "out_par1"
        lock_path = env.base_dir / "open.lock"
        holding = self.tmp_path / "holder_holding.txt"
        calls = env.state_dir / "colab_calls.jsonl"
        holder_py = "\n".join([
            "import fcntl, os, sys, time",
            "lock, flag, on, calls = sys.argv[1:5]",
            "fh = open(lock, 'a')",
            "fcntl.flock(fh, fcntl.LOCK_EX)",
            "os.makedirs(lock + '.lockdir', exist_ok=True)",
            "open(flag, 'w').write('1')",
            "t0 = time.time()",
            "while time.time() - t0 < 90:",
            "    try:",
            "        if '\"subcmd\": \"usage\"' in open(calls, encoding='utf-8').read():",
            "            break",
            "    except OSError:",
            "        pass",
            "    time.sleep(0.1)",
            "time.sleep(1.5)",
            "open(on, 'w').write('1')",
            "os.rmdir(lock + '.lockdir')",
            "fcntl.flock(fh, fcntl.LOCK_UN)",
        ])
        env.state_dir.mkdir(parents=True, exist_ok=True)
        holder = subprocess.Popen(
            [sys.executable, "-c", holder_py, str(lock_path), str(holding), str(env.state_dir / "neighbours_on.txt"), str(calls)],
            stdin=subprocess.DEVNULL,
        )
        try:
            t0 = time.time()
            while not holding.exists() and time.time() - t0 < 10:
                time.sleep(0.1)
            self.assertTrue(holding.exists(), "the holder never took the open lock")
            res = env.run_launch(self._par_args(proj, cap_cfg, out_dir), timeout=240.0)
        finally:
            holder.wait(timeout=90)
        self.assertEqual(res.returncode, 0, msg=res.stderr[-1500:])
        rc = json.loads((out_dir / "rate_check.json").read_text(encoding="utf-8"))
        self.assertEqual(rc["decision"], "ok")
        self.assertAlmostEqual(rc["window_rate"], 1.96, places=2)
        win = json.loads((out_dir / "window.json").read_text(encoding="utf-8"))
        self.assertNotEqual(win.get("refusal_code"), "RATE_ABOVE_MAX")
        self.assertTrue(win["account_shared"])
        self.assertEqual(win["pre_open_assignments"], 2)
        self.assertEqual(win["spend_basis"], "window_rate_x_elapsed")
        guards = (out_dir / "guards.log").read_text(encoding="utf-8")
        self.assertIn("PRE_OPEN_UNDER_LOCK rate=3.92 assignments=2", guards)
        self.assertIn("OPEN_LOCK_RELEASED", guards)
        usage = (out_dir / "usage.log").read_text(encoding="utf-8").splitlines()
        self.assertIn("Usage rate: 0.00/hr", usage[:3])

    def test_parallel_2_neighbours_opened_outside_the_lock_make_the_rate_unattributable(self) -> None:
        proj, cap_cfg = self.make_project("par2_src", flagged_count=2, unflagged_count=1)
        env = self.make_env("par2", config={"per_session_rate": 1.96, "neighbours": 2, "neighbours_when": "after_open"})
        out_dir = self.tmp_path / "out_par2"
        res = env.run_launch(self._par_args(proj, cap_cfg, out_dir), timeout=240.0)
        self.assertNotEqual(res.returncode, 0, msg=res.stdout + res.stderr)
        win = json.loads((out_dir / "window.json").read_text(encoding="utf-8"))
        self.assertEqual(win["refusal_code"], "RATE_UNATTRIBUTABLE", msg=res.stderr[-1500:])
        self.assertTrue(win.get("why"))
        subcmds = [c["subcmd"] for c in env.colab_calls()]
        self.assertIn("stop", subcmds)
        # Refused before any private upload; the only exec is the VM workdir cleanup that precedes every stop.
        self.assertNotIn("upload", subcmds)
        self.assertFalse((out_dir / "vitals.log").exists() and "SDLAB_REMOTE_RC" in (out_dir / "vitals.log").read_text(encoding="utf-8"))
        self.assertEqual(win["spend_basis"], "max_rate_x_elapsed_no_stable_rate")

    def test_parallel_3_a_neighbours_spend_on_the_shared_balance_never_trips_this_gpu_windows_cap(self) -> None:
        proj, cap_cfg = self.make_project("par3_src", flagged_count=2, unflagged_count=1)
        env = self.make_env("par3", config={"balance_start": 120.0, "balance_active": 90.5})
        out_dir = self.tmp_path / "out_par3"
        res = env.run_launch(self._par_args(proj, cap_cfg, out_dir), timeout=240.0)
        self.assertEqual(res.returncode, 0, msg=res.stderr[-1500:])
        win = json.loads((out_dir / "window.json").read_text(encoding="utf-8"))
        self.assertNotEqual(win.get("refusal_code"), "CAP_GUARD_TRIPPED")
        self.assertEqual(win["account_balance_drop"], 29.5)
        self.assertEqual(win["spend_basis"], "window_rate_x_elapsed")
        self.assertLess(win["spend_ccu"], 1.0)
        self.assertTrue((out_dir / "account_balance.json").is_file())
        self.assertIn("spend_ccu_at_priced_rate", win)

    def test_parallel_4_update_window_rate_attributes_only_the_pre_open_count_plus_one(self) -> None:
        lg = _load_module("launch_guards_parallel_test", LAUNCH_GUARDS_PY)

        def run(reads, pre_rate, pre_n):
            st = {}
            for i, (acc, n) in enumerate(reads):
                st = lg.update_window_rate(st, account_rate=acc, pre_open_rate=pre_rate, at="t",
                                           epoch=10.0 * i, assignments=n, pre_open_assignments=pre_n)
            return st

        a = run([(20.1, 2), (30.15, 3), (30.15, 3)], 20.1, 2)
        self.assertEqual(a.get("best_stable"), 10.05)
        self.assertTrue(a["readings"][0].get("not_registered"))
        self.assertTrue(a.get("shared_account"))
        self.assertEqual(lg.window_rate_decision(a, 10.5), "ok")
        b = run([(30.15, 3), (30.15, 3)], 0.0, 0)
        self.assertIsNone(b.get("best_stable"))
        self.assertEqual(lg.window_rate_decision(b, 10.5), "unattributable")
        c = run([(10.05, 1), (10.05, 1)], 0.0, 0)
        self.assertEqual(c.get("best_stable"), 10.05)
        self.assertFalse(c.get("shared_account"))
        self.assertEqual(lg.window_rate_decision(c, 10.5), "ok")

    def test_parallel_5_lib_refuses_when_the_pre_open_read_fails_or_the_wall_is_short_after_the_lock(self) -> None:
        import subprocess

        env = self.make_env("par5")
        results = {}
        for name, setup in (
            ("wall", 'echo "$(( $(date -u +%s) - 590 ))" > "$O/wall_start_epoch.txt"; echo 600 > "$O/wall_limit_s.txt"'),
            ("preopen", "export SDLAB_FAULT_POLL_HELPER=empty"),
        ):
            out = self.tmp_path / f"par5_{name}"
            out.mkdir()
            for fname, val in (("gpu_rate_gate_armed.txt", "1\n"), ("max_rate.txt", "2.25\n"), ("shape_requested.txt", "standard\n")):
                (out / fname).write_text(val, encoding="utf-8")
            script = (
                f'set -u; export COLAB_CARD_CLASS=gpu PY_BIN="{sys.executable}"; O="{out}"; '
                f'source "{COLAB_WINDOW_SH}"; {setup}; '
                f'rc=0; ( colab_open_session_locked par5-{name} L4 "$O" ) || rc=$?; echo "rc=$rc"'
            )
            proc = subprocess.run(["bash", "-c", script], env=env.build_env(), stdin=subprocess.DEVNULL,
                                  capture_output=True, text=True, timeout=60, check=False)
            results[name] = (proc, out)
        self.assertIn("WALL_SHORT_AFTER_LOCK", (results["wall"][1] / "guards.log").read_text(encoding="utf-8"))
        self.assertIn("PRE_OPEN_READ_FAILED", (results["preopen"][1] / "guards.log").read_text(encoding="utf-8"))
        for name, (proc, out) in results.items():
            self.assertIn("rc=2", proc.stdout, msg=name + proc.stdout + proc.stderr)
            self.assertFalse((out / "session_open_started.txt").exists(), name)
        self.assertNotIn("new", [c["subcmd"] for c in env.colab_calls()])

    # ---- review round 3: JSON-safe scrub, normal-exit watcher, edge cases ----

    def test_scrub_keeps_json_and_jsonl_valid_around_escaped_quotes(self) -> None:
        helper = _load_module("desk_draft_helper_json_test", DESK_DRAFT_HELPER_PY)
        doc = {"m": 'HOME="/home/bob"', "w": 'C:\\Users\\bob\\x', "u": "/Users/Jane Doe/p", "n": ["/mnt/c/Users/bob/z"]}
        text = json.dumps(doc)
        cleaned, status = helper.scrub_text_for_file("a.json", text)
        self.assertEqual(status, "ok")
        parsed = json.loads(cleaned)
        self.assertNotRegex(json.dumps(parsed), r"(?i)bob|jane")
        jl = "\n".join(json.dumps(doc) for _ in range(3)) + "\n"
        cleaned_jl, status_jl = helper.scrub_text_for_file("a.jsonl", jl)
        self.assertEqual(status_jl, "ok")
        self.assertEqual(len([json.loads(ln) for ln in cleaned_jl.splitlines()]), 3)
        # A scrub that would break parsable JSON falls back to the decoded values; unparsable JSON keeps its original.
        self.assertEqual(helper.scrub_text_for_file("bad.json", '{"a": "/home/bob", ')[1], "json_error")
        out_dir = self.tmp_path / "json_scrub"
        out_dir.mkdir()
        (out_dir / "window.json").write_text(text, encoding="utf-8")
        (out_dir / "bad.json").write_text('{"a": "/home/bob", ', encoding="utf-8")
        (out_dir / "latin.txt").write_bytes("caf\xe9 /home/bob".encode("latin-1"))
        rep = helper._scrub_dir(out_dir, [])
        json.loads((out_dir / "window.json").read_text(encoding="utf-8"))
        self.assertIn("/home/bob", (out_dir / "bad.json").read_text(encoding="utf-8"))
        self.assertEqual(rep["scrub_errors"], ["bad.json (scrub would break the JSON; original kept)"])
        self.assertEqual((out_dir / "latin.txt").read_bytes(), "caf\xe9 /home/bob".encode("latin-1"))
        self.assertIn("latin.txt (not utf-8)", rep["skipped"])
        self.assertEqual([f.name for f in out_dir.iterdir() if ".scrub_tmp_" in f.name], [])

    def test_upload_refuses_json_the_scrub_cannot_verify_and_hard_linked_images(self) -> None:
        from dataset_desk import description_gate, desk

        helper = _load_module("desk_draft_helper_edge_test", DESK_DRAFT_HELPER_PY)
        proj, cap_cfg = self.make_project("edge_src", flagged_count=1, unflagged_count=1)
        out_dir = self.tmp_path / "edge_out"
        out_dir.mkdir()
        ns = types.SimpleNamespace(
            repo_root=str(REPO_ROOT), project=str(proj), caption_config=str(cap_cfg), out_dir=str(out_dir), isolate_user_config=True
        )
        (proj / "outputs" / "wave-1" / "generation.json").write_text('{"a": "/home/bob", ', encoding="utf-8")
        with self.assertRaises(helper.RefusalError) as ctx:
            helper.cmd_preflight_project(ns)
        self.assertEqual(ctx.exception.code, "UPLOAD_SCRUB_INVALID_JSON")
        (proj / "outputs" / "wave-1" / "generation.json").unlink()
        img = proj / "outputs" / "wave-1" / "flagged_00.png"
        try:
            os.link(img, proj / "outputs" / "wave-1" / "twin.png")
        except OSError:
            self.skipTest("hard links not permitted here")
        with self.assertRaises(helper.RefusalError) as ctx2:
            helper._collect_compose_upload_files(proj, desk, description_gate)
        self.assertEqual(ctx2.exception.code, "PROJECT_ASSET_HARDLINKED")

    def test_path_is_rejected_dotdot_name_and_fail_closed(self) -> None:
        helper = _load_module("desk_draft_helper_pr_test", DESK_DRAFT_HELPER_PY)
        proj = self.tmp_path / "pr_proj"
        (proj / "..old").mkdir(parents=True)
        (proj / "..old" / "a.png").write_bytes(b"x")
        self.assertFalse(helper.path_is_rejected(proj, "..old/a.png"))
        with mock.patch.object(pathlib.Path, "resolve", side_effect=OSError("boom")):
            self.assertTrue(helper.path_is_rejected(proj, "keep/a.png"))

    def test_sweep_logs_when_it_stops_at_the_entry_limit(self) -> None:
        helper = _load_module("desk_draft_helper_sweep2_test", DESK_DRAFT_HELPER_PY)
        parent = self.tmp_path / "many"
        parent.mkdir()
        for i in range(205):
            (parent / f"d{i:03d}").mkdir()
        mine = parent / "zz_mine"
        mine.mkdir()
        ns = types.SimpleNamespace(parent=str(parent), exclude=str(mine))
        self.assertEqual(helper.cmd_sweep_unscrubbed(ns), 0)
        self.assertIn("SWEEP_LIMIT_REACHED", (mine / "guards.log").read_text(encoding="utf-8"))

    def test_watcher_subshell_scrubs_only_on_escalation_and_a_normal_exit_keeps_closing_evidence(self) -> None:
        text = LAUNCH_SH.read_text(encoding="utf-8")
        a = text.index("colab_run_wall_watcher \\")
        watcher = text[a : text.index('WALL_WATCHER_PID="$!"', a)]
        guard = watcher.index('watcher_escalated.txt')
        self.assertLess(guard, watcher.index("desk_scrub_kept_paths"))
        proj, cap_cfg = self.make_project("norm_src", flagged_count=2, unflagged_count=1)
        env = self.make_env("norm_env")
        out_dir = self.tmp_path / "out_norm"
        res = env.run_launch(self._gpu_args(proj, cap_cfg, out_dir))
        self.assertEqual(res.returncode, 0, msg=res.stderr[-800:])
        self.assertIn("No active sessions.", (out_dir / "colab.log").read_text(encoding="utf-8"))
        self.assertIn("STOP_COMPLETED", (out_dir / "stop.log").read_text(encoding="utf-8") + (out_dir / "guards.log").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
