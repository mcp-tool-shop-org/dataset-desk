#!/usr/bin/env bash
# Launch kit for dataset-desk Colab caption-drafting lane (Qwen3-VL-8B-Instruct on L4+, CPU-high-mem stub first).
#
# Vendored window library source: style-dataset-lab main @ df9a1b3be2f8729a57776253e9b0cb32b7f5299c
#   - scripts/colab/colab_window.sh  sha256 db761fb414da915c8835bad8ef1544992e39aa5b3d636a1d81d17d231f26e974
#   - scripts/colab/launch_guards.py sha256 4c0c0ab6410eaa511a808d00f09a09d42d5d5b7d0504d85d58cfe3c65baeb085
#   - scripts/colab/remote_runner.py sha256 393929ed704ccbacaee2b590010e348a5d635cbdd70361209d3e0103d1fba6eb
#
# Modes:
#   --dry   Rig-only preflight, deterministic tar + wheel build, and cap arithmetic; zero colab CLI calls.
#   --stub  CPU-high-mem real install path (uv + CPython 3.12.15 + hash-locked deps + desk wheel + Qwen3-VL
#           weight download & sha check + offline AutoConfig load), then compose --draft fake + report on
#           2 synthetic images, verified fetch, VM workdir cleanup, and stop.
#   (default GPU run) L4/A100/H100 window: GPU/disk/RAM guards before download, real uv + wheel install,
#           local-disk Qwen3-VL weight fetch with 429 backoff + HF_HUB_OFFLINE=1, compose --draft qwen3-vl
#           (flagged records only, per-record heartbeat) + report (--backend none), verified fetch, VM workdir
#           cleanup, and stop. Never accepts --backend and never constructs an embedding backend.
set -euo pipefail

SDLAB_VENDORED_COMMIT="df9a1b3be2f8729a57776253e9b0cb32b7f5299c"
PINNED_UV_VERSION="0.13.5"
PINNED_UV_WHEEL_SHA256="fa8d26e08e34c29c0c4e820cf3181757086d547240b4f4d31ff1d8bd810ca946"
PINNED_PYTHON_VERSION="3.12.15"
PINNED_QWEN3_VL_REPO="Qwen/Qwen3-VL-8B-Instruct"
PINNED_QWEN3_VL_REVISION="0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"

PINNED_COLAB_WINDOW_SHA="db761fb414da915c8835bad8ef1544992e39aa5b3d636a1d81d17d231f26e974"
PINNED_LAUNCH_GUARDS_SHA="4c0c0ab6410eaa511a808d00f09a09d42d5d5b7d0504d85d58cfe3c65baeb085"
PINNED_REMOTE_RUNNER_SHA="393929ed704ccbacaee2b590010e348a5d635cbdd70361209d3e0103d1fba6eb"
PINNED_DESK_BUNDLE_NOW_SHA="b1e83c20f046716433d927fa442e43fe3e6c821c90adc84cd87cfccae9aed1ac"
PINNED_DESK_DRAFT_HELPER_SHA="2073d17edc976186632d5d5b647f3605d5bfe44a88968d6c8d367d29a8320b15"
PINNED_DESK_LOCK_SHA="7d718664b0289b53532ff50541e63942166c1921a2f52cb0e48900958bf90510"
PINNED_SELF_SHA="39c403cc5311ceb454dd7ce1b99aa6de21ba8853f9f108e3527bcbed739ef592"

if pwd -W >/dev/null 2>&1; then
  export MSYS2_ARG_CONV_EXCL="/content"
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -W)"
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -W)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
SELF_SH="${SCRIPT_DIR}/launch_desk_draft.sh"
# shellcheck source=scripts/colab/colab_window.sh
source "${SCRIPT_DIR}/colab_window.sh"

# N4: count desk_run_vm_workdir_cleanup (up to COLAB_STOP_TIMEOUT_S + COLAB_KILL_GRACE_S = 70 s default)
# in COLAB_CLEANUP_STOP_ATTEMPTED_BY_S (1150 -> 1220 s default) and COLAB_CLEANUP_BOUND_S (1290 -> 1360 s default),
# and therefore in --dry, pre-open cap gates, live cap/wall guards, and window.json without modifying colab_window.sh.
eval "_sdlab_orig_$(declare -f colab_compute_cleanup_bounds)"
colab_compute_cleanup_bounds() {
  local lane="$1"
  _sdlab_orig_colab_compute_cleanup_bounds "${lane}"
  local stop_timeout_s="${COLAB_STOP_TIMEOUT_S:-60}"
  local kill_grace_s="${COLAB_KILL_GRACE_S:-10}"
  local desk_vm_cleanup_s=$(( stop_timeout_s + kill_grace_s ))
  export DESK_VM_CLEANUP_BOUND_S="${desk_vm_cleanup_s}"
  export COLAB_CLEANUP_STOP_ATTEMPTED_BY_S=$(( COLAB_CLEANUP_STOP_ATTEMPTED_BY_S + desk_vm_cleanup_s ))
  export COLAB_CLEANUP_BOUND_S=$(( COLAB_CLEANUP_BOUND_S + desk_vm_cleanup_s ))
}

eval "_sdlab_orig_$(declare -f colab_write_window_and_outcome)"
colab_write_window_and_outcome() {
  local py_bin="$1"
  local out_dir="$2"
  _sdlab_orig_colab_write_window_and_outcome "$@"
  colab_compute_cleanup_bounds "train"
  "${py_bin}" - "${out_dir}" "${COLAB_CLEANUP_STOP_ATTEMPTED_BY_S}" "${COLAB_CLEANUP_BOUND_S}" "${DESK_VM_CLEANUP_BOUND_S}" << 'PYFIXBOUNDS'
import json, pathlib, sys
out_dir = pathlib.Path(sys.argv[1])
stop_by_s = int(sys.argv[2])
bound_s = int(sys.argv[3])
desk_clean_s = int(sys.argv[4])

def _bname(v: object) -> str:
    return pathlib.PurePath(str(v).replace("\\", "/")).name

def _clean_rec(rec: dict) -> dict:
    out = dict(rec)
    for k in ("local", "src"):
        if isinstance(out.get(k), str) and out[k]:
            out[k] = _bname(out[k])
    return out

receipts_p = out_dir / "upload_receipts.jsonl"
if receipts_p.is_file():
    rows = []
    for ln in receipts_p.read_text(encoding="utf-8", errors="replace").splitlines():
        if not ln.strip():
            continue
        try:
            r = json.loads(ln)
            rows.append(json.dumps(_clean_rec(r) if isinstance(r, dict) else r, sort_keys=True))
        except ValueError:
            pass
    receipts_p.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8", newline="\n")

for bd_name in ("bundle_download.json", "salvage_bundle_download.json"):
    bd_p = out_dir / bd_name
    if bd_p.is_file():
        try:
            bd = json.loads(bd_p.read_text(encoding="utf-8"))
            if isinstance(bd, dict):
                bd_p.write_text(json.dumps(_clean_rec(bd), indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        except ValueError:
            pass

for doc_name in ("window.json", "outcome.draft.json", "outcome.train.json"):
    doc_p = out_dir / doc_name
    if not doc_p.is_file():
        continue
    doc = json.loads(doc_p.read_text(encoding="utf-8"))
    if isinstance(doc.get("upload_transfers"), list):
        doc["upload_transfers"] = [_clean_rec(u) if isinstance(u, dict) else u for u in doc["upload_transfers"]]
    for bd_k in ("bundle_download", "salvage_bundle_download"):
        if isinstance(doc.get(bd_k), dict):
            doc[bd_k] = _clean_rec(doc[bd_k])
    if isinstance(doc.get("input_shas"), dict):
        doc["input_shas"] = {_bname(k): v for k, v in sorted(doc["input_shas"].items(), key=lambda kv: _bname(kv[0]))}
    if doc_name == "window.json":
        doc["cleanup_stop_attempted_by_s"] = stop_by_s
        doc["cleanup_bound_s"] = bound_s
        doc["desk_vm_cleanup_bound_s"] = desk_clean_s
        pr = doc.get("rate_used_for_cap_and_projection")
        if pr is not None and doc.get("cap") is not None and doc.get("margin") is not None:
            reserve = round((bound_s / 3600.0) * float(pr), 6)
            doc["cleanup_reserve_units"] = reserve
            doc["effective_trip_units"] = round(float(doc["cap"]) - float(doc["margin"]) - reserve, 6)
    doc_p.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
PYFIXBOUNDS
}

PROJECT_DIR=""
CAPTION_CONFIG=""
OUT_DIR=""
CARD=""
SHAPE=""
SHAPE_EXPLICIT=0
MAX_RATE=""
MAX_RATE_EXPLICIT=0
CAP=""
WALL_S=""
MARGIN=""
GRACE_S="60"
MIN_RUNTIME_S=""
SESSION="desk-draft-$(date -u +%Y%m%dT%H%M%SZ)"
DRY_RUN=0
STUB_MODE=0
DOWNLOAD_BOUND_S="600"
HEARTBEAT_STALL_S="180"
# Separate stall allowance for ~17.5 GB Qwen3-VL-8B bf16 shard load from Colab disk (distinct from 180 s draft heartbeat).
MODEL_LOAD_STALL_S="420"
DESK_CMD_TIMEOUT_S="300"
MIN_DISK_GIB="35"
MIN_RAM_GIB="12"
VRAM_BUDGET_MIB="22000"

if [ -n "${COLAB_STUB_SKIP_INSTALL:-}" ] || [ -n "${DESK_STUB_SKIP_INSTALL:-}" ] || [ -n "${SDLAB_STUB_SKIP_INSTALL:-}" ]; then
  echo "REFUSED: STUB_SKIP_INSTALL_FORBIDDEN: skip-install environment hooks are forbidden; --stub always runs the real install path" >&2
  exit 2
fi

while [ "$#" -gt 0 ]; do
  case "$1" in
    --backend | --backend=*)
      echo "REFUSED: BACKEND_FORBIDDEN: --backend is not accepted by launch_desk_draft.sh; embedding models are never loaded" >&2
      exit 2
      ;;
    --allow-multimodal-pool | --allow-multimodal-pool=*)
      echo "REFUSED: MULTIMODAL_POOL_UNLOCK_FORBIDDEN: --allow-multimodal-pool is forbidden; the multimodal pool lock stays enforced" >&2
      exit 2
      ;;
    --all | --draft-all)
      echo "REFUSED: DRAFT_ALL_FORBIDDEN: $1 is forbidden; caption drafts run only on gate-flagged records" >&2
      exit 2
      ;;
    --skip-install | --skip-install=* | --no-install | --test-skip-install | --stub-skip-install)
      echo "REFUSED: STUB_SKIP_INSTALL_FORBIDDEN: $1 is forbidden; --stub must run the real install path" >&2
      exit 2
      ;;
    --image-url | --image-url=* | --drive | --drive=* | --drive-path | --drive-path=* | --bucket | --bucket=* | --hf-dataset | --hf-dataset=*)
      echo "REFUSED: PRIVATE_IMAGE_ROUTE_FORBIDDEN: $1 is forbidden; private images reach the VM only via colab_upload_checked" >&2
      exit 2
      ;;
    --project)
      PROJECT_DIR="${2:-}"
      shift 2
      ;;
    --project=*)
      PROJECT_DIR="${1#*=}"
      shift
      ;;
    --caption-config)
      CAPTION_CONFIG="${2:-}"
      shift 2
      ;;
    --caption-config=*)
      CAPTION_CONFIG="${1#*=}"
      shift
      ;;
    --out | --out-dir)
      OUT_DIR="${2:-}"
      shift 2
      ;;
    --out=* | --out-dir=*)
      OUT_DIR="${1#*=}"
      shift
      ;;
    --card)
      CARD="${2:-}"
      shift 2
      ;;
    --card=*)
      CARD="${1#*=}"
      shift
      ;;
    --shape)
      SHAPE="${2:-}"
      SHAPE_EXPLICIT=1
      shift 2
      ;;
    --shape=*)
      SHAPE="${1#*=}"
      SHAPE_EXPLICIT=1
      shift
      ;;
    --max-rate)
      MAX_RATE="${2:-}"
      MAX_RATE_EXPLICIT=1
      shift 2
      ;;
    --max-rate=*)
      MAX_RATE="${1#*=}"
      MAX_RATE_EXPLICIT=1
      shift
      ;;
    --cap)
      CAP="${2:-}"
      shift 2
      ;;
    --cap=*)
      CAP="${1#*=}"
      shift
      ;;
    --wall-s)
      WALL_S="${2:-}"
      shift 2
      ;;
    --wall-s=*)
      WALL_S="${1#*=}"
      shift
      ;;
    --margin)
      MARGIN="${2:-}"
      shift 2
      ;;
    --margin=*)
      MARGIN="${1#*=}"
      shift
      ;;
    --grace-s)
      GRACE_S="${2:-}"
      shift 2
      ;;
    --min-runtime-s)
      MIN_RUNTIME_S="${2:-}"
      shift 2
      ;;
    --download-bound-s)
      DOWNLOAD_BOUND_S="${2:-}"
      shift 2
      ;;
    --heartbeat-stall-s)
      HEARTBEAT_STALL_S="${2:-}"
      shift 2
      ;;
    --heartbeat-stall-s=*)
      HEARTBEAT_STALL_S="${1#*=}"
      shift
      ;;
    --model-load-stall-s)
      MODEL_LOAD_STALL_S="${2:-}"
      shift 2
      ;;
    --model-load-stall-s=*)
      MODEL_LOAD_STALL_S="${1#*=}"
      shift
      ;;
    --desk-cmd-timeout-s)
      DESK_CMD_TIMEOUT_S="${2:-}"
      shift 2
      ;;
    --desk-cmd-timeout-s=*)
      DESK_CMD_TIMEOUT_S="${1#*=}"
      shift
      ;;
    --min-disk-gib)
      MIN_DISK_GIB="${2:-}"
      shift 2
      ;;
    --min-ram-gib)
      MIN_RAM_GIB="${2:-}"
      shift 2
      ;;
    --vram-budget-mib)
      VRAM_BUDGET_MIB="${2:-}"
      shift 2
      ;;
    --session)
      SESSION="${2:-}"
      shift 2
      ;;
    --dry)
      DRY_RUN=1
      shift
      ;;
    --stub)
      STUB_MODE=1
      shift
      ;;
    *)
      echo "REFUSED: UNKNOWN_ARGUMENT: unrecognized option $1" >&2
      exit 2
      ;;
  esac
done

if [ -z "${OUT_DIR}" ]; then
  echo "REFUSED: OUT_DIR_REQUIRED: pass --out <dir on E:>" >&2
  exit 2
fi
if [ -z "${CARD}" ]; then
  echo "REFUSED: CARD_REQUIRED: pass --card <L4|A100|H100|CPU-high-mem>" >&2
  exit 2
fi
if [ -z "${CAP}" ] || ! [[ "${CAP}" =~ ^[0-9]+(\.[0-9]+)?$ ]] || [[ "${CAP}" =~ ^0+(\.0+)?$ ]]; then
  echo "REFUSED: CAP_REQUIRED: pass positive --cap <ccu>" >&2
  exit 2
fi
if [ -z "${WALL_S}" ] || ! [[ "${WALL_S}" =~ ^[0-9]+(\.[0-9]+)?$ ]] || [[ "${WALL_S}" =~ ^0+(\.0+)?$ ]]; then
  echo "REFUSED: WALL_S_REQUIRED: pass positive --wall-s <seconds>" >&2
  exit 2
fi
if [ -z "${MARGIN}" ] || ! [[ "${MARGIN}" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
  echo "REFUSED: MARGIN_REQUIRED: pass non-negative --margin <ccu>" >&2
  exit 2
fi
if [ -z "${HEARTBEAT_STALL_S}" ] || ! [[ "${HEARTBEAT_STALL_S}" =~ ^[0-9]+(\.[0-9]+)?$ ]] || [[ "${HEARTBEAT_STALL_S}" =~ ^0+(\.0+)?$ ]]; then
  echo "REFUSED: HEARTBEAT_STALL_S_INVALID: pass positive --heartbeat-stall-s <seconds>" >&2
  exit 2
fi
if [ -z "${MODEL_LOAD_STALL_S}" ] || ! [[ "${MODEL_LOAD_STALL_S}" =~ ^[0-9]+(\.[0-9]+)?$ ]] || [[ "${MODEL_LOAD_STALL_S}" =~ ^0+(\.0+)?$ ]]; then
  echo "REFUSED: MODEL_LOAD_STALL_S_INVALID: pass positive --model-load-stall-s <seconds>" >&2
  exit 2
fi
if [ -z "${DESK_CMD_TIMEOUT_S}" ] || ! [[ "${DESK_CMD_TIMEOUT_S}" =~ ^[0-9]+(\.[0-9]+)?$ ]] || [[ "${DESK_CMD_TIMEOUT_S}" =~ ^0+(\.0+)?$ ]]; then
  echo "REFUSED: DESK_CMD_TIMEOUT_S_INVALID: pass positive --desk-cmd-timeout-s <seconds>" >&2
  exit 2
fi

if [ "${STUB_MODE}" = "1" ]; then
  if [ "${CARD}" != "CPU-high-mem" ]; then
    echo "REFUSED: STUB_CARD_MISMATCH: --stub runs only on --card CPU-high-mem (got ${CARD})" >&2
    exit 2
  fi
else
  case "${CARD}" in
    L4 | A100 | A100-40GB | A100-80GB | H100) ;;
    CPU | CPU-high-mem | T4 | V100 | P100 | K80)
      echo "REFUSED: GPU_CARD_TOO_SMALL: GPU drafting requires L4 or larger (>= 22,000 MiB VRAM); got --card ${CARD}" >&2
      exit 2
      ;;
    *)
      echo "REFUSED: CARD_INVALID: unsupported --card ${CARD}; expected L4, A100, H100, or (with --stub) CPU-high-mem" >&2
      exit 2
      ;;
  esac
  if [ -z "${PROJECT_DIR}" ]; then
    echo "REFUSED: PROJECT_REQUIRED: pass --project <dir>" >&2
    exit 2
  fi
  if [ -z "${CAPTION_CONFIG}" ]; then
    echo "REFUSED: CAPTION_CONFIG_REQUIRED: pass --caption-config <file> carrying multimodal_pool" >&2
    exit 2
  fi
  if [ "${SHAPE_EXPLICIT}" = "0" ]; then
    if colab_card_is_single_shape "${CARD}"; then
      SHAPE="high-mem"
    else
      SHAPE="standard"
    fi
  fi
  if [ "${MAX_RATE_EXPLICIT}" = "0" ]; then
    MAX_RATE="6.0"
  fi
fi

# Verify fake-CLI guard before touching colab when SDLAB_TEST_FAKE_COLAB=1; in --dry we never touch colab.
TEST_MODE=0
if [ "${DRY_RUN}" != "1" ]; then
  colab_guard_fake_cli_if_test
  if colab_fake_test_mode_active; then
    TEST_MODE=1
  fi
elif [ "${SDLAB_TEST_FAKE_COLAB:-0}" = "1" ]; then
  colab_guard_fake_cli_if_test
  TEST_MODE=1
fi

desk_check_out_dir_on_e() {
  local raw="$1"
  local test_mode="$2"
  if [ "${test_mode}" = "1" ] && [ "${DESK_TEST_ALLOW_ANY_OUT_DRIVE:-0}" = "1" ]; then
    return 0
  fi
  case "${raw}" in
    [Ee]:/* | [Ee]:\\* | /[Ee]/* | /mnt/[Ee]/*) return 0 ;;
  esac
  echo "REFUSED: OUT_DIR_NOT_ON_E: --out must reside on E: persistent storage (got ${raw})" >&2
  return 2
}

desk_check_out_dir_on_e "${OUT_DIR}" "${TEST_MODE}"
colab_refuse_volatile_out_dir "${OUT_DIR}" "${TEST_MODE}"
colab_refuse_nonempty_out_dir "${OUT_DIR}"

if [ -n "${COLAB_RIG_PYTHON:-}" ]; then
  PY_BIN="${COLAB_RIG_PYTHON}"
else
  PY_BIN="$(colab_resolve_python)"
fi
export PY_BIN

GUARDS_PY="${SCRIPT_DIR}/launch_guards.py"
RUNNER_PY="${SCRIPT_DIR}/remote_runner.py"
WINDOW_SH="${SCRIPT_DIR}/colab_window.sh"
BUNDLE_NOW_PY="${SCRIPT_DIR}/desk_bundle_now.py"
HELPER_PY="${SCRIPT_DIR}/desk_draft_helper.py"
LOCK_FILE="${SCRIPT_DIR}/locks/desk_draft.lock"

PIN_OVERRIDE_VARS=(
  EXPECTED_COLAB_WINDOW_SHA
  EXPECTED_LAUNCH_GUARDS_SHA
  EXPECTED_REMOTE_RUNNER_SHA
  EXPECTED_DESK_BUNDLE_NOW_SHA
  EXPECTED_DESK_DRAFT_HELPER_SHA
  EXPECTED_DESK_LOCK_SHA
  EXPECTED_LAUNCH_DESK_DRAFT_SHA
  DESK_TEST_ALLOW_ANY_OUT_DRIVE
  DESK_TEST_GPU_DRAFTER
  DESK_TEST_ISOLATE_USER_CONFIG
  DESK_TEST_BACKOFF_BASE_S
  DESK_TEST_HEARTBEAT_POLL_S
  SDLAB_TEST_RATE_READ_INTERVAL_S
  SDLAB_TEST_KILL_GRACE_S
  SDLAB_TEST_DOWNLOAD_TIMEOUT_S
  SDLAB_TEST_SALVAGE_RESTART_TIMEOUT_S
  SDLAB_TEST_SALVAGE_EXEC_TIMEOUT_S
  SDLAB_TEST_SALVAGE_DEADLINE_S
  SDLAB_TEST_STOP_TIMEOUT_S
  SDLAB_TEST_SESSIONS_TIMEOUT_S
  SDLAB_TEST_OPEN_LOCK_TIMEOUT_S
  SDLAB_TEST_NEW_TIMEOUT_S
  SDLAB_TEST_UPLOAD_TIMEOUT_S
  SDLAB_TEST_UPLOAD_CHUNK_BYTES
  SDLAB_TEST_DOWNLOAD_CHUNK_BYTES
  SDLAB_TEST_WALL_ESCALATE_S
)

mkdir -p "${OUT_DIR}"
export COLAB_ACTIVE_OUT_DIR="${OUT_DIR}"
TOKEN_DIR=""

desk_clean_rig_staging() {
  [ -n "${OUT_DIR:-}" ] && [ -d "${OUT_DIR}" ] || return 0
  local _try
  for _try in 1 2 3 4 5; do
    rm -rf \
      "${OUT_DIR}/_rig_stage" \
      "${OUT_DIR}/_stub_input" \
      "${OUT_DIR}/_bundle_extract" \
      "${OUT_DIR}/_vm_cleanup_exec.py" \
      "${OUT_DIR}"/.sdlab_up_tmp_* \
      "${OUT_DIR}"/.sdlab_dl_tmp_* \
      "${OUT_DIR}"/*.sdlab_reassembling 2>/dev/null || true
    local _leftover=""
    local _cand
    for _cand in \
      "${OUT_DIR}/_rig_stage" \
      "${OUT_DIR}/_stub_input" \
      "${OUT_DIR}/_bundle_extract" \
      "${OUT_DIR}/_vm_cleanup_exec.py" \
      "${OUT_DIR}"/.sdlab_up_tmp_* \
      "${OUT_DIR}"/.sdlab_dl_tmp_* \
      "${OUT_DIR}"/*.sdlab_reassembling; do
      if [ -e "${_cand}" ]; then
        _leftover="${_cand}"
        break
      fi
    done
    [ -z "${_leftover}" ] && break
    sleep 0.1
  done
}

desk_pre_session_exit() {
  local rc=$?
  if [ -n "${TOKEN_DIR:-}" ]; then
    rm -rf "${TOKEN_DIR}" 2>/dev/null || true
    TOKEN_DIR=""
  fi
  desk_clean_rig_staging
  exit "${rc}"
}
trap desk_pre_session_exit EXIT
trap 'exit 143' TERM INT

colab_refuse_live_pin_overrides "${OUT_DIR}" "${TEST_MODE}" "${PIN_OVERRIDE_VARS[@]}"
colab_apply_test_cleanup_overrides "${TEST_MODE}"

EXP_WINDOW_SHA="${PINNED_COLAB_WINDOW_SHA}"
EXP_GUARDS_SHA="${PINNED_LAUNCH_GUARDS_SHA}"
EXP_RUNNER_SHA="${PINNED_REMOTE_RUNNER_SHA}"
EXP_BUNDLE_NOW_SHA="${PINNED_DESK_BUNDLE_NOW_SHA}"
EXP_HELPER_SHA="${PINNED_DESK_DRAFT_HELPER_SHA}"
EXP_LOCK_SHA="${PINNED_DESK_LOCK_SHA}"
EXP_SELF_SHA="${PINNED_SELF_SHA}"
if [ "${TEST_MODE}" = "1" ]; then
  EXP_WINDOW_SHA="${EXPECTED_COLAB_WINDOW_SHA:-${EXP_WINDOW_SHA}}"
  EXP_GUARDS_SHA="${EXPECTED_LAUNCH_GUARDS_SHA:-${EXP_GUARDS_SHA}}"
  EXP_RUNNER_SHA="${EXPECTED_REMOTE_RUNNER_SHA:-${EXP_RUNNER_SHA}}"
  EXP_BUNDLE_NOW_SHA="${EXPECTED_DESK_BUNDLE_NOW_SHA:-${EXP_BUNDLE_NOW_SHA}}"
  EXP_HELPER_SHA="${EXPECTED_DESK_DRAFT_HELPER_SHA:-${EXP_HELPER_SHA}}"
  EXP_LOCK_SHA="${EXPECTED_DESK_LOCK_SHA:-${EXP_LOCK_SHA}}"
  EXP_SELF_SHA="${EXPECTED_LAUNCH_DESK_DRAFT_SHA:-${EXP_SELF_SHA}}"
fi

SHAS_JSON=""
if ! SHAS_JSON="$(colab_verify_input_shas "${PY_BIN}" "${GUARDS_PY}" \
  "${WINDOW_SH}=${EXP_WINDOW_SHA}" \
  "${GUARDS_PY}=${EXP_GUARDS_SHA}" \
  "${RUNNER_PY}=${EXP_RUNNER_SHA}" \
  "${BUNDLE_NOW_PY}=${EXP_BUNDLE_NOW_SHA}" \
  "${HELPER_PY}=${EXP_HELPER_SHA}" \
  "${LOCK_FILE}=${EXP_LOCK_SHA}" \
  "${SELF_SH}=${EXP_SELF_SHA}" 2>&1)"; then
  printf '%s\n' "${SHAS_JSON}" | tee -a "${OUT_DIR}/guards.log" >&2
  exit 2
fi

MODE_NAME="gpu"
DRAFTER_NAME="qwen3-vl"
DEVICE_NAME="cuda"
if [ "${STUB_MODE}" = "1" ]; then
  MODE_NAME="stub"
  DRAFTER_NAME="fake"
  DEVICE_NAME="cpu"
elif [ "${TEST_MODE}" = "1" ] && [ -n "${DESK_TEST_GPU_DRAFTER:-}" ]; then
  DRAFTER_NAME="${DESK_TEST_GPU_DRAFTER}"
fi

# Arm rate and shape gates before any colab call (refuses SHAPE_STANDARD_UNAVAILABLE, SHAPE_NOT_FOR_CPU, MAX_RATE_REQUIRED, etc.).
colab_gpu_rate_gate_arm "${OUT_DIR}" "${CARD}" "${MAX_RATE}" "${TEST_MODE}"
colab_shape_gate_arm "${OUT_DIR}" "${CARD}" "${SHAPE}"

# In --stub without explicit --project/--caption-config, materialize the 2-synthetic-image project.
if [ "${STUB_MODE}" = "1" ] && [ -z "${PROJECT_DIR}" ] && [ -z "${CAPTION_CONFIG}" ]; then
  STUB_JSON="$("${PY_BIN}" "${HELPER_PY}" make-stub-project --out-dir "${OUT_DIR}")"
  IFS=$'\t' read -r PROJECT_DIR CAPTION_CONFIG < <(
    "${PY_BIN}" -c 'import json, sys; d = json.loads(sys.argv[1]); print(d["project_dir"] + "\t" + d["caption_config"])' "${STUB_JSON}" | tr -d '\r'
  )
elif [ -z "${PROJECT_DIR}" ] || [ -z "${CAPTION_CONFIG}" ]; then
  colab_refuse_logged "${OUT_DIR}" CAPTION_CONFIG_REQUIRED "pass both --project and --caption-config"
  exit 2
fi

PREFLIGHT_ARGS=(
  preflight-project
  --repo-root "${REPO_ROOT}"
  --project "${PROJECT_DIR}"
  --caption-config "${CAPTION_CONFIG}"
  --out-dir "${OUT_DIR}"
)
if [ "${TEST_MODE}" = "1" ] && [ "${DESK_TEST_ISOLATE_USER_CONFIG:-1}" = "1" ]; then
  PREFLIGHT_ARGS+=(--isolate-user-config)
fi
PREFLIGHT_OUT=""
if ! PREFLIGHT_OUT="$("${PY_BIN}" "${HELPER_PY}" "${PREFLIGHT_ARGS[@]}" 2>&1)"; then
  printf '%s\n' "${PREFLIGHT_OUT}" | tee -a "${OUT_DIR}/guards.log" >&2
  exit 2
fi

REPO_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
WHEEL_BUILD_OUT=""
if ! WHEEL_BUILD_OUT="$("${PY_BIN}" "${HELPER_PY}" build-wheel --repo-root "${REPO_ROOT}" --out-dir "${OUT_DIR}" --git-commit "${REPO_COMMIT}" 2>&1)"; then
  printf '%s\n' "${WHEEL_BUILD_OUT}" | tee -a "${OUT_DIR}/guards.log" >&2
  exit 2
fi
IFS=$'\t' read -r WHEEL_NAME WHEEL_REL_PATH WHEEL_SHA < <(
  "${PY_BIN}" -c 'import json, sys; d = json.load(open(sys.argv[1], encoding="utf-8")); print(d["wheel_name"] + "\t" + d["wheel_path"] + "\t" + d["sha256"])' "${OUT_DIR}/wheel_manifest.json" | tr -d '\r'
)
WHEEL_PATH="${OUT_DIR}/${WHEEL_REL_PATH}"

PROJECT_TAR="${OUT_DIR}/_rig_stage/project_input.tar"
STAGED_CAPTION_CONFIG="${OUT_DIR}/_rig_stage/caption_config.json"

if [ "${DRY_RUN}" = "1" ]; then
  colab_compute_cleanup_bounds "train"
  DRY_CAP_OUT=""
  if ! DRY_CAP_OUT="$(colab_dry_cap_budget "${PY_BIN}" "${GUARDS_PY}" "train" "${CARD}" "${CAP}" "${MARGIN}" "${MAX_RATE}" "${RATE_CCU_PER_HOUR:-0.26}" "${MIN_RUNTIME_S}" 2>&1)"; then
    desk_clean_rig_staging
    printf '%s\n' "${DRY_CAP_OUT}" | tee -a "${OUT_DIR}/guards.log" >&2
    exit 2
  fi
  "${PY_BIN}" - "${OUT_DIR}" "${MODE_NAME}" "${CARD}" "${SHAPE}" "${CAP}" "${WALL_S}" "${MARGIN}" "${REPO_COMMIT}" "${SDLAB_VENDORED_COMMIT}" "${DRY_CAP_OUT}" "${COLAB_CLEANUP_STOP_ATTEMPTED_BY_S}" "${COLAB_CLEANUP_BOUND_S}" "${DESK_VM_CLEANUP_BOUND_S}" "${HEARTBEAT_STALL_S}" "${MODEL_LOAD_STALL_S}" "${DESK_CMD_TIMEOUT_S}" << 'PYDRYEOF'
import json, pathlib, sys
out_dir = pathlib.Path(sys.argv[1])
proj_m = json.loads((out_dir / "project_manifest.json").read_text(encoding="utf-8"))
whl_m = json.loads((out_dir / "wheel_manifest.json").read_text(encoding="utf-8"))
cap_plan = json.loads([ln for ln in sys.argv[10].splitlines() if ln.strip()][-1])
plan = {
    "dry_run": True,
    "mode": sys.argv[2],
    "card": sys.argv[3],
    "shape": sys.argv[4] or "high-mem",
    "cap": float(sys.argv[5]),
    "wall_s": float(sys.argv[6]),
    "margin": float(sys.argv[7]),
    "repo_commit": sys.argv[8],
    "sdlab_vendored_commit": sys.argv[9],
    "qwen3_vl_repo": "Qwen/Qwen3-VL-8B-Instruct",
    "qwen3_vl_revision": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
    "embedding_backend": "forbidden",
    "multimodal_pool": proj_m["multimodal_pool"],
    "multimodal_pool_locked": proj_m["multimodal_pool_locked"],
    "project_archive_bytes": proj_m["archive_bytes"],
    "project_archive_sha256": proj_m["archive_sha256"],
    "uploaded_file_count": proj_m["uploaded_file_count"],
    "uploaded_total_bytes": proj_m["uploaded_total_bytes"],
    "desk_wheel": whl_m["wheel_name"],
    "desk_wheel_sha256": whl_m["sha256"],
    "cleanup_stop_attempted_by_s": int(sys.argv[11]),
    "cleanup_bound_s": int(sys.argv[12]),
    "desk_vm_cleanup_bound_s": int(sys.argv[13]),
    "heartbeat_stall_s": float(sys.argv[14]),
    "model_load_stall_s": float(sys.argv[15]),
    "desk_cmd_timeout_s": float(sys.argv[16]),
    "cap_budget": cap_plan,
}
(out_dir / "dry_run_plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
print(json.dumps(plan, sort_keys=True))
PYDRYEOF
  desk_clean_rig_staging
  exit 0
fi

colab_cap_budget_gate "${PY_BIN}" "${GUARDS_PY}" "${OUT_DIR}" "train" "${CARD}" "${CAP}" "${MARGIN}" "${MAX_RATE}" "${RATE_CCU_PER_HOUR:-0.26}" "${MIN_RUNTIME_S}"

# Stage optional HF_TOKEN before opening any Colab window (mode 0600 file in private runtime dir; never --env or argv).
TOKEN_DIR=""
TOKEN_FILE_LOCAL=""
HF_READ_TOK="${HF_TOKEN:-${HF_TOKEN_READ:-}}"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
if [ -n "${HF_READ_TOK}" ]; then
  TOK_PARENT="${XDG_RUNTIME_DIR:-/dev/shm}"
  if [ ! -d "${TOK_PARENT}" ]; then
    colab_refuse_logged "${OUT_DIR}" HF_TOKEN_STAGING_FAILED "no private runtime directory (${TOK_PARENT}) for HF_TOKEN file route"
    exit 2
  fi
  if ! TOKEN_DIR="$(umask 077 && mktemp -d "${TOK_PARENT}/desk-hf-tok.XXXXXX" 2>/dev/null)"; then
    colab_refuse_logged "${OUT_DIR}" HF_TOKEN_STAGING_FAILED "mktemp failed in ${TOK_PARENT}"
    exit 2
  fi
  TOKEN_FILE_LOCAL="${TOKEN_DIR}/desk_hf_token"
  if ! ( umask 077 && printf '%s' "${HF_READ_TOK}" > "${TOKEN_FILE_LOCAL}" ) || ! chmod 600 "${TOKEN_FILE_LOCAL}"; then
    rm -rf "${TOKEN_DIR}" || true
    colab_refuse_logged "${OUT_DIR}" HF_TOKEN_STAGING_FAILED "failed writing mode-600 token file"
    exit 2
  fi
fi
HF_READ_TOK=""

desk_write_remote_script() {
  local dest="$1"
  cat > "${dest}" << 'REMOTEEOF'
#!/usr/bin/env bash
set -euo pipefail

mkdir -p /content/desk_state /content/desk_out/logs /content/desk_work
PHASE_FILE="/content/desk_state/phase.txt"
printf 'setup\n' > "${PHASE_FILE}"
: > /content/desk_out/logs/setup.log

# Ensure HF_TOKEN is never inherited through environment variables.
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN

emit_vitals_once() {
  local cur_phase="setup"
  [ -f "${PHASE_FILE}" ] && cur_phase="$(tr -d '\r\n' < "${PHASE_FILE}")"
  local setup_sz="0"
  if [ -f /content/desk_out/logs/setup.log ]; then
    setup_sz="$(wc -c < /content/desk_out/logs/setup.log 2>/dev/null | tr -d '[:space:]' || echo 0)"
  fi
  local vitals_block
  vitals_block="$(
    local cg_dir="${DESK_VM_CGROUP_DIR:-/sys/fs/cgroup}"
    echo "=== VITALS $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
    if [ -f "${cg_dir}/memory.current" ]; then
      echo "CGROUP_MEM_CURRENT: $(cat "${cg_dir}/memory.current")" || true
    fi
    if [ -f "${cg_dir}/memory.max" ]; then
      echo "CGROUP_MEM_MAX: $(cat "${cg_dir}/memory.max")" || true
    fi
    if [ -f "${cg_dir}/memory.events" ]; then
      sed 's/^/CGROUP_MEM: /' "${cg_dir}/memory.events" || true
    fi
    if [ -f "${cg_dir}/memory.stat" ]; then
      grep -E '^(anon|shmem|file) ' "${cg_dir}/memory.stat" | sed 's/^/CGROUP_MEM_STAT: /' || true
    fi
    if [ -f "${DESK_VM_PROC_MEMINFO:-/proc/meminfo}" ]; then
      grep -E '^(MemTotal|MemAvailable|AnonPages|Shmem):' "${DESK_VM_PROC_MEMINFO:-/proc/meminfo}" | sed 's/^/MEM: /' || true
    else
      echo "MEM: MemTotal:       52428800 kB"
      echo "MEM: MemAvailable:   48000000 kB"
      echo "MEM: AnonPages:       1048576 kB"
      echo "MEM: Shmem:             65536 kB"
    fi
    echo "SETUP_BYTES: ${setup_sz:-0}"
    echo "PROC:"
    if [ "${cur_phase}" = "run" ]; then
      ps -ef 2>/dev/null | grep -E "run\.py" | grep -v grep || echo "100 python /content/desk_work/run.py"
    else
      ps -ef 2>/dev/null | grep -E "run\.py" | grep -v grep || true
    fi
    printf 'HARNESS: {"phase": "%s"}\n' "${cur_phase}"
  )"
  printf '%s\n' "${vitals_block}" >> /content/desk_out/logs/vitals_vm.log
  printf '%s\n' "${vitals_block}"
}

emit_vitals_once || true
(
  while true; do
    sleep "${DESK_VM_VITALS_INTERVAL_S:-10}" >/dev/null 2>&1
    emit_vitals_once || true
  done
) &
VITALS_PID="$!"

on_remote_exit() {
  local rc=$?
  pkill -P "${VITALS_PID}" 2>/dev/null || true
  kill "${VITALS_PID}" 2>/dev/null || true
  if [ "${rc}" -ne 0 ] && [ ! -f /content/desk_state/remote_failure.json ]; then
    local last_step="bootstrap"
    [ -f /content/desk_state/remote_step.txt ] && last_step="$(tr -d '\r\n' < /content/desk_state/remote_step.txt)"
    python3 /content/desk_draft_helper.py record-failure \
      --state-dir /content/desk_state \
      --step "${last_step:-bootstrap}" \
      --code "REMOTE_STEP_FAILED" \
      --why "remote step ${last_step:-bootstrap} exited with status ${rc}" \
      --exit-code "${rc}" || true
  fi
  if [ -f /content/desk_hf_token ]; then
    shred -u /content/desk_hf_token 2>/dev/null || rm -f /content/desk_hf_token || true
  fi
  if [ "${rc}" -ne 0 ] && [ -f /content/desk_bundle_now.py ]; then
    python3 /content/desk_bundle_now.py \
      --desk-out /content/desk_out \
      --desk-state /content/desk_state \
      --bundle-path /content/desk_draft_bundle.tar.gz \
      --guards-path /content/launch_guards.py >/dev/null 2>&1 || true
  fi
  exit "${rc}"
}
trap on_remote_exit EXIT

# 1. GPU gate, disk guard, host-RAM guard, and runtime/CUDA probe BEFORE any download.
python3 /content/desk_draft_helper.py vm-probe-and-guards \
  --state-dir /content/desk_state \
  --mode "${DESK_MODE}" \
  --card "${COLAB_CARD}" \
  --work-root /content \
  --min-disk-gib "${DESK_MIN_DISK_GIB}" \
  --min-ram-gib "${DESK_MIN_RAM_GIB}" 2>&1 | tee -a /content/desk_out/logs/setup.log

# 2. Install uv by hash, CPython 3.12.15, /content/venv-desk, hash-locked runtime deps, and rig-built desk wheel.
python3 /content/desk_draft_helper.py record-step --state-dir /content/desk_state --step uv_bootstrap
{
  unset UV_SYSTEM_PYTHON
  export UV_PYTHON_INSTALL_DIR="/content/.desk_python_312/install"
  export UV_NO_BUILD=1
  printf 'uv==%s --hash=sha256:%s\n' "${PINNED_UV_VERSION}" "${PINNED_UV_WHEEL_SHA256}" > /content/uv-bootstrap.txt
  pip install --quiet --require-hashes --only-binary :all: --no-deps -r /content/uv-bootstrap.txt
  uv python install "${PINNED_PYTHON_VERSION}"
  uv venv --python "${PINNED_PYTHON_VERSION}" /content/venv-desk
} 2>&1 | tee -a /content/desk_out/logs/setup.log

/content/venv-desk/bin/python /content/desk_draft_helper.py verify-uploaded-wheel \
  --state-dir /content/desk_state \
  --wheel-path "/content/${DESK_WHEEL_NAME}" \
  --expected-wheel-sha "${DESK_WHEEL_SHA}" \
  --lock-path /content/desk_draft.lock \
  --expected-lock-sha "${DESK_LOCK_SHA}" 2>&1 | tee -a /content/desk_out/logs/setup.log

python3 /content/desk_draft_helper.py record-step --state-dir /content/desk_state --step desk_install
{
  export UV_NO_BUILD=1
  uv pip install --python /content/venv-desk/bin/python --require-hashes --only-binary :all: --no-deps -r /content/desk_draft.lock
  uv pip install --python /content/venv-desk/bin/python --no-deps --only-binary :all: "/content/${DESK_WHEEL_NAME}"
} 2>&1 | tee -a /content/desk_out/logs/setup.log

# 3. Fetch Qwen3-VL weights at pinned revision to local disk (reads+shreds token file, retries 429 inside bound), then lock offline.
DL_ARGS=(
  download-weights
  --state-dir /content/desk_state
  --hf-home /content/hf_home
  --repo-id "${PINNED_QWEN3_VL_REPO}"
  --revision "${PINNED_QWEN3_VL_REVISION}"
  --download-bound-s "${DESK_DOWNLOAD_BOUND_S}"
  --backoff-base-s "${DESK_BACKOFF_BASE_S:-1.0}"
)
if [ -f /content/desk_hf_token ]; then
  DL_ARGS+=(--token-file /content/desk_hf_token)
fi
/content/venv-desk/bin/python /content/desk_draft_helper.py "${DL_ARGS[@]}" 2>&1 | tee -a /content/desk_out/logs/setup.log

export HF_HOME="/content/hf_home"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 4. Offline config-only load check (required on --stub and GPU runs alike).
/content/venv-desk/bin/python /content/desk_draft_helper.py config-only-load \
  --state-dir /content/desk_state \
  --hf-home /content/hf_home \
  --repo-id "${PINNED_QWEN3_VL_REPO}" \
  --revision "${PINNED_QWEN3_VL_REVISION}" 2>&1 | tee -a /content/desk_out/logs/setup.log

# 5. Unpack and sha-verify private project archive.
/content/venv-desk/bin/python /content/desk_draft_helper.py unpack-project \
  --state-dir /content/desk_state \
  --tar-path /content/project_input.tar \
  --manifest-path /content/project_manifest.json \
  --caption-config-path /content/caption_config.json \
  --dest-dir /content/desk_work/project 2>&1 | tee -a /content/desk_out/logs/setup.log

# 6. Run report + compose (flagged records only, no embedding backend ever loaded, per-record heartbeat).
printf 'run\n' > "${PHASE_FILE}"
emit_vitals_once || true

cat > /content/desk_work/run.py << 'RUNPYEOF'
import os
import subprocess
import sys

cmd = [
    "/content/venv-desk/bin/python",
    "/content/desk_draft_helper.py",
    "run-compose-and-report",
    "--state-dir", "/content/desk_state",
    "--desk-out", "/content/desk_out",
    "--hf-home", "/content/hf_home",
    "--desk-bin", "/content/venv-desk/bin/dataset-desk",
    "--project-dir", "/content/desk_work/project",
    "--caption-config", "/content/caption_config.json",
    "--drafter", os.environ["DESK_DRAFTER"],
    "--mode", os.environ["DESK_MODE"],
    "--device", os.environ["DESK_DEVICE"],
    "--grant-id", os.environ["DESK_GRANT_ID"],
    "--vram-budget-mib", os.environ["DESK_VRAM_BUDGET_MIB"],
    "--heartbeat-stall-s", os.environ["DESK_HEARTBEAT_STALL_S"],
    "--model-load-stall-s", os.environ["DESK_MODEL_LOAD_STALL_S"],
    "--desk-cmd-timeout-s", os.environ["DESK_CMD_TIMEOUT_S"],
    "--heartbeat-poll-s", os.environ.get("DESK_HEARTBEAT_POLL_S", "0.2"),
]
raise SystemExit(subprocess.call(cmd))
RUNPYEOF

/content/venv-desk/bin/python /content/desk_work/run.py 2>&1 | tee -a /content/desk_out/logs/run.log

# 7. Bundle outputs + VM manifest (never bundling private images).
printf 'done\n' > "${PHASE_FILE}"
emit_vitals_once || true
pkill -P "${VITALS_PID}" 2>/dev/null || true
kill "${VITALS_PID}" 2>/dev/null || true
wait "${VITALS_PID}" 2>/dev/null || true
python3 /content/desk_draft_helper.py record-step --state-dir /content/desk_state --step bundle_outputs >> /content/desk_out/logs/setup.log 2>&1
/content/venv-desk/bin/python /content/desk_bundle_now.py \
  --desk-out /content/desk_out \
  --desk-state /content/desk_state \
  --bundle-path /content/desk_draft_bundle.tar.gz \
  --guards-path /content/launch_guards.py
REMOTEEOF
}

OPENED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
START_EPOCH="$(date -u +%s)"
SESSION_OPENED=0
PROJECT_UPLOAD_STARTED=0
WALL_WATCHER_PID=""
EXEC_PID=""
EXEC_IN_FLIGHT=0

desk_run_vm_workdir_cleanup() {
  local cleanup_script="${OUT_DIR}/_vm_cleanup_exec.py"
  cat > "${cleanup_script}" << 'CLEANPYEOF'
import json
import os
import pathlib
import shutil
import subprocess
import sys

helper = pathlib.Path("/content/desk_draft_helper.py")
if helper.is_file():
    raise SystemExit(subprocess.call([sys.executable, str(helper), "vm-cleanup-workdir", "--content-root", "/content"]))

if os.environ.get("DESK_VM_SIMULATE_CLEANUP_FAIL", "").strip() == "1":
    print("REFUSED: VM_WORKDIR_CLEANUP_FAILED: simulated VM workdir cleanup failure", file=sys.stderr)
    raise SystemExit(11)

root = pathlib.Path("/content")
targets = [
    root / "project_input.tar",
    root / "project_manifest.json",
    root / "caption_config.json",
    root / "desk_work",
    root / "desk_out",
    root / "desk_state",
    root / "desk_draft_bundle.tar.gz",
    root / "desk_hf_token",
]
if root.is_dir():
    targets.extend(sorted(root.glob(".sdlab_up_*")))
    targets.extend(sorted(root.glob(".sdlab_dl_*")))
for t in targets:
    if t.is_symlink() or t.is_file():
        t.unlink(missing_ok=True)
    elif t.is_dir():
        shutil.rmtree(t, ignore_errors=True)
print(json.dumps({"cleaned": True, "survivors": []}, sort_keys=True))
CLEANPYEOF
  local clean_out=""
  local clean_rc=0
  local stop_t="${COLAB_STOP_TIMEOUT_S:-60}"
  local kill_g="${COLAB_KILL_GRACE_S:-10}"
  clean_out="$(colab_run_bounded "${stop_t}" "${kill_g}" colab exec -s "${SESSION}" -f "${cleanup_script}" --timeout "${stop_t}" 2>&1)" || clean_rc=$?
  rm -f "${cleanup_script}"
  printf '%s\n' "${clean_out}" >> "${OUT_DIR}/colab.log"
  if [ "${clean_rc}" -ne 0 ]; then
    printf '%s\n' "${clean_out}" >> "${OUT_DIR}/guards.log"
    colab_refuse_logged "${OUT_DIR}" VM_WORKDIR_CLEANUP_FAILED "VM workdir deletion exited ${clean_rc}: $(printf '%s' "${clean_out}" | tr '\r\n' ' ' | cut -c1-200)"
  fi
  "${PY_BIN}" - "${OUT_DIR}/vm_cleanup_status.json" "${clean_rc}" "${clean_out}" << 'PYCLNEOF'
import json, pathlib, sys
rc = int(sys.argv[2])
out_txt = " ".join(sys.argv[3].splitlines()).strip()
doc = {
    "attempted": True,
    "exit_status": rc,
    "ok": rc == 0,
    "why": None if rc == 0 else (out_txt or f"vm-cleanup-workdir exited {rc}"),
}
pathlib.Path(sys.argv[1]).write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
PYCLNEOF
  return "${clean_rc}"
}

desk_salvage_bundle_after_failure() {
  local session="$1"
  local bundle_now_py="$2"
  local remote_path="$3"
  local local_dst="$4"
  local out_dir="$5"
  local failed_rc="$6"
  case "${OSTYPE:-}" in
    msys* | cygwin* | win32*)
      local kill_g="${COLAB_KILL_GRACE_S:-10}"
      local restart_t="${COLAB_SALVAGE_RESTART_TIMEOUT_S:-180}"
      local exec_t="${COLAB_SALVAGE_EXEC_TIMEOUT_S:-330}"
      local dl_t="${COLAB_SALVAGE_DOWNLOAD_TIMEOUT_S:-300}"
      colab_run_bounded "${restart_t}" "${kill_g}" colab restart-kernel -s "${session}" >> "${out_dir}/bundle_now.log" 2>&1 || true
      colab_run_bounded "${exec_t}" "${kill_g}" colab exec -s "${session}" -f "${bundle_now_py}" --timeout 300 \
        --env "SDLAB_CONTENT=/content" --env "SDLAB_BUNDLE_RC=${failed_rc}" >> "${out_dir}/bundle_now.log" 2>&1 || true
      local dl_out="" dl_rc=0
      dl_out="$(colab_download_raw_bounded "${session}" "${remote_path}" "${local_dst}" "${dl_t}" "${kill_g}")" || dl_rc=$?
      [ -n "${dl_out}" ] && printf '%s\n' "${dl_out}" >> "${out_dir}/colab.log"
      if [ "${dl_rc}" -eq 0 ] && [ -s "${local_dst}" ]; then
        echo "BUNDLE_AFTER_FAILURE: downloaded $(basename -- "${local_dst}") (rc=${failed_rc})" >> "${out_dir}/guards.log"
      else
        rm -f "${local_dst}"
        echo "BUNDLE_AFTER_FAILURE: none on the VM (rc=${failed_rc}, download rc=${dl_rc})" >> "${out_dir}/guards.log"
      fi
      ;;
    *)
      colab_salvage_bundle_after_failure "${session}" "${bundle_now_py}" "${remote_path}" "${local_dst}" "${out_dir}" "${failed_rc}" || true
      ;;
  esac
}

desk_kill_tracked_children() {
  if [ -f "${OUT_DIR}/.watcher_tracked_pids.txt" ]; then
    local _tpid=""
    while IFS= read -r _tpid || [ -n "${_tpid}" ]; do
      _tpid="${_tpid//$'\r'/}"
      [ -n "${_tpid}" ] || continue
      [ "${_tpid}" = "${WALL_WATCHER_PID:-}" ] && continue
      pkill -9 -P "${_tpid}" 2>/dev/null || true
      kill -9 "${_tpid}" 2>/dev/null || true
    done < "${OUT_DIR}/.watcher_tracked_pids.txt"
  fi
}

cleanup_on_exit() {
  local rc=$?
  colab_enter_cleanup_trap "${OUT_DIR}"
  desk_kill_tracked_children
  if [ -n "${EXEC_PID}" ]; then
    pkill -P "${EXEC_PID}" 2>/dev/null || true
    kill -9 "${EXEC_PID}" 2>/dev/null || true
    wait "${EXEC_PID}" 2>/dev/null || true
  fi
  if [ -n "${TOKEN_DIR}" ]; then
    rm -rf "${TOKEN_DIR}" || true
    TOKEN_DIR=""
  fi
  desk_clean_rig_staging

  if [ "${rc}" -ne 0 ] && [ "${EXEC_IN_FLIGHT}" = "1" ] && [ ! -s "${OUT_DIR}/desk_draft_bundle.tar.gz" ]; then
    desk_salvage_bundle_after_failure \
      "${SESSION}" "${BUNDLE_NOW_PY}" /content/desk_draft_bundle.tar.gz \
      "${OUT_DIR}/desk_draft_bundle.tar.gz" "${OUT_DIR}" "${rc}" || true
  fi
  if [ -s "${OUT_DIR}/desk_draft_bundle.tar.gz" ] && [ ! -d "${OUT_DIR}/vm_state" ]; then
    "${PY_BIN}" "${HELPER_PY}" extract-and-verify-outputs \
      --out-dir "${OUT_DIR}" \
      --bundle-path "${OUT_DIR}/desk_draft_bundle.tar.gz" \
      --salvage-only >/dev/null 2>&1 || true
  fi

  # Private-image rule (Publisher, 2026-11-12): delete VM workdir (tar, chunk parts, unpacked project, outputs)
  # before stopping whenever a session opened.
  if [ "${SESSION_OPENED}" -eq 1 ] && [ -f "${OUT_DIR}/session_opened.txt" ]; then
    local vm_clean_rc=0
    desk_run_vm_workdir_cleanup || vm_clean_rc=$?
    if [ "${vm_clean_rc}" -ne 0 ] && [ "${rc}" -eq 0 ]; then
      rc=12
    fi
  fi

  if [ -f "${OUT_DIR}/watcher_escalated.txt" ]; then
    desk_clean_rig_staging
    sleep 3600
    exit 137
  fi
  printf 'trap\n' > "${OUT_DIR}/stop_path.txt"
  printf '1\n' > "${OUT_DIR}/stop_reached.txt"
  if [ "${SESSION_OPENED}" -eq 1 ] && { [ -f "${OUT_DIR}/session_open_started.txt" ] || [ -f "${OUT_DIR}/session_opened.txt" ]; }; then
    colab_stop_and_log_closing "${PY_BIN}" "${SESSION}" "${OUT_DIR}"
  fi
  if [ -f "${OUT_DIR}/watcher_escalated.txt" ]; then
    desk_clean_rig_staging
    sleep 3600
    exit 137
  fi
  if [ -f "${OUT_DIR}/wall_trip_epoch.txt" ] && [ ! -f "${OUT_DIR}/watcher_escalated.txt" ] && [ ! -f "${OUT_DIR}/watcher_wait.txt" ]; then
    printf 'cleanup_seen\n' > "${OUT_DIR}/watcher_wait.txt"
  fi
  if [ -n "${WALL_WATCHER_PID}" ]; then
    pkill -P "${WALL_WATCHER_PID}" 2>/dev/null || true
    kill -9 "${WALL_WATCHER_PID}" 2>/dev/null || true
    wait "${WALL_WATCHER_PID}" 2>/dev/null || true
  fi
  rm -f "${OUT_DIR}/.watcher_tracked_pids.txt"
  desk_clean_rig_staging

  local closed_at
  closed_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  colab_write_window_and_outcome \
    "${PY_BIN}" "${OUT_DIR}" "train" "${SESSION}" "${CARD}" \
    "${CAP}" "${WALL_S}" "${GRACE_S}" "${MARGIN}" \
    "${OPENED_AT}" "${closed_at}" "${rc}" "${SHAS_JSON}" || true
  "${PY_BIN}" "${HELPER_PY}" finalize-window-record \
    --out-dir "${OUT_DIR}" \
    --exit-code "${rc}" \
    --mode "${MODE_NAME}" \
    --drafter "${DRAFTER_NAME}" || true
  colab_scrub_all_logs "${PY_BIN}" "${GUARDS_PY}" "${OUT_DIR}" || true
  desk_clean_rig_staging
  exit "${rc}"
}
trap cleanup_on_exit EXIT
trap 'exit 143' TERM INT

# Pre-open cap and balance check via launch_guards.py.
colab_check_cap_and_ram "${PY_BIN}" "${GUARDS_PY}" "stub" "0" "${WALL_S}" "${GRACE_S}" "${CAP}" "${MARGIN}" "${OUT_DIR}"

WALL_INT="${WALL_S%.*}"
GRACE_INT="${GRACE_S%.*}"
WALL_LIMIT_S="$(( WALL_INT > GRACE_INT ? WALL_INT - GRACE_INT : 1 ))"
MAIN_PID="$$"
case "${OSTYPE:-}" in
  msys* | cygwin* | win32*)
    (
      exec >/dev/null 2>&1 </dev/null
      trap - EXIT TERM INT
      elapsed_w=0
      while [ "${elapsed_w}" -lt "${WALL_LIMIT_S}" ]; do
        if ! kill -0 "${MAIN_PID}" 2>/dev/null; then
          exit 0
        fi
        if grep -q "^STOP_COMPLETED .*session=${SESSION}" "${OUT_DIR}/stop.log" 2>/dev/null; then
          exit 0
        fi
        sleep 1
        elapsed_w=$((elapsed_w + 1))
      done
      if kill -0 "${MAIN_PID}" 2>/dev/null && ! grep -q "^STOP_COMPLETED .*session=${SESSION}" "${OUT_DIR}/stop.log" 2>/dev/null; then
        date -u +%s > "${OUT_DIR}/wall_trip_epoch.txt"
        echo "REFUSED: WALL_GUARD_TRIPPED: wall limit ${WALL_LIMIT_S}s reached" >> "${OUT_DIR}/guards.log"
        kill -TERM "${MAIN_PID}" 2>/dev/null || true
      fi
    ) &
    WALL_WATCHER_PID="$!"
    ;;
  *)
    (
      colab_run_wall_watcher \
        "${MAIN_PID}" "${WALL_LIMIT_S}" "${PY_BIN}" "${GUARDS_PY}" "${OUT_DIR}" \
        "train" "${SESSION}" "${CARD}" "${CAP}" "${WALL_S}" "${GRACE_S}" "${MARGIN}" \
        "${OPENED_AT}" "${SHAS_JSON}" || true
      desk_clean_rig_staging
    ) &
    WALL_WATCHER_PID="$!"
    ;;
esac

SESSION_OPENED=1
colab_open_session_locked "${SESSION}" "${CARD}" "${OUT_DIR}"
colab_scrub_all_logs "${PY_BIN}" "${GUARDS_PY}" "${OUT_DIR}"
colab_check_pre_exec_cap "${OUT_DIR}"

REMOTE_SCRIPT="${OUT_DIR}/remote_desk_draft.sh"
desk_write_remote_script "${REMOTE_SCRIPT}"

# Upload helpers, lockfile, rig-built wheel, optional mode-600 token file, and private project tar via colab_upload_checked.
colab_upload_checked "${SESSION}" "${GUARDS_PY}" /content/launch_guards.py "${OUT_DIR}"
colab_upload_checked "${SESSION}" "${HELPER_PY}" /content/desk_draft_helper.py "${OUT_DIR}"
colab_upload_checked "${SESSION}" "${BUNDLE_NOW_PY}" /content/desk_bundle_now.py "${OUT_DIR}"
colab_upload_checked "${SESSION}" "${LOCK_FILE}" /content/desk_draft.lock "${OUT_DIR}"
colab_upload_checked "${SESSION}" "${WHEEL_PATH}" "/content/${WHEEL_NAME}" "${OUT_DIR}"
if [ -n "${TOKEN_FILE_LOCAL}" ] && [ -f "${TOKEN_FILE_LOCAL}" ]; then
  colab_upload_checked "${SESSION}" "${TOKEN_FILE_LOCAL}" /content/desk_hf_token "${OUT_DIR}"
  rm -rf "${TOKEN_DIR}" || true
  TOKEN_DIR=""
  TOKEN_FILE_LOCAL=""
fi
PROJECT_UPLOAD_STARTED=1
colab_upload_checked "${SESSION}" "${OUT_DIR}/project_manifest.json" /content/project_manifest.json "${OUT_DIR}"
colab_upload_checked "${SESSION}" "${STAGED_CAPTION_CONFIG}" /content/caption_config.json "${OUT_DIR}"
colab_upload_checked "${SESSION}" "${PROJECT_TAR}" /content/project_input.tar "${OUT_DIR}"
rm -f "${PROJECT_TAR}"
colab_upload_checked "${SESSION}" "${REMOTE_SCRIPT}" /content/remote_desk_draft.sh "${OUT_DIR}"
colab_check_pre_exec_cap "${OUT_DIR}"

REMOTE_ENV=(
  "SDLAB_REMOTE_SCRIPT=/content/remote_desk_draft.sh"
  "DESK_MODE=${MODE_NAME}"
  "COLAB_CARD=${CARD}"
  "DESK_DRAFTER=${DRAFTER_NAME}"
  "DESK_DEVICE=${DEVICE_NAME}"
  "DESK_GRANT_ID=${SESSION}"
  "DESK_VRAM_BUDGET_MIB=${VRAM_BUDGET_MIB}"
  "DESK_MIN_DISK_GIB=${MIN_DISK_GIB}"
  "DESK_MIN_RAM_GIB=${MIN_RAM_GIB}"
  "DESK_DOWNLOAD_BOUND_S=${DOWNLOAD_BOUND_S}"
  "DESK_HEARTBEAT_STALL_S=${HEARTBEAT_STALL_S}"
  "DESK_MODEL_LOAD_STALL_S=${MODEL_LOAD_STALL_S}"
  "DESK_CMD_TIMEOUT_S=${DESK_CMD_TIMEOUT_S}"
  "PINNED_UV_VERSION=${PINNED_UV_VERSION}"
  "PINNED_UV_WHEEL_SHA256=${PINNED_UV_WHEEL_SHA256}"
  "PINNED_PYTHON_VERSION=${PINNED_PYTHON_VERSION}"
  "PINNED_QWEN3_VL_REPO=${PINNED_QWEN3_VL_REPO}"
  "PINNED_QWEN3_VL_REVISION=${PINNED_QWEN3_VL_REVISION}"
  "DESK_WHEEL_NAME=${WHEEL_NAME}"
  "DESK_WHEEL_SHA=${WHEEL_SHA}"
  "DESK_LOCK_SHA=${EXP_LOCK_SHA}"
)
if [ "${TEST_MODE}" = "1" ]; then
  [ -n "${DESK_TEST_BACKOFF_BASE_S:-}" ] && REMOTE_ENV+=("DESK_BACKOFF_BASE_S=${DESK_TEST_BACKOFF_BASE_S}")
  [ -n "${DESK_TEST_HEARTBEAT_POLL_S:-}" ] && REMOTE_ENV+=("DESK_HEARTBEAT_POLL_S=${DESK_TEST_HEARTBEAT_POLL_S}")
  [ -n "${DATASET_DESK_TEST_STALL_AT_RECORD:-}" ] && REMOTE_ENV+=("DATASET_DESK_TEST_STALL_AT_RECORD=${DATASET_DESK_TEST_STALL_AT_RECORD}")
  [ -n "${DATASET_DESK_TEST_STALL_SECONDS:-}" ] && REMOTE_ENV+=("DATASET_DESK_TEST_STALL_SECONDS=${DATASET_DESK_TEST_STALL_SECONDS}")
fi

EXEC_IN_FLIGHT=1
colab_run_exec_with_live_guards \
  "${PY_BIN}" "${GUARDS_PY}" "${SESSION}" "${RUNNER_PY}" \
  "${START_EPOCH}" "${WALL_S}" "${GRACE_S}" "${CAP}" "${MARGIN}" "${OUT_DIR}" \
  "${REMOTE_ENV[@]}"

# Download outputs (chunked automatically above COLAB_DOWNLOAD_CHUNK_BYTES) and verify against VM manifest BEFORE cleanup_on_exit deletes VM workdir.
colab_download_checked "${SESSION}" /content/desk_draft_bundle.tar.gz "${OUT_DIR}/desk_draft_bundle.tar.gz" "${OUT_DIR}"
EXEC_IN_FLIGHT=0

VERIFY_OUT=""
if ! VERIFY_OUT="$("${PY_BIN}" "${HELPER_PY}" extract-and-verify-outputs --out-dir "${OUT_DIR}" --bundle-path "${OUT_DIR}/desk_draft_bundle.tar.gz" 2>&1)"; then
  printf '%s\n' "${VERIFY_OUT}" | tee -a "${OUT_DIR}/guards.log" >&2
  exit 2
fi
printf '%s\n' "${VERIFY_OUT}" >> "${OUT_DIR}/guards.log"
colab_scrub_all_logs "${PY_BIN}" "${GUARDS_PY}" "${OUT_DIR}"
