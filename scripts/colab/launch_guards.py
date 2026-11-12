#!/usr/bin/env python3
"""C2 Colab launcher guards (DESIGN-v3 section 9.2 item 6 — lessons v4k, v4p, v4q).

Provides:
  1. evaluate_wall_guard (v4k: WALL_S and GRACE_S honoured in stub and live modes)
  2. evaluate_cap_guard (v4k: CAP and MARGIN honoured in stub and live modes)
  3. parse_vitals_log & evaluate_vitals_and_stall_guard (v4p: DEAD_HARNESS_TRIPPED on
     cgroup OOM or 2 consecutive phase: run samples with 0 worker procs, STALL_GUARD_TRIPPED on stall)
  4. check_host_ram_projection & run_between_item_fetch (v4q: pre-load cgroup/host RAM
     check against kernel SIGKILL rc=-9, plus between-item fetch before current.json reset)
  5. verify_input_shas & scrub_colab_tokens (pre-open sha256 gate and log token scrubber)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

WORKER_PROC_RE = re.compile(
    r"\b(comfy|main\.py|ai-toolkit|run\.py|llama-server|ollama|lane1_render\.py)\b",
    re.I,
)
OOM_KILL_RE = re.compile(r"\boom_kill\s+([1-9]\d*)\b|\bOut of memory: Killed process\b", re.I)

TOKEN_SCRUB_PATTERNS = [
    (re.compile(r"(colab-runtime-proxy-token=)[^\s&\"']+"), r"\1[REDACTED]"),
    (re.compile(r"(access_token=)[^\s&\"']+"), r"\1[REDACTED]"),
    (re.compile(r"(Bearer\s+)[A-Za-z0-9._\-]+"), r"\1[REDACTED]"),
    (re.compile(r"\bhf_[A-Za-z0-9]{16,}\b"), "hf_[REDACTED]"),
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


SELF_PIN_LINE_RE = re.compile(rb'^(PINNED_SELF_SHA=")[0-9a-fA-F]{64}(")', flags=re.M)


def sha256_self_normalized(path: Path) -> tuple[str | None, str]:
    raw = Path(path).read_bytes()
    m = re.search(rb'^PINNED_SELF_SHA="([0-9a-fA-F]{64})"', raw, flags=re.M)
    embedded_pin = m.group(1).decode("ascii").lower() if m else None
    norm_bytes = SELF_PIN_LINE_RE.sub(rb"\1\2", raw, count=1)
    return embedded_pin, hashlib.sha256(norm_bytes).hexdigest()


def normalize_hex_sha(val: str) -> str:
    s = str(val).strip().lower()
    if s.startswith("sha256:"):
        s = s[7:]
    return s


def verify_input_shas(expected_shas: dict[str, str]) -> dict[str, str]:
    verified: dict[str, str] = {}
    for file_str, exp_raw in expected_shas.items():
        p = Path(file_str)
        if not p.exists():
            raise ValueError(f"REFUSED: INPUT_FILE_MISSING: {p} does not exist")
        exp = normalize_hex_sha(exp_raw)
        got = sha256_file(p)
        if got != exp:
            embedded_pin, norm_got = sha256_self_normalized(p)
            if embedded_pin is not None and exp == norm_got and embedded_pin == norm_got:
                got = norm_got
            else:
                raise ValueError(
                    f"REFUSED: INPUT_SHA_MISMATCH for {p}: got {got}, expected {exp}"
                )
        verified[file_str] = got
    return verified


def scrub_colab_tokens(text: str) -> str:
    out = text
    for pattern, repl in TOKEN_SCRUB_PATTERNS:
        out = pattern.sub(repl, out)
    return out


def scrub_file_in_place(path: Path) -> None:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return
    raw = p.read_text(encoding="utf-8", errors="replace")
    scrubbed = scrub_colab_tokens(raw)
    if scrubbed != raw:
        p.write_text(scrubbed, encoding="utf-8", newline="\n")


def evaluate_wall_guard(
    *,
    mode: str = "live",
    elapsed_s: float,
    wall_s: float = 3000.0,
    grace_s: float = 60.0,
) -> dict[str, Any]:
    threshold_s = round(max(0.0, float(wall_s) - float(grace_s)), 6)
    elapsed_val = round(float(elapsed_s), 6)
    remaining_s = round(threshold_s - elapsed_val, 6)
    tripped = elapsed_val >= threshold_s
    return {
        "mode": mode,
        "tripped": tripped,
        "code": "WALL_GUARD_TRIPPED" if tripped else None,
        "remaining_s": remaining_s,
        "threshold_s": threshold_s,
    }


CURRENT_BALANCE_RE = re.compile(
    r"^\s*Current\s+balance:\s*([+-]?\d+(?:\.\d+)?)\s*(?:compute\s+units)?\s*$",
    re.I | re.M,
)
USAGE_RATE_RE = re.compile(
    r"^\s*Usage\s+rate:\s*([+-]?\d+(?:\.\d+)?)\s*/\s*hr?\s*$",
    re.I | re.M,
)
ACTIVE_ASSIGNMENTS_RE = re.compile(
    r"^\s*Active\s+assignments:\s*(\d+)\s*$",
    re.I | re.M,
)


class RateUnread(Exception):
    """A GPU window has no positive live post-open rate. `partial` carries what the usage text did
    give (balance, the readings), so a caller can record it; the caller decides whether this
    means "not yet" (polling right after open) or a refusal (RATE_UNREAD)."""

    code = "RATE_UNREAD"

    def __init__(self, partial: dict[str, Any]) -> None:
        super().__init__("RATE_UNREAD: no positive live post-open rate for a GPU window")
        self.partial = partial


CARD_CLASSES = ("gpu", "cpu")

# The window rate. `colab usage` reports the ACCOUNT's rate (docs/colab-cli-contract.md): a window's
# own rate is the account reading minus the neighbour baseline (at first the pre-open reading). It
# counts only when it is stable (two reads at least RATE_STABLE_GAP_S apart agreeing within
# RATE_STABLE_TOLERANCE) and at or above GPU_RATE_FLOOR. The floor is below the cheapest GPU
# measured, the T4 at 1.07/hr (16 reads, receipt windowC2-12-t4-s1a-20261027), and above the highest
# CPU rate, 0.26/hr. A GPU window is priced on max(--max-rate, the highest stable window rate seen),
# never on a window or account rate alone.
#
# "Active assignments" is recorded with every reading. When it changes, another window opened or
# closed. Nothing is judged until the count settles: two counted reads at the new count, at least
# RATE_STABLE_GAP_S apart, whose account rates agree within the tolerance (the count can arrive a
# read before the neighbour's rate). Then the neighbour is re-baselined: baseline = account - the
# highest stable rate of ours seen (not the last one: a low stable read taken while a neighbour's
# rate led its count would otherwise inflate the baseline and hide our later rise). Pending or settling reads are never judged against the max and never raise our
# priced rate, which stays max(--max-rate, highest stable seen). Only a read at an unchanged count
# may raise our rate or trip the max. Known limit: a neighbour opening on the same read our own rate
# rises folds that rise into the baseline; pricing at max(--max-rate, seen) bounds the cost.
#
# Before our window's first stable read, a missing count or 0 means "not yet registered": the
# receipts show the rate and the count arriving together (windowC2-12-t4-s1a-20261027 reads 0.00
# with 0 assignments until the session registers, then 1.07 with 1). Such a read is recorded,
# uncounted, and spends only the RATE_UNREAD poll budget. After the first stable read, a missing
# count or 0 is a bad read: recorded, the rate not judged, and ASSIGNMENTS_UNREADABLE after
# ASSIGNMENTS_BAD_LIMIT consecutive bad reads. After the first stable read the floor applies too: a
# stable pair below GPU_RATE_FLOOR (our window cannot bill under the cheapest GPU) is never
# accepted; it is recorded `below_floor` and counts as a bad read toward the same limit. Known
# limit: a rate that leads its count by two or more reads at open waits for the count before it
# can be stable.
GPU_RATE_FLOOR = 1.00
RATE_STABLE_TOLERANCE = 0.01
RATE_STABLE_GAP_S = 5.0
ASSIGNMENTS_BAD_LIMIT = 3


def parse_active_assignments(raw_usage: str) -> int | None:
    m = re.search(r"^\s*Active\s+assignments:\s*(\d+)\s*$", raw_usage or "", re.I | re.M)
    return int(m.group(1)) if m else None


def update_window_rate(state: dict[str, Any], *, account_rate: float | None, pre_open_rate: float | None,
                       at: str, epoch: float, assignments: int | None,
                       min_gap_s: float = RATE_STABLE_GAP_S, floor: float = GPU_RATE_FLOOR,
                       tolerance: float = RATE_STABLE_TOLERANCE,
                       pre_open_assignments: int | None = None) -> dict[str, Any]:
    """Fold one post-open `colab usage` reading into the window-rate state (kept in the out dir).
    Parallel windows (2026-11-12): with `pre_open_assignments` known, the first counted reading must
    show exactly one more assignment than before the open. The same count means this window has
    not registered yet (not counted, poll again); any other count means a neighbour opened or closed
    in between, so (account − pre-open) is not this window's rate: `unattributable`, never guessed.
    Any other assignment before or during the window sets `shared_account` (the balance is shared).
    Every reading is recorded. A bad count (missing, or 0) is recorded and judged nothing; three in
    a row set `assignments_unreadable`. A read less than `min_gap_s` after the last counted one is
    recorded as `counted: false`. A changed count starts a pending re-baseline that settles on two
    agreeing reads at the new count. Otherwise the window rate is `stable` when it is at or above
    the floor and agrees with the last counted window rate within the tolerance; `best_stable` is
    the highest stable one."""
    readings = state.setdefault("readings", [])
    state["floor"] = floor
    if "neighbour_baseline" not in state:
        state["neighbour_baseline"] = float(pre_open_rate) if pre_open_rate is not None else 0.0
    if pre_open_assignments is not None and "pre_open_assignments" not in state:
        state["pre_open_assignments"] = int(pre_open_assignments)
    pre_count = state.get("pre_open_assignments")
    if (pre_count is not None and int(pre_count) > 0) or (assignments is not None and int(assignments) > 1):
        state["shared_account"] = True
    base = float(state["neighbour_baseline"])
    rec: dict[str, Any] = {"at": at, "epoch": round(float(epoch), 3), "account_rate": account_rate,
                           "pre_open_rate": pre_open_rate, "assignments": assignments,
                           "pre_open_assignments": pre_count,
                           "neighbour_baseline": base, "window_rate": None, "stable": False}
    first_count = "last_assignments" not in state and state.get("best_stable") is None
    if (first_count and pre_count is not None and assignments is not None
            and int(assignments) == int(pre_count) and int(pre_count) > 0):
        # This window is not on the account's count yet (registration lag): judge nothing.
        rec.update({"counted": False, "not_registered": True})
        readings.append(rec)
        return state
    if (assignments is None or int(assignments) <= 0) and state.get("best_stable") is None:
        rec.update({"counted": False, "not_registered": True})
        readings.append(rec)
        return state
    if assignments is None or int(assignments) <= 0:
        state["assignments_bad_streak"] = int(state.get("assignments_bad_streak", 0)) + 1
        if state["assignments_bad_streak"] >= ASSIGNMENTS_BAD_LIMIT:
            state["assignments_unreadable"] = True
        rec.update({"counted": False, "assignments_bad": True})
        readings.append(rec)
        return state
    bad_streak_before = int(state.get("assignments_bad_streak", 0))
    state["assignments_bad_streak"] = 0
    last_epoch = state.get("last_counted_epoch")
    if last_epoch is not None and float(epoch) - float(last_epoch) < float(min_gap_s):
        rec["counted"] = False
        readings.append(rec)
        return state
    rec["counted"] = True
    state["last_counted_epoch"] = float(epoch)
    known = "last_assignments" in state
    if (not known and pre_count is not None and state.get("best_stable") is None
            and int(assignments) != int(pre_count) + 1):
        state["unattributable"] = True
        state["last_assignments"] = assignments
        rec.update({"unattributable": True, "expected_assignments": int(pre_count) + 1})
        readings.append(rec)
        return state
    prev_count = state.get("last_assignments")
    state["last_assignments"] = assignments
    pending = state.get("rebaseline_pending")
    if known and assignments != prev_count:
        # A new count: start (or restart) the pending re-baseline. Judge nothing.
        state["rebaseline_pending"] = {"count": assignments, "from": prev_count, "last_account": account_rate}
        rec.update({"rebaseline_pending": True, "assignments_before": prev_count})
        readings.append(rec)
        return state
    if pending is not None:
        last_acc = pending.get("last_account")
        if (account_rate is None or last_acc is None
                or abs(float(account_rate) - float(last_acc)) > tolerance):
            pending["last_account"] = account_rate
            rec["rebaseline_pending"] = True
            readings.append(rec)
            return state
        # Settled: two agreeing reads at the new count. The neighbour is what the account reads above
        # the highest stable rate of ours seen: the conservative direction (our later reads can only
        # come out higher, never hidden under an inflated baseline).
        ours = state.get("best_stable")
        if ours is not None:
            base = round(max(float(account_rate) - float(ours), 0.0), 4)
            state["neighbour_baseline"] = base
        state["last_window"] = ours
        state.pop("rebaseline_pending", None)
        rec.update({"rebaselined": True, "window_rate": ours, "neighbour_baseline": base,
                    "assignments_before": pending.get("from")})
        readings.append(rec)
        return state
    window = None if account_rate is None else round(max(float(account_rate) - base, 0.0), 4)
    prev = state.get("last_window")
    stable = (window is not None and window >= floor and prev is not None
              and abs(window - float(prev)) <= tolerance)
    below_floor = (window is not None and window < floor and prev is not None
                   and abs(window - float(prev)) <= tolerance and state.get("best_stable") is not None)
    rec.update({"window_rate": window, "stable": stable})
    if below_floor:
        # Mid-run, a stable own rate below the floor is a bad read, never an accepted rate.
        state["assignments_bad_streak"] = bad_streak_before + 1
        if state["assignments_bad_streak"] >= ASSIGNMENTS_BAD_LIMIT:
            state["assignments_unreadable"] = True
        rec["below_floor"] = True
    readings.append(rec)
    state["last_window"] = window
    if stable:
        state["last_stable"] = window
        if state.get("best_stable") is None or window > state["best_stable"]:
            state["best_stable"] = window
    return state


def gpu_projection_rate(state: dict[str, Any], max_rate: float) -> float:
    """The rate a GPU window's cap is projected with: max(--max-rate, highest stable window rate)."""
    best = state.get("best_stable")
    return max(float(max_rate), float(best)) if best is not None else float(max_rate)


def window_rate_decision(state: dict[str, Any], max_rate: float) -> str:
    """`assignments_unreadable`, `unread` (no stable window rate at or above the floor yet),
    `above_max` or `ok`."""
    if state.get("assignments_unreadable"):
        return "assignments_unreadable"
    if state.get("unattributable") and state.get("best_stable") is None:
        return "unattributable"
    best = state.get("best_stable")
    if best is None:
        return "unread"
    return "above_max" if float(best) > float(max_rate) else "ok"


def parse_usage_text(
    raw_usage: str,
    card_class: str,
    *,
    usage_rc: int = 0,
    configured_rate: float = 0.26,
    session_opened: bool = True,
    post_open_rate: float | None = None,
    pre_open_rate: float | None = None,
) -> dict[str, Any]:
    """Parse `colab usage`. `card_class` ("gpu" or "cpu", required) decides the rate policy: a GPU
    window is priced only by its live post-open rate, and `_select_rate` raises RateUnread when it
    has none, so no GPU caller can reach a configured-rate branch. A CPU stub keeps the labelled
    configured rate."""
    if card_class not in CARD_CLASSES:
        raise ValueError(f"card_class must be one of {CARD_CLASSES}, got {card_class!r}")
    cfg = float(configured_rate)

    def _select_rate(
        cur_measured: float | None,
    ) -> tuple[float | None, float | None, float, str]:
        eff_pre = cur_measured if not session_opened else pre_open_rate
        existing_pos_post = post_open_rate if (post_open_rate is not None and post_open_rate > 0.0) else None
        if existing_pos_post is not None:
            eff_post: float | None = existing_pos_post
        elif session_opened and cur_measured is not None:
            eff_post = cur_measured
        elif post_open_rate is not None:
            eff_post = post_open_rate
        else:
            eff_post = None

        if eff_post is not None and eff_post > 0.0:
            return eff_pre, eff_post, float(eff_post), "post_open_measured"
        if card_class == "gpu":
            raise RateUnread({"rate_measured_pre_open": eff_pre, "rate_measured_post_open": eff_post})
        if eff_post is not None and eff_post == 0.0:
            pos_pre = float(eff_pre) if (eff_pre is not None and eff_pre > 0.0) else 0.0
            return eff_pre, eff_post, max(cfg, pos_pre), "configured_pending_post_open_rate"
        if eff_pre is not None:
            return eff_pre, None, max(float(eff_pre), cfg), "max_of_pre_open_and_configured"
        return None, None, cfg, "configured"

    def _failure_rates() -> tuple[float | None, float | None, float | None, str]:
        # A usage read that failed is COLAB_USAGE_PARSE_FAILED (the cap guard trips on it); for a
        # GPU window without a live rate it carries no projection rate at all, never a configured one.
        try:
            return _select_rate(None)
        except RateUnread as exc:
            return (exc.partial["rate_measured_pre_open"], exc.partial["rate_measured_post_open"], None,
                    "live_rate_unread")

    if usage_rc != 0:
        eff_pre, eff_post, rate_used, rate_src = _failure_rates()
        return {
            "ok": False,
            "code": "COLAB_USAGE_PARSE_FAILED",
            "balance": None,
            "rate_measured": None,
            "rate_measured_pre_open": eff_pre,
            "rate_measured_post_open": eff_post,
            "rate_for_projection": rate_used,
            "rate_used_for_cap_and_projection": rate_used,
            "rate_source": rate_src,
            "active_assignments": None,
            "error": f"COLAB_USAGE_PARSE_FAILED: colab usage exited with status {usage_rc}",
        }
    text = raw_usage or ""
    m_bal = CURRENT_BALANCE_RE.search(text)
    if not m_bal:
        eff_pre, eff_post, rate_used, rate_src = _failure_rates()
        return {
            "ok": False,
            "code": "COLAB_USAGE_PARSE_FAILED",
            "balance": None,
            "rate_measured": None,
            "rate_measured_pre_open": eff_pre,
            "rate_measured_post_open": eff_post,
            "rate_for_projection": rate_used,
            "rate_used_for_cap_and_projection": rate_used,
            "rate_source": rate_src,
            "active_assignments": None,
            "error": "COLAB_USAGE_PARSE_FAILED: missing 'Current balance: <float> compute units'",
        }
    balance = float(m_bal.group(1))
    if balance < 0.0:
        eff_pre, eff_post, rate_used, rate_src = _failure_rates()
        return {
            "ok": False,
            "code": "COLAB_USAGE_PARSE_FAILED",
            "balance": None,
            "rate_measured": None,
            "rate_measured_pre_open": eff_pre,
            "rate_measured_post_open": eff_post,
            "rate_for_projection": rate_used,
            "rate_used_for_cap_and_projection": rate_used,
            "rate_source": rate_src,
            "active_assignments": None,
            "error": f"COLAB_USAGE_PARSE_FAILED: negative balance {balance} is impossible",
        }
    m_rate = USAGE_RATE_RE.search(text)
    if m_rate:
        rate_measured: float | None = float(m_rate.group(1))
        if rate_measured < 0.0:
            eff_pre, eff_post, rate_used, rate_src = _failure_rates()
            return {
                "ok": False,
                "code": "COLAB_USAGE_PARSE_FAILED",
                "balance": None,
                "rate_measured": None,
                "rate_measured_pre_open": eff_pre,
                "rate_measured_post_open": eff_post,
                "rate_for_projection": rate_used,
                "rate_used_for_cap_and_projection": rate_used,
                "rate_source": rate_src,
                "active_assignments": None,
                "error": f"COLAB_USAGE_PARSE_FAILED: negative usage rate {rate_measured} is impossible",
            }
    else:
        rate_measured = None
    m_assign = ACTIVE_ASSIGNMENTS_RE.search(text)
    active_assignments = int(m_assign.group(1)) if m_assign else None
    try:
        eff_pre, eff_post, rate_for_projection, rate_source = _select_rate(rate_measured)
    except RateUnread as exc:
        exc.partial.update({"balance": balance, "rate_measured": rate_measured,
                            "active_assignments": active_assignments})
        raise
    return {
        "ok": True,
        "code": None,
        "balance": balance,
        "rate_measured": rate_measured,
        "rate_measured_pre_open": eff_pre,
        "rate_measured_post_open": eff_post,
        "rate_for_projection": rate_for_projection,
        "rate_used_for_cap_and_projection": rate_for_projection,
        "rate_source": rate_source,
        "active_assignments": active_assignments,
        "error": None,
    }


CAP_TRIP_FORMULA = (
    "effective_trip = cap - margin - cleanup_reserve; cleanup_reserve = (cleanup_bound_s / 3600) * priced_rate"
)


def compute_cap_trip(
    cap: float,
    margin: float,
    cleanup_bound_s: float = 0.0,
    priced_rate: float = 0.0,
) -> dict[str, Any]:
    bound_s = max(0.0, float(cleanup_bound_s))
    rate = max(0.0, float(priced_rate))
    cleanup_reserve = round((bound_s / 3600.0) * rate, 6)
    effective_trip = round(float(cap) - float(margin) - cleanup_reserve, 6)
    ceiling = round(max(0.0, effective_trip), 6)
    return {
        "cap_ccu": round(float(cap), 6),
        "margin_ccu": round(float(margin), 6),
        "cleanup_bound_s": int(bound_s) if bound_s == int(bound_s) else round(bound_s, 6),
        "priced_rate": round(rate, 6),
        "cleanup_reserve_units": cleanup_reserve,
        "effective_trip_units": effective_trip,
        "ceiling_ccu": ceiling,
        "formula": CAP_TRIP_FORMULA,
        "cap_trip_formula": CAP_TRIP_FORMULA,
    }


def evaluate_cap_budget(
    *,
    cap: float,
    margin: float,
    cleanup_bound_s: float,
    priced_rate: float,
    max_rate: float | None = None,
    min_runtime_s: float | None = None,
) -> dict[str, Any]:
    trip = compute_cap_trip(cap=cap, margin=margin, cleanup_bound_s=cleanup_bound_s, priced_rate=priced_rate)
    eff_max_rate = round(float(max_rate), 6) if max_rate is not None else trip["priced_rate"]
    effective_trip = trip["effective_trip_units"]
    cleanup_reserve = trip["cleanup_reserve_units"]
    p_rate = trip["priced_rate"]
    bound_s = trip["cleanup_bound_s"]
    cap_v = trip["cap_ccu"]
    margin_v = trip["margin_ccu"]

    runtime_at_priced_rate_s = round((effective_trip / p_rate) * 3600.0, 2) if p_rate > 0.0 else 0.0
    runtime_at_max_rate_s = (
        round((effective_trip / eff_max_rate) * 3600.0, 2) if eff_max_rate > 0.0 else runtime_at_priced_rate_s
    )
    death_time_exact_s = max(0.0, (effective_trip / p_rate) * 3600.0) if p_rate > 0.0 else 0.0
    death_time_max_s = max(0.0, runtime_at_priced_rate_s)
    worst_case_cap_path = round(effective_trip + margin_v + cleanup_reserve, 6)
    worst_case_launcher_death_path = round(((death_time_exact_s + bound_s) / 3600.0) * p_rate, 6)
    worst_case_total_spend = round(max(worst_case_cap_path, worst_case_launcher_death_path), 6)
    assumptions = {
        "cap_path": "live launcher trips cap guard and runs cleanup_on_exit",
        "launcher_death_path": (
            "wall watcher survives launcher death to poll kill -0 every 1 s and run bounded colab stop"
        ),
    }

    tripped = False
    code: str | None = None
    why: str | None = None
    if effective_trip <= 0.0:
        tripped = True
        code = "CAP_TOO_SMALL_FOR_CLEANUP"
        why = (
            f"effective_trip = cap {cap_v} - margin {margin_v} - cleanup_reserve {cleanup_reserve} "
            f"(cleanup_bound_s {bound_s} / 3600 * priced_rate {p_rate}/h) = {effective_trip} <= 0"
        )
    elif min_runtime_s is not None and runtime_at_priced_rate_s < float(min_runtime_s):
        min_rt = (
            int(min_runtime_s)
            if float(min_runtime_s) == int(float(min_runtime_s))
            else round(float(min_runtime_s), 2)
        )
        tripped = True
        code = "CAP_TOO_SMALL_FOR_CLEANUP"
        why = (
            f"effective_trip = cap {cap_v} - margin {margin_v} - cleanup_reserve {cleanup_reserve} "
            f"(cleanup_bound_s {bound_s} / 3600 * priced_rate {p_rate}/h) = {effective_trip} units "
            f"leaves {runtime_at_priced_rate_s} s at {p_rate}/h < --min-runtime-s {min_rt} s"
        )

    rec: dict[str, Any] = {
        "check": "cap_budget",
        "ok": not tripped,
        "tripped": tripped,
        "code": code,
        "why": why,
        "cap": cap_v,
        "cap_ccu": cap_v,
        "margin": margin_v,
        "margin_ccu": margin_v,
        "cleanup_bound_s": bound_s,
        "priced_rate": p_rate,
        "max_rate": eff_max_rate,
        "reserve": cleanup_reserve,
        "cleanup_reserve": cleanup_reserve,
        "cleanup_reserve_units": cleanup_reserve,
        "effective_trip": effective_trip,
        "effective_trip_units": effective_trip,
        "runtime_to_trip_s": runtime_at_priced_rate_s,
        "runtime_at_priced_rate_s": runtime_at_priced_rate_s,
        "runtime_at_max_rate_s": runtime_at_max_rate_s,
        "death_time_max_s": death_time_max_s,
        "min_runtime_s": (
            None
            if min_runtime_s is None
            else (
                int(min_runtime_s)
                if float(min_runtime_s) == int(float(min_runtime_s))
                else round(float(min_runtime_s), 2)
            )
        ),
        "worst_case_cap_path": worst_case_cap_path,
        "worst_case_launcher_death_path": worst_case_launcher_death_path,
        "worst_case_total_spend": worst_case_total_spend,
        "worst_case_total_spend_units": worst_case_total_spend,
        "assumptions": assumptions,
        "formula": CAP_TRIP_FORMULA,
        "cap_trip_formula": CAP_TRIP_FORMULA,
    }
    if tripped and code and why:
        rec["refusal"] = {"code": code, "why": why}
    return rec


def format_dry_cap_budget(rec: dict[str, Any]) -> str:
    assumptions = rec.get("assumptions") or {}
    lines = [
        f"formula: {rec['formula']}",
        f"cap: {rec['cap']}",
        f"margin: {rec['margin']}",
        f"cleanup_bound_s: {rec['cleanup_bound_s']}",
        f"priced_rate: {rec['priced_rate']}/h",
        f"max_rate: {rec['max_rate']}/h",
        f"reserve (cleanup_reserve): {rec['cleanup_reserve_units']} ({rec['cleanup_bound_s']} / 3600 * {rec['priced_rate']})",
        f"effective_trip: {rec['effective_trip_units']} ({rec['cap']} - {rec['margin']} - {rec['cleanup_reserve_units']})",
        f"runtime_to_trip (priced_rate {rec['priced_rate']}/h): {rec['runtime_at_priced_rate_s']} s",
        f"runtime_to_trip (--max-rate {rec['max_rate']}/h): {rec['runtime_at_max_rate_s']} s",
        f"worst_case_cap_path: {rec['worst_case_cap_path']} (= cap {rec['cap']})",
        (
            f"worst_case_launcher_death_path: {rec['worst_case_launcher_death_path']} "
            f"((death_time <= {rec['death_time_max_s']} s + cleanup_bound_s {rec['cleanup_bound_s']} s) / 3600 * {rec['priced_rate']}/h)"
        ),
        (
            f"worst_case_total_spend: {rec['worst_case_total_spend']} "
            f"(= max(cap_path {rec['worst_case_cap_path']}, launcher_death_path {rec['worst_case_launcher_death_path']}) = cap {rec['cap']})"
        ),
        (
            f"assumptions: cap_path assumes {assumptions.get('cap_path', '')}; "
            f"launcher_death_path assumes {assumptions.get('launcher_death_path', '')}"
        ),
        json.dumps(rec),
    ]
    return "\n".join(lines)


def evaluate_cap_guard(
    *,
    mode: str = "live",
    cap: float = 0.05,
    margin: float = 0.01,
    cleanup_bound_s: float = 0.0,
    start_balance: float | None = None,
    current_balance: float | None = None,
    rate_ccu_per_hour: float = 0.0,
    elapsed_s: float = 0.0,
    usage_parse_error: str | None = None,
    phase: str | None = None,
    checkpoint: str | None = None,
) -> dict[str, Any]:
    trip = compute_cap_trip(
        cap=cap, margin=margin, cleanup_bound_s=cleanup_bound_s, priced_rate=rate_ccu_per_hour
    )
    ceiling = trip["ceiling_ccu"]
    extra: dict[str, Any] = {
        "cap_ccu": trip["cap_ccu"],
        "margin_ccu": trip["margin_ccu"],
        "cleanup_bound_s": trip["cleanup_bound_s"],
        "priced_rate": trip["priced_rate"],
        "cleanup_reserve_units": trip["cleanup_reserve_units"],
        "effective_trip_units": trip["effective_trip_units"],
        "formula": CAP_TRIP_FORMULA,
        "cap_trip_formula": CAP_TRIP_FORMULA,
    }
    if phase is not None:
        extra["phase"] = phase
    if checkpoint is not None:
        extra["checkpoint"] = checkpoint
    if usage_parse_error:
        return {
            "mode": mode,
            "tripped": True,
            "code": "COLAB_USAGE_PARSE_FAILED",
            "spend_ccu": None,
            "ceiling_ccu": ceiling,
            **extra,
            "usage_parse_error": usage_parse_error,
        }
    if (start_balance is not None and float(start_balance) < 0.0) or (
        current_balance is not None and float(current_balance) < 0.0
    ):
        return {
            "mode": mode,
            "tripped": True,
            "code": "COLAB_USAGE_PARSE_FAILED",
            "spend_ccu": None,
            "ceiling_ccu": ceiling,
            **extra,
            "usage_parse_error": "COLAB_USAGE_PARSE_FAILED: negative balance is impossible",
        }
    if mode not in ("fake", "dry_run", "offline") and (start_balance is None or current_balance is None):
        return {
            "mode": mode,
            "tripped": True,
            "code": "COLAB_USAGE_PARSE_FAILED",
            "spend_ccu": None,
            "ceiling_ccu": ceiling,
            **extra,
            "usage_parse_error": "COLAB_USAGE_PARSE_FAILED: missing start_balance or current_balance in non-fake mode",
        }
    drop = (
        round(max(0.0, float(start_balance) - float(current_balance)), 6)
        if start_balance is not None and current_balance is not None
        else 0.0
    )
    rate_est = round(max(0.0, (float(rate_ccu_per_hour) * float(elapsed_s)) / 3600.0), 6)
    spend = round(max(drop, rate_est), 6)
    tripped = spend >= ceiling
    return {
        "mode": mode,
        "tripped": tripped,
        "code": "CAP_GUARD_TRIPPED" if tripped else None,
        "spend_ccu": spend,
        "ceiling_ccu": ceiling,
        **extra,
    }


def parse_vitals_log(vitals_text: str) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for block in _vitals_blocks(vitals_text):
        if not _HARNESS_LINE_RE.search(block):
            continue
        lines = block.splitlines()
        header = lines[0] if lines else ""
        ts = header.split()[0] if header else ""
        phase = None
        worker_count = 0
        oom_kill = False
        setup_bytes: int | None = None
        in_proc = False
        for line in lines[1:]:
            if OOM_KILL_RE.search(line):
                oom_kill = True
            if line.startswith("SETUP_BYTES:"):
                in_proc = False
                raw_sb = line[len("SETUP_BYTES:") :].strip()
                if raw_sb.isdigit():
                    setup_bytes = int(raw_sb)
                continue
            if line.startswith("PROC:"):
                in_proc = True
                continue
            if line.startswith("HARNESS:"):
                in_proc = False
                payload = line[len("HARNESS:") :].strip()
                try:
                    phase = json.loads(payload).get("phase")
                except Exception:
                    m = re.search(r'"phase"\s*:\s*"([^"]+)"', payload) or re.search(
                        r"phase:\s*([a-z0-9_-]+)", payload, re.I
                    )
                    phase = m.group(1) if m else None
                continue
            if in_proc and WORKER_PROC_RE.search(line):
                worker_count += 1
        samples.append(
            {
                "timestamp": ts,
                "phase": phase,
                "worker_proc_count": worker_count,
                "oom_kill": oom_kill,
                "setup_bytes": setup_bytes,
            }
        )
    return samples


def evaluate_vitals_and_stall_guard(
    *,
    vitals_text: str,
    idle_seconds: float = 0.0,
    setup_idle_seconds: float | None = None,
    stall_s: float = 300.0,
    setup_stall_s: float = 600.0,
) -> dict[str, Any]:
    raw_blocks = _vitals_blocks(vitals_text)
    samples = parse_vitals_log(vitals_text)
    current_phase: str | None = None
    if samples:
        last = samples[-1]
        raw_phase = last["phase"]
        if not isinstance(raw_phase, str) or not raw_phase.strip():
            return {
                "tripped": True,
                "code": "VITALS_PHASE_UNREADABLE",
                "cgroup_oom": False,
                "reason": "Newest complete vitals block has no HARNESS phase; refusing rather than assuming setup",
                "samples_count": len(samples),
            }
        current_phase = raw_phase.strip()
        if (
            (current_phase.startswith("run") or current_phase.startswith("setup"))
            and last["worker_proc_count"] == 0
            and last.get("oom_kill")
        ):
            return {
                "tripped": True,
                "code": "DEAD_HARNESS_TRIPPED",
                "cgroup_oom": True,
                "reason": f"cgroup OOM kill detected in phase={current_phase} with 0 active worker processes",
                "samples_count": len(samples),
            }
    elif len(raw_blocks) >= 2:
        return {
            "tripped": True,
            "code": "VITALS_PHASE_UNREADABLE",
            "cgroup_oom": False,
            "reason": f"{len(raw_blocks)} vitals blocks and none is complete (missing HARNESS: line); refusing rather than assuming setup",
            "samples_count": 0,
        }
    if len(samples) >= 2:
        last_two = samples[-2:]
        if all(
            isinstance(s["phase"], str) and s["phase"].startswith("run") and s["worker_proc_count"] == 0
            for s in last_two
        ):
            return {
                "tripped": True,
                "code": "DEAD_HARNESS_TRIPPED",
                "cgroup_oom": any(bool(s.get("oom_kill")) for s in last_two),
                "reason": "2 consecutive vitals samples in phase=run with 0 active worker processes",
                "samples_count": len(samples),
            }
    if current_phase is not None and current_phase.startswith("setup"):
        eff_setup_idle = float(setup_idle_seconds if setup_idle_seconds is not None else idle_seconds)
        if eff_setup_idle >= float(setup_stall_s):
            return {
                "tripped": True,
                "code": "SETUP_STALL_TRIPPED",
                "cgroup_oom": False,
                "reason": f"No new setup.log output (SETUP_BYTES unchanged) in phase=setup for {eff_setup_idle}s (setup_stall_s={setup_stall_s}s)",
                "samples_count": len(samples),
            }
    elif float(idle_seconds) >= float(stall_s):
        return {
            "tripped": True,
            "code": "STALL_GUARD_TRIPPED",
            "cgroup_oom": False,
            "reason": f"No output row or vitals progress for {idle_seconds}s (stall_s={stall_s}s)",
            "samples_count": len(samples),
        }
    return {
        "tripped": False,
        "code": None,
        "cgroup_oom": False,
        "reason": None,
        "samples_count": len(samples),
    }


# The host-RAM guard measures PRIVATE memory: pages the kernel cannot drop to make room, so
# pages that end in an OOM kill when they outgrow the VM. That is anonymous memory plus shmem
# (tmpfs, /dev/shm, PyTorch's shared-memory tensors): Colab VMs run without swap, so shmem is
# as unreclaimable as anon. It is NOT cgroup memory.current, which also counts the page cache
# (file pages: a downloaded or mmap'd model), and the kernel reclaims that cache before it
# OOM-kills anything. Window sdlab-l3-rung-g4a (2026-11-06) tripped on memory.current: 173,117
# MiB of which AnonPages was 52,492 MiB, while MemAvailable still read 125,790 MiB.
#
# Two checks, either one trips HOST_RAM_PROJECTION_EXCEEDED:
#   1. (private + delta) x factor > ceiling, ceiling = min(MemTotal, cgroup memory.max when it is
#      a number; `max` is no cgroup limit).
#   2. MemAvailable - delta x factor < floor. MemAvailable is the kernel's own estimate of what
#      can be had without swapping, reclaimable cache included; it also falls for unreclaimable
#      memory that is not anon (kernel slab, page tables, driver-pinned buffers), which check 1
#      cannot see. The floor (2048 MiB) is the one the lane-3 walks stop at (walk_probe.py
#      HOST_RAM_CEILING: MemAvailable < 2 GiB).
MEM_AVAILABLE_FLOOR_MB = 2048.0
RAM_SAFETY_FACTOR = 1.05
# Field lines end in a newline: a number the poll caught mid-write ("anon 55" at the end of the
# buffer) is not a reading.
_KB_FIELD_RE = {
    name: re.compile(rf"^MEM:[ \t]*{name}:[ \t]*(\d+)[ \t]*kB[ \t]*\r?\n", re.M)
    for name in ("MemTotal", "MemAvailable", "AnonPages", "Shmem")
}
_CGROUP_STAT_RE = re.compile(r"^CGROUP_MEM_STAT:[ \t]*(anon|shmem|file)[ \t]+(\d+)[ \t]*\r?\n", re.M)
_CGROUP_MAX_RE = re.compile(r"^CGROUP_MEM_MAX:[ \t]*(\S+)[ \t]*\r?\n", re.M)
_CGROUP_CUR_RE = re.compile(r"^CGROUP_MEM_CURRENT:[ \t]*(\d+)[ \t]*\r?\n", re.M)
_HARNESS_LINE_RE = re.compile(r"^HARNESS:.*\r?\n", re.M)
# The newest complete sample must be the newest or the one before it; older than that and the
# guard would be steering on a stale reading (HOST_RAM_UNREADABLE).
RAM_READING_MAX_BLOCKS_BACK = 1


def _ram_from_block(block: str) -> dict[str, Any] | None:
    """The private-memory reading of ONE complete vitals sample, or None.

    Complete means the emitter reached its last line (HARNESS:) and the sample has MemTotal,
    MemAvailable and at least one private-memory source: the full cgroup memory.stat triple (anon,
    shmem, file) and/or /proc/meminfo AnonPages (+ Shmem). Both sources are recorded when present,
    and the larger private reading is selected for projection and ceiling checks so a cgroup slice
    that misses the trainer never under-reads private memory. A sample cut mid-write anywhere is
    None, so a torn tail never reads as 0 MB private or as a missing MemTotal."""
    if not _HARNESS_LINE_RE.search(block):
        return None
    kb: dict[str, float] = {}
    for name, rx in _KB_FIELD_RE.items():
        found = rx.findall(block)
        if found:
            kb[name] = round(int(found[-1]) / 1024.0, 1)
    if "MemTotal" not in kb or "MemAvailable" not in kb:
        return None
    stat = {k: int(v) for k, v in _CGROUP_STAT_RE.findall(block)}
    if {"anon", "shmem", "file"} <= stat.keys():
        cgroup_anon_mb: float | None = round(stat["anon"] / 1048576.0, 1)
        cgroup_shmem_mb: float | None = round(stat["shmem"] / 1048576.0, 1)
        cgroup_private_mb: float | None = round(cgroup_anon_mb + cgroup_shmem_mb, 1)
    else:
        cgroup_anon_mb = None
        cgroup_shmem_mb = None
        cgroup_private_mb = None
    if "AnonPages" in kb:
        meminfo_anon_mb: float | None = kb["AnonPages"]
        meminfo_shmem_mb: float | None = kb.get("Shmem", 0.0)
        meminfo_private_mb: float | None = round(meminfo_anon_mb + meminfo_shmem_mb, 1)
        meminfo_source = "meminfo:AnonPages+Shmem" if "Shmem" in kb else "meminfo:AnonPages"
    else:
        meminfo_anon_mb = None
        meminfo_shmem_mb = None
        meminfo_private_mb = None
        meminfo_source = None
    if cgroup_private_mb is None and meminfo_private_mb is None:
        return None
    if meminfo_private_mb is not None and (
        cgroup_private_mb is None or meminfo_private_mb > cgroup_private_mb
    ):
        private_mb = meminfo_private_mb
        anon_mb = meminfo_anon_mb
        shmem_mb = meminfo_shmem_mb
        source = meminfo_source
    else:
        private_mb = cgroup_private_mb
        anon_mb = cgroup_anon_mb
        shmem_mb = cgroup_shmem_mb
        source = "cgroup_memory.stat:anon+shmem"
    cg_max_raw = _CGROUP_MAX_RE.findall(block)
    cg_max_mb = None
    if cg_max_raw and cg_max_raw[-1].isdigit():
        cg_max_mb = round(int(cg_max_raw[-1]) / 1048576.0, 1)
    cg_cur = _CGROUP_CUR_RE.findall(block)
    return {
        "timestamp": block.split()[0] if block.split() else "",
        "private_mb": private_mb,
        "anon_mb": anon_mb,
        "shmem_mb": shmem_mb,
        "source": source,
        "cgroup_private_mb": cgroup_private_mb,
        "cgroup_anon_mb": cgroup_anon_mb,
        "cgroup_shmem_mb": cgroup_shmem_mb,
        "meminfo_private_mb": meminfo_private_mb,
        "meminfo_anon_mb": meminfo_anon_mb,
        "meminfo_shmem_mb": meminfo_shmem_mb,
        "mem_total_mb": kb["MemTotal"],
        "mem_available_mb": kb["MemAvailable"],
        "cgroup_max_mb": cg_max_mb,
        "cgroup_max_raw": cg_max_raw[-1] if cg_max_raw else None,
        "cgroup_current_mb_not_used": round(int(cg_cur[-1]) / 1048576.0, 1) if cg_cur else None,
    }


# A sample header is split on wherever it appears, not only at column 0: job output without a
# trailing newline (a "\r" progress bar) can put "=== VITALS" mid-line. Each block is then cut
# at the end of its first HARNESS: line, so no block can combine fields of two samples.
_VITALS_HEADER_RE = re.compile(r"===[ \t]*VITALS[ \t]+")


def _vitals_blocks(vitals_text: str) -> list[str]:
    blocks = []
    for part in _VITALS_HEADER_RE.split(vitals_text)[1:]:
        m = _HARNESS_LINE_RE.search(part)
        if m:
            part = part[: m.end()]
        if part.strip():
            blocks.append(part)
    return blocks


def host_ram_reading(vitals_text: str) -> dict[str, Any] | None:
    """The newest COMPLETE vitals sample's private-memory reading (see _ram_from_block), or None.

    Private memory records both cgroup memory.stat anon + shmem and /proc/meminfo AnonPages + Shmem
    from that one sample and selects the larger of the two. cgroup memory.current is recorded for
    the record only and never used. `blocks_back` is how many samples older than the newest it is."""
    blocks = _vitals_blocks(vitals_text)
    for back, block in enumerate(reversed(blocks)):
        reading = _ram_from_block(block)
        if reading is not None:
            reading["blocks_back"] = back
            reading["blocks_seen"] = len(blocks)
            return reading
    return None


def host_ram_status(vitals_text: str | None) -> dict[str, Any]:
    """What the live RAM guard can do with this vitals stream. No default value ever stands in for
    a reading:
      - "no_vm": no vitals stream yet (the pre-open check, or before the exec starts): skip.
      - "waiting": fewer than two samples and none complete (the first one is still being
        written): skip this poll.
      - "ok": a complete sample that is the newest or the one before it (a torn newest sample
        falls back to the previous one).
      - "unreadable": two or more samples and neither the newest nor the one before is complete.
        Samples come 15 s apart and each is written in milliseconds, so this is an emitter that
        does not print what the guard needs, not a race: the guard is blind, and the window stops
        (HOST_RAM_UNREADABLE) rather than run unguarded."""
    if vitals_text is None:
        return {"state": "no_vm", "reading": None, "blocks_seen": 0}
    blocks = _vitals_blocks(vitals_text)
    reading = host_ram_reading(vitals_text)
    if reading is not None and reading["blocks_back"] <= RAM_READING_MAX_BLOCKS_BACK:
        return {"state": "ok", "reading": reading, "blocks_seen": len(blocks)}
    if len(blocks) < 2 and reading is None:
        return {"state": "waiting", "reading": None, "blocks_seen": len(blocks)}
    newest = "none" if reading is None else f"the newest is {reading['blocks_back']} samples back"
    return {
        "state": "unreadable",
        "reading": None,
        "blocks_seen": len(blocks),
        "why": (f"{len(blocks)} vitals samples and neither of the last two is complete ({newest}); a complete "
                "sample has MemTotal, MemAvailable and AnonPages or memory.stat anon/shmem/file before HARNESS:"),
    }


# Every line the vitals emitters print (the `=== VITALS` loop in launch_lane1_stub.sh and
# launch_lane3_stub.sh, and the fake CLI's emit_vitals). The stall guard counts only run output,
# so these lines must never count as progress; tests/colab/test_host_ram_guard.py checks every
# emitter against this list, so a line added to an emitter cannot slip past it.
VITALS_LINE_PREFIXES = (
    "=== VITALS",
    "CGROUP_MEM_CURRENT:",
    "CGROUP_MEM_MAX:",
    "CGROUP_MEM:",
    "CGROUP_MEM_STAT:",
    "MEM:",
    "SETUP_BYTES:",
    "PROC:",
    "HARNESS:",
)
# A `ps -ef` row for a worker in a PROC: section ("root  2216  2101 21 19:01 ? 00:00:02 ...run.py
# ..."), or the fake's shorter "<pid> <cmd>" row. The old shell filter matched only rows that
# start with a number, so every real `ps -ef` row (it starts with the user) counted as progress.
_PS_WORKER_ROW_RE = re.compile(r"^\s*(?:[A-Za-z_][\w.+-]*\s+)?\d+\s+.*" + WORKER_PROC_RE.pattern, re.I)


def is_vitals_line(line: str) -> bool:
    stripped = line.rstrip("\r\n")
    if stripped.startswith(VITALS_LINE_PREFIXES[1:]) or re.match(r"^===\s*VITALS", stripped):
        return True
    return bool(_PS_WORKER_ROW_RE.match(stripped))


def run_output_bytes(vitals_text: str) -> int:
    """Bytes of the exec stream that are not vitals lines: what the stall guard counts as progress."""
    return sum(len(line.encode("utf-8", errors="replace")) for line in vitals_text.splitlines(keepends=True)
               if not is_vitals_line(line))


def evaluate_host_ram_projection(
    *,
    current_anon_mb: float,
    projected_delta_mb: float,
    mem_total_mb: float,
    cgroup_max_mb: float | None = None,
    mem_available_mb: float | None = None,
    mem_available_floor_mb: float = MEM_AVAILABLE_FLOOR_MB,
    safety_factor: float = RAM_SAFETY_FACTOR,
    source: str | None = None,
) -> dict[str, Any]:
    """Both host-RAM checks (see the comment above MEM_AVAILABLE_FLOOR_MB). The record carries every
    number either check used, tripped or not, so guards.log shows why a window stopped."""
    limit_mb = float(mem_total_mb)
    limit_source = "MemTotal"
    if cgroup_max_mb is not None and float(cgroup_max_mb) < limit_mb:
        limit_mb = float(cgroup_max_mb)
        limit_source = "cgroup memory.max"
    factor = float(safety_factor)
    delta = float(projected_delta_mb)
    projected_mb = round((float(current_anon_mb) + delta) * factor, 1)
    tripped_on: list[str] = []
    why: list[str] = []
    if projected_mb > limit_mb:
        tripped_on.append("private_projection")
        why.append(f"private {float(current_anon_mb):.1f} MB ({source or 'unknown source'}) + delta {delta:.1f} MB "
                   f"x {factor} = {projected_mb:.1f} MB > ceiling {limit_mb:.1f} MB ({limit_source})")
    avail_after = None
    if mem_available_mb is not None:
        avail_after = round(float(mem_available_mb) - delta * factor, 1)
        if avail_after < float(mem_available_floor_mb):
            tripped_on.append("mem_available_floor")
            why.append(f"MemAvailable {float(mem_available_mb):.1f} MB - delta {delta:.1f} MB x {factor} = "
                       f"{avail_after:.1f} MB < floor {float(mem_available_floor_mb):.1f} MB")
    tripped = bool(tripped_on)
    return {
        "check": "host_ram",
        "ok": not tripped,
        "tripped": tripped,
        "code": "HOST_RAM_PROJECTION_EXCEEDED" if tripped else None,
        "tripped_on": tripped_on,
        "source": source,
        "private_mb": round(float(current_anon_mb), 1),
        "delta_mb": round(delta, 1),
        "factor": factor,
        "projected_mb": projected_mb,
        "limit_mb": limit_mb,
        "limit_source": limit_source,
        "mem_total_mb": float(mem_total_mb),
        "cgroup_max_mb": None if cgroup_max_mb is None else float(cgroup_max_mb),
        "mem_available_mb": None if mem_available_mb is None else float(mem_available_mb),
        "mem_available_after_delta_mb": avail_after,
        "mem_available_floor_mb": float(mem_available_floor_mb),
        "why": "; ".join(why) if why else None,
    }


def check_host_ram_projection(
    *,
    current_anon_mb: float,
    projected_delta_mb: float,
    mem_total_mb: float,
    cgroup_max_mb: float | None = None,
    safety_factor: float = RAM_SAFETY_FACTOR,
    mem_available_mb: float | None = None,
    mem_available_floor_mb: float = MEM_AVAILABLE_FLOOR_MB,
    source: str | None = None,
) -> dict[str, Any]:
    """evaluate_host_ram_projection, raising ValueError (HOST_RAM_PROJECTION_EXCEEDED: <why>) on a trip."""
    rec = evaluate_host_ram_projection(
        current_anon_mb=current_anon_mb,
        projected_delta_mb=projected_delta_mb,
        mem_total_mb=mem_total_mb,
        cgroup_max_mb=cgroup_max_mb,
        mem_available_mb=mem_available_mb,
        mem_available_floor_mb=mem_available_floor_mb,
        safety_factor=safety_factor,
        source=source,
    )
    if rec["tripped"]:
        raise ValueError(f"HOST_RAM_PROJECTION_EXCEEDED: {rec['why']}")
    return rec


def run_between_item_fetch(
    *,
    remote_dir: Path,
    local_dir: Path,
    next_item: dict[str, Any] | None = None,
) -> dict[str, Any]:
    local_dir.mkdir(parents=True, exist_ok=True)
    current_json = remote_dir / "current.json"
    copied: list[str] = []
    previous_item = None
    if current_json.exists():
        previous_item = json.loads(current_json.read_text(encoding="utf-8"))
        for rel in previous_item.get("files") or []:
            src = remote_dir / rel
            if src.exists():
                dst = local_dir / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                copied.append(rel)
        shutil.copy2(current_json, local_dir / "current.json")
    if next_item is not None:
        current_json.write_text(json.dumps(next_item, indent=2) + "\n", encoding="utf-8")
    return {"copied_files": copied, "previous_item": previous_item}


def wall_cap_refusal(wall: dict[str, Any], cap: dict[str, Any]) -> dict[str, Any] | None:
    """The kit's named refusal for a wall/cap trip, with the numbers that tripped it. When both trip,
    the cap's code is the one named (a failed usage read or a spent cap is the costlier finding)."""
    if cap.get("tripped"):
        phase = cap.get("phase")
        prefix = f"cap guard (phase={phase}):" if phase else "cap guard:"
        if cap.get("code") == "COLAB_USAGE_PARSE_FAILED":
            why = f"{prefix} {cap.get('usage_parse_error')} (ceiling {cap.get('ceiling_ccu')} ccu)"
        elif cap.get("cleanup_bound_s", 0):
            why = (
                f"{prefix} spend {cap.get('spend_ccu')} ccu >= ceiling {cap.get('ceiling_ccu')} ccu "
                f"(effective_trip {cap.get('effective_trip_units')} = cap {cap.get('cap_ccu')} "
                f"- margin {cap.get('margin_ccu')} - cleanup_reserve {cap.get('cleanup_reserve_units')})"
            )
        else:
            why = f"{prefix} spend {cap.get('spend_ccu')} ccu >= ceiling {cap.get('ceiling_ccu')} ccu"
        ref: dict[str, Any] = {"code": cap.get("code"), "why": why}
        if phase is not None:
            ref["phase"] = phase
        if cap.get("checkpoint") is not None:
            ref["checkpoint"] = cap.get("checkpoint")
        return ref
    if wall.get("tripped"):
        return {
            "code": wall.get("code"),
            "why": f"wall guard: {wall.get('remaining_s')} s remaining of threshold {wall.get('threshold_s')} s (wall - grace)",
        }
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="sdlab C2 launcher guards CLI")
    parser.add_argument("--mode", default="stub")
    parser.add_argument("--phase", default=None, help="Execution phase (e.g. pre_exec or live)")
    parser.add_argument("--checkpoint", default=None, help="Pre-exec or live check point name")
    parser.add_argument("--elapsed-s", type=float, default=0.0)
    parser.add_argument("--wall-s", type=float, default=3000.0)
    parser.add_argument("--grace-s", type=float, default=60.0)
    parser.add_argument("--cap", type=float, default=0.05)
    parser.add_argument("--margin", type=float, default=0.01)
    parser.add_argument("--cleanup-bound-s", type=float, default=0.0)
    parser.add_argument("--rate", type=float, default=0.0)
    parser.add_argument("--max-rate", type=float, default=None)
    parser.add_argument("--min-runtime-s", type=float, default=None)
    parser.add_argument("--check-cap-budget", action="store_true", help="Check cap budget against cleanup reserve and --min-runtime-s")
    parser.add_argument("--dry", action="store_true", help="Print cap trip arithmetic and exit")
    parser.add_argument("--start-balance", type=float, default=None)
    parser.add_argument("--current-balance", type=float, default=None)
    parser.add_argument("--usage-parse-error", default=None, help="Usage parse failure message causing cap guard to trip")
    parser.add_argument("--verify-sha", action="append", default=[], help="Verify PATH=SHA256 before opening window")
    parser.add_argument("--scrub-file", action="append", default=[], help="Scrub Colab/OAuth tokens from file in place")
    parser.add_argument("--vitals-file", type=Path, default=None, help="Evaluate vitals.log and stall guard")
    parser.add_argument("--idle-seconds", type=float, default=0.0)
    parser.add_argument("--setup-idle-seconds", type=float, default=None)
    parser.add_argument("--stall-s", type=float, default=300.0)
    parser.add_argument("--setup-stall-s", type=float, default=600.0)
    parser.add_argument("--check-ram", action="store_true", help="Run host/cgroup RAM projection check")
    parser.add_argument("--current-anon-mb", type=float, default=512.0)
    parser.add_argument("--projected-delta-mb", type=float, default=1024.0)
    parser.add_argument("--mem-total-mb", type=float, default=12288.0)
    parser.add_argument("--cgroup-max-mb", type=float, default=None)
    parser.add_argument("--mem-available-mb", type=float, default=None)
    parser.add_argument("--mem-available-floor-mb", type=float, default=MEM_AVAILABLE_FLOOR_MB)
    parser.add_argument("--ram-source", default=None, help="Where --current-anon-mb was read (recorded in guards.log)")
    parser.add_argument("--ram-vitals-file", type=Path, default=None,
                        help="With --check-ram: read private memory from this vitals.log (host_ram_status)")
    parser.add_argument("--run-output-bytes", type=Path, default=None,
                        help="Print the bytes of this vitals.log that are not vitals lines (stall guard progress)")
    args = parser.parse_args(argv)

    if args.dry or args.check_cap_budget:
        eff_rate = (
            float(args.rate)
            if float(args.rate) > 0.0
            else (float(args.max_rate) if args.max_rate is not None else 0.26)
        )
        if args.max_rate is not None:
            priced_rate = max(eff_rate, float(args.max_rate))
            max_rate_val = float(args.max_rate)
        else:
            priced_rate = eff_rate
            max_rate_val = eff_rate
        rec = evaluate_cap_budget(
            cap=args.cap,
            margin=args.margin,
            cleanup_bound_s=args.cleanup_bound_s,
            priced_rate=priced_rate,
            max_rate=max_rate_val,
            min_runtime_s=args.min_runtime_s,
        )
        if args.dry:
            print(format_dry_cap_budget(rec))
        else:
            print(json.dumps(rec))
        if rec["tripped"]:
            print(f"REFUSED: {rec['code']}: {rec['why']}", file=sys.stderr)
            return 2
        return 0

    if args.run_output_bytes is not None:
        f = args.run_output_bytes
        print(run_output_bytes(f.read_text(encoding="utf-8", errors="replace")) if f.exists() else 0)
        return 0

    if args.verify_sha:
        pairs: dict[str, str] = {}
        for item in args.verify_sha:
            if "=" not in item:
                print(f"REFUSED: invalid --verify-sha argument {item!r}", file=sys.stderr)
                return 2
            k, v = item.split("=", 1)
            pairs[k] = v
        try:
            verified = verify_input_shas(pairs)
            print(json.dumps({"verified_shas": verified}))
            return 0
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.scrub_file:
        for fpath in args.scrub_file:
            scrub_file_in_place(Path(fpath))
        return 0

    if args.check_ram and args.ram_vitals_file is not None:
        f = args.ram_vitals_file
        status = host_ram_status(f.read_text(encoding="utf-8", errors="replace") if f.exists() else None)
        if status["state"] in ("no_vm", "waiting"):
            print(json.dumps({"check": "host_ram", "state": status["state"], "ok": True, "tripped": False,
                              "skipped": True, "blocks_seen": status["blocks_seen"]}))
            return 0
        if status["state"] == "unreadable":
            rec = {"check": "host_ram", "state": "unreadable", "ok": False, "tripped": True,
                   "code": "HOST_RAM_UNREADABLE", "blocks_seen": status["blocks_seen"], "why": status["why"]}
            rec["refusal"] = {"code": rec["code"], "why": rec["why"]}
            print(json.dumps(rec))
            print(f"HOST_RAM_UNREADABLE: {status['why']}", file=sys.stderr)
            return 1
        r = status["reading"]
        res = evaluate_host_ram_projection(
            current_anon_mb=r["private_mb"],
            projected_delta_mb=args.projected_delta_mb,
            mem_total_mb=r["mem_total_mb"],
            cgroup_max_mb=r["cgroup_max_mb"],
            mem_available_mb=r["mem_available_mb"],
            mem_available_floor_mb=args.mem_available_floor_mb,
            source=r["source"],
        )
        res.update({
            "state": "ok",
            "sample": r["timestamp"],
            "blocks_back": r["blocks_back"],
            "cgroup_private_mb": r["cgroup_private_mb"],
            "cgroup_anon_mb": r["cgroup_anon_mb"],
            "cgroup_shmem_mb": r["cgroup_shmem_mb"],
            "meminfo_private_mb": r["meminfo_private_mb"],
            "meminfo_anon_mb": r["meminfo_anon_mb"],
            "meminfo_shmem_mb": r["meminfo_shmem_mb"],
            "cgroup_current_mb_not_used": r["cgroup_current_mb_not_used"],
        })
        if res["tripped"]:
            res["refusal"] = {"code": res["code"], "why": res["why"]}
            print(json.dumps(res))
            print(f"{res['code']}: {res['why']}", file=sys.stderr)
            return 1
        print(json.dumps(res))
        return 0

    if args.check_ram:
        # The record goes to stdout (guards.log) whether or not it trips, so the tripping numbers
        # reach the rig; a trip also names itself on stderr and carries `refusal` for the kit.
        res = evaluate_host_ram_projection(
            current_anon_mb=args.current_anon_mb,
            projected_delta_mb=args.projected_delta_mb,
            mem_total_mb=args.mem_total_mb,
            cgroup_max_mb=args.cgroup_max_mb,
            mem_available_mb=args.mem_available_mb,
            mem_available_floor_mb=args.mem_available_floor_mb,
            source=args.ram_source,
        )
        if res["tripped"]:
            res["refusal"] = {"code": res["code"], "why": res["why"]}
            print(json.dumps(res))
            print(f"{res['code']}: {res['why']}", file=sys.stderr)
            return 1
        print(json.dumps(res))
        return 0

    if args.vitals_file is not None:
        txt = args.vitals_file.read_text(encoding="utf-8", errors="replace") if args.vitals_file.exists() else ""
        vit = evaluate_vitals_and_stall_guard(
            vitals_text=txt,
            idle_seconds=args.idle_seconds,
            setup_idle_seconds=args.setup_idle_seconds,
            stall_s=args.stall_s,
            setup_stall_s=args.setup_stall_s,
        )
        if vit["tripped"]:
            vit["refusal"] = {"code": vit["code"], "why": vit["reason"]}
        print(json.dumps(vit))
        return 1 if vit["tripped"] else 0

    wall = evaluate_wall_guard(mode=args.mode, elapsed_s=args.elapsed_s, wall_s=args.wall_s, grace_s=args.grace_s)
    cap = evaluate_cap_guard(
        mode=args.mode,
        cap=args.cap,
        margin=args.margin,
        cleanup_bound_s=args.cleanup_bound_s,
        start_balance=args.start_balance,
        current_balance=args.current_balance,
        rate_ccu_per_hour=args.rate,
        elapsed_s=args.elapsed_s,
        usage_parse_error=args.usage_parse_error,
        phase=args.phase,
        checkpoint=args.checkpoint,
    )
    out: dict[str, Any] = {"wall": wall, "cap": cap}
    if args.phase is not None:
        out["phase"] = args.phase
    if args.checkpoint is not None:
        out["checkpoint"] = args.checkpoint
    refusal = wall_cap_refusal(wall, cap)
    if refusal is not None:
        out["refusal"] = refusal
    print(json.dumps(out))
    return 1 if (wall["tripped"] or cap["tripped"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
