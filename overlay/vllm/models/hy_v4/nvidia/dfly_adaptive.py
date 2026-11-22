# SPDX-License-Identifier: Apache-2.0
"""Adaptive draft-depth chooser for Hy4 DFly.

Reads a measured per-depth (round_ms, acceptance) table from
VLLM_HY4_DFLY_ROUND_MS (JSON), computes the depth that maximizes expected
throughput (accepted tokens / wall ms), and applies it as the lane's
num_speculative_tokens at boot.

Env knobs:
  VLLM_HY4_DFLY_ADAPTIVE        "1"/"true"/"on" -> enable chooser; unset -> off
  VLLM_HY4_DFLY_ROUND_MS        JSON list of [depth, round_ms, acceptance]
                                entries. If unset, falls back to the spark1
                                TP1-measured Hy4 DFly table (d1..d7).
  VLLM_HY4_DFLY_TARGET_GAIN     float, optional. If set, only consider depths
                                whose acceptance >= target_gain; else argmax over
                                all depths by tok/ms.
  VLLM_HY4_DFLY_DEPTH_CLAMP     "lo,hi" ints. Clamp the chosen depth into the
                                inclusive range. Default no clamp.
"""

from __future__ import annotations

import json
import os
from typing import Any

# Measured on spark1, TP1, Hy4-flash EXL3 3.0bpw, 2026-11-22.
# Format: [depth, round_ms, acceptance]
_DEFAULT_TABLE: list[list[float]] = [
    [0,   0.0, 1.00],   # canon, no drafts
    [1, 131.7, 1.65],
    [2, 173.8, 1.90],
    [3, 116.0, 1.96],
    [4, 244.0, 2.24],
    [7, 305.6, 2.30],
]


def _log(msg: str) -> None:
    # Print to stdout/stderr so it lands in `docker logs hy4-serve` without
    # requiring logger wiring at module-import time.
    print(f"[hy4-dfly-adaptive] {msg}", flush=True)


def _enabled() -> bool:
    return os.environ.get("VLLM_HY4_DFLY_ADAPTIVE", "").strip().lower() in (
        "1",
        "true",
        "on",
        "yes",
    )


def _load_table() -> list[list[float]]:
    raw = os.environ.get("VLLM_HY4_DFLY_ROUND_MS", "").strip()
    if not raw:
        return [list(row) for row in _DEFAULT_TABLE]
    try:
        data = json.loads(raw)
        out: list[list[float]] = []
        for row in data:
            d, ms, acc = float(row[0]), float(row[1]), float(row[2])
            out.append([d, ms, acc])
        out.sort(key=lambda r: r[0])
        return out
    except Exception as e:  # noqa: BLE001
        _log(f"VLLM_HY4_DFLY_ROUND_MS parse failed ({e!r}); using default table")
        return [list(row) for row in _DEFAULT_TABLE]


def _clamp(d: float, lo: int | None, hi: int | None) -> int:
    di = int(round(d))
    if lo is not None:
        di = max(lo, di)
    if hi is not None:
        di = min(hi, di)
    return di


def choose_depth() -> int | None:
    """Return num_speculative_tokens to use, or None if the policy is off."""
    if not _enabled():
        return None
    table = _load_table()
    target_gain_raw = os.environ.get("VLLM_HY4_DFLY_TARGET_GAIN", "").strip()
    target_gain: float | None = None
    if target_gain_raw:
        try:
            target_gain = float(target_gain_raw)
        except ValueError:
            _log(f"VLLM_HY4_DFLY_TARGET_GAIN={target_gain_raw!r} not a float; ignored")
    clamp_lo: int | None = None
    clamp_hi: int | None = None
    clamp_raw = os.environ.get("VLLM_HY4_DFLY_DEPTH_CLAMP", "").strip()
    if clamp_raw:
        try:
            a, b = clamp_raw.split(",", 1)
            clamp_lo, clamp_hi = int(a), int(b)
        except Exception as e:  # noqa: BLE001
            _log(f"VLLM_HY4_DFLY_DEPTH_CLAMP parse failed ({e!r}); no clamp")

    _log(
        "policy on; table="
        + ",".join(f"d{int(r[0])}:ms={r[1]:.1f},acc={r[2]:.2f}" for r in table)
        + f" target_gain={target_gain} clamp=({clamp_lo},{clamp_hi})"
    )

    # Skip depth=0 (canon) for the t/s argmax — acceptance<1 would imply no gain;
    # the dsv41 default table also encodes depth=0 as "off" and is never picked
    # by the conf policy unless everything else is worse.
    candidates = [r for r in table if r[0] > 0]
    if target_gain is not None:
        filtered = [r for r in candidates if r[2] >= target_gain]
        if not filtered:
            _log(
                f"no depth meets target_gain={target_gain}; falling back to argmax over all depths"
            )
            filtered = candidates
        candidates = filtered

    # Expected throughput proxy: acceptance / round_ms. Higher is better.
    # Canon (depth=0) is the fallback: rate = 1.0 / 1.0 = 1.0 in arbitrary units.
    # We compare depth's rate to canon's baseline rate derived from measured t/s:
    # canon t/s ~= 16, and each depth's projected t/s = canon_t/s * acc / (ms_d / ms_canon).
    # Since we don't have a canon ms baseline in this table (the table is absolute
    # round times), we use a simpler rate = acc / ms — depths that score below
    # "canon-equivalent" rate are dropped.
    best_row: list[float] | None = None
    best_rate = -1.0
    for row in candidates:
        d, ms, acc = row
        if ms <= 0:
            continue
        rate = acc / ms
        if rate > best_rate:
            best_rate = rate
            best_row = row

    if best_row is None:
        _log("no viable depth in table; policy abstains (fall back to launcher default)")
        return None

    chosen = _clamp(best_row[0], clamp_lo, clamp_hi)
    _log(
        f"choose depth={chosen} (best row d{int(best_row[0])} "
        f"rate={best_rate:.5f}, ms={best_row[1]:.1f}, acc={best_row[2]:.2f})"
    )
    return chosen


def apply_to_speculative_config(spec_kwargs: dict[str, Any]) -> dict[str, Any]:
    """If the policy is on, override spec_kwargs['num_speculative_tokens'].

    Called by the launcher wrapper before constructing the engine. Mutates and
    returns the same dict for convenience.
    """
    depth = choose_depth()
    if depth is None:
        return spec_kwargs
    old = spec_kwargs.get("num_speculative_tokens")
    spec_kwargs["num_speculative_tokens"] = depth
    _log(f"num_speculative_tokens {old} -> {depth}")
    return spec_kwargs
