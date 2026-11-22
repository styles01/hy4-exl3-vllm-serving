#!/usr/bin/env python3
"""Hy4 DFly adaptive-depth launcher shim.

Reads --speculative-config JSON from sys.argv, applies the adaptive policy
from vllm.models.hy_v4.nvidia.dfly_adaptive (VLLM_HY4_DFLY_* env knobs),
rewrites argv, and exec's the vllm OpenAI entrypoint.

Used by /home/jaita/hy4_tp1_spec_adaptive.sh.
"""
from __future__ import annotations

import json
import os
import sys


def main() -> None:
    # Make the overlay importable.
    overlay = os.environ.get("PYTHONPATH", "/home/jaita/hy4serve/site-packages")
    if overlay not in sys.path:
        sys.path.insert(0, overlay)

    from vllm.models.hy_v4.nvidia.dfly_adaptive import (  # type: ignore
        apply_to_speculative_config,
        _log,
    )

    argv = list(sys.argv)
    new_argv: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--speculative-config" and i + 1 < len(argv):
            raw = argv[i + 1]
            try:
                spec = json.loads(raw)
            except json.JSONDecodeError:
                _log(f"--speculative-config not JSON ({raw!r}); passing through")
                new_argv.extend([a, raw])
                i += 2
                continue
            spec = apply_to_speculative_config(spec)
            new_argv.extend([a, json.dumps(spec)])
            i += 2
            continue
        new_argv.append(a)
        i += 1

    sys.argv = new_argv
    _log("argv rewritten; " + " ".join(a for a in new_argv if "speculative" in a or "spec" in a))

    import vllm_exl3  # type: ignore

    vllm_exl3.register()
    from vllm.entrypoints.launchers.api_server.entry import main  # type: ignore

    main()


if __name__ == "__main__":
    main()
