"""
Thin wrapper around edmd.tune_quad_2d for the 2D quadrotor **stabilization** task.

Overrides default data / output paths; all other flags (--quick, --retrain-best,
--p-cond-max, --verbose) are forwarded unchanged.

Usage
-----
  python -m edmd.tune_quad_2d_stab --quick
  python -m edmd.tune_quad_2d_stab
  python -m edmd.tune_quad_2d_stab --quick --retrain-best
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"


def main() -> int:
    from edmd.tune_quad_2d import main as _tune_main

    argv = list(sys.argv[1:])
    _add_if_absent(argv, "--data",   str(RESULTS / "edmd" / "data" / "quadrotor_2d_stab.npz"))
    _add_if_absent(argv, "--output", str(RESULTS / "edmd" / "quadrotor_2d_stab"))

    sys.argv = [sys.argv[0]] + argv
    return _tune_main()


def _add_if_absent(argv: list[str], flag: str, value: str) -> None:
    if flag not in argv:
        argv += [flag, value]


if __name__ == "__main__":
    raise SystemExit(main())
