"""
Grid-search EDMD + LQR hyperparameters for 3D quadrotor **stabilisation** error dynamics.

Thin wrapper around ``edmd.tune_quad_3d`` that sets stab-specific default paths.
All flags (``--quick``, ``--retrain-best``, ``--p-cond-max``, ``--verbose``) are forwarded.

Requires ``results/edmd/data/quadrotor_3d_stab.npz``; run
``python -m edmd.collect_quad_3d_stab`` first.

Usage
-----
  python -m edmd.tune_quad_3d_stab --quick
  python -m edmd.tune_quad_3d_stab
  python -m edmd.tune_quad_3d_stab --retrain-best --verbose
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS   = REPO_ROOT / "results"


def _add_if_absent(argv: list[str], flag: str, value: str) -> None:
    if flag not in argv:
        argv += [flag, value]


def main() -> int:
    from edmd.tune_quad_3d import main as _tune_main

    argv = list(sys.argv[1:])
    _add_if_absent(argv, "--data",   str(RESULTS / "edmd" / "data"          / "quadrotor_3d_stab.npz"))
    _add_if_absent(argv, "--output", str(RESULTS / "edmd" / "quadrotor_3d_stab"))

    sys.argv = [sys.argv[0]] + argv
    return _tune_main()


if __name__ == "__main__":
    raise SystemExit(main())
