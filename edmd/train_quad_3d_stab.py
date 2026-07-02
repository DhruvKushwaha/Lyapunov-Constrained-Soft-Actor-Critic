"""
Fit EDMDc on 3D quadrotor **stabilisation error** and export LQR matrices.

Thin wrapper around ``edmd.train_quad_3d`` that sets stab-specific default paths.
All ``--n-rbf-centers``, ``--rbf-width``, ``--regularization``, ``--q-x``,
``--q-phi``, and ``--data`` flags are forwarded unchanged.

Requires ``results/edmd/data/quadrotor_3d_stab.npz``; run
``python -m edmd.collect_quad_3d_stab`` first.

Examples::

  python -m edmd.train_quad_3d_stab
  python -m edmd.train_quad_3d_stab --n-rbf-centers 5 --rbf-width 0.25 --q-x 0.01 --q-phi 1e-3
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS   = REPO_ROOT / "results"


def _add_if_absent(argv: list[str], flag: str, value: str) -> None:
    if flag not in argv:
        argv += [flag, value]


def main() -> int:
    from edmd.train_quad_3d import main as _train_main

    argv = list(sys.argv[1:])
    _add_if_absent(argv, "--data",   str(RESULTS / "edmd" / "data"          / "quadrotor_3d_stab.npz"))
    _add_if_absent(argv, "--output", str(RESULTS / "edmd" / "quadrotor_3d_stab"))

    sys.argv = [sys.argv[0]] + argv
    return _train_main()


if __name__ == "__main__":
    raise SystemExit(main())
