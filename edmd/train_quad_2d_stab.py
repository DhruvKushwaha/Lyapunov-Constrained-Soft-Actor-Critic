#!/usr/bin/env python3
"""
Fit EDMDc on 2D quadrotor **stabilization error** and export LQR matrices.

Thin wrapper around ``edmd.train_quad_2d`` that sets stabilization-specific
default paths.  All ``--n-rbf-centers``, ``--rbf-width``, ``--regularization``,
``--q-x``, ``--q-phi``, and ``--data`` flags are forwarded.

Requires ``Saved_data/data_EDMD_2D_stab.npz``; run
``python -m edmd.collect_quad_2d_stab`` first.

Examples::

  python -m edmd.train_quad_2d_stab
  python -m edmd.train_quad_2d_stab --n-rbf-centers 2 --rbf-width 0.25
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"


def main() -> int:
    from edmd.train_quad_2d import main as _train_main

    # Inject stab-specific defaults unless the caller already specified them
    argv = list(sys.argv[1:])
    _add_if_absent(argv, "--data",           str(RESULTS / "edmd" / "data" / "quadrotor_2d_stab.npz"))
    _add_if_absent(argv, "--model-output",   str(RESULTS / "edmd" / "quadrotor_2d_stab" / "edmd_model.pkl"))
    _add_if_absent(argv, "--riccati-output", str(RESULTS / "edmd" / "quadrotor_2d_stab" / "lqr_matrices.npz"))
    _add_if_absent(argv, "--metrics-prefix", "edmd_2d_stab_")

    sys.argv = [sys.argv[0]] + argv
    return _train_main()


def _add_if_absent(argv: list[str], flag: str, value: str) -> None:
    """Append --flag value only if flag not already present in argv."""
    if flag not in argv:
        argv += [flag, value]


if __name__ == "__main__":
    raise SystemExit(main())
