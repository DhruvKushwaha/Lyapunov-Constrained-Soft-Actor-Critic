#!/usr/bin/env python3
"""
Unified reinforcement-learning CLI at the repository root.

Delegates to :mod:`rl.train` (full argparse, presets, SAC / LC-SAC / offline baselines).

- ``python train_rl.py --help`` — list algorithms and flags.
- ``python -m rl.train`` — identical behavior.
- ``python -m rl.compare_summaries`` — tabulate ``train_summary.json`` from multiple runs.

Offline algorithms (CQL, TD3+BC, MOPO, COMBO) live under :mod:`rl.offline` and require
``pip install -e OfflineRL-Kit`` (or the bundled tree on ``sys.path``).
"""
from __future__ import annotations

from rl.train import main

if __name__ == "__main__":
    raise SystemExit(main())
