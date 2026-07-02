#!/usr/bin/env python3
"""
CLI: compare ``train_summary.json`` files from ``train_rl`` / ``rl.train`` runs.

Usage: ``python -m rl.compare_summaries <path> [<path> ...]`` where each path is a JSON file or a directory
containing ``train_summary.json``. Prints algo, preset, best eval reward (online), offline kit metric, etc.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "summaries",
        nargs="+",
        type=str,
        help="Paths to train_summary.json (or directories containing it)",
    )
    args = p.parse_args()

    rows = []
    for raw in args.summaries:
        path = Path(raw)
        if path.is_dir():
            path = path / "train_summary.json"
        if not path.is_file():
            print(f"Missing {path}", flush=True)
            return 2
        with open(path, encoding="utf-8") as f:
            rows.append(json.load(f))

    print(f"{'algo':<10} {'preset':<22} {'best_eval':<14} {'offline_metric':<18} {'n_trans':<10} seed")
    for r in rows:
        algo = str(r.get("algo", ""))[:10]
        preset = str(r.get("preset", r.get("task_log_name", "")))[:22]
        best = r.get("best_eval_reward")
        best_s = f"{best:.4f}" if isinstance(best, (int, float)) else str(best)[:14]
        tr = r.get("trainer_return") or {}
        off_m = tr.get("last_10_performance", "")
        off_s = f"{float(off_m):.4f}" if isinstance(off_m, (int, float)) else str(off_m)[:18]
        nt = r.get("n_transitions", "")
        sd = r.get("seed", "")
        print(f"{algo:<10} {preset:<22} {best_s:<14} {off_s:<18} {str(nt):<10} {sd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
