# LC-SAC Quadrotor Trajectory Tracking

This repository implements and compares four online reinforcement learning algorithms for quadrotor and cartpole control in [safe-control-gym](https://github.com/utiasDSL/safe-control-gym), with **EDMD / EDMDc** (Koopman operator) models used as the shared Lyapunov function across three stability-constrained variants.

## Algorithms

| Algorithm | File | Description |
|-----------|------|-------------|
| **SAC** | `SAC.py` | Vanilla Soft Actor-Critic (Haarnoja et al. 2018). No Lyapunov constraint. |
| **LC-SAC** | `LC_SAC.py` + `LC_SAC_Train.py` | Lyapunov-constrained SAC . CVaR-based CLF decrease constraint via adaptive Lagrangian with ramp-in schedule. |
| **LC-SAC-Mean** | `LCSAC_Mean.py` | Ablation of LC-SAC with mean (not CVaR) violation aggregation, bidirectional λ update, and no ramp-in schedule. Uses the same Koopman CLF as LC-SAC. |
| **Lyap-RS-SAC** | `Lyap_RS_SAC.py` | Lyapunov potential reward shaping (Dong et al. 2020 ablation). Shapes reward as `r' = r + w*(V(z_t) - γ*V(z_{t+1}))` using the Koopman CLF as potential. |

> **Note on Koopman CLF:** All three Lyapunov algorithms use the same Koopman-based CLF, `V(z) = z^T P z`, where `z = Φ(x_error)` is the EDMD-lifted tracking error and `P` is the Riccati solution on the lifted `(A, B)` matrices. LC-SAC-Mean and Lyap-RS-SAC are **controlled ablations** that isolate the constraint enforcement mechanism (CVaR Lagrangian vs. mean Lagrangian vs. reward shaping) while holding the CLF constant.

## Environments and Presets

| Preset | Environment | Task | EDMD assets |
|--------|-------------|------|-------------|
| `quadrotor_2d_track` | 2D quadrotor | Trajectory tracking (circle) | `results/edmd/quadrotor_2d_track/` |
| `quadrotor_2d_stab` | 2D quadrotor | Stabilization to hover | `results/edmd/quadrotor_2d_stab/` |
| `cartpole_stab` | Cartpole | Stabilization to upright | `results/edmd/cartpole/` |
| `cartpole_track` | Cartpole | Trajectory tracking (circle in zx-plane) | `results/edmd/cartpole/` |
| `quadrotor_3d_track` | 3D quadrotor | Trajectory tracking (circle) | `results/edmd/quadrotor_3d/` |
| `quadrotor_3d_stab` | 3D quadrotor | Stabilization to hover | `results/edmd/quadrotor_3d_stab/` |

SAC runs on all presets without EDMD. All other algorithms require the corresponding EDMD assets.

## Features

| Area | Description |
|------|-------------|
| **Unified RL training** | `train_rl.py` / `python -m rl.train` — all 4 algorithms, 6 presets, YAML-driven |
| **Koopman / EDMD** | `edmd/` — PID data collection, EDMDc fitting, LQR on lifted `(A, B)`, hyperparameter tuning |
| **Experiment suite** | `experiments/run_online_suite.py` — multi-seed, multi-algo, multi-preset batch runner |
| **Results comparison** | `experiments/compare_results.py` — reward curves, Lyapunov loss, comparison tables |

## Installation

### Prerequisites

- Python 3.10 recommended
- CUDA optional; PyTorch uses GPU when available

### Python dependencies

```bash
pip install torch torchvision
pip install numpy scipy matplotlib pyyaml scikit-learn gymnasium
pip install pykoopman
pip install control   # python-control for edmd/lqr.py
```

### safe-control-gym

This repo vendors **`safe-control-gym/`**. Install in editable mode:

```bash
cd safe-control-gym
pip install -e .
cd ..
```

## Project Layout

```
LC-SAC-Quadrotor-Trajectory-Tracking/
├── train_rl.py                        # CLI entry → rl.train
├── rl/
│   └── train.py                       # run_sac / run_lcsac / run_lcsac_mean / run_lyap_rs_sac
├── SAC.py                             # SAC training loop + evaluation helpers
├── LC_SAC.py                          # LCSAC agent (networks, update, Koopman CLF)
├── LC_SAC_Train.py                    # train_lcsac() training loop
├── LCSAC_Mean.py                      # LCSACMeanAgent (subclass of LCSAC, mean violation)
├── Lyap_RS_SAC.py                     # Lyapunov reward-shaping training loop
├── Modified_SAC_Buffer.py             # Replay buffer with X_error field for Lyapunov algos
├── PID_controller_quadrotor.py        # PID control + EDMD data collection
├── edmd/
│   ├── collect_quad_2d.py             # PID rollouts → quadrotor_2d_track data
│   ├── collect_quad_2d_stab.py        # PID rollouts → quadrotor_2d_stab data
│   ├── collect_quad_3d.py             # PID rollouts → quadrotor_3d_track data
│   ├── collect_quad_3d_stab.py        # PID rollouts → quadrotor_3d_stab data
│   ├── collect_cartpole.py            # PID rollouts → cartpole data
│   ├── train_quad_2d.py               # EDMDc + LQR for 2D track
│   ├── train_quad_2d_stab.py          # EDMDc + LQR for 2D stab
│   ├── train_quad_3d.py               # EDMDc + LQR for 3D track
│   ├── train_quad_3d_stab.py          # EDMDc + LQR for 3D stab
│   ├── train_cartpole.py              # EDMDc + LQR for cartpole
│   ├── tune_quad_2d.py                # Hyperparameter grid search — 2D track
│   ├── tune_quad_2d_stab.py           # Hyperparameter grid search — 2D stab
│   ├── tune_quad_3d.py                # Hyperparameter grid search — 3D track
│   ├── tune_quad_3d_stab.py           # Hyperparameter grid search — 3D stab
│   ├── tune_cartpole.py               # Hyperparameter grid search — cartpole
│   ├── lqr.py                         # Discrete-time LQR / Riccati solver
│   ├── metrics.py                     # EDMD evaluation metrics
│   └── plotting.py                    # EDMD diagnostic figures
├── experiments/
│   ├── run_online_suite.py            # Multi-seed experiment runner
│   └── compare_results.py            # Aggregate plots and comparison tables
├── Params/
│   ├── Quadrotor_2D/                  # env_track.yaml, env_stab.yaml, sac.yaml, lcsac.yaml, …
│   ├── Quadrotor_3D/                  # env_track_rl.yaml, env_stab.yaml, sac.yaml, lcsac.yaml, …
│   ├── Cartpole/                      # env_stab.yaml, env_track.yaml, sac.yaml, lcsac.yaml, …
│   └── algorithms/                    # pid_drone_gains.yaml
├── results/
│   ├── edmd/                          # EDMD assets (models, LQR matrices, tuning results)
│   │   ├── data/                      # Raw .npz datasets from PID collection
│   │   ├── quadrotor_2d_track/        # edmd_model.pkl, lqr_matrices.npz, tune_results.json
│   │   ├── quadrotor_2d_stab/
│   │   ├── quadrotor_3d/              # (quadrotor_3d_track assets)
│   │   ├── quadrotor_3d_stab/
│   │   └── cartpole/
│   └── online/                        # RL training outputs
│       └── <algo>/<preset>/seed_<N>/ # train_summary.json, eval_rewards*.json, lyap_loss*.json
└── safe-control-gym/                  # Vendored safe-control-gym
```

## Usage

### Step 1 — Build EDMD models (required for Lyapunov algorithms)

Each preset needs its own EDMD model. Use `tune_*` to run the full hyperparameter grid search and save the best model. SAC skips this step entirely.

**2D Quadrotor:**
```bash
python -m edmd.collect_quad_2d
python -m edmd.tune_quad_2d --retrain-best

python -m edmd.collect_quad_2d_stab
python -m edmd.tune_quad_2d_stab --retrain-best
```

**3D Quadrotor:**
```bash
python -m edmd.collect_quad_3d
python -m edmd.tune_quad_3d --retrain-best

python -m edmd.collect_quad_3d_stab
python -m edmd.tune_quad_3d_stab --retrain-best
```

**Cartpole (shared across stab and track):**
```bash
python -m edmd.collect_cartpole
python -m edmd.tune_cartpole --retrain-best
```

Assets are saved to `results/edmd/<preset>/edmd_model.pkl` and `lqr_matrices.npz`.

### Step 2 — Train a single run

```bash
# SAC baseline (no EDMD needed)
python train_rl.py --algo sac --preset quadrotor_2d_track

# LC-SAC
python train_rl.py --algo lcsac --preset quadrotor_2d_track

# LC-SAC-Mean
python train_rl.py --algo lcsac_mean --preset quadrotor_2d_stab

# Lyap-RS-SAC
python train_rl.py --algo lyap_rs_sac --preset cartpole_stab

# 3D quadrotor
python train_rl.py --algo sac   --preset quadrotor_3d_track
python train_rl.py --algo lcsac --preset quadrotor_3d_stab

# Override seed and output directory
python train_rl.py --algo lcsac --preset quadrotor_2d_track --seed 3 \
    --output-dir results/online/lcsac/quadrotor_2d_track/seed_3

# Override trajectory type (tracking presets only)
python train_rl.py --algo sac --preset quadrotor_2d_track --trajectory figure8
```

Outputs land under `results/online/<algo>/<preset>/seed_<N>/`.

Available `--algo` values: `sac`, `lcsac`, `lcsac_mean`, `lyap_rs_sac`

Available `--preset` values: `quadrotor_2d_track`, `quadrotor_2d_stab`, `cartpole_stab`, `cartpole_track`, `quadrotor_3d_track`, `quadrotor_3d_stab`

```bash
python -m rl.train --help
```

### Step 3 — Multi-seed experiment suite

```bash
# Full suite — all 4 algos × all 6 presets × 5 seeds (120 runs)
python experiments/run_online_suite.py

# Filter by algo or preset
python experiments/run_online_suite.py --algos sac lcsac
python experiments/run_online_suite.py --presets quadrotor_3d_track quadrotor_3d_stab

# Resume — skip runs that already have train_summary.json
python experiments/run_online_suite.py --resume

# Dry run — print commands without executing
python experiments/run_online_suite.py --dry-run

# Override seeds
python experiments/run_online_suite.py --seeds 1 2 3
```

### Step 4 — Compare results

```bash
python experiments/compare_results.py
```

Reads `results/online/` and produces per-preset reward curves and Lyapunov loss comparison plots across all four algorithms. Output figures saved under `results/plots/`.

## Configuration

YAML files follow a two-level structure: **env config** (task, physics, reward weights) and **algo config** (network, optimizer, training hyperparameters).

### Environment configs

| File | Preset |
|------|--------|
| `Params/Quadrotor_2D/env_track.yaml` | `quadrotor_2d_track` |
| `Params/Quadrotor_2D/env_stab.yaml` | `quadrotor_2d_stab` |
| `Params/Quadrotor_3D/env_track_rl.yaml` | `quadrotor_3d_track` |
| `Params/Quadrotor_3D/env_stab.yaml` | `quadrotor_3d_stab` |
| `Params/Cartpole/env_stab.yaml` | `cartpole_stab` |
| `Params/Cartpole/env_track.yaml` | `cartpole_track` |

### Algorithm configs

| File | Algorithm | Env |
|------|-----------|-----|
| `Params/Quadrotor_2D/sac.yaml` | SAC | 2D quad |
| `Params/Quadrotor_2D/lcsac.yaml` | LC-SAC | 2D quad |
| `Params/Quadrotor_2D/lcsac_mean.yaml` | LC-SAC-Mean | 2D quad |
| `Params/Quadrotor_2D/lyap_rs_sac.yaml` | Lyap-RS-SAC | 2D quad |
| `Params/Quadrotor_3D/sac.yaml` | SAC | 3D quad |
| `Params/Quadrotor_3D/lcsac.yaml` | LC-SAC | 3D quad |
| `Params/Quadrotor_3D/lcsac_mean.yaml` | LC-SAC-Mean | 3D quad |
| `Params/Quadrotor_3D/lyap_rs_sac.yaml` | Lyap-RS-SAC | 3D quad |
| `Params/Cartpole/sac.yaml` | SAC | cartpole |
| `Params/Cartpole/lcsac.yaml` | LC-SAC | cartpole |
| `Params/Cartpole/lcsac_mean.yaml` | LC-SAC-Mean | cartpole |
| `Params/Cartpole/lyap_rs_sac.yaml` | Lyap-RS-SAC | cartpole |

### Key hyperparameters (Lyapunov algorithms)

| Parameter | LC-SAC | LC-SAC-Mean | Lyap-RS-SAC |
|-----------|--------|-------------|-------------|
| `lam_max` | `50.0` | `50.0` | — |
| `lam_lr` | `1e-3` | `1e-3` | — |
| `decay_rate` (CLF decrease margin) | `0.0` | — | — |
| `alpha_V` (CLF decrease margin) | — | `0.0` | — |
| `cvar_q` (CVaR quantile) | `0.75` | — (uses mean) | — |
| `lyap_ramp_steps` | 50k (2D) / 20k (cart) / 100k (3D) | — (no ramp) | — |
| `lyap_rs_weight` | — | — | auto-calibrated |
| `calibrate_steps` | — | — | `500` |

## Results Layout

| Path | Contents |
|------|----------|
| `results/online/<algo>/<preset>/seed_<N>/` | `train_summary.json`, best model checkpoint, `eval_rewards*.json`, `lyap_loss*.json` |
| `results/edmd/<preset>/` | `edmd_model.pkl`, `lqr_matrices.npz`, `tune_results.json`, `best_config.json`, diagnostic figures |
| `results/edmd/data/` | Raw `.npz` transition datasets from PID collection |
| `results/plots/` | Comparison figures from `compare_results.py` |

## Citations

### LC-SAC

```bibtex
@article{lcsac2026,
  title={LC-SAC: Lyapunov-Constrained Soft Actor-Critic via Koopman Operator Theory for Trajectory Tracking and Stabilization},
  author={Kushwaha, Dhruv S. and Biron, Zoleikha A.},
  journal={arXiv preprint arXiv:2602.04132},
  year={2026}
}
```

Paper: https://arxiv.org/abs/2602.04132

### BLAC (original — this codebase uses a Koopman CLF ablation, not the full method)

```bibtex
@inproceedings{zhao2023blac,
  title={Stable and Safe Reinforcement Learning via a Barrier-Lyapunov Actor-Critic Approach},
  author={Zhao, Liqun and Gan, Lu and Liu, Cunjia and Shi, Zhongke},
  booktitle={IEEE Conference on Decision and Control (CDC)},
  year={2023}
}
```

### Lyapunov reward shaping (original — this codebase uses a Koopman CLF ablation)

```bibtex
@article{dong2020lyapunov,
  title={Principled Reward Shaping for Reinforcement Learning via Lyapunov Stability Theory},
  author={Dong, Hanhan and Ding, Zihan and Huang, Shaocheng and Ding, Zhengtao},
  journal={Neurocomputing},
  volume={393},
  pages={83--90},
  year={2020}
}
```

### SAC

```bibtex
@inproceedings{haarnoja2018soft,
  title={Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor},
  author={Haarnoja, Tuomas and Zhou, Aurick and Abbeel, Pieter and Levine, Sergey},
  booktitle={International Conference on Machine Learning},
  year={2018},
  organization={PMLR}
}
```

### EDMD / Koopman operator

```bibtex
@article{williams2015data,
  title={A data-driven approximation of the Koopman operator: Extending dynamic mode decomposition},
  author={Williams, Matthew O and Kevrekidis, Ioannis G and Rowley, Clarence W},
  journal={Journal of Nonlinear Science},
  volume={25},
  number={6},
  pages={1307--1346},
  year={2015},
  publisher={Springer}
}
```

### safe-control-gym

```bibtex
@article{safe_control_gym,
  title={safe-control-gym: A Unified Benchmark Suite for Safe Learning-based Control and Reinforcement Learning},
  author={Yuan, Jingyun and Carrillo, Luis and Leung, Chi Hay and Abbeel, Pieter},
  journal={arXiv preprint arXiv:2109.12325},
  year={2021}
}
```

### PyKoopman

```bibtex
@software{pykoopman,
  author = {E. Kaiser and J. N. Kutz and S. L. Brunton},
  title = {PyKoopman: A Python Package for Data-Driven Approximation of the Koopman Operator},
  year = {2022},
  url = {https://github.com/dynamicslab/pykoopman}
}
```

## Authors

- Dhruv Kushwaha (dhruv.kushwaha@ufl.edu)

## Acknowledgments

- [safe-control-gym](https://github.com/utiasDSL/safe-control-gym)
- [PyKoopman](https://github.com/dynamicslab/pykoopman)
