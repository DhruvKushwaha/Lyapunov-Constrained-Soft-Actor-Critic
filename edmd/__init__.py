"""
**EDMD / EDMDc** (Koopman) — collect trajectories, fit models, discrete LQR, and metrics.

Modules
-------
``lqr``
    Discrete-time LQR / Riccati on lifted ``(A_lifted, B_lifted)``; P-matrix analysis helpers.
``metrics``
    Matrix stats, one-step prediction errors, open-loop rollout error (episode-aware if ``episode_id`` present).
``plotting``
    One-step prediction grids and multi-step rollout figures for papers.
``collect_quad_2d`` / ``collect_quad_3d``
    PID rollouts via ``PID_controller_quadrotor.collect_edmd_data`` → ``Saved_data/data_EDMD_*.npz`` (includes ``episode_id``).
``collect_cartpole``
    LQR rollouts → ``data_EDMD_cartpole.npz``.
``train_quad_2d`` / ``train_quad_3d`` / ``train_cartpole``
    Fit EDMDc, save ``.pkl`` model, LQR ``.npz``, figures, and JSON metrics.

**Episode IDs:** Saved ``.npz`` files may include ``episode_id`` per row. Training scripts pass it into
rollout metrics so multi-step starts do not span episode boundaries; older files omit it and use legacy sampling.

**Online RL:** After 2D EDMD + LQR, use ``python train_rl.py --algo lcsac`` (see repo README).

Examples (repo root)::

    python -m edmd.collect_quad_2d
    python -m edmd.train_quad_2d
"""
