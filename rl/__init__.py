"""
Project reinforcement-learning package.

Submodules
----------
``rl.train``
    Preset-based training CLI (``python train_rl.py`` or ``python -m rl.train``).
``rl.offline``
    Offline baselines: Gymnasium→OfflineRL-Kit eval adapter, transition ``.npz`` I/O,
    and training for CQL / TD3+BC / MOPO / COMBO.
``rl.compare_summaries``
    ``python -m rl.compare_summaries`` — table multiple ``train_summary.json`` paths.
"""
