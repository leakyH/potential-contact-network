"""Random seed helpers for US reopening simulations."""

from __future__ import annotations

import numpy as np


def initialize_numpy_random(seed: int | None = 0):
    if seed is None:
        np.random.seed()
        return np.random.default_rng()
    seed = int(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def derive_worker_seed(base_seed: int | None, offset: int) -> int | None:
    if base_seed is None:
        return None
    return int(base_seed) + int(offset)
