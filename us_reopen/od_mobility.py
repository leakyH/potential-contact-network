"""OD matrix preprocessing for the US reopening model."""

from __future__ import annotations

import numpy as np

from us_reopen.numeric import div_consider_zero


def process_sample_od(
    sample_od: np.ndarray,
    period: str,
    initpopulations: np.ndarray,
    flow_ratio_local: float,
) -> np.ndarray:
    if period == "commuting":
        sample_od = sample_od.astype(float)
        eyemask = np.eye(*sample_od.shape, dtype=bool)
        result = np.where(eyemask, 0, sample_od)

        now_osum = result.sum(axis=1)
        maxflow_ratio = div_consider_zero(
            initpopulations,
            now_osum,
            _allow_nonzero_over_zero=True,
        ) * (1 - 1e-5)
        maxflow_ratio[maxflow_ratio > flow_ratio_local] = flow_ratio_local
        result *= np.reshape(maxflow_ratio, (-1, 1))
        result[result < 5] = 0
        result[eyemask] = initpopulations - result.sum(axis=1)
        return result

    now_osum = sample_od.sum(axis=1)
    extendratio = initpopulations / now_osum
    extendratio[np.isinf(extendratio)] = 0
    normalized_od = sample_od * np.reshape(extendratio, (-1, 1))
    eyemask = np.eye(*normalized_od.shape, dtype=bool)
    result = np.where(eyemask, 0, normalized_od)

    maxflow_ratio = initpopulations / (result.sum(axis=1)) * (1 - 1e-5)
    maxflow_ratio[maxflow_ratio > flow_ratio_local] = flow_ratio_local
    result *= np.reshape(maxflow_ratio, (-1, 1))
    result[result < 5] = 0
    result[eyemask] = initpopulations - result.sum(axis=1)
    return result
