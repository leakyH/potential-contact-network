"""Mobility matrix helpers for the US reopening model."""

from __future__ import annotations

import numpy as np

from us_reopen.numeric import div_consider_zero


def relocate_people(towns: np.ndarray, initpopulations: np.ndarray, status_id_of_S: int, ODandLeft: np.ndarray, rng):
    worktown = np.zeros_like(towns)
    od_repeated = np.repeat(
        np.repeat(np.expand_dims(ODandLeft, axis=(1, 2)), towns.shape[1], axis=1),
        towns.shape[2],
        axis=2,
    )
    transformation = np.swapaxes(rng.multinomial(towns, od_repeated), 1, 3)
    sh2sw = transformation[:, :, status_id_of_S, :]
    worktown = np.swapaxes(transformation.sum(axis=0), 1, 2)
    return worktown, sh2sw


def linear_transform_sampleOD(sampleOD, ratio, population=None):
    if population is None:
        population = sampleOD.sum(axis=1)
    eyemask = np.eye(*sampleOD.shape, dtype=bool)
    mat_to = np.where(eyemask, 0, sampleOD)
    now_osum = mat_to.sum(axis=1)
    maxflow_ratio = div_consider_zero(
        population,
        now_osum,
        _allow_nonzero_over_zero=True,
    ) * (1 - 1e-5)
    maxflow_ratio[maxflow_ratio > ratio] = ratio
    mat_to *= np.reshape(maxflow_ratio, (-1, 1))
    mat_to[eyemask] = population - mat_to.sum(axis=1)
    return mat_to
