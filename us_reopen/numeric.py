"""Numeric helpers used by the US reopening model."""

from __future__ import annotations

from typing import Union

import numpy as np


def div_consider_zero(
    numerator: Union[np.ndarray, float],
    denominator: Union[np.ndarray, float],
    _allow_nonzero_over_zero=False,
    zero_over_zero=0,
    clip=None,
):
    numerator, denominator = np.broadcast_arrays(numerator, denominator)
    denominatoriszero = denominator == 0
    numeratoriszero = numerator == 0
    if (not _allow_nonzero_over_zero) and (numerator[denominatoriszero] != 0).any():
        raise Exception(
            "in div_consider_zero, numerator[denominatoriszero] includes 0, \n"
            "while nonzero/0 is not allowed here"
        )

    result_dtype = np.result_type(numerator, denominator, float)
    result = np.empty(numerator.shape, dtype=result_dtype)
    np.divide(numerator, denominator, out=result, where=~denominatoriszero)
    result[denominatoriszero & numeratoriszero] = zero_over_zero
    if _allow_nonzero_over_zero:
        nonzero_over_zero = denominatoriszero & ~numeratoriszero
        result[nonzero_over_zero] = np.sign(numerator[nonzero_over_zero]) * np.inf
    if clip is not None:
        np.clip(result, -clip, clip, out=result)
    return result


def getS2Eratio(
    beta_density,
    population_work,
    population,
    worktown,
    towns,
    infectables,
    statusbeta,
    alphamat,
    beta,
    town_area=None,
    REF_POP_DENSITY=None,
):
    beta_iovern_work = div_consider_zero(
        np.dot(worktown[:, :, infectables], statusbeta[infectables]),
        worktown.sum(axis=2),
    )
    beta_iovern_home = div_consider_zero(
        np.dot(towns[:, :, infectables], statusbeta[infectables]),
        towns.sum(axis=2),
    )
    beta_iovern_work = np.repeat(np.expand_dims(beta_iovern_work, 1), len(beta), 1) * beta.reshape(1, -1, 1)
    beta_iovern_home = np.repeat(np.expand_dims(beta_iovern_home, 1), len(beta), 1) * beta.reshape(1, -1, 1)

    if beta_density == "linear":
        assert town_area is not None, "town area is required"
        assert REF_POP_DENSITY is not None, "REF_POP_DENSITY is required"
        alphamat_work = np.expand_dims(alphamat, 0) * 0.8 * (population_work / town_area / REF_POP_DENSITY).reshape(-1, 1, 1)
        alphamat_home = np.expand_dims(alphamat, 0) * 0.2 * (population / town_area / REF_POP_DENSITY).reshape(-1, 1, 1)
    elif beta_density == "log2":
        alphamat_work = np.expand_dims(alphamat, 0) * 0.8 * np.log2(1 + population_work / town_area / REF_POP_DENSITY).reshape(-1, 1, 1)
        alphamat_home = np.expand_dims(alphamat, 0) * 0.2 * np.log2(1 + population / town_area / REF_POP_DENSITY).reshape(-1, 1, 1)
    elif beta_density == "log2R1":
        temp1 = np.clip(np.log2(1 + population_work / town_area / REF_POP_DENSITY), 1 / 1.4, None)
        temp2 = np.clip(np.log2(1 + population / town_area / REF_POP_DENSITY), 1 / 1.4, None)
        alphamat_work = np.expand_dims(alphamat, 0) * 0.8 * temp1.reshape(-1, 1, 1)
        alphamat_home = np.expand_dims(alphamat, 0) * 0.2 * temp2.reshape(-1, 1, 1)
    elif beta_density == "log2R1p5":
        temp1 = np.clip(np.log2(1 + population_work / town_area / REF_POP_DENSITY), 1.5 / 1.4, None)
        temp2 = np.clip(np.log2(1 + population / town_area / REF_POP_DENSITY), 1.5 / 1.4, None)
        alphamat_work = np.expand_dims(alphamat, 0) * 0.8 * temp1.reshape(-1, 1, 1)
        alphamat_home = np.expand_dims(alphamat, 0) * 0.2 * temp2.reshape(-1, 1, 1)
    elif beta_density == "log10":
        alphamat_work = np.expand_dims(alphamat, 0) * 0.8 * np.log10(1 + population_work / town_area / REF_POP_DENSITY).reshape(-1, 1, 1)
        alphamat_home = np.expand_dims(alphamat, 0) * 0.2 * np.log10(1 + population / town_area / REF_POP_DENSITY).reshape(-1, 1, 1)
    elif beta_density == "ln":
        alphamat_work = np.expand_dims(alphamat, 0) * 0.8 * np.log(1 + population_work / town_area / REF_POP_DENSITY).reshape(-1, 1, 1)
        alphamat_home = np.expand_dims(alphamat, 0) * 0.2 * np.log(1 + population / town_area / REF_POP_DENSITY).reshape(-1, 1, 1)
    elif beta_density in ["cfg", "fit", "fit_poisson", "fit_poisson_weekly"]:
        alphamat_work = alphamat * 0.8
        alphamat_home = alphamat * 0.2
    elif beta_density in ["cfgTwo", "cfgFull"]:
        alphamat_work = alphamat * 1
        alphamat_home = alphamat * 0
    elif beta_density is False or beta_density == "False":
        alphamat_work = np.expand_dims(alphamat, 0) * 0.8
        alphamat_home = np.expand_dims(alphamat, 0) * 0.2

    notinfect_work = np.prod(np.power(1 - beta_iovern_work, alphamat_work), axis=2)
    notinfect_home = np.prod(np.power(1 - beta_iovern_home, alphamat_home), axis=2)
    return notinfect_work, notinfect_home


def getS2Eratio_v0(
    beta_density,
    population_work,
    population,
    worktown,
    towns,
    infectables,
    statusbeta,
    alphamat,
    beta,
    town_area=None,
    REF_POP_DENSITY=None,
):
    if beta_density == "log":
        assert town_area is not None, "town area is required"
        assert REF_POP_DENSITY is not None, "REF_POP_DENSITY is required"
        s2e_work = np.repeat(
            np.expand_dims(
                np.where(
                    population_work.reshape(-1, 1) > 0,
                    (
                        np.matmul(np.dot(worktown[:, :, infectables], statusbeta[infectables]), alphamat * 0.8)
                    )
                    * beta.reshape(1, -1)
                    * (np.log2(population_work / town_area / REF_POP_DENSITY + 1) / population).reshape(-1, 1),
                    0,
                ),
                0,
            ),
            len(population),
            0,
        )
        s2e_home = np.where(
            population.reshape(-1, 1) > 0,
            (np.matmul(np.dot(towns[:, :, infectables], statusbeta[infectables]), alphamat * 0.2))
            * beta.reshape(1, -1)
            * (np.log2(population / town_area / REF_POP_DENSITY + 1) / population).reshape(-1, 1),
            0,
        )
    elif beta_density == "linear":
        assert town_area is not None, "town area is required"
        assert REF_POP_DENSITY is not None, "REF_POP_DENSITY is required"
        s2e_work = np.repeat(
            np.expand_dims(
                np.where(
                    population_work.reshape(-1, 1) > 0,
                    (np.matmul(np.dot(worktown[:, :, infectables], statusbeta[infectables]), alphamat * 0.8))
                    * beta.reshape(1, -1)
                    / (town_area * REF_POP_DENSITY).reshape(-1, 1),
                    0,
                ),
                0,
            ),
            len(population),
            0,
        )
        s2e_home = np.where(
            population.reshape(-1, 1) > 0,
            (np.matmul(np.dot(towns[:, :, infectables], statusbeta[infectables]), alphamat * 0.2))
            * beta.reshape(1, -1)
            / (town_area * REF_POP_DENSITY).reshape(-1, 1),
            0,
        )
    elif beta_density is False:
        s2e_work = np.repeat(
            np.expand_dims(
                np.where(
                    population_work.reshape(-1, 1) > 0,
                    (np.matmul(np.dot(worktown[:, :, infectables], statusbeta[infectables]), alphamat * 0.8))
                    * beta.reshape(1, -1)
                    / population_work.reshape(-1, 1),
                    0,
                ),
                0,
            ),
            len(population),
            0,
        )
        s2e_home = np.where(
            population.reshape(-1, 1) > 0,
            (np.matmul(np.dot(towns[:, :, infectables], statusbeta[infectables]), alphamat * 0.2))
            * beta.reshape(1, -1)
            / population.reshape(-1, 1),
            0,
        )
    else:
        raise NotImplementedError("Specify log or linear or BOOL False")
    return s2e_work, s2e_home


def getS2Eratio_v0f(
    beta_density,
    population_work,
    population,
    worktown,
    towns,
    infectables,
    statusbeta,
    alphamat,
    beta,
    town_area=None,
    REF_POP_DENSITY=None,
):
    if beta_density == "log":
        assert town_area is not None, "town area is required"
        assert REF_POP_DENSITY is not None, "REF_POP_DENSITY is required"
        s2e_work = np.repeat(
            np.expand_dims(
                np.where(
                    population_work.reshape(-1, 1) > 0,
                    (np.matmul(np.dot(worktown[:, :, infectables], statusbeta[infectables]), alphamat * 0.8))
                    * beta.reshape(1, -1)
                    * (np.log2(population_work / town_area / REF_POP_DENSITY + 1) / population).reshape(-1, 1),
                    0,
                ),
                0,
            ),
            len(population),
            0,
        )
        s2e_home = np.where(
            population.reshape(-1, 1) > 0,
            (np.matmul(np.dot(towns[:, :, infectables], statusbeta[infectables]), alphamat * 0.2))
            * beta.reshape(1, -1)
            * (np.log2(population / town_area / REF_POP_DENSITY + 1) / population).reshape(-1, 1),
            0,
        )
    elif beta_density == "linear":
        assert town_area is not None, "town area is required"
        assert REF_POP_DENSITY is not None, "REF_POP_DENSITY is required"
        s2e_work = np.repeat(
            np.expand_dims(
                np.where(
                    population_work.reshape(-1, 1) > 0,
                    (np.matmul(np.dot(worktown[:, :, infectables], statusbeta[infectables]), alphamat * 0.8))
                    * beta.reshape(1, -1)
                    / (town_area * REF_POP_DENSITY).reshape(-1, 1),
                    0,
                ),
                0,
            ),
            len(population),
            0,
        )
        s2e_home = np.where(
            population.reshape(-1, 1) > 0,
            (np.matmul(np.dot(towns[:, :, infectables], statusbeta[infectables]), alphamat * 0.2))
            * beta.reshape(1, -1)
            / (town_area * REF_POP_DENSITY).reshape(-1, 1),
            0,
        )
    elif beta_density is False:
        s2e_work = np.repeat(
            np.expand_dims(
                np.matmul(
                    div_consider_zero(
                        np.dot(worktown[:, :, infectables], statusbeta[infectables]),
                        worktown.sum(axis=2),
                    ),
                    alphamat * 0.8,
                )
                * beta.reshape(1, -1),
                0,
            ),
            len(population),
            0,
        )
        s2e_home = np.matmul(
            div_consider_zero(
                np.dot(towns[:, :, infectables], statusbeta[infectables]),
                towns.sum(axis=2),
            ),
            alphamat * 0.2,
        ) * beta.reshape(1, -1)
    elif beta_density == "cfg":
        s2e_work = np.repeat(
            np.expand_dims(
                np.matmul(
                    div_consider_zero(
                        np.dot(worktown[:, :, infectables], statusbeta[infectables]),
                        worktown.sum(axis=2),
                    ),
                    alphamat * 0.8,
                )
                * beta,
                0,
            ),
            len(population),
            0,
        )
        s2e_home = np.matmul(
            div_consider_zero(
                np.dot(towns[:, :, infectables], statusbeta[infectables]),
                towns.sum(axis=2),
            ),
            alphamat * 0.2,
        ) * beta
    else:
        raise NotImplementedError("Specify log or linear or BOOL False")
    return s2e_work, s2e_home


def weighted_gini(values, weights, axis=None):
    values = np.array(values)
    weights = np.array(weights)
    if axis is None:
        values = values.flatten()
        weights = np.broadcast_to(weights, values.shape) if weights.ndim == 1 else weights.flatten()
        return _weighted_gini_1d(values, weights)
    if axis == 0:
        return np.apply_along_axis(lambda v: _weighted_gini_1d(v, weights), axis=1, arr=values)
    if axis == 1:
        return np.apply_along_axis(lambda v: _weighted_gini_1d(v, weights), axis=0, arr=values.T)
    raise ValueError("Axis must be 0, 1, or None.")


def _weighted_gini_1d(values, weights):
    sorted_indices = np.argsort(values)
    values = values[sorted_indices]
    weights = weights[sorted_indices]
    cumulative_weights = np.cumsum(weights)
    cumulative_weighted_values = np.cumsum(values * weights)
    total_weighted_values = cumulative_weighted_values[-1]
    total_weight = cumulative_weights[-1]
    weighted_sum = np.sum(weights * (cumulative_weighted_values - values * weights / 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1 - (2 * weighted_sum) / (total_weight * total_weighted_values)


def get_Gini(arr, pop=None, axis=None):
    if pop is None:
        pop = np.ones(arr.shape[axis])
    return weighted_gini(arr, pop.squeeze(), axis)


def get_Theil_L(arr, pop, axis=None):
    assert (arr >= 0).all()
    with np.errstate(divide="ignore", invalid="ignore"):
        if axis is None:
            y = arr / sum(arr)
            p = pop / sum(pop)
            temp = np.log(p / y)
            temp[np.isinf(temp)] = 0
            temp = p * temp
            return sum(temp)
        y = arr / np.sum(arr, axis=axis, keepdims=True)
        p = pop / np.sum(pop)
        temp = np.log(p / y)
        temp[np.isinf(temp)] = 0
        temp = p * temp
        return np.sum(temp, axis=axis)


def get_sum_from_lists(arr, list_of_list):
    result = np.zeros(len(list_of_list))
    for idx, lst in enumerate(list_of_list):
        result[idx] = arr[lst].sum()
    return result


def get_sum_from_dict(arr, dct):
    return get_sum_from_lists(arr, dct.values())
