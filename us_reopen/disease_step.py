"""Daily disease-state advancement for US reopening simulations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from us_reopen.mobility import relocate_people
from us_reopen.numeric import getS2Eratio, getS2Eratio_v0, getS2Eratio_v0f


@dataclass
class DiseaseStepResult:
    s2e: np.ndarray
    s2e_rate: np.ndarray
    r2s: np.ndarray
    i2r: np.ndarray
    i2p: np.ndarray
    tested_by_region: np.ndarray
    delta_d: np.ndarray
    e_date_new: np.ndarray
    i_date_new: np.ndarray
    ia_date_new: np.ndarray
    r_date_new: np.ndarray


def advance_disease_step(
    *,
    towns: np.ndarray,
    population: np.ndarray,
    od_and_left: np.ndarray,
    rng,
    s2eversion: str,
    beta_density,
    infectables,
    statusbeta,
    alphamat,
    alphamat_county,
    beta,
    alpha_ratio: float,
    town_area,
    ref_pop_density,
    moveable,
    status2id,
    e_date: np.ndarray,
    i_date: np.ndarray,
    it_date: np.ndarray,
    ia_date: np.ndarray,
    h_date: np.ndarray,
    r_date: np.ndarray,
    e_days_prob,
    i_days_prob,
    h_days_prob,
    r_days_prob,
    ia_ratio: float,
    ia_test: float,
    i_test: float,
    hratio_base,
    pratio_num: float,
    get_h2d_ratio,
    day_index: int,
) -> DiseaseStepResult:
    worktown, sh2sw = relocate_people(
        towns[:, :, moveable],
        population,
        0,
        od_and_left,
        rng,
    )
    population_work = worktown.sum(axis=(1, 2))
    if s2eversion == "v0":
        s2e_rate_work, s2e_rate_home = getS2Eratio_v0(
            beta_density,
            population_work,
            population,
            worktown,
            towns[:, :, moveable],
            infectables,
            statusbeta,
            alphamat,
            beta * alpha_ratio,
            town_area,
            ref_pop_density,
        )
        s2e_rate = s2e_rate_work + np.expand_dims(s2e_rate_home, 1)
    elif s2eversion == "v0f":
        s2e_rate_work, s2e_rate_home = getS2Eratio_v0f(
            beta_density,
            population_work,
            population,
            worktown,
            towns[:, :, moveable],
            infectables,
            statusbeta,
            alphamat,
            beta * alpha_ratio,
            town_area,
            ref_pop_density,
        )
        s2e_rate = s2e_rate_work + np.expand_dims(s2e_rate_home, 1)
    elif s2eversion == "v1":
        no_s2e_rate_work, no_s2e_rate_home = getS2Eratio(
            beta_density,
            population_work,
            population,
            worktown,
            towns[:, :, moveable],
            infectables,
            statusbeta,
            alphamat_county * alpha_ratio,
            beta,
            town_area,
            ref_pop_density,
        )
        s2e_rate = 1 - (no_s2e_rate_work * np.expand_dims(no_s2e_rate_home, 1))
    elif s2eversion == "compare":
        s2e_rate_work, s2e_rate_home = getS2Eratio_v0f(
            beta_density,
            population_work,
            population,
            worktown,
            towns[:, :, moveable],
            infectables,
            statusbeta,
            alphamat,
            beta * alpha_ratio,
            town_area,
            ref_pop_density,
        )
        no_s2e_rate_work, no_s2e_rate_home = getS2Eratio(
            beta_density,
            population_work,
            population,
            worktown,
            towns[:, :, moveable],
            infectables,
            statusbeta,
            alphamat,
            beta * alpha_ratio,
            town_area,
            ref_pop_density,
        )
        s2e_rate_v0 = s2e_rate_work + np.expand_dims(s2e_rate_home, 1)
        s2e_rate_v1 = 1 - (no_s2e_rate_work * np.expand_dims(no_s2e_rate_home, 1))
        raise NotImplementedError(
            "s2eversion='compare' is diagnostic-only; choose v0 or v1 "
            f"(maximum rate difference: {np.max(np.abs(s2e_rate_v0 - s2e_rate_v1))})"
        )
    else:
        raise ValueError(f"unknown s2eversion {s2eversion}")
    s2e_rate[s2e_rate > 1] = 1
    r2s = r_date[:, :, -1]
    s2e = rng.binomial(sh2sw, s2e_rate).sum(axis=1)
    e2i = e_date[:, :, -1]
    e2ia = rng.binomial(e2i, ia_ratio)
    e2is = e2i - e2ia
    tested_ia = rng.binomial(ia_date, ia_test)
    tested_is = rng.binomial(i_date, i_test)
    tested_by_region = (tested_ia + tested_is).sum(axis=(1, 2))
    it_date += tested_ia + tested_is
    ia_date -= tested_ia
    i_date -= tested_is
    is2prh = i_date[:, :, -1]
    ia2prh = ia_date[:, :, -1]
    it2prh = it_date[:, :, -1]
    h2rdp = h_date[:, :, -1]
    i2prh = is2prh + ia2prh + it2prh
    i2h = rng.binomial(i2prh, hratio_base)
    i2p = rng.binomial(i2prh - i2h, pratio_num)
    i2r = i2prh - i2h - i2p
    h2d = rng.binomial(h2rdp, get_h2d_ratio(day_index))
    h2rp = h2rdp - h2d
    h2p = rng.binomial(h2rp, pratio_num)
    h2r = h2rp - h2p
    delta_s = r2s - s2e
    delta_e = s2e - e2i
    delta_is = e2is - is2prh - tested_is.sum(axis=2)
    delta_ia = e2ia - ia2prh - tested_ia.sum(axis=2)
    delta_it = tested_ia.sum(axis=2) + tested_is.sum(axis=2) - it2prh
    delta_h = i2h - h2rdp
    delta_r = h2r + i2r - r2s
    delta_p = i2p + h2p
    delta_d = h2d
    i_date_new = np.zeros_like(i_date)
    ia_date_new = np.zeros_like(ia_date)
    e_date_new = np.zeros_like(e_date)
    r_date_new = np.zeros_like(r_date)
    h_date_new = np.zeros_like(h_date)
    i_date_new[:, :, len(i_days_prob) - 1 :: -1] += rng.multinomial(e2is, i_days_prob)
    ia_date_new[:, :, len(i_days_prob) - 1 :: -1] += rng.multinomial(e2ia, i_days_prob)
    e_date_new[:, :, len(e_days_prob) - 1 :: -1] += rng.multinomial(s2e, e_days_prob)
    r_date_new[:, :, len(r_days_prob) - 1 :: -1] += rng.multinomial(i2r + h2r, r_days_prob)
    h_date_new[:, :, len(h_days_prob) - 1 :: -1] += rng.multinomial(i2h, h_days_prob)
    i_date[:, :, 1:] = i_date[:, :, 0:-1]
    i_date[:, :, 0] = 0
    ia_date[:, :, 1:] = ia_date[:, :, 0:-1]
    ia_date[:, :, 0] = 0
    it_date[:, :, 1:] = it_date[:, :, 0:-1]
    it_date[:, :, 0] = 0
    e_date[:, :, 1:] = e_date[:, :, 0:-1]
    e_date[:, :, 0] = 0
    r_date[:, :, 1:] = r_date[:, :, 0:-1]
    r_date[:, :, 0] = 0
    h_date[:, :, 1:] = h_date[:, :, 0:-1]
    h_date[:, :, 0] = 0
    i_date += i_date_new
    ia_date += ia_date_new
    e_date += e_date_new
    r_date += r_date_new
    h_date += h_date_new
    towns[:, :, status2id["S"]] += delta_s
    towns[:, :, status2id["E"]] += delta_e
    towns[:, :, status2id["I"]] += delta_is
    towns[:, :, status2id["Ia"]] += delta_ia
    towns[:, :, status2id["It"]] += delta_it
    towns[:, :, status2id["R"]] += delta_r
    towns[:, :, status2id["H"]] += delta_h
    towns[:, :, status2id["P"]] += delta_p
    towns[:, :, status2id["D"]] += delta_d
    return DiseaseStepResult(
        s2e=s2e,
        s2e_rate=s2e_rate,
        r2s=r2s,
        i2r=i2r,
        i2p=i2p,
        tested_by_region=tested_by_region,
        delta_d=delta_d,
        e_date_new=e_date_new,
        i_date_new=i_date_new,
        ia_date_new=ia_date_new,
        r_date_new=r_date_new,
    )
