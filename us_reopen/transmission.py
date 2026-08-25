"""Transmission-parameter builders for the US reopening model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from us_reopen.config import RECOVER_DATE, RECOVER_RATE, SHUTDOWN_NAME, SHUTDOWN_RATE


@dataclass
class CountyTransmission:
    countyalpha: np.ndarray | None
    alphamat_county: np.ndarray
    xmax_date: np.ndarray | int


def build_county_transmission(
    beta_density: str | bool,
    period: str,
    town_ids: dict[str, int],
    alphamat: np.ndarray,
    alphaoffset: float,
    test_over: int,
) -> CountyTransmission:
    if beta_density in ["cfg", "cfgFull"]:
        alpha_csv = pd.read_csv("graphs/graphR/testR0_max.csv", dtype={"fips": str})
        alpha_csv["fips"] = alpha_csv["fips"].str.zfill(5)
        county_cfg = dict((line["fips"], line["r_max"]) for _, line in alpha_csv.iterrows())
        betamedian = np.median(list(county_cfg.values()))
        countyalpha = (np.array([county_cfg.get(item, betamedian) for item in town_ids]) + (1 / 12)) / (
            0.896
        )
        with open(f"ext-data/us-counties/cases_coef_R0_local_reconstruct_{period}.json", "r") as f:
            alphamat_county_dict = json.load(f)
            countyalpha = np.array([alphamat_county_dict[item] for item in town_ids]) + alphaoffset
            countyalpha[countyalpha < 0] = 0
        alphamat_county = countyalpha[:, np.newaxis, np.newaxis] * alphamat
        xmax_date = 50
    elif beta_density == "fit":
        assert test_over in [3, 5, 7, 10]
        alpha_csv = pd.read_csv("graphs/graphR/testR0_max_fitted_beta.csv", dtype={"fips": str})
        alpha_csv["fips"] = alpha_csv["fips"].str.zfill(5)
        county_cfg = dict((line["fips"], line[f"TO{test_over}_beta"]) for _, line in alpha_csv.iterrows())
        betamin = np.min(list(county_cfg.values()))
        countyalpha = np.array([county_cfg.get(item, betamin) for item in town_ids])
        alphamat_county = countyalpha[:, np.newaxis, np.newaxis] * alphamat
        xmax_date = np.ones(len(town_ids)) * 50
    elif beta_density == "fit_poisson":
        assert test_over in [3, 5, 7, 10]
        alpha_csv = pd.read_csv("graphs/graphR/testR0_max_fitted_beta_poisson.csv", dtype={"fips": str})
        alpha_csv["fips"] = alpha_csv["fips"].str.zfill(5)
        county_cfg = dict((line["fips"], line[f"TO{test_over}_beta"]) for _, line in alpha_csv.iterrows())
        xmax_cfg = dict((line["fips"], line["xmax"] + 9) for _, line in alpha_csv.iterrows())
        betamin = np.min(list(county_cfg.values()))
        xmax_default = 50
        xmax_date = np.array([xmax_cfg.get(item, xmax_default) for item in town_ids])
        countyalpha = np.array([county_cfg.get(item, betamin) for item in town_ids])
        alphamat_county = countyalpha[:, np.newaxis, np.newaxis] * alphamat
    elif beta_density == "fit_poisson_weekly":
        assert test_over in [5, 7, 10]
        alpha_csv = pd.read_csv("graphs/graphR/testR0_max_fitted_beta_poisson_weekly.csv", dtype={"fips": str})
        alpha_csv["fips"] = alpha_csv["fips"].str.zfill(5)
        county_cfg = dict((line["fips"], line[f"TO{test_over}_beta"]) for _, line in alpha_csv.iterrows())
        xmax_cfg = dict((line["fips"], line["xmax"] * 7 + 7) for _, line in alpha_csv.iterrows())
        betamin = np.min(list(county_cfg.values()))
        xmax_default = 50
        xmax_date = np.array([xmax_cfg.get(item, xmax_default) for item in town_ids])
        countyalpha = np.array([county_cfg.get(item, betamin) for item in town_ids])
        alphamat_county = countyalpha[:, np.newaxis, np.newaxis] * alphamat
    else:
        countyalpha = None
        alphamat_county = alphamat
        xmax_date = 50
    return CountyTransmission(
        countyalpha=countyalpha,
        alphamat_county=alphamat_county,
        xmax_date=xmax_date,
    )


def load_early_reopen_flags(town_ids: dict[str, int], threshold_day: int) -> np.ndarray:
    early_reopen_df = pd.read_csv(
        "graphs/graph_map/multi_wave_idx_onset_origin.csv", dtype={"FIPS": str}
    )
    early_reopen_set = set(
        early_reopen_df.loc[
            early_reopen_df["Dates of Second Onsets"] < threshold_day, "FIPS"
        ].values.tolist()
    )
    return np.array([item in early_reopen_set for item in town_ids])


def make_pcf_beta_fn(
    beta_density: str | bool,
    countyalpha: np.ndarray | None,
    recover_rate: float,
    town_area: np.ndarray,
    ref_pop_density: float,
) -> Callable[[np.ndarray], np.ndarray | None]:
    def get_pcf_beta(current_od: np.ndarray) -> np.ndarray | None:
        if beta_density in ["cfg", "cfgFull"]:
            return countyalpha
        if beta_density in ["fit", "fit_poisson", "fit_poisson_weekly"]:
            return recover_rate * countyalpha
        if beta_density is False:
            return None
        pcf_workpop = current_od.sum(axis=0)
        pcfbeta = 1 + pcf_workpop / town_area / ref_pop_density
        if beta_density == "log10":
            return np.log10(pcfbeta)
        if beta_density == "log2":
            return np.log2(pcfbeta)
        if beta_density == "ln":
            return np.log(pcfbeta)
        return None

    return get_pcf_beta


def update_county_transmission_for_day(
    *,
    day_index: int,
    beta_density: str | bool,
    alphamat: np.ndarray,
    countyalpha: np.ndarray,
    xmax_date: np.ndarray | int,
    flag_early_reopen: np.ndarray,
) -> np.ndarray:
    """Return the existing county transmission matrix for one day."""
    if "log" in beta_density or "ln" in beta_density:
        if day_index <= xmax_date:
            return alphamat * 1.0
        if day_index > xmax_date and day_index < RECOVER_DATE:
            return alphamat * SHUTDOWN_RATE
        if day_index >= RECOVER_DATE:
            return alphamat * RECOVER_RATE

    if day_index < RECOVER_DATE:
        flag_stop = day_index > xmax_date
        if "fix" in SHUTDOWN_NAME:
            county_values = np.where(flag_stop, SHUTDOWN_RATE, countyalpha)
        else:
            county_values = np.where(
                flag_stop, countyalpha * SHUTDOWN_RATE, countyalpha
            )
        if day_index >= RECOVER_DATE - 30:
            county_values = np.where(
                flag_early_reopen,
                countyalpha * RECOVER_RATE,
                county_values,
            )
        return county_values[:, np.newaxis, np.newaxis] * alphamat
    return countyalpha[:, np.newaxis, np.newaxis] * RECOVER_RATE * alphamat
