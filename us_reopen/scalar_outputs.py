"""Scalar output writing for US reopening simulation runs."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from us_reopen.model_utils import PlotSelectedScalar


def write_scalar_outputs(metrics, towns_list, csa2county, initpopulations, subdir_name: str, suffix: str) -> None:
    largest_s2e_date_df = pd.DataFrame(
        {"first_peak_sim": metrics.largest_s2e_date},
        index=[ct.townname for ct in towns_list],
    )
    largest_s2e_date_df.to_csv(os.path.join("output/csvs/", subdir_name, f"largest_S2E_date{suffix}.csv"))
    largest_s2e_date_csa_df = pd.DataFrame(
        {"first_peak_sim_csa": metrics.largest_s2e_date_csa},
        index=csa2county.keys(),
    )
    largest_s2e_date_csa_df.to_csv(os.path.join("output/csvs/", subdir_name, f"largest_S2E_date_csa{suffix}.csv"))
    ref = np.loadtxt("ext-data/us-counties/AmericaPlot_merge.txt")
    ts_7window = lambda x: np.lib.stride_tricks.sliding_window_view(x, 7, 0)[::7, :].sum(axis=1)
    death_age_ref = pd.read_csv("ext-data/us-Omicron-data/death_age_agg.csv").loc[3:, :]
    death_ref = death_age_ref.loc[:, "All Ages"].values
    new_cases_by_age = np.array(metrics.new_cases_by_age)
    with np.errstate(divide="ignore", invalid="ignore"):
        before_ifr = (
            death_age_ref.iloc[:17, :].sum(axis=0).values[1:4]
            / new_cases_by_age[:100, :].sum(axis=0)
        )
        after_ifr = (
            death_age_ref.iloc[17:, :].sum(axis=0).values[1:4]
            / new_cases_by_age[100:380, :].sum(axis=0)
        )
    print("before 120: IFR should be real death / sim cases ")
    print(before_ifr)
    print("after 120: IFR should be real death / sim cases ")
    print(after_ifr)
    PlotSelectedScalar(
        metrics.selected,
        metrics.i_in_selected,
        "I_count",
        os.path.join("output/scalars/", subdir_name, f"I_count_init{suffix}.png"),
        False,
        extraInfo={"sum": metrics.i_in_all, "reference": ref},
        ref={"TheilL": metrics.i_theil_l},
    )
    PlotSelectedScalar(
        metrics.selected,
        metrics.i_ratio_in_selected,
        "I_ratio",
        os.path.join("output/scalars/", subdir_name, f"I_ratio_init{suffix}.png"),
        True,
        extraInfo={"all": metrics.i_ratio_in_all, "reference": ref / np.sum(initpopulations)},
        ref={"TheilL": metrics.i_theil_l},
    )
    PlotSelectedScalar(
        metrics.selected,
        metrics.s_ratio_in_selected,
        "S_ratio",
        os.path.join("output/scalars/", subdir_name, f"S_ratio_init{suffix}.png"),
        False,
        extraInfo={
            "S": metrics.s_ratio_in_all,
            "I": metrics.i_ratio_in_all,
            "R": metrics.r_ratio_in_all,
            "P": metrics.p_ratio_in_all,
        },
        ref={"Cum. Reported Cases": np.cumsum(metrics.total_tested_i) / np.sum(initpopulations)},
    )
    PlotSelectedScalar(
        metrics.selected,
        metrics.s2e_in_selected,
        "S2E",
        os.path.join("output/scalars/", subdir_name, f"S2E{suffix}.png"),
        True,
        extraInfo={"sum": metrics.s2e_in_all, "reference": ref},
        ref={"TheilL": metrics.s2e_theil_l},
    )
    PlotSelectedScalar(
        [],
        [],
        "I_tested",
        os.path.join("output/scalars/", subdir_name, f"I_tested_log{suffix}.png"),
        True,
        extraInfo={"sum": metrics.total_tested_i, "reference": ref},
        ref={"TheilL": metrics.tested_i_theil_l},
    )
    PlotSelectedScalar(
        [],
        [],
        "I_tested",
        os.path.join("output/scalars/", subdir_name, f"I_tested_linear{suffix}.png"),
        False,
        extraInfo={"sum": metrics.total_tested_i, "reference": ref},
        ref={"TheilL": metrics.tested_i_theil_l},
    )
    PlotSelectedScalar(
        [],
        [],
        "Death",
        os.path.join("output/scalars/", subdir_name, f"I2D_init{suffix}.png"),
        False,
        extraInfo={"daily_death": ts_7window(np.array(metrics.daily_death).sum(axis=1))},
        ref={"reference": death_ref, "reference_7days_ahead": death_ref[1:]},
        same_ylim=True,
    )
    PlotSelectedScalar(
        [],
        [],
        "Death-Age",
        os.path.join("output/scalars/", subdir_name, f"I2D_age_init{suffix}.png"),
        True,
        extraInfo={
            "daily_death_y": ts_7window(np.array(metrics.daily_death)[:, 0]),
            "daily_death_m": ts_7window(np.array(metrics.daily_death)[:, 1]),
            "daily_death_o": ts_7window(np.array(metrics.daily_death)[:, 2]),
        },
        ref={
            "reference_y": death_age_ref.loc[:, "Under 14"].values,
            "reference_m": death_age_ref.loc[:, "15-64 Years"].values,
            "reference_o": death_age_ref.loc[:, "65 Years and Over"].values,
        },
        same_ylim=True,
    )
    validate_data = np.hstack(
        [
            np.array(metrics.total_tested_i).reshape(-1, 1),
            np.array(metrics.tested_i_theil_l).reshape(-1, 1),
            metrics.daily_death,
        ]
    )
    np.savetxt(
        os.path.join("output/csvs/", subdir_name, f"validate_data{suffix}.csv"),
        validate_data,
        delimiter=",",
        header="totalTestedI,TheilL,daily_death_y,daily_death_m,daily_death_o",
        comments="",
    )
