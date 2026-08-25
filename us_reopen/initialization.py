"""Initial infection and regional grouping helpers."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd


def select_initial_infections(
    init_method: str,
    initpopulations: np.ndarray,
    town_ids: dict[str, int],
    beta_density: str | bool,
    test_over: int,
) -> list[int]:
    if init_method == "pop10t10":
        start_at = np.argsort(initpopulations)[-10:].tolist() * 10
    elif init_method == "airport50k":
        airport_passengers = _read_airport_passengers("PASSENGERS>50000")
        start_at = sum(
            [
                [town_ids[line["FIPS"]] - 1] * int(line["PASSENGERS"] / 50000) * 10
                for _, line in airport_passengers.iterrows()
            ],
            [],
        )
    elif init_method == "airport100k":
        airport_passengers = _read_airport_passengers("PASSENGERS>100000")
        start_at = sum(
            [
                [town_ids[line["FIPS"]] - 1] * int(line["PASSENGERS"] / 100000) * 30
                for _, line in airport_passengers.iterrows()
            ],
            [],
        )
    elif init_method == "airport5k":
        airport_passengers = _read_airport_passengers("PASSENGERS>5000")
        start_at = sum(
            [
                [town_ids[line["FIPS"]] - 1] * int(line["PASSENGERS"] / 5000)
                for _, line in airport_passengers.iterrows()
            ],
            [],
        )
    elif init_method == "airport10k":
        airport_passengers = _read_airport_passengers("PASSENGERS>10000")
        start_at = sum(
            [
                [town_ids[line["FIPS"]] - 1] * int(line["PASSENGERS"] / 10000) * 2
                for _, line in airport_passengers.iterrows()
            ],
            [],
        )
    elif init_method in ["cfg", "cfgFull"]:
        with open("ext-data/us-counties/cases_start_day.json") as f:
            county_start_day = json.load(f)
        sorted_county_start_day = sorted(county_start_day.items(), key=lambda x: x[1])
        start_at = []
        for fips, _ in sorted_county_start_day:
            if fips in town_ids:
                start_at.append(town_ids[fips])
                if len(start_at) == 10:
                    break
        start_at *= 10
    elif init_method == "everywhere":
        start_at = np.arange(len(initpopulations)).tolist()
    elif init_method == "pop10k":
        in_each_county = (initpopulations // (1e4)).astype(int)
        start_at = np.repeat(np.arange(len(initpopulations)), in_each_county).tolist()
    elif init_method == "pop100k":
        in_each_county = (initpopulations // (1e5)).astype(int) * 2
        start_at = np.repeat(np.arange(len(initpopulations)), in_each_county).tolist()
    elif init_method == "pop1k":
        in_each_county = (initpopulations // (1e3)).astype(int)
        start_at = np.repeat(np.arange(len(initpopulations)), in_each_county).tolist()
    elif init_method == "random1k":
        start_at = np.random.choice(len(initpopulations), size=1000, replace=True).tolist()
    elif init_method == "random100k":
        start_at = np.random.choice(len(initpopulations), size=100_000, replace=True).tolist()
    elif init_method == "random500k":
        start_at = np.random.choice(len(initpopulations), size=500_000, replace=True).tolist()
    elif init_method == "fitpoisson":
        alpha_csv = pd.read_csv(
            "graphs/graphR/testR0_max_fitted_beta_poisson.csv", dtype={"fips": str}
        )
        alpha_csv["fips"] = alpha_csv["fips"].str.zfill(5)
        alpha_csv["init"] = np.exp(alpha_csv["bias"] + 3 * alpha_csv["growth_rate"]) * test_over
        init_cfg = dict((line["fips"], line["init"]) for _, line in alpha_csv.iterrows())
        init_min = 0
        countyinit = np.array([init_cfg.get(item, init_min) for item in town_ids])
        countyinit[countyinit < 1] = 0
        start_at = np.repeat(np.arange(len(countyinit)), countyinit.astype(int)).tolist()
        print(f"init_method=fit_poisson,len(START_AT)={len(start_at)}")
    else:
        raise ValueError(f"unsupported init_method: {init_method}")

    if "airport" in init_method:
        if beta_density == "fit_poisson":
            start_at *= 20
        elif beta_density == "fit":
            start_at *= 200
        else:
            start_at *= 500

    if not isinstance(start_at, list):
        return [start_at]
    return start_at


def load_csa_county_indices(town_ids: dict[str, int]) -> tuple[dict[str, list[str]], dict[str, list[int]]]:
    with open("ext-data/us-counties/CSA2County_1.json", "r") as f:
        csa2county = json.load(f)
    csa2countyidx = {}
    for key, value in csa2county.items():
        csa2countyidx[key] = [town_ids[item] - 1 for item in value]
    return csa2county, csa2countyidx


def _read_airport_passengers(query: str) -> pd.DataFrame:
    airport_passengers = pd.read_csv(
        "ext-data/us-counties/iata-icao-UScounty/fips_passengers_count.csv",
        dtype={"FIPS": str},
    )
    return airport_passengers.query(query)
