"""US county and mobility data loaders used by the experiment model."""

from __future__ import annotations

import itertools
import json
import pickle as pkl
from typing import Literal
from warnings import warn

import numpy as np
import pandas as pd

from us_reopen.build_od_network import buildUSNetwork, summarize_graph


def process_FIPS(df: pd.DataFrame, colname="FIPS"):
    if colname not in df.columns:
        if "STATEFP" in df.columns and "COUNTYFP" in df.columns:
            df[colname] = df.loc[:, "STATEFP"].astype("int").astype("str").str.zfill(2) + df.loc[:, "COUNTYFP"].astype("int").astype("str").str.zfill(3)
        elif "GEOID" in df.columns:
            df[colname] = df.loc[:, "GEOID"].str.removeprefix("05000US").str.zfill(5)
        else:
            raise NotImplementedError("cannot build FIPS")
    if not pd.api.types.is_string_dtype(df[colname]):
        df[colname] = df.loc[:, colname].astype(int).astype(str).str.zfill(5)
    if (df.loc[:, colname].str.len() != 5).any():
        df[colname] = df.loc[:, colname].str.zfill(5)
    return df


def getInformation(
    target_counties,
    counties=None,
    REF_DENSITY=None,
    records_wide=None,
    ageProperties=None,
    ageMethod: Literal["activity", "death"] = "activity",
):
    townIDs = {}
    town_population = {}
    if counties is None:
        try:
            counties = pd.read_csv("ext-data/us-counties/co-est2022-density.csv")
            counties = process_FIPS(counties)
        except FileNotFoundError:
            print("run calculate_pop_density.py and calculate_max_14day_ratio_JHU.py first")
            return

    if ageProperties is None:
        ageProperties = pd.read_csv("ext-data/us-counties/B01001.csv")
        ageProperties = process_FIPS(ageProperties)
        ageProperties.set_index("FIPS", drop=True, inplace=True)

    target_df = counties.query("FIPS in @target_counties")
    if len(target_df) != len(target_counties):
        notincounties = []
        for item in target_counties:
            if item not in counties["FIPS"].values:
                notincounties.append(item)
        target_df = target_df.drop(index=target_df.query("FIPS in @notincounties").index)
    target_df = target_df.sort_values(by="density", ascending=False)
    ageprops = []
    for idx, (_, line) in enumerate(target_df.iterrows()):
        townIDs[line["FIPS"]] = idx + 1
        town_population[str(idx + 1)] = line["ESTIMATESBASE2020"]
        try:
            if ageMethod == "activity":
                ageprops.append(
                    np.array(
                        [
                            ageProperties.loc[line["FIPS"], [f"B01001e{i}" for i in itertools.chain(range(3, 7), range(27, 31))]].sum().item(),
                            ageProperties.loc[line["FIPS"], [f"B01001e{i}" for i in itertools.chain(range(7, 18), range(31, 42))]].sum().item(),
                            ageProperties.loc[line["FIPS"], [f"B01001e{i}" for i in itertools.chain(range(18, 26), range(42, 50))]].sum().item(),
                        ]
                    )
                )
            elif ageMethod == "death":
                ageprops.append(
                    np.array(
                        [
                            ageProperties.loc[line["FIPS"], [f"B01001e{i}" for i in itertools.chain(range(3, 6), range(27, 30))]].sum().item(),
                            ageProperties.loc[line["FIPS"], [f"B01001e{i}" for i in itertools.chain(range(6, 20), range(30, 44))]].sum().item(),
                            ageProperties.loc[line["FIPS"], [f"B01001e{i}" for i in itertools.chain(range(20, 26), range(44, 50))]].sum().item(),
                        ]
                    )
                )
            else:
                raise ValueError(f"unknown ageMethod {ageMethod}")
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            fips = line.get("FIPS", "<missing>")
            raise ValueError(
                f"failed to compute {ageMethod!r} age properties for FIPS {fips}"
            ) from exc
    town_area = target_df.loc[:, "ALAND"].values
    max_ratios = None

    if records_wide is None:
        records_wide = pd.read_csv("ext-data/us-counties/2022wide.csv")
        records_wide.ffill(inplace=True)
        records_wide.fillna(0, inplace=True)
    record_fips = list(set(target_df["FIPS"]).intersection(set(records_wide.columns)))
    csa_ts = records_wide.loc[:, record_fips].values.sum(axis=1)
    csa_max = max(csa_ts)
    csa_max_ratio = csa_max / sum(town_population.values())

    if REF_DENSITY is None:
        REF_POP_DENSITY = np.median(np.array(list(town_population.values())) / town_area)
    else:
        REF_POP_DENSITY = REF_DENSITY
    return townIDs, town_population, town_area, max_ratios, csa_max_ratio, ageprops, REF_POP_DENSITY


def getCSAInformation_read_file(csa):
    temp = getAllCSAInformation_read_file()
    if csa in temp.keys():
        return temp[csa]
    warn(
        "there is an implicit file read in getCSAInformation_read_file() "
        "to transform CSAName to GeoID as a key. Try use GeoID instead to reduce file io."
    )
    with open("ext-data/us-counties/CSAName2GeoID.json", "r") as f:
        CSAName2GeoID = json.load(f)
    return temp[CSAName2GeoID[csa]]


def getAllCSAInformation_read_file():
    print("INFO: reading from ext-data/us-counties/CSA2CountyInfo.pkl")
    with open("ext-data/us-counties/CSA2CountyInfo.pkl", "rb") as f:
        return pkl.load(f)


__all__ = [
    "buildUSNetwork",
    "getCSAInformation_read_file",
    "getInformation",
    "process_FIPS",
    "summarize_graph",
]
