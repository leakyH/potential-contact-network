"""Numeric configuration for the US reopening experiments."""

from __future__ import annotations

import numpy as np


DEFAULT_WEEKCOUNT = 60
FLOW_RATIO = 1.0

AGES = ["young", "mid", "old"]
AGE_BETA = [0.1, 0.1, 0.15]
ALPHAMAT = np.ones([3, 3]) * 2
ALPHAMAT[[0, 1, 2], [0, 1, 2]] = 6
ALPHAMAT[2, :] = [1, 1, 3]
ALPHAMAT[:, 2] = [1, 1, 3]

STATUS = ["S", "E", "Ia", "I", "It", "R", "P", "H", "D"]
STATUS_INDEX = {status: idx for idx, status in enumerate(STATUS)}
MOVEABLE_STATUS_IDS = [0, 1, 2, 3, 5, 6]
CONTACTABLE_STATUS_IDS = [0, 1, 2, 3, 5, 6]
INFECTABLE_STATUS_IDS = [
    STATUS_INDEX["E"],
    STATUS_INDEX["Ia"],
    STATUS_INDEX["I"],
]

E_BETA = 1 / 10
IA_BETA = 1
IA_RATIO = 0.9
STATUS_BETA = np.array([0, E_BETA, IA_BETA, 1.0, 0, 0, 0, 0])

E_DAYS_PROB = [1 / 3] * 3
I_DAYS_PROB = [1 / 5] * 5
H_DAYS_PROB = [1 / 8] * 8

TEST_OVER = 5
RMAX = 365 * 2
RECOVER_DATE = 120
SHUTDOWN_NAME = "0p60"
RECOVER_NAME = "1"
ALPHA_OFFSET_NAME = "0"
ALPHA_OFFSET = float(ALPHA_OFFSET_NAME.replace("p", "."))
RECOVER_RATE = float(RECOVER_NAME.replace("p", "."))
SHUTDOWN_RATE = float(SHUTDOWN_NAME.removeprefix("fix").replace("p", "."))

I_TEST_STAGE1 = 1 / TEST_OVER
I_TEST = 1 / TEST_OVER
IA_TEST = I_TEST * 1 / 2

DRATIO_BASE = np.array([3e-5, 1.5e-3, 2e-2])
H2DRATIO_BASE = np.array([0.01, 0.04, 0.1])
HRATIO_BASE = DRATIO_BASE / H2DRATIO_BASE

FIXED_PROCESS_METHODS = {31, 34, 40, 58, 67, 70}
REALTIME_PROCESS_METHODS = {33, 36}
COMMUNITY_PROCESS_METHODS = {40, 58, 67, 70}
MINIMUM_MOBILITY_PROCESS_METHODS = {34, 36, 67, 70}
CONFIRMED_PROCESS_METHODS = FIXED_PROCESS_METHODS | REALTIME_PROCESS_METHODS

PERIOD_CHOICES = [
    "preCovid",
    "Delta",
    "Alpha",
    "AlphaRestrict",
    "preCovidlikeAlphaRestrict",
    "Omicron",
    "Omicron_lm",
    "commuting",
]

INIT_METHOD_CHOICES = [
    "pop10t10",
    "airport50k",
    "airport5k",
    "airport100k",
    "airport10k",
    "cfg",
    "everywhere",
    "pop10k",
    "pop100k",
    "pop1k",
    "random1k",
    "random100k",
    "random500k",
    "fitpoisson",
]

BETA_DENSITY_CHOICES = [
    "fit",
    "fit_poisson",
    "fit_poisson_weekly",
    "cfg",
    "cfgFull",
    "False",
    "log2",
    "log10",
    "ln",
    "log2R1",
    "log2R1p5",
]

S2E_VERSION = "v1"

PARAMETER_PROVENANCE = {
    "DRATIO_BASE_H2DRATIO_BASE": {
        "values": {
            "DRATIO_BASE": DRATIO_BASE.tolist(),
            "H2DRATIO_BASE": H2DRATIO_BASE.tolist(),
        },
        "original_note": "The original formal script placed this link immediately after the death/hospitalization ratio constants.",
        "source": "https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(21)02867-1/fulltext",
    },
    "US_OD_MOBILITY_INPUTS": {
        "paths": [
            "graphs/graph1/average_graph_full_daily_<period>_workday.pkl",
            "graphs/graph1/average_graph_full_<period>.pkl",
            "ext-data/COVID19USFlows-DailyFlows/",
            "ext-data/COVID19USFlows-WeeklyFlows/",
        ],
        "source": "Kang et al., Scientific Data 7, 390 (2020), https://www.nature.com/articles/s41597-020-00734-5",
        "data_project": "https://github.com/GeoDS/COVID19USFlows",
        "underlying_provider": "SafeGraph, https://www.safegraph.com/",
    },
    "DEATH_REFERENCES": {
        "paths": [
            "ext-data/us-Omicron-data/death_age_agg.csv",
            "ext-data/us-counties/AmericaPlot_merge.txt",
        ],
        "sources": [
            "https://data.cdc.gov/NCHS/Provisional-COVID-19-Death-Counts-by-Week-Ending-D/r8kw-7aab",
            "https://data.cdc.gov/NCHS/Provisional-COVID-19-Deaths-by-Week-Sex-and-Age/vsak-wrfu/about_data",
        ],
    },
    "AIRPORT_INITIALIZATION_INPUT": {
        "path": "ext-data/us-counties/iata-icao-UScounty/fips_passengers_count.csv",
        "source": "IP2Location IATA/ICAO List, https://www.ip2location.com",
        "license": "Creative Commons Attribution-ShareAlike 4.0 International",
    },
}


def get_h2d_ratio(day: int) -> np.ndarray:
    if day < RECOVER_DATE:
        return H2DRATIO_BASE
    return H2DRATIO_BASE * np.array([1, 1 / 3, 1 / 2])
