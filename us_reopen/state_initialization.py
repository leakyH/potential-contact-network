"""Initial SEIR state construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from us_reopen.immunity import getR
from us_reopen.population import AgePopulation, Town, init_i


@dataclass
class InitialState:
    towns_list: list[Town]
    selected: list
    e_date: np.ndarray
    i_date: np.ndarray
    it_date: np.ndarray
    ia_date: np.ndarray
    h_date: np.ndarray
    r_date: np.ndarray


def build_initial_state(
    town_ids: dict[str, int],
    initpopulations: np.ndarray,
    ageprops,
    ages,
    status,
    status2id: dict[str, int],
    start_at: list[int],
    rmax: int,
    rng,
) -> InitialState:
    towns_list = []
    e_date = np.zeros([len(initpopulations), len(ages), 4], int)
    i_date = np.zeros([len(initpopulations), len(ages), 8], int)
    it_date = np.zeros([len(initpopulations), len(ages), 8], int)
    ia_date = np.zeros([len(initpopulations), len(ages), 8], int)
    h_date = np.zeros([len(initpopulations), len(ages), 14], int)
    r_date = np.zeros([len(initpopulations), len(ages), rmax], int)

    for (town, townid), ageprop in zip(town_ids.items(), ageprops):
        if town != "-1":
            population_t = initpopulations[townid - 1]
            townseir = AgePopulation(population_t, len(ages), len(status), np.array(ageprop) / sum(ageprop))
            former_protected = rng.binomial(townseir.array[:, status2id["S"]], 1 - 1 / 2)
            townseir.array[:, status2id["P"]] = rng.binomial(former_protected, 1)
            townseir.array[:, status2id["R"]] = former_protected - townseir.array[:, status2id["P"]]
            townseir.array[:, status2id["S"]] -= former_protected
            towns_list.append(Town(town, townid, townseir))

    selected = init_i(towns_list, status2id["S"], status2id["E"], init_town_index=start_at)
    r_days_prob = getR("R12mGeom", Rmin=0, Rmax=rmax)
    for town_idx, town in enumerate(towns_list):
        e_date[town_idx, :, 0] += town.array[:, status2id["E"]]
        r_date[town_idx, :, len(r_days_prob) - 1 :: -1] += rng.multinomial(
            town.array[:, status2id["R"]],
            r_days_prob,
        )

    return InitialState(
        towns_list=towns_list,
        selected=selected,
        e_date=e_date,
        i_date=i_date,
        it_date=it_date,
        ia_date=ia_date,
        h_date=h_date,
        r_date=r_date,
    )
