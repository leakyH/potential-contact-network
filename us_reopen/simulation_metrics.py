"""Metric collection for US reopening simulation runs."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from us_reopen.numeric import get_Gini, get_Theil_L, get_sum_from_dict


@dataclass
class SimulationMetrics:
    selected: list
    towns_list: list
    initpopulations: np.ndarray
    csa2county: dict
    csa2countyidx: dict
    start_at: list
    result_frame: list = field(default_factory=list)
    daily_new: list = field(default_factory=list)
    daily_death: list = field(default_factory=list)
    daily_tested: list = field(default_factory=list)
    i_in_selected: list = field(init=False)
    i_in_all: list = field(default_factory=list)
    i_gini: list = field(default_factory=list)
    i_theil_l: list = field(default_factory=list)
    i_ratio_in_all: list = field(default_factory=list)
    s_in_all: list = field(default_factory=list)
    s_ratio_in_all: list = field(default_factory=list)
    p_ratio_in_all: list = field(default_factory=list)
    r_ratio_in_all: list = field(default_factory=list)
    s_in_selected: list = field(init=False)
    i_ratio_in_selected: list = field(init=False)
    daily_tested_in_selected: dict = field(init=False)
    s_ratio_in_selected: list = field(init=False)
    s2e_in_selected: list = field(init=False)
    s2e_ratio_in_selected: list = field(init=False)
    s2e_in_all: list = field(default_factory=list)
    total_tested_i: list = field(default_factory=list)
    s2e_ratio_in_all: list = field(default_factory=list)
    s2e_gini: list = field(default_factory=list)
    s2e_theil_l: list = field(default_factory=list)
    tested_i_theil_l: list = field(default_factory=list)
    largest_s2e: np.ndarray = field(init=False)
    largest_s2e_date: np.ndarray = field(init=False)
    largest_s2e_csa: np.ndarray = field(init=False)
    largest_s2e_date_csa: np.ndarray = field(init=False)
    i2r_in_selected: list = field(init=False)
    r2s_in_selected: list = field(init=False)
    i2p_in_selected: list = field(init=False)
    population_in_selected: list = field(init=False)
    e_bydate: list = field(default_factory=list)
    i_bydate: list = field(default_factory=list)
    r_bydate: list = field(default_factory=list)
    e_bydatenew: list = field(default_factory=list)
    i_bydatenew: list = field(default_factory=list)
    r_bydatenew: list = field(default_factory=list)
    new_cases_by_age: list = field(default_factory=list)

    def __post_init__(self) -> None:
        self.i_in_selected = [[] for _ in self.selected]
        self.s_in_selected = [[] for _ in self.selected]
        self.i_ratio_in_selected = [[] for _ in self.selected]
        self.daily_tested_in_selected = {k[1]: [] for k in self.selected}
        self.s_ratio_in_selected = [[] for _ in self.selected]
        self.s2e_in_selected = [[] for _ in self.selected]
        self.s2e_ratio_in_selected = [[] for _ in self.selected]
        self.largest_s2e = np.zeros(len(self.towns_list))
        self.largest_s2e_date = np.zeros(len(self.towns_list))
        self.largest_s2e_csa = np.zeros(len(self.csa2county))
        self.largest_s2e_date_csa = np.zeros(len(self.csa2county))
        self.i2r_in_selected = [[] for _ in self.selected]
        self.r2s_in_selected = [[] for _ in self.selected]
        self.i2p_in_selected = [[] for _ in self.selected]
        self.population_in_selected = [[] for _ in self.selected]

    def record_population_state(self, towns: np.ndarray, population: np.ndarray, status2id: dict[str, int]) -> None:
        self.result_frame.append(towns[:, :, status2id["I"]].sum(axis=1))
        for s, i_item, i_rt, s_item, s_rt, pop_item in zip(
            self.selected,
            self.i_in_selected,
            self.i_ratio_in_selected,
            self.s_in_selected,
            self.s_ratio_in_selected,
            self.population_in_selected,
        ):
            idx = s[0] - 1
            total = towns[idx, :, :].sum()
            i_item.append(towns[idx, :, [status2id["I"]]].sum())
            s_item.append(towns[idx, :, [status2id["S"]]].sum())
            i_rt.append(towns[idx, :, [status2id["I"]]].sum() / total)
            s_rt.append(towns[idx, :, [status2id["S"]]].sum() / total)
            pop_item.append(population[idx])
        self.i_ratio_in_all.append(towns[:, :, [status2id["I"]]].sum() / population.sum())
        self.i_in_all.append(towns[:, :, [status2id["I"]]].sum())
        self.i_gini.append(get_Gini(towns[:, :, status2id["I"]].sum(axis=1) / self.initpopulations, self.initpopulations))
        self.i_theil_l.append(get_Theil_L(towns[:, :, status2id["I"]].sum(axis=1), self.initpopulations))
        self.s_ratio_in_all.append(towns[:, :, [status2id["S"]]].sum() / population.sum())
        self.p_ratio_in_all.append(towns[:, :, [status2id["P"]]].sum() / population.sum())
        self.r_ratio_in_all.append(towns[:, :, [status2id["R"]]].sum() / population.sum())
        self.s_in_all.append(towns[:, :, [status2id["S"]]].sum())

    def record_transition_state(
        self,
        day_index: int,
        s2e: np.ndarray,
        s2e_rate: np.ndarray,
        r2s: np.ndarray,
        i2r: np.ndarray,
        i2p: np.ndarray,
        tested_by_region: np.ndarray,
        delta_d: np.ndarray,
        population: np.ndarray,
    ) -> None:
        self.daily_new.append(s2e.sum(axis=1) / self.initpopulations)
        self.daily_death.append(delta_d.sum(axis=0))
        self.daily_tested.append(tested_by_region)
        for s, s2e_s, s2e_rs, i2r_s, r2s_s, i2p_s in zip(
            self.selected,
            self.s2e_in_selected,
            self.s2e_ratio_in_selected,
            self.i2r_in_selected,
            self.r2s_in_selected,
            self.i2p_in_selected,
        ):
            idx = s[0] - 1
            s2e_s.append(s2e[idx, :].sum())
            s2e_rs.append(s2e_rate[idx, idx, :].mean())
            self.daily_tested_in_selected[s[1]].append(tested_by_region[idx + 1])
            i2r_s.append(i2r[idx, :].sum())
            r2s_s.append(r2s[idx, :].sum())
            i2p_s.append(i2p[idx, :].sum())
        self.new_cases_by_age.append(s2e.sum(axis=0))
        self.s2e_in_all.append(s2e.sum())
        self.total_tested_i.append(tested_by_region.sum())
        self.s2e_ratio_in_all.append(s2e.sum() / population.sum())
        self.s2e_gini.append(get_Gini(s2e.sum(axis=1) / self.initpopulations, self.initpopulations))
        self.s2e_theil_l.append(get_Theil_L(s2e.sum(axis=1), self.initpopulations))
        self.tested_i_theil_l.append(get_Theil_L(tested_by_region, self.initpopulations))
        county_s2e = s2e.sum(axis=1)
        self.largest_s2e_date[county_s2e > self.largest_s2e] = day_index
        self.largest_s2e[county_s2e > self.largest_s2e] = county_s2e[county_s2e > self.largest_s2e]
        s2e_csa = get_sum_from_dict(county_s2e, self.csa2countyidx)
        self.largest_s2e_date_csa[s2e_csa > self.largest_s2e_csa] = day_index
        self.largest_s2e_csa[s2e_csa > self.largest_s2e_csa] = s2e_csa[s2e_csa > self.largest_s2e_csa]

    def record_delay_profiles(
        self,
        e_date: np.ndarray,
        i_date: np.ndarray,
        ia_date: np.ndarray,
        r_date: np.ndarray,
        e_date_new: np.ndarray,
        i_date_new: np.ndarray,
        ia_date_new: np.ndarray,
        r_date_new: np.ndarray,
        population: np.ndarray,
    ) -> None:
        total_population = population.sum()
        self.e_bydate.append(e_date.sum(axis=(0, 1)) / total_population)
        self.i_bydate.append((i_date + ia_date).sum(axis=(0, 1)) / total_population)
        self.r_bydate.append(r_date.sum(axis=(0, 1)) / total_population)
        self.e_bydatenew.append(e_date_new.sum(axis=(0, 1)) / total_population)
        self.i_bydatenew.append((i_date_new + ia_date_new).sum(axis=(0, 1)) / total_population)
        self.r_bydatenew.append(r_date_new.sum(axis=(0, 1)) / total_population)

    def result_tuple(self):
        return (
            self.result_frame,
            self.daily_new,
            self.selected,
            self.i_in_selected,
            self.i_ratio_in_selected,
            self.s_in_selected,
            self.s_ratio_in_selected,
            self.start_at,
            self.total_tested_i,
            self.daily_tested_in_selected,
        )
