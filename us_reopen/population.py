"""Population container helpers for initial county SEIR state."""

from __future__ import annotations

from typing import List

import numpy as np


class AgePopulation:
    def __init__(self, population, ageCount, statusCount, _ageProp) -> None:
        self.array = np.zeros([ageCount, statusCount], dtype=int)
        self.array[:, 0] = np.random.multinomial(population, _ageProp)
        self.ageprop = np.array(_ageProp)


class Town:
    def __init__(self, town, townid, townseir) -> None:
        self.townname = town
        self.townid = townid
        self.array = townseir.array
        self.ageprop = townseir.ageprop

    def init_i(self, _from, _to):
        ageidx = np.random.choice([0, 1, 2], p=self.array[:, _from] / self.array[:, _from].sum())
        self.array[ageidx, _to] += 1
        self.array[ageidx, _from] -= 1


def buildnp(tl: List[Town], ageCount, statusCount) -> np.ndarray:
    array = np.ndarray([len(tl), ageCount, statusCount])
    for idx, town in enumerate(tl):
        array[idx, :, :] = town.array
    return array


def init_i(towns, _from, _to, local_symptom_rate=None, init_town_index=None, count=None):
    probs = [town.array.sum() for town in towns]
    probs = np.array(probs) / sum(probs)
    selectedtowns = []
    assert (init_town_index is None) ^ (count is None)
    if init_town_index is not None:
        for _id in init_town_index:
            randomtown = towns[_id]
            selectedtowns.append((randomtown.townid, randomtown.townname))
            if local_symptom_rate is None:
                randomtown.init_i(_from, _to)
            else:
                assert len(_to) == 2, "local_symptom_rate requires two target status ids"
                if np.random.random() < local_symptom_rate:
                    randomtown.init_i(_from, _to[0])
                else:
                    randomtown.init_i(_from, _to[1])
    else:
        for _ in range(count):
            randomtown = np.random.choice(towns, p=probs)
            selectedtowns.append((randomtown.townid, randomtown.townname))
            if local_symptom_rate is None:
                randomtown.init_i(_from, _to)
            else:
                assert len(_to) == 2, "local_symptom_rate requires two target status ids"
                if np.random.random() < local_symptom_rate:
                    randomtown.init_i(_from, _to[0])
                else:
                    randomtown.init_i(_from, _to[1])
    return selectedtowns
