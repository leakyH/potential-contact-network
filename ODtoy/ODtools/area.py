from typing import List
import numpy as np
class Town():
    def __init__(self,town,townid,townseir) -> None:
        self.townname = town
        self.townid = townid
        self.array = townseir.array
        self.ageprop = townseir.ageprop
    def init_i(self,_from,_to):
        ageidx = 1
        self.array[ageidx,_to]+=1
        self.array[ageidx,_from]-=1


def buildnp(tl:List[Town],ageCount,statusCount)->np.ndarray:
    array = np.ndarray([len(tl),ageCount,statusCount])
    for idx,town in enumerate(tl):
        array[idx,:,:] = town.array
    return array