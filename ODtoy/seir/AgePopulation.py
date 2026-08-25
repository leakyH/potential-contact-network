import numpy as np
class AgePopulation():
    def __init__(self,population,ageCount,statusCount,_ageProp) -> None:
        self.array = np.zeros([ageCount,statusCount])
        self.array[:,0] = population*np.array(_ageProp)
        self.ageprop = np.array(_ageProp)
