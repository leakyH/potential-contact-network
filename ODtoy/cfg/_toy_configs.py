import numpy as np
import pandas as pd

flow_ratio = 1.0



STATUS  = ["S","E","I","R"]
status2id = {}
for idx,s in enumerate(STATUS):
    status2id[s]=idx
moveable = [0,1,2,3]
E_beta = 1/10 
statusbeta = np.array([0,E_beta,1.0,0])


ages=["young","mid","old"]
ageprop= np.array([0.18,0.64,1-0.18-0.64])

init_count = 10 

weekcount=52
alphamat = np.ones([3,3])*2
alphamat[[0,1,2],[0,1,2]]  = 6
alphamat*=2
agebeta = [0.2, 0.2, 0.3]



E_days = 3
I_days = 7
R_days = 30
gamma = 1/E_days
ita =  1/I_days

recover_rate = 1/R_days