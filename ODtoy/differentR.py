import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
Rnames = [
          'R4mGamma','R4mGammaw',
          'R6mGamma','R6mGammaw',
        "R0t7ma","R0t12ma",
          'R12mGeom','R5mGeom',
          ]
parameters = {
}
def _getR(Rtype,loc=None,scale= None):
    if Rtype == 'Gamma':
        gaussian = np.diff(stats.gamma.cdf(np.arange(1,365,1),a=loc,scale=scale))
    elif Rtype =='Gaussian':
        gaussian = np.diff(stats.norm.cdf(np.arange(1,365,1),loc=loc,scale=scale))
    elif Rtype == 'Uniform':
        temp = np.ones(364)
        temp[np.arange(1,365)<loc[0]] = 0
        temp[np.arange(1,365)>loc[1]] = 0
        gaussian = temp
    elif Rtype =='Geom':
        gaussian = stats.geom.pmf(np.arange(1,366),1/loc)
    else:
        raise NotImplementedError(f"Rtype = {Rtype}")
    gaussian /=sum(gaussian)
    return gaussian
def resolveRname(Rname:str=None,Rtype=None,loc=None,scale = None):
    if Rname is None:
        return Rtype,loc,scale
    if Rname in parameters:
        return parameters[Rname]
    

    if "Gamma" in Rname:
        Rtype = 'Gamma'
        if "Gammaw" in Rname:
            scale = 60
        elif "Gammas" in Rname:
            scale = 15
        else:
            scale = 30
        loc = int(Rname[1])*30/scale

    elif Rname.endswith("ma"):
        Rtype = 'Uniform'
        if 't' in Rname:
            locstart_str,locend_str = Rname.lstrip("R").rstrip('ma').split('t')
            loc = (int(locstart_str)*30,int(locend_str)*30)
        else:
            loc = (int(Rname[1])*30,int(Rname[3])*30)
        scale = None
    elif "Geom" in Rname:
        Rtype = 'Geom'
        loc = int(Rname.lstrip("R").rstrip('mGeom'))*30
        scale = None
    else:
        Rtype = 'Gaussian'
        loc = int(Rname[1])*30
        if 'mw' in Rname:
            scale = loc
        elif 'ms' in Rname:
            scale = loc/3
        else:
            scale = loc*2/3
    return Rtype,loc,scale


def getR(Rname:str=None,Rtype=None,loc=None,scale= None):
    return _getR(*resolveRname(Rname,Rtype,loc,scale))

if __name__ == "__main__":
        
    for Rtype in Rnames:
        gaussian = getR(Rtype)
        plt.plot(np.cumsum(gaussian),label = Rtype)
    plt.legend()
    plt.savefig("tmp/Rtypes.jpg")