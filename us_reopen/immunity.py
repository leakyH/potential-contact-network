"""Recovered-state immunity duration distributions."""

from __future__ import annotations

import numpy as np
from scipy import stats


Rnames = [
    "R5mGammas",
    "R5mGamma",
    "R5mGammaw",
    "R22wGammas",
    "R23wGammas",
    "R2t8ma",
    "R5mGeom",
    "R12mGeom",
]
parameters = {}


def _getR(Rtype, loc=None, scale=None, Rmin=90, Rmax=365):
    if Rtype == "Gamma":
        gaussian = np.diff(stats.gamma.cdf(np.arange(1, Rmax, 1), a=loc, scale=scale))
    elif Rtype == "Gaussian":
        gaussian = np.diff(stats.norm.cdf(np.arange(1, Rmax, 1), loc=loc, scale=scale))
    elif Rtype == "Uniform":
        temp = np.ones(Rmax - 1)
        temp[np.arange(1, Rmax) < loc[0]] = 0
        temp[np.arange(1, Rmax) > loc[1]] = 0
        gaussian = temp
    elif Rtype == "Geometry":
        gaussian = stats.geom.pmf(np.arange(1, Rmax - 1), 1 / loc)
    elif Rtype == "empirical":
        assert Rmin == 90
        original_x = np.arange(90, Rmax - 1)
        original_y = 0.064 + (0.368 - 0.064) * np.power(original_x - 90, 3.2) / (
            np.power(65.7, 3.2) + np.power(original_x - 90, 3.2)
        )
        proby = np.diff(original_y) / original_y[-1]
        proby0 = original_y[0]
        gaussian = np.concatenate([np.array([proby0]), np.zeros(89), proby])
    elif Rtype == "empirical2":
        assert Rmin == 0
        original_x = np.arange(0, Rmax - 1)
        original_y = 0 + (0.368 - 0) * np.power(original_x, 3.2) / (
            np.power(65.7 + 90, 3.2) + np.power(original_x, 3.2)
        )
        gaussian = np.diff(original_y) / original_y[-1]
    else:
        raise NotImplementedError(f"Rtype = {Rtype}")
    gaussian[0] = gaussian[0:Rmin].sum()
    gaussian[1:Rmin] = 0
    gaussian /= sum(gaussian)
    return gaussian


def resolveRname(Rname: str = None, Rtype=None, loc=None, scale=None):
    if Rname is None:
        return Rtype, loc, scale
    if Rname in parameters:
        return parameters[Rname]

    if "Gamma" in Rname:
        Rtype = "Gamma"
        if "Gammaw" in Rname:
            scale = 60
        elif "Gammas" in Rname:
            scale = 9
        else:
            scale = 30

        if "mGamma" in Rname:
            loc = (int(Rname.removeprefix("R").removesuffix("w").removesuffix("s").removesuffix("mGamma")) * 30) / scale
        elif "wGamma" in Rname:
            loc = (int(Rname.removeprefix("R").removesuffix("w").removesuffix("s").removesuffix("wGamma")) * 7) / scale
    elif Rname.endswith("ma"):
        Rtype = "Uniform"
        if "t" in Rname:
            locstart_str, locend_str = Rname.lstrip("R").rstrip("ma").split("t")
            loc = (int(locstart_str) * 30, int(locend_str) * 30)
        else:
            loc = (int(Rname[1]) * 30, int(Rname[3]) * 30)
        scale = None
    elif "Geom" in Rname:
        Rtype = "Geometry"
        loc = int(Rname.lstrip("R").rstrip("mGeom")) * 30
        scale = None
    elif Rname == "empirical":
        Rtype = "empirical"
        loc = None
        scale = None
    else:
        Rtype = "Gaussian"
        loc = int(Rname[1]) * 30
        if "mw" in Rname:
            scale = loc
        elif "ms" in Rname:
            scale = loc / 3
        else:
            scale = loc * 2 / 3
    return Rtype, loc, scale


def getR(Rname: str = None, Rtype=None, loc=None, scale=None, Rmin=90, Rmax=365):
    return _getR(*resolveRname(Rname, Rtype, loc, scale), Rmin=Rmin, Rmax=Rmax)


def labelfromRname(Rname: str = None, Rtype=None, loc=None, scale=None):
    Rtype, loc, scale = resolveRname(Rname, Rtype, loc, scale)
    label = f"Type:{Rtype}, "
    if Rtype == "Gamma":
        label += "{}={}, {}={}".format(r"$\alpha$", int(loc * scale), r"$\beta$", scale)
    elif Rtype == "Uniform":
        label += f"min={loc[0]}, max={loc[1]}"
    elif Rtype == "Geometry":
        label += "{}={}".format(r"$\frac{1}{p}$", loc)
    elif Rtype == "Gaussian":
        label += "{}={}, {}={}".format(r"$\mu$", loc, r"$\sigma$", scale)
    return label
