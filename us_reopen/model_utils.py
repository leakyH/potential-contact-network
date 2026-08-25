"""Shared helpers for the US reopening model."""

from __future__ import annotations

from copy import copy, deepcopy
import os
import warnings

import matplotlib
import matplotlib.cm
from matplotlib import animation, axes, colorbar
from matplotlib import pyplot as plt


CMAP = copy(matplotlib.cm.get_cmap("inferno"))
CMAP.set_under("gray")

linestyle_tuple = [
    ("solid", "solid"),
    ("densely dotted", (0, (1, 1))),
    ("dashed", (0, (5, 5))),
    ("densely dashed", (0, (5, 1))),
    ("densely dashdotted", (0, (3, 1, 1, 1))),
    ("dashdotdotted", (0, (3, 5, 1, 5, 1, 5))),
    ("loosely dashdotdotted", (0, (3, 10, 1, 10, 1, 10))),
    ("densely dashdotdotted", (0, (3, 1, 1, 1, 1, 1))),
]


def initMapGif(savepath):
    fig, ax = plt.subplots(1, 1, figsize=[6, 5], dpi=200, facecolor="white")
    fig.subplots_adjust(right=0.8)
    position = fig.add_axes([0.85, 0.2, 0.015, 0.6])
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0, clip=True)
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=position)
    ax.axis("off")
    gifwriter = animation.FFMpegWriter()
    gifwriter.setup(fig, savepath, dpi=200)
    return gifwriter, fig, ax, cb


def DrawMapGif(
    gifwriter: animation.FFMpegFileWriter,
    ax: axes.Axes,
    cb: colorbar.Colorbar,
    data,
    map_flag,
    title,
    vmin=0.0,
    vmax=1.0,
    log_norm=False,
):
    ax.clear()
    if log_norm:
        norm = matplotlib.colors.LogNorm(vmin=vmin + 1e-6, vmax=vmax, clip=True)
    else:
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    cb.update_normal(matplotlib.cm.ScalarMappable(norm=norm, cmap=CMAP))
    map_flag_copy = deepcopy(map_flag)
    for i in range(len(data)):
        map_flag_copy[map_flag == i + 1] = data[i]
    alpha = deepcopy(map_flag)
    alpha[map_flag >= 0] = 1.0
    alpha[map_flag < 0] = 0.0
    ax.imshow(map_flag_copy, alpha=alpha, norm=norm, cmap=CMAP)
    ax.set_title(title)
    gifwriter.grab_frame()


def saveMapGif_mpa(gifwriter: animation.FFMpegFileWriter):
    gifwriter.finish()


def PlotSelectedScalar(
    selectedtowns,
    Indexinselected,
    title,
    filename,
    logscale=False,
    extraInfo=None,
    ref=None,
    ylim=None,
    same_ylim=False,
):
    for s, item in zip(selectedtowns, Indexinselected):
        _id, _name, _popcount, _odcount, _pop_od = s
        plt.plot(item, label=str(_id) + "|" + f"{_popcount:.2f}_{_odcount:.2f}_{_pop_od:.2f}")
    plt.yscale("log" if logscale else "linear")
    if ylim is not None:
        plt.ylim(*ylim)
    if extraInfo is not None:
        if isinstance(extraInfo, dict):
            for (k, v), (_, ls) in zip(extraInfo.items(), linestyle_tuple):
                plt.plot(v, color="black", label=k, linestyle=ls)
        else:
            plt.plot(extraInfo, color="black", label="overall")
    if ref is not None:
        ax = plt.gca()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Data has no positive values, and therefore cannot be log-scaled.",
                category=UserWarning,
            )
            bx = plt.twinx(ax)
        if isinstance(ref, dict):
            for (k, v), (_, ls) in zip(ref.items(), linestyle_tuple):
                bx.plot(v, color="chocolate", label=k, linestyle=ls)
        else:
            bx.plot(ref, color="chocolate", label="reference")
        if ylim is not None:
            bx.set_ylim(*ylim)
        if logscale:
            ax.set_yscale("log")
            bx.set_yscale("log")
        if same_ylim:
            axylim = ax.get_ylim()
            bxylim = bx.get_ylim()
            if logscale:
                if min(axylim[0], bxylim[0]) > 0:
                    bx.set_ylim(min(axylim[0], bxylim[0]), max(axylim[1], bxylim[1]))
                    ax.set_ylim(min(axylim[0], bxylim[0]), max(axylim[1], bxylim[1]))
                elif max(axylim[0], bxylim[0]) < 0:
                    pass
                else:
                    bx.set_ylim(max(axylim[0], bxylim[0]), max(axylim[1], bxylim[1]))
                    ax.set_ylim(max(axylim[0], bxylim[0]), max(axylim[1], bxylim[1]))
            else:
                bx.set_ylim(min(axylim[0], bxylim[0]), max(axylim[1], bxylim[1]))
                ax.set_ylim(min(axylim[0], bxylim[0]), max(axylim[1], bxylim[1]))

    plt.legend()
    plt.title(title)
    plt.savefig(filename)
    plt.close()


def process_kwargs(**kwargs):
    flow_ratio_local = kwargs.get("flow_ratio", None)
    dryrun = kwargs.get("dryrun", False)
    gif = kwargs.get("gif", False)
    scalar = kwargs.get("scalar", False)
    suffix = kwargs.get("suffix", None)
    beta_density = kwargs.get("beta_density", False)
    if suffix is None:
        _suffix = ""
    else:
        _suffix = "_" + suffix
    return flow_ratio_local, dryrun, gif, scalar, _suffix, beta_density


def MapInit():
    if not os.path.exists("output/pngs"):
        os.makedirs("output/pngs")
    if not os.path.exists("output/gifs"):
        os.makedirs("output/gifs")
    _vmax = 0.001
    _vmax2 = 1
    map_flag = None
    return _vmax, _vmax2, map_flag


def addSelected_manual(selected: list, towns: list, node_list, initpopulations, town_area):
    for item in node_list:
        selected.append((towns[item].townid, towns[item].townname, initpopulations[item], -1, initpopulations[item] / town_area[item]))
