"""Plotting color helpers shared by graph reproduction scripts."""

import colorsys

import numpy as np
from matplotlib.colors import Colormap, LinearSegmentedColormap


def hex_rgb_to_rgb_arr(color, range=1):
    color = color.lstrip("#")
    return [int(item, 16) / 255 * range for item in [color[0:2], color[2:4], color[4:6]]]


def _fit_different_rgb(color):
    if len(color) == 3:
        return color
    if len(color) == 4:
        return color[0:3]
    if len(color) == 6 or color[0] == "#":
        return hex_rgb_to_rgb_arr(color)
    return color


def create_hls_colormap(
    color,
    lightness_start=None,
    lightness_end=None,
    saturation_start=None,
    saturation_end=None,
    name="CustomHLS",
):
    color = _fit_different_rgb(color)
    hue, lightness, saturation = colorsys.rgb_to_hls(*color)

    if lightness_start is None:
        lightness_start = lightness
    if lightness_end is None:
        lightness_end = lightness
    if saturation_start is None:
        saturation_start = saturation
    if saturation_end is None:
        saturation_end = saturation
    colors = [
        colorsys.hls_to_rgb(
            hue,
            lightness_start * (1 - v) + lightness_end * v,
            saturation_start * (1 - v) + saturation_end * v,
        )
        for v in np.linspace(0, 1, 256)
    ]

    return LinearSegmentedColormap.from_list(name, colors, N=256)


class DynamicHLSColormap(Colormap):
    def __init__(
        self,
        color,
        lightness_start=0,
        lightness_end=1,
        saturation_start=0.8,
        saturation_end=0.8,
        name="hls_custom",
    ):
        super().__init__(name, N=256)
        self.hue = colorsys.rgb_to_hls(*color)[0]
        self.lightness_start = lightness_start
        self.lightness_end = lightness_end
        self.saturation_start = saturation_start
        self.saturation_end = saturation_end

    def __call__(self, value, alpha=1.0, bytes=False):
        if isinstance(value, np.ndarray):
            value = np.array(value, copy=True).squeeze()
            rgba = np.zeros((value.shape[0], 4), dtype=float)
            hls_values = np.array(
                [
                    (
                        self.hue,
                        self.lightness_start * (1 - v) + self.lightness_end * v,
                        self.saturation_start * (1 - v) + self.saturation_end * v,
                    )
                    for v in value
                ]
            )
            rgba[:, :3] = np.array([colorsys.hls_to_rgb(*hls) for hls in hls_values])
            rgba[:, 3] = alpha
            if bytes:
                rgba = (rgba * 255).astype(np.uint8)
            return rgba

        l = self.lightness_start * (1 - value) + self.lightness_end * value
        s = self.saturation_start * (1 - value) + self.saturation_end * value
        rgba = (*colorsys.hls_to_rgb(self.hue, l, s), alpha)
        if bytes:
            return (np.array(rgba) * 255).astype(np.uint8)
        return rgba


class SequentialColormap(Colormap):
    def __init__(self, color_start, color_end, name="sequential_custom"):
        super().__init__(name)
        self.color_start = np.array(_fit_different_rgb(color_start))
        self.color_end = np.array(_fit_different_rgb(color_end))

    def __call__(self, value, alpha=1.0, bytes=False):
        value = np.clip(value, 0, 1)
        if isinstance(value, np.ndarray):
            value = np.array(value, copy=True).squeeze()
            rgba = np.zeros((value.shape[0], 4), dtype=float)
            rgba[:, :3] = self.color_start * (1 - value) + self.color_end * value
            rgba[:, 3] = alpha
            if bytes:
                rgba = (rgba * 255).astype(np.uint8)
            return rgba

        rgba = self.color_start * (1 - value) + self.color_end * value
        rgba = (*rgba, alpha)
        if bytes:
            return (np.array(rgba) * 255).astype(np.uint8)
        return rgba
