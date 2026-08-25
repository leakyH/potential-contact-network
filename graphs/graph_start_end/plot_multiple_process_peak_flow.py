"""Overlay prevalence peak versus flow/population for several process methods.

The script reads existing curve outputs; it does not run the epidemic model.
"""

import argparse
import csv
from pathlib import Path
import pickle as pkl
import re
import sys
import os

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.append("./")
from paper_repro_paths import font_dir
from graphs.graph_common.figure_output import add_output_args, configure_from_args, graph_output_path
from graphs.graph_common.plot_colors import create_hls_colormap
from us_reopen.network_processing import (
    analysis_process_method,
    criterias_name,
    process_number_to_fns,
)
from us_reopen.us_data import getInformation, buildUSNetwork


DEFAULT_EXPERIMENT = "us_preCovid_fit_empirical_P0p632_a2_fit_f1"
DEFAULT_SUFFIX = (
    "preCovid_Iairport5k_empirical_P0p632_a2_fit_f1_aftershutdown_v1_"
    "prc{process_method}_ar0p60120_1_TO5_fitted500kIa0p9initPF120_120_inf_"
    "{method}_{run_id}"
)
# Shared marker, line, and color settings for the start/end figures.
MARKERS = ['o', 's', '^', 'D', 'v', '*', 'p', 'X', '<', '>', '+', 'd']
LINESTYLES = ['--', '-', '-.', ':']
LINECOLORS = [
    "#aa3474", "#3b5249", "#845e48", "#639aab", "#b2a2ba", "#fc8d62",
    "#ff9d9f", "#98FBCB", "#6E18DE", "#FFE176", "#0AA137", "#76EAFF",
]
TEST_OVER = 5
SHUTDOWN_NAME = '0p60'
REOPEN_DATE = 120
REOPEN_NAME = '1'
XLIM = (-0.009973271357008916, 0.20943869849718724)
PEAK_AFTER_YLIM = (8.3e4, 1.1e5)
FIG4H_PEAK_AFTER_YLIM = (8.4e4, 1.1e5)
PLOT_CURVE_OFFSET = -30
LINEAR_PROCESS_COLORS = {
    31: "#fc8d62",
    34: "#fc8d62",
    40: "#66c2a5",
    67: "#66c2a5",
    58: "#8da0cb",
    70: "#8da0cb",
}


def configure_plot_style(is_fig4h=False):
    """Apply the typography settings used by the start/end figures."""
    plt.rcParams['svg.fonttype'] = 'none'
    for font_file in fm.findSystemFonts(fontpaths=[font_dir()]):
        fm.fontManager.addfont(font_file)
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 7.0 if is_fig4h else 7.5
    plt.rcParams['pdf.fonttype'] = 42


def parse_int_list(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def load_point(input_root, experiment, method, suffix_template, process_method, run_id,
               process_start, peak_days):
    suffix = suffix_template.format(
        process_method=process_method, method=method, run_id=run_id
    )
    base = Path(input_root)
    prevalence_path = base / "pkls" / experiment / method / f"I_exist_count_{suffix}.npy"
    flow_path = base / "csvs" / experiment / method / f"final_flow_{suffix}.csv"
    prevalence = np.load(prevalence_path)
    if prevalence.ndim != 2:
        raise ValueError(f"Expected a time x county array: {prevalence_path}")
    with flow_path.open(newline="") as handle:
        row = next(csv.reader(handle))
    flow_over_population = float(row[2])
    prevalence_total = prevalence.sum(axis=1)
    stop = (
        len(prevalence_total)
        if peak_days is None
        else min(process_start + peak_days, len(prevalence_total))
    )
    if process_start >= stop:
        raise ValueError(f"Empty peak window for {prevalence_path}")
    window = prevalence_total[process_start:stop]
    peak_offset = int(np.argmax(window))
    return {
        "process_method": process_method,
        "method": method,
        "run_id": run_id,
        "flow_over_population": flow_over_population,
        "prevalence_peak": float(window[peak_offset]),
        "peak_day": process_start + peak_offset,
        "prevalence_path": str(prevalence_path),
    }, prevalence


def load_multiple_points(input_root, experiment, method, suffix_template,
                         process_method, process_start, peak_days):
    """Load every available single-flow repeated-seed output for one method."""
    base = Path(input_root)
    pkl_dir = base / "pkls" / experiment / method
    csv_dir = base / "csvs" / experiment / method
    wildcard_suffix = suffix_template.format(
        process_method=process_method, method=method, run_id="*"
    )
    prefix, suffix = wildcard_suffix.split("*", maxsplit=1)
    pattern = f"I_exist_count_{prefix}*{suffix}_multiple.npy"
    rows = []
    for prevalence_path in sorted(pkl_dir.glob(pattern)):
        filename = prevalence_path.name
        token = filename.removeprefix(f"I_exist_count_{prefix}").removesuffix(
            f"{suffix}_multiple.npy"
        )
        match = re.fullmatch(r"a(\d+(?:p\d+)?)", token)
        if match:
            a_value = float(match.group(1).replace("p", "."))
            curve_index = int(round(a_value * 20))
        elif token.isdigit():
            curve_index = int(token) - 300
            a_value = curve_index / 20
        else:
            continue

        flow_paths = sorted(csv_dir.glob(f"final_flow_{prefix}{token}{suffix}rep*.csv"))
        flow_values = []
        for flow_path in flow_paths:
            with flow_path.open(newline="") as handle:
                flow_values.append(float(next(csv.reader(handle))[2]))
        if not flow_values:
            continue

        prevalence = np.load(prevalence_path, mmap_mode="r")
        if prevalence.ndim != 3:
            raise ValueError(f"Expected repeat x time x county: {prevalence_path}")
        stop = (
            prevalence.shape[1]
            if peak_days is None
            else min(process_start + peak_days, prevalence.shape[1])
        )
        totals = prevalence[:, process_start:stop, :].sum(axis=2)
        peaks = np.max(totals, axis=1)
        mean_flow = float(np.mean(flow_values))
        for rep, peak in enumerate(peaks):
            rows.append({
                "process_method": process_method,
                "method": method,
                "curve_index": curve_index,
                "a_value": a_value,
                "rep": rep,
                "flow_over_population": mean_flow,
                "prevalence_peak": float(peak),
                "multiple_path": str(prevalence_path),
            })
    return rows


def replace_curve_points_with_multiple_means(points, multiple_points):
    """Replace curve points with repeated-run means where available."""
    if multiple_points.empty:
        return points
    points = points.copy()
    for (process_method, method, curve_index), group in multiple_points.groupby(
        ["process_method", "method", "curve_index"]
    ):
        mask = (
            (points.process_method == process_method)
            & (points.method == method)
            & (points.run_id == 300 + curve_index)
        )
        if mask.any():
            points.loc[mask, "flow_over_population"] = group.flow_over_population.mean()
            points.loc[mask, "prevalence_peak"] = group.prevalence_peak.mean()
    return points


def add_relative_flow_and_epp(points, multiple_points):
    """Add plot coordinates using one unrestricted/isolation reference curve.

    The unrestricted reference is the curve containing the largest observed
    inter-county flow.  Its minimum-flow endpoint supplies the isolation peak,
    so the right axis can remain a single absolute EPP scale for the full plot.
    """
    if points.empty:
        raise ValueError("Cannot define unrestricted flow from an empty point set")

    unrestricted_flow = float(points.flow_over_population.max())
    if not np.isfinite(unrestricted_flow) or unrestricted_flow <= 0:
        raise ValueError(
            "Unrestricted inter-county flow must be finite and greater than zero"
        )

    unrestricted_candidates = points[np.isclose(
        points.flow_over_population, unrestricted_flow
    )].sort_values(["process_method", "method", "run_id"])
    unrestricted_point = unrestricted_candidates.iloc[0]
    reference_curve = points[
        (points.process_method == unrestricted_point.process_method)
        & (points.method == unrestricted_point.method)
    ].sort_values(["flow_over_population", "run_id"])
    if reference_curve.empty:
        raise ValueError("The unrestricted reference curve has no endpoints")

    isolation_point = reference_curve.iloc[0]
    isolation_peak = float(isolation_point.prevalence_peak)
    isolation_reference_method = unrestricted_point.method

    # When repeated seeds are part of the figure, anchor EPP=0 to the mean of
    # the repeated linear isolation endpoint (the 21st/a=1 curve point).  The
    # main curve has already been replaced by the same repeated-seed mean, but
    # compute the reference directly from those repeated runs.
    if not multiple_points.empty:
        reference_linear = points[
            (points.process_method == unrestricted_point.process_method)
            & (points.method == "linear")
        ].sort_values(["flow_over_population", "run_id"])
        if not reference_linear.empty:
            linear_isolation_flow = float(
                reference_linear.iloc[0].flow_over_population
            )
            repeated_linear_isolation = multiple_points[
                (multiple_points.process_method == unrestricted_point.process_method)
                & (multiple_points.method == "linear")
                & np.isclose(
                    multiple_points.flow_over_population,
                    linear_isolation_flow,
                )
            ]
            if not repeated_linear_isolation.empty:
                isolation_peak = float(
                    repeated_linear_isolation.prevalence_peak.mean()
                )
                isolation_reference_method = "linear_multiple_mean"
    unrestricted_peak = float(unrestricted_point.prevalence_peak)
    epp_max = unrestricted_peak - isolation_peak
    if not np.isfinite(epp_max) or epp_max <= 0:
        raise ValueError(
            "EPP_max must be positive: unrestricted peak must exceed isolation peak"
        )

    points = points.copy()
    points["remaining_flow_pct"] = (
        points.flow_over_population / unrestricted_flow * 100
    )
    points["epp"] = points.prevalence_peak - isolation_peak

    multiple_points = multiple_points.copy()
    multiple_points["remaining_flow_pct"] = (
        multiple_points.flow_over_population / unrestricted_flow * 100
    )
    multiple_points["epp"] = multiple_points.prevalence_peak - isolation_peak

    baseline = {
        "reference_process_method": int(unrestricted_point.process_method),
        "reference_method": unrestricted_point.method,
        "isolation_reference_method": isolation_reference_method,
        "unrestricted_flow": unrestricted_flow,
        "unrestricted_peak": unrestricted_peak,
        "isolation_peak": isolation_peak,
        "epp_max": epp_max,
    }
    for name in ("unrestricted_flow", "unrestricted_peak", "isolation_peak", "epp_max"):
        points[name] = baseline[name]
        multiple_points[name] = baseline[name]
    return points, multiple_points, baseline


def remaining_flow_for_reduction(reduction_pct, minimum_remaining_pct=0):
    """Map intervention reduction to remaining flow with an optional floor."""
    return minimum_remaining_pct + (
        100 - minimum_remaining_pct
    ) * (1 - reduction_pct / 100)


def pch_flow_reduction_points(points, minimum_remaining_pct=0):
    """Return the PCH points nearest 15% and 35% flow reduction."""
    pch = points[points.method == "realtime_county_pchDextS_both"]
    # Full-isolation Fig. 5 uses dynamic method 33; the 0.1-flow-floor
    # counterpart uses method 36.  Only one should be present in a preset.
    for process_method in (33, 36):
        candidate = pch[pch.process_method == process_method]
        if not candidate.empty:
            pch = candidate
            break
    if pch.empty:
        return {}
    selected = {}
    for reduction_pct in (15, 35):
        target_remaining = remaining_flow_for_reduction(
            reduction_pct, minimum_remaining_pct
        )
        row = pch.loc[
            (pch.remaining_flow_pct - target_remaining).abs().idxmin()
        ]
        selected[reduction_pct] = row
    return selected


def add_fig5_flow_reduction_guides(ax, points, minimum_remaining_pct=0):
    """Add long- and short-term flow-reduction annotations to Fig. 5."""
    guide_color = "0.65"
    half_reduction_x = remaining_flow_for_reduction(
        50, minimum_remaining_pct
    )
    ax.axvline(
        half_reduction_x,
        color=guide_color,
        linestyle="--",
        linewidth=0.8,
        zorder=1,
    )
    ax.text(
        half_reduction_x,
        0.98,
        "Flow reduction = 50%",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize = 7
    )

    y_half_length = 3000
    label_offsets = {
        15: (-2.0, -2200, "left", "top"),
        35: (2.0, 3200, "right", "bottom"),
    }
    for reduction_pct, row in pch_flow_reduction_points(
        points, minimum_remaining_pct
    ).items():
        x = float(row.remaining_flow_pct)
        y = float(row.prevalence_peak)
        ax.vlines(
            x,
            y - y_half_length,
            y + y_half_length,
            color=guide_color,
            linestyle="--",
            linewidth=0.8,
            zorder=1,
        )
        dx, dy, ha, va = label_offsets[reduction_pct]
        ax.text(
            x + dx,
            y + dy,
            f"Flow reduction\n= {reduction_pct}%",
            ha=ha,
            va=va,
            zorder=7,
            fontsize = 5,
            # bbox={
            #     "facecolor": "none",
            #     "edgecolor": "none",
            #     "alpha": 0.75,
            #     "pad": 0.5,
            # },
        )


def add_minimum_mobility_flow_reduction_guides(ax):
    """Draw only the 50% and 100% guides for the 0.1-flow-floor preset."""
    guide_color = "0.65"
    for reduction_pct in (50, 100):
        x = remaining_flow_for_reduction(
            reduction_pct, minimum_remaining_pct=10
        )
        ax.axvline(
            x,
            color=guide_color,
            linestyle="--",
            linewidth=0.8,
            zorder=1,
        )
        ax.text(
            x,
            0.98,
            f"Flow reduction = {reduction_pct}%",
            transform=ax.get_xaxis_transform(),
            ha="left",
            va="bottom",
            fontsize=7,
        )


def preset_guide_configuration(figure_style):
    """Return the reference lines enabled by a figure preset."""
    return {
        "fig4h": {
            "epp_levels": (0, 50, 100),
            "flow_reductions": (),
            "pch_reductions": (),
        },
        "fig5": {
            "epp_levels": (0, 50, 100),
            "flow_reductions": (50,),
            "pch_reductions": (15, 35),
        },
        "fig5_min_mobility": {
            "epp_levels": (0, 100),
            "flow_reductions": (50, 100),
            "pch_reductions": (),
        },
    }.get(
        figure_style,
        {"epp_levels": (), "flow_reductions": (), "pch_reductions": ()},
    )


def compare_methods(points, reference_method):
    rows = []
    if reference_method not in points.process_method.unique():
        return pd.DataFrame(rows)
    for method in points.method.unique():
        reference = points[
            (points.process_method == reference_method) & (points.method == method)
        ].set_index("run_id")
        if reference.empty:
            continue
        for process_method in sorted(points.process_method.unique()):
            if process_method == reference_method:
                continue
            candidate = points[
                (points.process_method == process_method) & (points.method == method)
            ].set_index("run_id")
            shared = reference.index.intersection(candidate.index)
            for run_id in shared:
                ref = reference.loc[run_id]
                cur = candidate.loc[run_id]
                ref_array = np.load(ref.prevalence_path)
                cur_array = np.load(cur.prevalence_path)
                rows.append({
                    "reference_process_method": reference_method,
                    "process_method": process_method,
                    "method": method,
                    "run_id": run_id,
                    "flow_abs_diff": abs(ref.flow_over_population - cur.flow_over_population),
                    "peak_abs_diff": abs(ref.prevalence_peak - cur.prevalence_peak),
                    "peak_relative_diff": abs(ref.prevalence_peak - cur.prevalence_peak)
                    / max(abs(ref.prevalence_peak), np.finfo(float).eps),
                    "prevalence_array_identical": np.array_equal(ref_array, cur_array),
                    "prevalence_array_max_abs_diff": (
                        float(np.max(np.abs(ref_array - cur_array)))
                        if ref_array.shape == cur_array.shape else np.nan
                    ),
                })
    return pd.DataFrame(rows)


def method_style(process_method, method):
    if method == 'linear':
        return {
            'color': LINEAR_PROCESS_COLORS.get(process_method, 'black'),
            'linestyle': '-', 'marker': None, 'linewidth': 3,
        }
    method_ana = analysis_process_method(method)
    color = next(
        (color for criteria, color in zip(criterias_name, LINECOLORS)
         if criteria == method_ana['criteria']),
        LINECOLORS[0],
    )
    return {
        'color': color,
        'linestyle': LINESTYLES[0 if method_ana['inverse'] else 1],
        'marker': MARKERS[0 if method_ana['both_direction'] else 1],
        'linewidth': (
            3 if (
                not method_ana['inverse'] and method_ana['both_direction']
                and method_ana['criteria'] == 'pcf_compare'
            ) else 1.5
        ),
    }



def load_curve_group_definition(period, beta_density):
    """Build four population-weighted county groups."""
    period = period.split('-', maxsplit=1)[0]
    if period in [
        'preCovid', 'Alpha', 'Delta', 'AlphaRestrict',
        'preCovidlikeAlphaRestrict',
    ]:
        graph_path = Path(
            f"graphs/graph1/average_graph_full_daily_{period}_workday.pkl"
        )
        with graph_path.open('rb') as handle:
            graph = pkl.load(handle)
    elif period in ['Omicron', 'Omicron_lm']:
        graph_path = Path(f"graphs/graph1/average_graph_full_{period}.pkl")
        with graph_path.open('rb') as handle:
            graph = pkl.load(handle)
    elif period == 'commuting':
        graph = buildUSNetwork(
            "full", basedir="./ext-data/us-counties/", recompute=False
        )
    else:
        raise ValueError(f"Unsupported period for grouped curve plotting: {period}")

    csainfo = getInformation(graph.nodes())
    populations = np.asarray(list(csainfo[1].values()))

    if 'log' in beta_density:
        density = populations / np.asarray(csainfo[2])
        group_basis = 'density'
    elif beta_density in ['cfg', 'cfgFull']:
        alpha_csv = pd.read_csv("graphs/graphR/testR0.csv", dtype={"fips": str})
        county_cfg = {line["fips"]: line["r"] for _, line in alpha_csv.iterrows()}
        beta_median = np.median(list(county_cfg.values()))
        density = np.asarray([
            county_cfg.get(item, beta_median) for item in csainfo[0]
        ])
        group_basis = 'county_alpha'
    elif beta_density == 'fit':
        alpha_csv = pd.read_csv(
            "graphs/graphR/testR0_max_fitted_beta.csv", dtype={"fips": str}
        )
        alpha_csv['fips'] = alpha_csv['fips'].str.zfill(5)
        county_cfg = {
            line["fips"]: line[f"TO{TEST_OVER}_beta"]
            for _, line in alpha_csv.iterrows()
        }
        beta_min = np.min(list(county_cfg.values()))
        density = np.asarray([county_cfg.get(item, beta_min) for item in csainfo[0]])
        group_basis = 'county_alpha'
    elif beta_density == 'fit_poisson':
        alpha_csv = pd.read_csv(
            "graphs/graphR/testR0_max_fitted_beta_poisson.csv",
            dtype={"fips": str},
        )
        alpha_csv['fips'] = alpha_csv['fips'].str.zfill(5)
        county_cfg = {
            line["fips"]: line[f"TO{TEST_OVER}_beta"]
            for _, line in alpha_csv.iterrows()
        }
        beta_min = np.min(list(county_cfg.values()))
        density = np.asarray([county_cfg.get(item, beta_min) for item in csainfo[0]])
        group_basis = 'county_alpha'
    else:
        raise ValueError(
            "Grouped curve plotting follows plot_start_end_aftershutdown.py and "
            f"does not define groups for beta_density={beta_density!r}."
        )

    order = np.argsort(density)
    cumulative_population = np.cumsum(populations[order]) / populations.sum()
    groups = []
    for qlevel in range(4):
        groups.append(order[
            (qlevel / 4 <= cumulative_population)
            & (cumulative_population <= (qlevel + 1) / 4)
        ])
    return groups, group_basis, populations


def _a_encoded_run_id(run_id):
    curve_index = int(run_id) - 300
    a_value = curve_index / 20
    return "a" + f"{a_value:g}".replace("-", "m").replace(".", "p")


def _curve_output_paths(input_root, experiment, method, suffix):
    pkl_dir = Path(input_root) / "pkls" / experiment / method
    csv_dir = Path(input_root) / "csvs" / experiment / method
    return {
        "prevalence": pkl_dir / f"I_exist_count_{suffix}.npy",
        "s2e": pkl_dir / f"S2E_ratio_{suffix}.npy",
        "prevalence_multiple": pkl_dir / f"I_exist_count_{suffix}_multiple.npy",
        "s2e_multiple": pkl_dir / f"S2E_ratio_{suffix}_multiple.npy",
        "flow_rep0": csv_dir / f"final_flow_{suffix}rep0.csv",
    }


def _read_flow_over_population(path, fallback):
    if not path.exists():
        return float(fallback)
    with path.open(newline="") as handle:
        return float(next(csv.reader(handle))[2])


def load_curve_seed_zero(point, input_root, experiment, suffix_template):
    """Load I and S2E curves, selecting repeat 0 when repeated outputs exist."""
    numeric_suffix = suffix_template.format(
        process_method=int(point.process_method),
        method=point.method,
        run_id=int(point.run_id),
    )
    encoded_suffix = suffix_template.format(
        process_method=int(point.process_method),
        method=point.method,
        run_id=_a_encoded_run_id(point.run_id),
    )

    for suffix in (numeric_suffix, encoded_suffix):
        paths = _curve_output_paths(
            input_root, experiment, point.method, suffix
        )
        repeated_exists = (
            paths["prevalence_multiple"].exists()
            or paths["s2e_multiple"].exists()
        )
        if not repeated_exists:
            continue
        if not paths["prevalence_multiple"].exists():
            raise FileNotFoundError(paths["prevalence_multiple"])
        if not paths["s2e_multiple"].exists():
            raise FileNotFoundError(paths["s2e_multiple"])

        prevalence_multiple = np.load(
            paths["prevalence_multiple"], mmap_mode="r"
        )
        s2e_multiple = np.load(paths["s2e_multiple"], mmap_mode="r")
        if prevalence_multiple.ndim != 3:
            raise ValueError(
                "Expected repeat x time x county for repeated prevalence: "
                f"{paths['prevalence_multiple']}"
            )
        if s2e_multiple.ndim not in (2, 3):
            raise ValueError(
                "Expected repeat x time or repeat x time x county for repeated "
                f"S2E: {paths['s2e_multiple']}"
            )
        flow_over_population = _read_flow_over_population(
            paths["flow_rep0"], point.flow_over_population
        )
        return (
            np.asarray(prevalence_multiple[0]),
            np.asarray(s2e_multiple[0]),
            flow_over_population,
            paths["prevalence_multiple"],
            paths["s2e_multiple"],
        )

    paths = _curve_output_paths(
        input_root, experiment, point.method, numeric_suffix
    )
    prevalence = np.load(paths["prevalence"])
    s2e = np.load(paths["s2e"])
    if prevalence.ndim != 2:
        raise ValueError(
            f"Expected time x county prevalence: {paths['prevalence']}"
        )
    if s2e.ndim not in (1, 2):
        raise ValueError(
            f"Expected time or time x county S2E: {paths['s2e']}"
        )
    return (
        prevalence,
        s2e,
        float(point.flow_over_population),
        paths["prevalence"],
        paths["s2e"],
    )


def collect_prevalence_curves(points, groups, populations, process_start,
                              input_root, experiment, suffix_template):
    plot_start = max(0, process_start + PLOT_CURVE_OFFSET)
    max_group_index = max(np.max(group) for group in groups if len(group))
    curves = []
    for point in points.itertuples(index=False):
        prevalence, s2e, flow_over_population, prevalence_path, s2e_path = (
            load_curve_seed_zero(
                point, input_root, experiment, suffix_template
            )
        )
        if prevalence.shape[1] <= max_group_index:
            raise ValueError(
                "County dimension does not match grouped county indices: "
                f"{prevalence_path}"
            )
        if prevalence.shape[1] != len(populations):
            raise ValueError(
                "Population vector and prevalence county dimensions differ: "
                f"{len(populations)} != {prevalence.shape[1]} ({prevalence_path})"
            )

        prevalence = prevalence[plot_start:, :]
        if s2e.ndim == 1:
            total_s2e = s2e[plot_start:]
            grouped_s2e = [None for _ in groups]
        else:
            if s2e.shape[1] != len(populations):
                raise ValueError(
                    "Population vector and S2E county dimensions differ: "
                    f"{len(populations)} != {s2e.shape[1]} ({s2e_path})"
                )
            s2e = s2e[plot_start:, :]
            total_s2e = (s2e * populations[np.newaxis, :]).sum(axis=1)
            grouped_s2e = [
                (s2e[:, group] * populations[np.newaxis, group]).sum(axis=1)
                for group in groups
            ]

        curves.append({
            "process_method": int(point.process_method),
            "method": point.method,
            "run_id": int(point.run_id),
            "curve_index": int(point.run_id) - 300,
            "flow_over_population": flow_over_population,
            "total_I": prevalence.sum(axis=1),
            "total_s2e": total_s2e,
            "group_I": [prevalence[:, group].sum(axis=1) for group in groups],
            "group_s2e": grouped_s2e,
            "prevalence_path": str(prevalence_path),
            "s2e_path": str(s2e_path),
        })
    return curves, plot_start


def _curve_method_groups(curves):
    grouped = {}
    for curve in curves:
        key = (curve["process_method"], curve["method"])
        grouped.setdefault(key, []).append(curve)
    return sorted(grouped.items())


def _draw_curve_set(ax, method_curves, value_key, plot_start, group_index=None):
    color_mappables = []
    for (process_method, method), curves in method_curves:
        curves = sorted(curves, key=lambda item: item["curve_index"])
        style = method_style(process_method, method)
        flow_values = np.asarray([
            curve["flow_over_population"] for curve in curves
        ])
        vmin = float(flow_values.min())
        vmax = float(flow_values.max())
        if np.isclose(vmin, vmax):
            vmax = vmin + np.finfo(float).eps
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = create_hls_colormap(
            color=style['color'], saturation_start=0.0, saturation_end=None
        )
        color_mappables.append((
            f"{process_method}: {method}", ScalarMappable(norm=norm, cmap=cmap)
        ))

        has_index_zero = any(curve["curve_index"] == 0 for curve in curves)
        for curve_number, curve in enumerate(curves):
            values = (
                curve[value_key]
                if group_index is None
                else curve[value_key][group_index]
            )
            if values is None:
                continue
            if curve["curve_index"] == 0:
                zorder = 3
                alpha = 1
                linewidth = 2
                label = f"{process_method}: {method}"
            else:
                if str(process_method)  not in ["31","34"]:
                    continue
                zorder = 2
                alpha = 1.0
                linewidth = 1
                label = (
                    f"{process_method}: {method}"
                    if not has_index_zero and curve_number == 0 else None
                )
            ax.plot(
                np.arange(len(values)) + plot_start,
                values,
                linestyle=style['linestyle'],
                marker=None,
                color=cmap(norm(curve["flow_over_population"])),
                linewidth=linewidth,
                zorder=zorder,
                alpha=alpha,
                label=label,
            )
    return color_mappables


def _format_curve_axis(ax):
    ax.ticklabel_format(
        style='sci', axis='y', scilimits=(0, 0), useOffset=True,
        useLocale=False, useMathText=True,
    )


def _add_curve_key(fig, axes, color_mappables):
    axes = np.atleast_1d(axes).ravel()
    if len(color_mappables) == 1:
        fig.colorbar(color_mappables[0][1], ax=list(axes))
    elif color_mappables:
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            axes[0].legend(
                handles, labels, fontsize='small', frameon=True
            )


def draw_prevalence_curves(points, process_start, input_root, experiment,
                           suffix_template, period, beta_density, output_stem):
    """Overlay all-node and four-group I/S2E curves across methods."""
    configure_plot_style()
    groups, group_basis, populations = load_curve_group_definition(
        period, beta_density
    )
    curves, plot_start = collect_prevalence_curves(
        points, groups, populations, process_start, input_root, experiment,
        suffix_template,
    )
    method_curves = _curve_method_groups(curves)

    # Use two stacked panels for the all-node infection and incidence curves.
    fig, (ax_I,ax_s2e) = plt.subplots(2,1,figsize = [4,4],dpi = 300,sharex= True)
    color_mappables = _draw_curve_set(
        ax_I, method_curves, "total_I", plot_start
    )
    _draw_curve_set(ax_s2e, method_curves, "total_s2e", plot_start)
    ax_I.set_ylabel("Exist I")
    ax_s2e.set_ylabel("Daily S2E")
    _format_curve_axis(ax_I)
    _format_curve_axis(ax_s2e)
    _add_curve_key(fig, [ax_I, ax_s2e], color_mappables)
    os.makedirs(f"{output_stem}_curves", exist_ok=True)
    for extension in ("jpg", "svg"):
        fig.savefig(f"{output_stem}_curves/curve.{extension}")
    plt.close(fig)

    # Draw infection and incidence as separate 2x2 county-group figures.
    fig_Q, axs_Q = plt.subplots(2,2,figsize = [4,4],dpi = 300)
    fig_inf_Q, axs_inf_Q = plt.subplots(2,2,figsize = [4,4],dpi = 300)
    q_color_mappables = []
    inf_q_color_mappables = []
    for group_index, (ax_q, ax_inf_q) in enumerate(
        zip(axs_Q.flatten(), axs_inf_Q.flatten())
    ):
        q_color_mappables = _draw_curve_set(
            ax_q, method_curves, "group_I", plot_start,
            group_index=group_index,
        )
        inf_q_color_mappables = _draw_curve_set(
            ax_inf_q, method_curves, "group_s2e", plot_start,
            group_index=group_index,
        )
        title = f"{group_basis}-Q{group_index}(pop weighted)"
        ax_q.set_title(title)
        ax_inf_q.set_title(title)
        _format_curve_axis(ax_q)
        _format_curve_axis(ax_inf_q)
        if group_index % 2 == 0:
            ax_q.set_ylabel("Exist I")
            ax_inf_q.set_ylabel("daily S2E")

    _add_curve_key(fig_Q, axs_Q, q_color_mappables)
    _add_curve_key(fig_inf_Q, axs_inf_Q, inf_q_color_mappables)
    for extension in ("jpg", "svg"):
        fig_Q.savefig(f"{output_stem}_curves/curve_Q.{extension}")
        fig_inf_Q.savefig(f"{output_stem}_curves/curve_inf_Q.{extension}")
    plt.close(fig_Q)
    plt.close(fig_inf_Q)

    print(
        f"Wrote {output_stem}_curves/curve.svg, "
        f"{output_stem}_curves/curve_Q.svg, and "
        f"{output_stem}_curves/curve_inf_Q.svg"
    )

def draw(points, multiple_points, output_stem, peak_after, baseline,
         figure_style=""):
    is_fig4h = figure_style == "fig4h"
    configure_plot_style(is_fig4h)
    figsize = (2.6, 2.0) if is_fig4h else (4.4, 3.2)
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    for (process_method, method), group in points.groupby(
        ["process_method", "method"], sort=True
    ):
        group = group.sort_values("remaining_flow_pct")
        style = method_style(process_method, method)
        ax.plot(
            group.remaining_flow_pct,
            group.prevalence_peak,
            color=style['color'], linestyle=style['linestyle'],
            marker=style['marker'], linewidth=style['linewidth'], markersize=4,
            label=f"{process_method}: {method}",
        )
        multiple_group = multiple_points[
            (multiple_points.process_method == process_method)
            & (multiple_points.method == method)
        ]
        if not multiple_group.empty:
            boxwidth = 0.8 if multiple_group.remaining_flow_pct.nunique() > 1 else 5
            sns.violinplot(
                data=multiple_group,
                x="remaining_flow_pct",
                y="prevalence_peak",
                color=style['color'],
                width=boxwidth,
                linewidth=0.5,
                zorder=4,
                native_scale=True,
                inner="box",
                inner_kws={
                    'zorder': 5, 'box_width': 2, 'linewidth': 0.5,
                    'whis_width': 0.5,
                },
                ax=ax,
            )
    ax.set_xlabel("Remaining flow proportion (%)")
    ax.set_ylabel("Prevalence Peak")
    _format_curve_axis(ax)
    ax.set_xlim(-10, 110) if is_fig4h else ax.set_xlim(-5, 105)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    if peak_after:
        ax.set_ylim(*(FIG4H_PEAK_AFTER_YLIM if is_fig4h else PEAK_AFTER_YLIM))
    ax.spines['bottom'].set_bounds(0, 100)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if peak_after:
        isolation_peak = baseline["isolation_peak"]
        epp_max = baseline["epp_max"]
        half_epp_peak = isolation_peak + 0.5 * epp_max
        unrestricted_peak = baseline["unrestricted_peak"]

        epp_axis = ax.secondary_yaxis(
            "right",
            functions=(
                lambda peak: peak - isolation_peak,
                lambda epp: epp + isolation_peak,
            ),
        )
        _format_curve_axis(epp_axis)
        epp_axis.set_ylabel("Extra prevalence peak")
        # epp_axis.set_yticks(
        #     [0, 0.5 * epp_max, epp_max],
        #     labels=["0", r"50% EPP$_{max}$", r"EPP$_{max}$"],
        # )
        guide_config = preset_guide_configuration(figure_style)
        if 50 in guide_config["epp_levels"]:
            ax.axhline(
                half_epp_peak,
                color="0.7",
                linestyle="--",
                linewidth=0.8,
                zorder=1,
            )
        if 0 in guide_config["epp_levels"]:
            ax.axhline(
                isolation_peak,
                color="0.7",
                linestyle="--",
                linewidth=0.8,
                zorder=1,
            )
        # ax.scatter(
        #     [100], [unrestricted_peak],
        #     facecolor="white", edgecolor="0.2", linewidth=0.7,
        #     s=16, zorder=6,
        # )
        if 100 in guide_config["epp_levels"]:
            ax.hlines(
                unrestricted_peak,
                xmin=100,
                xmax=ax.get_xlim()[1],
                colors="0.7",
                linestyles="--",
                linewidth=0.8,
                zorder=1,
            )
    if figure_style == "fig5":
        add_fig5_flow_reduction_guides(ax, points)
    elif figure_style == "fig5_min_mobility":
        add_minimum_mobility_flow_reduction_guides(ax)
    if not is_fig4h:
        ax.legend(
            loc='center left', title='Process method: function', fontsize='small',
            title_fontsize='small', frameon=True,
        )
    fig.tight_layout()
    fig.subplots_adjust(right=0.82 if is_fig4h else 0.82)
    for extension in ("jpg", "svg"):
        fig.savefig(f"{output_stem}.{extension}", bbox_inches="tight")
    plt.close(fig)


def sanitize_output_label(label):
    return label.strip().replace("/", "_").replace(" ", "_")


def apply_figure_preset(args):
    """Resolve the Fig. 4h and Fig. 5 plotting configurations."""
    if not args.figure_preset:
        return args
    explicit_output_label = args.output_label.strip()
    if args.figure_preset == "fig4h":
        args.process_methods = "31,40,58"
        args.method = "linear"
        args.no_multiple = True
        preset_output_label = "Fig4h"
    elif args.figure_preset == "fig5":
        args.process_methods = "31,33,40,58"
        args.method = ""
        args.no_multiple = False
        preset_output_label = "Fig5"
    elif args.figure_preset == "fig5_min_mobility":
        args.process_methods = "34,36,67,70"
        args.method = ""
        args.no_multiple = False
        preset_output_label = "Fig5MinMobility0p1"
    args.output_label = preset_output_label
    if explicit_output_label:
        args.output_label += f"_{explicit_output_label}"
    return args


def build_input_contract(args):
    process_start = 125 if args.process_threshold is None else args.process_threshold
    process_start_str = "" if args.process_threshold is None else f"_{process_start}"
    suffix = (
        f"{args.period}_I{args.init_method}_{args.Rtype}_P{args.Pratio}_"
        f"a{args.alpha_ratio_name}{'_' + args.beta_density if args.beta_density else ''}_"
        f"f{args.flow_ratio_name}_{args.simmode}_v1{{}}_"
        f"ar{SHUTDOWN_NAME}{REOPEN_DATE}_{REOPEN_NAME}_TO{TEST_OVER}_"
        f"fitted500kIa0p9initPF120{{}}"
    )
    if args.process_inf:
        suffix += '_inf'
    subdir = f"us_{args.period}_{str(args.beta_density)}"
    subdir += "_" + "_".join(suffix.split('_')[2:7])
    suffix_template = suffix.format(
        "_prc{process_method}", process_start_str
    ) + "_{method}_{run_id}"
    return process_start, subdir, suffix_template


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha_ratio_name", required=True)
    parser.add_argument("--flow_ratio_name", required=True)
    parser.add_argument("--period", required=True)
    parser.add_argument("--init_method", default="airport50k")
    parser.add_argument("--beta_density", default="cfg")
    parser.add_argument("--process_threshold", type=int, default=None)
    parser.add_argument("--simmode", default="aftershutdown")
    parser.add_argument(
        "--process_method", "--process_methods", dest="process_methods",
        default="34,67,70", help="comma-separated process methods",
    )
    parser.add_argument("--process_inf", action="store_true")
    parser.add_argument("--Rtype", default="R4m")
    parser.add_argument("--Pratio", default="0p3")
    parser.add_argument("--output_label", default="")
    parser.add_argument(
        "--figure_preset",
        choices=("fig4h", "fig5", "fig5_min_mobility"),
        default="",
        help=(
            "fig4h: small, process methods 31/40/58, linear only, no repeats; "
            "fig5: large, process methods 31/33/40/58, all configured functions "
            "with repeated-seed distributions; fig5_min_mobility: large, "
            "process methods 34/36/67/70 with a 0.1-flow mobility floor"
        ),
    )
    parser.add_argument(
        "--method", default="",
        help="optional single function name; default loads all configured functions per process method",
    )
    parser.add_argument("--run_ids", default="300-320", help="Inclusive range, e.g. 300-320")
    parser.add_argument("--input_root", default="output")
    parser.add_argument("--experiment", default="")
    parser.add_argument("--suffix_template", default="")
    parser.add_argument(
        "--peak_days", type=int, default=0,
        help="0 uses peak_after for the Fig. 5 y-axis range",
    )
    parser.add_argument(
        "--no_multiple", action="store_true",
        help="disable automatic loading of available repeated-seed outputs",
    )
    parser.add_argument(
        "--plot_curve", action="store_true",
        help=(
            "also draw prevalence curves for all counties and four "
            "population-weighted county groups"
        ),
    )
    parser.add_argument("--reference_method", type=int, default=34)
    parser.add_argument(
        "--output_stem",
        default="",
    )
    add_output_args(parser)
    args = parser.parse_args()
    args = apply_figure_preset(args)
    configure_from_args(args)

    process_start, experiment, suffix_template = build_input_contract(args)
    if args.experiment:
        experiment = args.experiment
    if args.suffix_template:
        suffix_template = args.suffix_template
    peak_days = args.peak_days or None

    run_start, run_stop = (int(value) for value in args.run_ids.split("-", maxsplit=1))
    records = []
    multiple_records = []
    for process_method in parse_int_list(args.process_methods):
        methods = [args.method] if args.method else list(process_number_to_fns(process_method))
        for method in methods:
            for run_id in range(run_start, run_stop + 1):
                point, prevalence = load_point(
                    args.input_root, experiment, method, suffix_template,
                    process_method, run_id, process_start, peak_days,
                )
                records.append(point)
            if not args.no_multiple:
                multiple_records.extend(load_multiple_points(
                    args.input_root, experiment, method, suffix_template,
                    process_method, process_start, peak_days,
                ))

    points = pd.DataFrame(records)
    multiple_points = pd.DataFrame(multiple_records, columns=[
        "process_method", "method", "curve_index", "a_value", "rep",
        "flow_over_population", "prevalence_peak", "multiple_path",
    ])
    points = replace_curve_points_with_multiple_means(points, multiple_points)
    points, multiple_points, baseline = add_relative_flow_and_epp(
        points, multiple_points
    )
    comparisons = compare_methods(points, args.reference_method)
    output_label = sanitize_output_label(args.output_label)
    process_method_tag = "-".join(
        str(process_method) for process_method in parse_int_list(args.process_methods)
    )
    output_stem = Path(args.output_stem) if args.output_stem else Path(
        f"graphs/graph_start_end/{experiment}/"
        f"multiple_process_prc{process_method_tag}_peak_after"
        f"{'_' + output_label if output_label else ''}"
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    points.drop(columns="prevalence_path").to_csv(f"{output_stem}_points.csv", index=False)
    multiple_points.drop(columns="multiple_path").to_csv(
        f"{output_stem}_multiple_points.csv", index=False
    )
    comparisons.to_csv(f"{output_stem}_comparison.csv", index=False)
    draw(
        points, multiple_points, output_stem,
        peak_after=peak_days is None,
        baseline=baseline,
        figure_style=args.figure_preset,
    )
    if args.plot_curve:
        draw_prevalence_curves(
            points=points,
            process_start=process_start,
            input_root=args.input_root,
            experiment=experiment,
            suffix_template=suffix_template,
            period=args.period,
            beta_density=args.beta_density,
            output_stem=output_stem,
        )

    print(f"Wrote {output_stem}.jpg and {output_stem}.svg")
    if not comparisons.empty:
        for process_method, group in comparisons.groupby("process_method"):
            print(
                f"{args.reference_method} vs {process_method}: "
                f"identical arrays {int(group.prevalence_array_identical.sum())}/{len(group)}, "
                f"max flow diff={group.flow_abs_diff.max():.6g}, "
                f"max peak diff={group.peak_abs_diff.max():.6g}, "
                f"max relative peak diff={group.peak_relative_diff.max():.6g}"
            )


if __name__ == "__main__":
    main()
