"""Replot the bundled toy-model trajectories from stored simulation outputs."""

import argparse
from pathlib import Path
import pickle
import sys

import numpy as np

sys.path.append("./")
from ODtoy.utils.utils import PlotSelectedMultiScalar, init_towns


DEFAULT_INPUT_ROOT = Path("toyoutput/scalars/PCFoverFlow_log_ED-1_RD200")
DEFAULT_DIRECTIONS = (
    "direct",
    "singleFull",
    "singleCBD",
    "singleIndustry",
)


def load_stored_arrays(result_dir):
    """Load and validate the two stored trajectory bundles for one network."""
    result_path = result_dir / "result_frame_same_para.pkl"
    incidence_path = result_dir / "daily_new_same_para.pkl"
    with result_path.open("rb") as handle:
        prevalence, selected = pickle.load(handle)
    with incidence_path.open("rb") as handle:
        incidence, incidence_selected = pickle.load(handle)

    prevalence = np.asarray(prevalence)
    incidence = np.asarray(incidence)
    if prevalence.ndim != 3 or incidence.ndim != 3:
        raise ValueError(
            "Expected repeat x time x node arrays in the bundled toy outputs"
        )
    if prevalence.shape != incidence.shape:
        raise ValueError(
            f"Toy output shapes differ: {prevalence.shape} != {incidence.shape}"
        )
    if selected != incidence_selected:
        raise ValueError("Toy output bundles use different selected-node metadata")
    if prevalence.shape[2] != len(selected):
        raise ValueError(
            "Stored node dimension does not match the selected-node metadata"
        )
    return prevalence, incidence, selected


def plot_direction(input_root, output_root, direction, output_format):
    """Render the stored active-case and incidence summaries for one network."""
    input_dir = input_root / direction / "4"
    output_dir = output_root / direction / "4"
    output_dir.mkdir(parents=True, exist_ok=True)
    prevalence, incidence, selected = load_stored_arrays(input_dir)

    _, _, populations, _, _ = init_towns(
        node_count=4,
        node_population=1_000_000,
        ODtype="PCFoverFlow",
        direction=direction,
    )
    selected_axis_first = prevalence.transpose(0, 2, 1)
    incidence_axis_first = incidence.transpose(0, 2, 1)
    population_scale = populations[np.newaxis, :, np.newaxis]
    reopen_marker = {"reopen:200": 200}

    common = {
        "selectedtowns": selected,
        "logscale": True,
        "extrax": reopen_marker,
        "legend": False,
        "linewidth": 2,
        "agg": 1,
    }
    PlotSelectedMultiScalar(
        multiTownFrames=selected_axis_first,
        title="Active cases",
        filename=output_dir / f"I_count_multi.{output_format}",
        ylim=(1, None),
        **common,
    )
    PlotSelectedMultiScalar(
        multiTownFrames=selected_axis_first / population_scale,
        title="Prevalence",
        filename=output_dir / f"I_ratio_multi.{output_format}",
        ylim=(1e-4, None),
        **common,
    )
    PlotSelectedMultiScalar(
        multiTownFrames=incidence_axis_first,
        title="# of newly infected cases",
        filename=output_dir / f"daily_new_multi.{output_format}",
        ylim=(1, None),
        **common,
    )
    PlotSelectedMultiScalar(
        multiTownFrames=incidence_axis_first / population_scale * 1e4,
        title="# of newly infected cases per 10k population",
        filename=output_dir / f"daily_new_ratio_multi.{output_format}",
        ylim=(1e-2, None),
        **common,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="defaults to --input-root, reproducing figures beside the PKLs",
    )
    parser.add_argument(
        "--directions",
        nargs="+",
        choices=DEFAULT_DIRECTIONS,
        default=list(DEFAULT_DIRECTIONS),
    )
    parser.add_argument("--format", choices=("svg", "png"), default="svg")
    args = parser.parse_args()

    output_root = args.input_root if args.output_root is None else args.output_root
    for direction in args.directions:
        plot_direction(args.input_root, output_root, direction, args.format)
        print(f"Wrote toy plots for {direction} under {output_root}")


if __name__ == "__main__":
    main()
