"""CLI parsing helpers for the US reopening experiment."""

from __future__ import annotations

import argparse

from us_reopen.config import (
    BETA_DENSITY_CHOICES,
    CONFIRMED_PROCESS_METHODS,
    DEFAULT_WEEKCOUNT,
    INIT_METHOD_CHOICES,
    PERIOD_CHOICES,
    RECOVER_DATE,
    RECOVER_NAME,
    S2E_VERSION,
    SHUTDOWN_NAME,
    TEST_OVER,
)


def build_parser(default_weekcount: int = DEFAULT_WEEKCOUNT) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alpha_ratio_name",
        type=str,
        required=True,
        help="float, use p to replace . in a number",
    )
    parser.add_argument(
        "--flow_ratio_name",
        type=str,
        required=True,
        help="float, use p to replace . in a number",
    )
    parser.add_argument(
        "--period",
        type=str,
        required=True,
        help="commuting, Omicron preCovid, Delta or Alpha ",
        choices=PERIOD_CHOICES,
    )
    parser.add_argument(
        "--init_method",
        type=str,
        default="airport50k",
        help="pop10t10, airport50k,cfg,everywhere",
        choices=INIT_METHOD_CHOICES,
    )
    parser.add_argument(
        "--beta_density",
        type=str,
        default="fit",
        help="cfg or False",
        choices=BETA_DENSITY_CHOICES,
    )
    parser.add_argument("--process_threshold", type=int, default=125, help="process_threshold")
    parser.add_argument(
        "--process_method",
        type=int,
        default=0,
        help="process_method, 0 for nothing",
    )
    parser.add_argument("--process_inf", action="store_true", help="process_inf store True")
    parser.add_argument("--Rtype", type=str, default="R4m", help="Rtype")
    parser.add_argument("--Pratio", type=str, default="0p3", help="Pratio - 0p3 - 0.3")
    parser.add_argument(
        "--mode",
        type=str,
        default="curve",
        choices=["single", "curve", "multiple"],
        help="single: one a=0 run; curve: sweep a values; multiple: repeat fixed a values",
    )
    parser.add_argument(
        "--multiple_idx",
        type=int,
        nargs="*",
        help="fixed-a indices for multiple mode on the --multiple_grid_points grid",
    )
    parser.add_argument(
        "--a_values",
        type=str,
        default="",
        help="comma-separated explicit a values for curve mode, e.g. 0,0p5,1; multiple mode uses --multiple_idx",
    )
    parser.add_argument("--weekcount", type=int, default=default_weekcount, help="number of simulation weeks")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="numpy random seed (default: 0)",
    )
    parser.add_argument("--curve_points", type=int, default=21, help="number of curve-mode a values")
    parser.add_argument("--poolsize", type=int, default=21, help="curve-mode multiprocessing pool size")
    parser.add_argument(
        "--repeat_count",
        type=int,
        default=100,
        help="number of repeated stochastic runs for each fixed a in multiple mode",
    )
    parser.add_argument(
        "--multiple_poolsize",
        type=int,
        default=50,
        help="multiple-mode multiprocessing pool size",
    )
    parser.add_argument(
        "--multiple_grid_points",
        type=int,
        default=21,
        help="number of grid points used by --multiple_idx in multiple mode",
    )
    parser.add_argument(
        "--process_fn_names",
        type=str,
        default="",
        help="comma-separated process function names to keep in curve or multiple mode",
    )
    parser.add_argument(
        "--process_fn_limit",
        type=int,
        default=0,
        help="limit curve/multiple mode to the first N process functions; 0 keeps all",
    )
    parser.add_argument(
        "--run_label",
        type=str,
        default="",
        help="optional suffix/subdir label for short or diagnostic runs",
    )
    return parser


def parse_cli_args(argv=None, default_weekcount: int = DEFAULT_WEEKCOUNT):
    parser = build_parser(default_weekcount)
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    try:
        explicit_a_values = parse_a_values(args.a_values)
    except ValueError as exc:
        parser.error(str(exc))
    if args.mode == "multiple" and explicit_a_values is not None:
        parser.error("--mode multiple uses --multiple_idx only; use grid index 300+idx naming")
    if args.mode == "multiple" and not args.multiple_idx:
        parser.error("--mode multiple needs --multiple_idx")
    return args, explicit_a_values


def parse_a_values(a_values: str):
    if not a_values:
        return None
    values = []
    for raw in a_values.split(","):
        token = raw.strip()
        if not token:
            continue
        values.append(float(token.replace("p", ".")))
    if not values:
        raise ValueError("--a_values did not contain any numeric values")
    return values


def safe_a_label(a_value: float) -> str:
    return "a" + f"{a_value:g}".replace("-", "m").replace(".", "p")


def sanitize_run_label(run_label: str) -> str:
    return run_label.strip().replace("/", "_").replace(" ", "_")


def build_suffix(args, beta_density: str | bool, init_method: str, run_label: str) -> str:
    suffix = (
        f"{args.period}_I{init_method}_{args.Rtype}_P{args.Pratio}_a{args.alpha_ratio_name}"
        f"{'_' + beta_density if beta_density else ''}_f{args.flow_ratio_name}_aftershutdown_{S2E_VERSION}"
        f"_prc{args.process_method}_ar{SHUTDOWN_NAME}{RECOVER_DATE}_{RECOVER_NAME}_TO{TEST_OVER}"
        f"_fitted500kIa0p9initPF120_{args.process_threshold}"
    )
    if args.process_inf:
        suffix += "_inf"
    if run_label:
        suffix += f"_{run_label}"
    return suffix


def build_subdir(args, beta_density: str | bool, suffix: str, run_label: str) -> str:
    subdir = f"us_{args.period}_{str(beta_density)}"
    subdir += "_" + "_".join(suffix.split("_")[2:7])
    if run_label:
        subdir += f"_{run_label}"
    return subdir


def _validate_args(parser: argparse.ArgumentParser, args) -> None:
    if args.weekcount < 1:
        parser.error("--weekcount must be at least 1")
    if args.seed is not None and (args.seed < 0 or args.seed >= 2**32):
        parser.error("--seed must be between 0 and 2**32 - 1")
    if args.curve_points < 1:
        parser.error("--curve_points must be at least 1")
    if args.poolsize < 1:
        parser.error("--poolsize must be at least 1")
    if args.repeat_count < 1:
        parser.error("--repeat_count must be at least 1")
    if args.multiple_poolsize < 1:
        parser.error("--multiple_poolsize must be at least 1")
    if args.multiple_grid_points < 1:
        parser.error("--multiple_grid_points must be at least 1")
    if args.process_fn_limit < 0:
        parser.error("--process_fn_limit must be 0 or greater")
    if args.process_method == 0 and args.mode != "single":
        parser.error("--process_method 0 is only supported with --mode single")
    if args.process_method not in CONFIRMED_PROCESS_METHODS and args.process_method != 0:
        parser.error(
            "--process_method must be one of "
            + ",".join(str(code) for code in sorted(CONFIRMED_PROCESS_METHODS))
            + " for the paper experiment configuration"
        )
