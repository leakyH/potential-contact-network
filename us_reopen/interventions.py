"""Mobility intervention algorithms and daily simulation scheduling."""

from __future__ import annotations

import os

import numpy as np

from us_reopen.config import (
    COMMUNITY_PROCESS_METHODS,
    FIXED_PROCESS_METHODS,
    MINIMUM_MOBILITY_PROCESS_METHODS,
    REALTIME_PROCESS_METHODS,
)
from us_reopen.mobility import linear_transform_sampleOD

from us_reopen.network_processing import (
    available_device,
    community_processing_on_matrix,
    hotspot_county_transform,
    linear_transform,
    process_number_to_fns,
)


MINIMUM_MOBILITY_RETAINED_FLOW = 0.1
MINIMUM_MOBILITY_SELECTION_TOL = 1e-3


def shutdown_county_mask(processed_od, mat_from, process_method):
    """Return counties whose off-diagonal mobility meets the shutdown rule."""
    offdiag_processed = np.sum(processed_od, axis=1) - np.diag(processed_od)
    offdiag_original = np.sum(mat_from, axis=1) - np.diag(mat_from)
    retained_ratio = np.divide(
        offdiag_processed,
        offdiag_original,
        out=np.full_like(offdiag_processed, np.inf, dtype=float),
        where=offdiag_original > 0,
    )
    if process_method in MINIMUM_MOBILITY_PROCESS_METHODS:
        return retained_ratio <= MINIMUM_MOBILITY_RETAINED_FLOW + MINIMUM_MOBILITY_SELECTION_TOL
    return retained_ratio < 1e-3


def _write_final_flow(subdir_name, suffix, processed_od):
    final_flow = np.sum(processed_od) - np.diag(processed_od).sum()
    filename = os.path.join("output/csvs/", subdir_name, f"final_flow{suffix}.csv")
    with open(filename, "w") as handle:
        handle.write(
            ",".join(
                [
                    str(final_flow),
                    str(np.sum(processed_od)),
                    str(final_flow / np.sum(processed_od)),
                ]
            )
        )


def _write_shutdown_counties(
    subdir_name,
    suffix,
    day_index,
    town_ids,
    processed_od,
    mat_from,
    process_method,
):
    fipslist = np.array(list(town_ids.keys()))
    filename = os.path.join(
        "output/csvs/", subdir_name, f"shutdown{suffix}_{day_index}.txt"
    )
    np.savetxt(
        filename,
        fipslist[shutdown_county_mask(processed_od, mat_from, process_method)],
        fmt="%s",
    )


def apply_daily_mobility_intervention(
    *,
    day_index,
    process_method,
    process_threshold_date,
    process_period,
    sample_od,
    od_and_left,
    initpopulations,
    population,
    towns,
    town_ids,
    status2id,
    infectables,
    statusbeta,
    agebeta,
    get_pcf_beta,
    mat_process_func,
    subdir_name,
    suffix,
):
    """Apply the existing mobility schedule for one simulation day."""
    if process_method in FIXED_PROCESS_METHODS:
        mat_from = None
        mat_to = None
        if process_method in [31, 34]:
            mat_from = sample_od
            if process_method == 31:
                mat_to = np.diag(initpopulations).astype(float)
            elif process_method == 34:
                mat_to = linear_transform_sampleOD(sample_od, 0.1)

        if day_index == process_threshold_date:
            if process_method in COMMUNITY_PROCESS_METHODS:
                block_diag_od, community_lists = community_processing_on_matrix(
                    sample_od,
                    pop=initpopulations,
                    beta=get_pcf_beta,
                    ref_name="flow",
                )
                if process_method == 40:
                    mat_from = block_diag_od
                    mat_to = linear_transform_sampleOD(block_diag_od, 0)
                elif process_method == 58:
                    mat_from = sample_od
                    mat_to = block_diag_od
                elif process_method == 67:
                    mat_from = block_diag_od * 0.9 + sample_od * 0.1
                    mat_to = linear_transform_sampleOD(sample_od, 0.1)
                elif process_method == 70:
                    mat_from = sample_od
                    mat_to = block_diag_od * 0.9 + sample_od * 0.1
                np.save(
                    os.path.join(
                        "output/pkls/",
                        subdir_name,
                        f"community_lists{suffix}_{day_index}.npy",
                    ),
                    community_lists,
                )
            processed_od, _ = mat_process_func(
                mat_from=mat_from,
                mat_to=mat_to,
                pop=population,
                beta=get_pcf_beta,
                amount="flowmat",
            )
            _write_final_flow(subdir_name, suffix, processed_od)
            if process_method not in COMMUNITY_PROCESS_METHODS:
                _write_shutdown_counties(
                    subdir_name,
                    suffix,
                    day_index,
                    town_ids,
                    processed_od,
                    mat_from,
                    process_method,
                )

        if day_index == 50:
            if process_method in MINIMUM_MOBILITY_PROCESS_METHODS:
                od_and_left = linear_transform_sampleOD(sample_od, 0.1)
                od_and_left = od_and_left / od_and_left.sum(axis=1, keepdims=True)
            else:
                od_and_left = np.diag(np.ones(len(initpopulations), dtype=float))
        elif day_index == process_threshold_date:
            od_and_left = processed_od / processed_od.sum(axis=1, keepdims=True)
        elif day_index == process_threshold_date + process_period:
            od_and_left = sample_od / sample_od.sum(axis=1, keepdims=True)

    elif process_method in REALTIME_PROCESS_METHODS:
        mat_from = sample_od
        if process_method == 33:
            mat_to = np.diag(initpopulations).astype(float)
        elif process_method == 36:
            mat_to = linear_transform_sampleOD(sample_od, 0.1)

        if day_index == 50:
            if process_method in MINIMUM_MOBILITY_PROCESS_METHODS:
                od_and_left = linear_transform_sampleOD(sample_od, 0.1)
                od_and_left = od_and_left / od_and_left.sum(axis=1, keepdims=True)
            else:
                od_and_left = np.diag(np.ones(len(initpopulations), dtype=float))
        elif process_threshold_date <= day_index < process_threshold_date + process_period:
            if (day_index - process_threshold_date) % 7 == 0:
                processed_od, _ = mat_process_func(
                    mat_from=mat_from,
                    mat_to=mat_to,
                    pop=population,
                    beta=get_pcf_beta,
                    s_arr=np.dot(towns[:, :, status2id["S"]], agebeta) / population,
                    i_arr=np.dot(
                        towns[:, :, infectables], statusbeta[infectables]
                    ).sum(axis=1)
                    / population,
                    local_device="cuda:{}".format(
                        available_device[os.getpid() % len(available_device)]
                    ),
                    amount="flowmat",
                    _internal_used_idx=day_index,
                )
                od_and_left = processed_od / processed_od.sum(axis=1, keepdims=True)
                _write_shutdown_counties(
                    subdir_name,
                    suffix,
                    day_index,
                    town_ids,
                    processed_od,
                    mat_from,
                    process_method,
                )
            if day_index - process_threshold_date == 0:
                _write_final_flow(subdir_name, suffix, processed_od)
        elif day_index == process_threshold_date + process_period:
            od_and_left = sample_od / sample_od.sum(axis=1, keepdims=True)

    return od_and_left

__all__ = [
    "available_device",
    "community_processing_on_matrix",
    "hotspot_county_transform",
    "linear_transform",
    "process_number_to_fns",
    "apply_daily_mobility_intervention",
    "shutdown_county_mask",
]
