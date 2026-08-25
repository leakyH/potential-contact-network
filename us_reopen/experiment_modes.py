"""Experiment-mode dispatch for US reopening runs."""

from __future__ import annotations

from functools import partial
import os

import numpy as np
import pandas as pd
from us_reopen.interventions import process_number_to_fns
from us_reopen.period_inputs import attach_period_inputs
from us_reopen.randomness import derive_worker_seed


def apply_args_and_kwargs(fn, args, kwargs):
    return fn(*args, **kwargs)


def run_experiment_batch(expbatch, worker_count):
    if worker_count < 1:
        raise ValueError("worker_count must be at least 1")
    if worker_count == 1:
        return [apply_args_and_kwargs(fn, args, kwargs) for fn, args, kwargs in expbatch]
    import multiprocessing as mp
    context = mp.get_context("spawn") if os.environ.get("US_REOPEN_USE_CUDA") == "1" else mp.get_context()
    with context.Pool(worker_count) as pool:
        return pool.starmap(apply_args_and_kwargs, expbatch)


def select_process_functions(process_number, process_fn_names="", process_fn_limit=0):
    fns = process_number_to_fns(process_number)
    if process_fn_names:
        requested = [name.strip() for name in process_fn_names.split(",") if name.strip()]
        missing = [name for name in requested if name not in fns]
        if missing:
            raise ValueError(f"unknown process function names for method {process_number}: {missing}")
        fns = {name: fns[name] for name in requested}
    elif process_fn_limit:
        fns = dict(list(fns.items())[:process_fn_limit])
    if not fns:
        raise ValueError("no process functions selected")
    return fns


def run_single_experiment(simulation_fn, *args, **kwargs):
    attach_period_inputs(kwargs)
    astr = "0"
    process_method = kwargs["process_inequal"][1]
    if process_method == 0:
        kwargs["suffix"] = kwargs["suffix"] + "_no_intervention_testSingle" + astr
        kwargs.pop("mat_process_func", None)
    else:
        fns = process_number_to_fns(process_method)
        fnname, fnfunc = next(iter(fns.items()))
        a = float(astr.replace("p", "."))
        kwargs["suffix"] = kwargs["suffix"] + "_" + fnname + "_testSingle" + astr
        kwargs["mat_process_func"] = partial(
            fnfunc,
            a=a,
            _internal_used_filename=kwargs["suffix"],
        )
    result_frame = simulation_fn(*args, **kwargs)
    os.makedirs("tmp", exist_ok=True)
    suffix = kwargs.get("suffix", None)
    if suffix is None:
        np.save("output/pkls/result_frame_us_test.npy", result_frame[0])
        np.save("output/pkls/S2E_ratio.npy", result_frame[1])
        if len(result_frame) >= 9:
            np.save("output/pkls/testedI_count.npy", result_frame[8])
    else:
        np.save(f"output/pkls/result_frame_test_{suffix}.npy", result_frame[0])
        np.save(f"output/pkls/S2E_ratio_{suffix}.npy", result_frame[1])
        if len(result_frame) >= 9:
            np.savetxt(f"output/csvs/total_testedI_count_{suffix}.txt", result_frame[8])
            pd.DataFrame(data=result_frame[9]).to_csv(
                f"output/csvs/select_testedI_count_{suffix}.csv",
                index=False,
            )


def run_curve_experiment(simulation_fn, *args, **kwargs):
    poolsize = kwargs.get("poolsize", 21)
    base_seed = kwargs.get("seed", 0)
    a_values = kwargs.get("a_values", None)
    if a_values is None:
        curve_points = kwargs.get("curve_points", 21)
        a_values = np.linspace(0, 1, curve_points)
    else:
        curve_points = len(a_values)
    if poolsize < 1:
        raise ValueError("poolsize must be at least 1")
    if curve_points < 1:
        raise ValueError("curve mode must have at least one a value")
    subdir = kwargs["subdir"]
    os.makedirs(f"output/pkls/{subdir}", exist_ok=True)
    os.makedirs(f"output/scalars/{subdir}", exist_ok=True)
    os.makedirs(f"output/csvs/{subdir}", exist_ok=True)
    attach_period_inputs(kwargs)
    fns = select_process_functions(
        kwargs["process_inequal"][1],
        kwargs.get("process_fn_names", ""),
        kwargs.get("process_fn_limit", 0),
    )
    exp_batch_offset = 300
    if base_seed is None:
        print("[curve] seed: unseeded; each worker initializes numpy from OS entropy")
    else:
        print(
            "[curve] seed: worker_seed = "
            f"{int(base_seed)} + curve_index; 40/67 index 0 reuse the last full-curve seed"
        )
    for fnname, fnfunc in fns.items():
        os.makedirs(f"output/csvs/{subdir}/{fnname}", exist_ok=True)
        os.makedirs(f"output/scalars/{subdir}/{fnname}", exist_ok=True)
        os.makedirs(f"output/pkls/{subdir}/{fnname}", exist_ok=True)
        expbatch = []
        for curve_index, a_mat in enumerate(a_values):
            i = curve_index + exp_batch_offset
            kwcp = _copy_without_runtime_controls(kwargs)
            kwcp["suffix"] = kwargs["suffix"] + "_" + fnname + "_" + str(i)
            kwcp["mat_process_func"] = partial(fnfunc, a=a_mat)
            kwcp["subdir_name"] = f"{subdir}/{fnname}"
            if (kwargs['process_inequal'][1] in [40,67]) and (curve_index == 0):
                # 40/67 start from the same mobility matrix as the 58/70
                # endpoint, so reuse the last full-curve seed to join them.
                last_curve_index = kwargs.get("curve_points", 21) - 1
                kwcp["seed"] = derive_worker_seed(base_seed, last_curve_index)
            else:
                kwcp["seed"] = derive_worker_seed(base_seed, curve_index)
            # kwcp["seed"] = base_seed
            expbatch.append((simulation_fn, args, kwcp))
        list_of_frames = run_experiment_batch(expbatch, min(poolsize, curve_points))
        for i, result_frame in enumerate(list_of_frames):
            i += exp_batch_offset
            np.save(
                f"output/pkls/{subdir}/{fnname}/I_exist_count_{kwargs['suffix']}_{fnname}_{str(i)}.npy",
                result_frame[0],
            )
        for i, result_frame in enumerate(list_of_frames):
            i += exp_batch_offset
            np.save(
                f"output/pkls/{subdir}/{fnname}/S2E_ratio_{kwargs['suffix']}_{fnname}_{str(i)}.npy",
                result_frame[1],
            )


def run_multiple_experiment(
    simulation_fn,
    expi=None,
    count=20,
    a_value=None,
    a_label=None,
    *args,
    **kwargs,
):
    if count < 1:
        raise ValueError("count must be at least 1")
    multiple_grid_points = kwargs.get("multiple_grid_points", 21)
    if multiple_grid_points < 1:
        raise ValueError("multiple_grid_points must be at least 1")
    a_grid = np.linspace(0, 1, multiple_grid_points)
    if a_value is None:
        if expi is None:
            raise ValueError("multiple mode needs expi")
    else:
        if a_label is not None:
            raise ValueError("multiple mode uses grid index output ids; provide multiple_idx")
        expi = int(round(float(a_value) * (multiple_grid_points - 1)))
        if expi < 0 or expi >= len(a_grid) or not np.isclose(float(a_value), a_grid[expi]):
            raise ValueError("multiple a_value must lie on the multiple grid; use multiple_idx")
    if expi < 0 or expi >= len(a_grid):
        raise ValueError(f"multiple_idx {expi} outside 0..{len(a_grid) - 1}")
    a_mat = a_grid[expi]
    a_output_id = expi + 300
    poolsize = kwargs.get("multiple_poolsize", kwargs.get("poolsize", min(50, count)))
    if poolsize < 1:
        raise ValueError("multiple_poolsize must be at least 1")
    subdir = kwargs["subdir"]
    print(subdir)
    os.makedirs(f"output/pkls/{subdir}", exist_ok=True)
    os.makedirs(f"output/scalars/{subdir}", exist_ok=True)
    os.makedirs(f"output/csvs/{subdir}", exist_ok=True)
    attach_period_inputs(kwargs)
    fns = select_process_functions(
        kwargs["process_inequal"][1],
        kwargs.get("process_fn_names", ""),
        kwargs.get("process_fn_limit", 0),
    )
    for fnname, fnfunc in fns.items():
        os.makedirs(f"output/csvs/{subdir}/{fnname}", exist_ok=True)
        os.makedirs(f"output/scalars/{subdir}/{fnname}", exist_ok=True)
        os.makedirs(f"output/pkls/{subdir}/{fnname}", exist_ok=True)
        expbatch = []
        for rep in range(count):
            kwcp = _copy_without_runtime_controls(kwargs)
            kwcp.pop("multiple_poolsize", None)
            kwcp.pop("multiple_grid_points", None)
            kwcp["suffix"] = kwargs["suffix"] + "_" + fnname + "_" + str(a_output_id) + f"rep{rep}"
            kwcp["mat_process_func"] = partial(fnfunc, a=a_mat)
            kwcp["subdir_name"] = f"{subdir}/{fnname}"
            kwcp["seed"] = derive_worker_seed(kwargs.get("seed", 0), rep)
            expbatch.append((simulation_fn, args, kwcp))
        list_of_frames = run_experiment_batch(expbatch, min(poolsize, count))
        result_frames = np.array([item[0] for item in list_of_frames])
        np.save(
            f"output/pkls/{subdir}/{fnname}/I_exist_count_{kwargs['suffix']}_{fnname}_{str(a_output_id)}_multiple.npy",
            result_frames,
        )
        result_frames = np.array([item[1] for item in list_of_frames])
        np.save(
            f"output/pkls/{subdir}/{fnname}/S2E_ratio_{kwargs['suffix']}_{fnname}_{str(a_output_id)}_multiple.npy",
            result_frames,
        )


def _copy_without_runtime_controls(kwargs):
    kwcp = kwargs.copy()
    kwcp.pop("poolsize", None)
    kwcp.pop("curve_points", None)
    kwcp.pop("a_values", None)
    kwcp.pop("process_fn_names", None)
    kwcp.pop("process_fn_limit", None)
    return kwcp
