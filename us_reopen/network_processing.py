"""Mobility intervention algorithms for the US reopening model."""

from __future__ import annotations

from functools import partial
import multiprocessing as mp
import os
import time

import networkx as nx
from networkx.algorithms import community
import numpy as np


_torch_import_error = None
try:
    import torch
except (ImportError, OSError) as exc:
    torch = None
    _torch_import_error = exc


criterias = [
    "pchDextS",
    "flow",
    "pop",
    "beta",
    "rt",
    "linear",
    "random",
]
criterias_name = [
    "PCH",
    "flow",
    "pop",
    "R0",
    "Rt",
    "linear",
    "random",
]


def _cuda_enabled() -> bool:
    if os.environ.get("US_REOPEN_USE_CUDA") != "1":
        return False
    if torch is None:
        raise RuntimeError(
            "US_REOPEN_USE_CUDA=1 was requested, but PyTorch could not be imported"
        ) from _torch_import_error
    try:
        cuda_available = torch.cuda.is_available()
    except (AttributeError, RuntimeError) as exc:
        raise RuntimeError(
            "US_REOPEN_USE_CUDA=1 was requested, but PyTorch CUDA support could not be checked"
        ) from exc
    if not cuda_available:
        raise RuntimeError(
            "US_REOPEN_USE_CUDA=1 was requested, but PyTorch reports that CUDA is unavailable"
        )
    return True


def _parse_cuda_devices(cuda_count: int) -> list[int]:
    raw_devices = os.environ.get("US_REOPEN_CUDA_DEVICES", "").strip()
    if raw_devices:
        devices = [int(item.strip()) for item in raw_devices.split(",") if item.strip()]
    else:
        devices = [idx for idx in [1, 2, 3] if idx < cuda_count] or list(range(cuda_count))
    invalid = [idx for idx in devices if idx < 0 or idx >= cuda_count]
    if invalid:
        raise ValueError(
            f"US_REOPEN_CUDA_DEVICES contains invalid logical CUDA device(s) {invalid}; "
            f"visible device count is {cuda_count}"
        )
    return devices


if _cuda_enabled():
    _cuda_count = torch.cuda.device_count()
    available_device = _parse_cuda_devices(_cuda_count)
    useGPU = True
else:
    available_device = [0]
    useGPU = False


def _default_device() -> str:
    if not useGPU:
        return "cpu"
    if os.environ.get("US_REOPEN_CUDA_DEVICE_POLICY") == "round_robin":
        identity = mp.current_process()._identity
        if identity:
            device_idx = (identity[0] - 1) % len(available_device)
        else:
            device_idx = os.getpid() % len(available_device)
        return f"cuda:{available_device[device_idx]}"
    return f"cuda:{np.random.choice(available_device)}"


def _to_tensor(value, local_device: str):
    if value is None:
        return None
    return torch.as_tensor(value, device=local_device)


def _to_numpy(value):
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _linear_transform(mat_from, mat_to, a):
    if mat_to is None:
        mat_to = np.eye(*mat_from.shape)
    return (1 - a) * mat_from + a * mat_to


def linear_transform(mat_from, mat_to=None, a=None, **kwargs):
    if a is None:
        a = np.linspace(0, 1, 20)[:, np.newaxis, np.newaxis]
    if mat_to is None:
        mat_to = np.eye(*mat_from.shape)
    if hasattr(a, "__iter__"):
        return [_linear_transform(mat_from, mat_to, item) for item in a]
    return _linear_transform(mat_from, mat_to, a), None


def _connection_matrix_cpu(adjmat: np.ndarray, pop=None, beta=None):
    adjmat_cp = np.asarray(adjmat).copy()
    if pop is None:
        pop = adjmat_cp.sum(axis=1)
    pop = np.asarray(pop)
    adjmat_cp *= 1 - np.eye(*adjmat_cp.shape)
    now_osum = adjmat_cp.sum(axis=1)
    od_and_left = np.diag(pop - now_osum) + adjmat_cp
    pop_work = od_and_left.sum(axis=0)
    if beta is None:
        od_corr = od_and_left @ np.diag(1 / pop_work) @ od_and_left.T
    else:
        od_corr = od_and_left @ np.diag(beta / pop_work) @ od_and_left.T
    return adjmat, od_corr


def _connection_matrix_gpu(adjmat, pop=None, beta=None):
    local_device = adjmat.device
    adjmat_cp = adjmat.to(local_device)
    if beta is not None:
        beta = beta.to(local_device)
    if pop is not None:
        pop = pop.to(local_device)
        od_and_left = adjmat_cp / adjmat_cp.sum(dim=1, keepdim=True) * pop.reshape(-1, 1)
    else:
        od_and_left = adjmat_cp
    pop_work = od_and_left.sum(dim=0)
    if beta is None:
        od_corr = od_and_left @ torch.diag(1 / pop_work) @ od_and_left.T
    else:
        od_corr = od_and_left @ torch.diag(beta / pop_work) @ od_and_left.T
    return adjmat, od_corr


def _infect_at_place_cpu(adjmat, pop=None, beta=None, i=None, s=None):
    adjmat = np.asarray(adjmat)
    if i is not None:
        i = np.asarray(i)
    if s is not None:
        s = np.asarray(s)
    if pop is not None:
        pop = np.asarray(pop)
        od_and_left = adjmat / adjmat.sum(axis=1, keepdims=True) * pop.reshape(-1, 1)
    else:
        od_and_left = adjmat
    pop_work = od_and_left.sum(axis=0)
    if beta is None:
        diag = 1 / pop_work
    else:
        diag = beta / pop_work
    if i is not None and s is not None:
        return s.reshape(-1, 1) * od_and_left * ((i @ od_and_left) * diag).reshape(1, -1)
    return od_and_left * beta.reshape(1, -1)


def _infect_at_place_gpu(adjmat, pop=None, beta=None, i=None, s=None):
    local_device = adjmat.device
    adjmat_cp = adjmat.to(local_device)
    beta = beta.to(local_device)
    if pop is not None:
        pop = pop.to(local_device)
        od_and_left = adjmat_cp / adjmat_cp.sum(dim=1, keepdim=True) * pop.reshape(-1, 1)
    else:
        od_and_left = adjmat_cp
    pop_work = od_and_left.sum(dim=0)
    diag = beta / pop_work if beta is not None else 1 / pop_work
    if i is not None and s is not None:
        i = torch.as_tensor(i, device=local_device)
        s = torch.as_tensor(s, device=local_device)
        return s.reshape(-1, 1) * od_and_left * ((i @ od_and_left) * diag).reshape(1, -1)
    return od_and_left * beta.reshape(1, -1)


def _clear_gpu_cache(clear_cache: bool) -> None:
    if useGPU and clear_cache:
        torch.cuda.empty_cache()


def community_processing_on_matrix(mat_from: np.ndarray, mat_to=None, ref_name="flow", **kwargs):
    if ref_name != "flow":
        raise ValueError("community processing uses ref_name='flow'")
    connect_mat = mat_from.copy()
    if kwargs.get("i_arr") is not None:
        connect_mat = kwargs["i_arr"].reshape(-1, 1) * connect_mat
    if kwargs.get("s_arr") is not None:
        connect_mat = connect_mat * kwargs["s_arr"].reshape(1, -1)
    connect_mat = connect_mat - np.diag(np.diag(connect_mat))
    graph: nx.DiGraph = nx.from_numpy_array(connect_mat, create_using=nx.DiGraph)
    now = time.time()
    community_lists = community.louvain_communities(graph, seed=123)
    print(time.time() - now)
    newmat = mat_from.copy()
    for commset in community_lists:
        comm = list(commset)
        temp = np.zeros_like(mat_from[comm, :])
        temp[:, comm] = mat_from[comm, :][:, comm]
        newmat[comm, :] = temp
    print("community average size:", sum(map(len, community_lists)) / len(community_lists))
    newmat += np.diag(np.sum(mat_from, axis=1) - np.sum(newmat, axis=1))
    return newmat, community_lists


def _hotspot_county_transform(
    mat_from,
    mat_to=None,
    a=None,
    ref=None,
    both_direction=False,
    inverse=False,
    amount=None,
    _internal_used_filename=None,
):
    if not both_direction:
        ref = (ref,)
    if inverse:
        argsort = np.argsort(ref[0], kind="stable")
    else:
        argsort = np.argsort(ref[0], kind="stable")[::-1]
    if both_direction and len(amount.shape) == 2:
        inverse_rank = argsort[::-1]
        newmat = amount[inverse_rank, :]
        newmat = newmat[:, inverse_rank]
        cumsum_inter_flow_diag = np.diag(newmat.cumsum(axis=0).cumsum(axis=1))
        cumsum_inter_flow = cumsum_inter_flow_diag / cumsum_inter_flow_diag[-1]
        if a < 0:
            scale = -a
            ref_norm = ((ref[0] - np.min(ref[0])) / (np.max(ref[0]) - np.min(ref[0])))[inverse_rank]
            if inverse:
                block_amount = len(cumsum_inter_flow) - np.argmin((cumsum_inter_flow) ** 2 + (ref_norm * scale) ** 2)
            else:
                block_amount = len(cumsum_inter_flow) - np.argmin((cumsum_inter_flow - 1) ** 2 + (ref_norm * scale) ** 2)
        else:
            threshold_idx = len(cumsum_inter_flow[cumsum_inter_flow <= 1 - a])
            block_amount = len(cumsum_inter_flow) - threshold_idx
        idx = argsort[0:block_amount]
        mat_from_cp = mat_from.copy()
        mat_from_cp[idx, :] = mat_to[idx, :]
        mat_from_cp[:, idx] = mat_to[:, idx]
        mat_from_cp += np.diag(mat_from.sum(axis=1) - mat_from_cp.sum(axis=1))
    else:
        mat_from_cp = mat_from.copy()
        if amount is None:
            idx = argsort[0 : int(len(argsort) * a)]
        else:
            total_amt = np.sum(amount)
            if total_amt == 0:
                idx = []
            else:
                cumsum_amount = np.cumsum(amount[argsort]) / total_amt
                idx = argsort[cumsum_amount <= a]
        mat_from_cp[idx, :] = mat_to[idx, :]
        if both_direction:
            if inverse:
                argsort = np.argsort(ref[1], kind="stable")
            else:
                argsort = np.argsort(ref[1], kind="stable")[::-1]
            if amount is None:
                idx = argsort[0 : int(len(argsort) * a)]
            else:
                total_amt = np.sum(amount)
                if total_amt == 0:
                    idx = []
                else:
                    cumsum_amount = np.cumsum(amount[argsort]) / total_amt
                    idx = argsort[cumsum_amount <= a]
            mat_from_cp[:, idx] = mat_to[:, idx]
            mat_from_cp += np.diag(mat_from.sum(axis=1) - mat_from_cp.sum(axis=1))
    if _internal_used_filename is not None:
        ref_rank_dir = "graphs/graph_map/ref_rank/"
        os.makedirs(ref_rank_dir, exist_ok=True)
        np.save(os.path.join(ref_rank_dir, _internal_used_filename), np.vstack((ref[0], ref[1], argsort)).T)
    return mat_from_cp


def _setup_gpu_inputs(pcfkwargs, enable_gpu: bool):
    if not useGPU or not enable_gpu:
        return None, False, pcfkwargs.get("pop")
    local_device = pcfkwargs.get("local_device")
    clear_cache = local_device is None
    if local_device is None:
        local_device = _default_device()
    pop = _to_tensor(pcfkwargs.get("pop"), local_device)
    return local_device, clear_cache, pop


def _reference_from_name(mat_from, mat_to, ref_name, both_direction, pcfkwargs, local_device, clear_cache, pop):
    if ref_name == "flow":
        adjmat = mat_from.copy()
        if "i_arr" in pcfkwargs:
            adjmat = pcfkwargs["i_arr"].reshape(-1, 1) * adjmat
        if "s_arr" in pcfkwargs:
            adjmat = adjmat * pcfkwargs["s_arr"].reshape(1, -1)
        if both_direction:
            pop_array = _to_numpy(pop if useGPU else pcfkwargs.get("pop"))
            ref_out = (adjmat.sum(axis=1) - np.diag(adjmat)) / pop_array
            return (ref_out, ref_out)
        return adjmat.sum(axis=1) - np.diag(adjmat)
    if ref_name == "rt":
        ref = pcfkwargs["beta"](mat_from).copy() * pcfkwargs["s_arr"]
        return (ref, ref) if both_direction else ref
    if ref_name == "beta":
        # Match the paper's 31/34 county-hotspot intervention: rank counties by
        # their beta under the current OD matrix.  This is intentionally a
        # county vector (rather than the pairwise beta-difference matrix used
        # by the county hotspot-pair transform).
        ref = pcfkwargs["beta"](mat_from).copy()
        return (ref, ref) if both_direction else ref
    if ref_name == "pop":
        ref = pcfkwargs["pop"].copy()
        return (ref, ref) if both_direction else ref
    if ref_name == "prevalence":
        ref = pcfkwargs["i_arr"].copy()
        return (ref, ref) if both_direction else ref
    if ref_name == "random":
        ref = np.random.random(mat_from.shape[0])
        return (ref, ref) if both_direction else ref
    if ref_name == "pchDextS":
        ref = _reference_pch_dext_s(mat_from, mat_to, pcfkwargs, local_device, clear_cache, pop)
        return (ref, ref) if both_direction else ref
    if ref_name == "pchofPCF":
        ref = _reference_pch_of_pcf(mat_from, mat_to, pcfkwargs, local_device, clear_cache, pop)
        return (ref, ref) if both_direction else ref
    raise ValueError(f"unsupported ref_name {ref_name}")


def _reference_pch_dext_s(mat_from, mat_to, pcfkwargs, local_device, clear_cache, pop):
    beta_func = pcfkwargs["beta"]
    if useGPU and local_device is not None:
        infect_from = _infect_at_place_gpu(
            torch.as_tensor(mat_from, device=local_device),
            pop=pop,
            beta=torch.as_tensor(beta_func(mat_from), device=local_device),
            i=pcfkwargs.get("i_arr"),
            s=pcfkwargs.get("s_arr"),
        )
        infect_to = _infect_at_place_gpu(
            torch.as_tensor(mat_to, device=local_device),
            pop=pop,
            beta=torch.as_tensor(beta_func(mat_to), device=local_device),
            i=pcfkwargs.get("i_arr"),
            s=pcfkwargs.get("s_arr"),
        )
        infect_from = infect_from.cpu().numpy()
        infect_to = infect_to.cpu().numpy()
    else:
        infect_from = _infect_at_place_cpu(
            mat_from,
            pop=pcfkwargs.get("pop"),
            beta=beta_func(mat_from),
            i=pcfkwargs.get("i_arr"),
            s=pcfkwargs.get("s_arr"),
        )
        infect_to = _infect_at_place_cpu(
            mat_to,
            pop=pcfkwargs.get("pop"),
            beta=beta_func(mat_to),
            i=pcfkwargs.get("i_arr"),
            s=pcfkwargs.get("s_arr"),
        )
    con_from = infect_from.sum(axis=0) - np.diag(infect_from)
    con_to = infect_to.sum(axis=0) - np.diag(infect_to)
    ref = (con_from - con_to) / ((mat_from.sum(axis=0) - np.diag(mat_from)) - (mat_to.sum(axis=0) - np.diag(mat_to)))
    _clear_gpu_cache(clear_cache)
    return ref


def _reference_pch_of_pcf(mat_from, mat_to, pcfkwargs, local_device, clear_cache, pop):
    beta_func = pcfkwargs["beta"]
    if useGPU and local_device is not None:
        _, odcorr_from = _connection_matrix_gpu(
            torch.as_tensor(mat_from, device=local_device),
            pop=pop,
            beta=torch.as_tensor(beta_func(mat_from), device=local_device),
        )
        _, odcorr_to = _connection_matrix_gpu(
            torch.as_tensor(mat_to, device=local_device),
            pop=pop,
            beta=torch.as_tensor(beta_func(mat_to), device=local_device),
        )
        infect_from = _infect_at_place_gpu(
            odcorr_from,
            pop=pop,
            beta=torch.as_tensor(beta_func(mat_from), device=local_device),
            i=pcfkwargs.get("i_arr"),
            s=pcfkwargs.get("s_arr"),
        )
        infect_to = _infect_at_place_gpu(
            odcorr_to,
            pop=pop,
            beta=torch.as_tensor(beta_func(mat_to), device=local_device),
            i=pcfkwargs.get("i_arr"),
            s=pcfkwargs.get("s_arr"),
        )
        infect_from = infect_from.cpu().numpy()
        infect_to = infect_to.cpu().numpy()
    else:
        _, odcorr_from = _connection_matrix_cpu(mat_from, pop=pcfkwargs.get("pop"), beta=beta_func(mat_from))
        _, odcorr_to = _connection_matrix_cpu(mat_to, pop=pcfkwargs.get("pop"), beta=beta_func(mat_from))
        infect_from = _infect_at_place_cpu(
            odcorr_from,
            pop=pcfkwargs.get("pop"),
            beta=beta_func(mat_from),
            i=pcfkwargs.get("i_arr"),
            s=pcfkwargs.get("s_arr"),
        )
        infect_to = _infect_at_place_cpu(
            odcorr_to,
            pop=pcfkwargs.get("pop"),
            beta=beta_func(mat_to),
            i=pcfkwargs.get("i_arr"),
            s=pcfkwargs.get("s_arr"),
        )
    con_from = infect_from.sum(axis=0) - np.diag(infect_from)
    con_to = infect_to.sum(axis=0) - np.diag(infect_to)
    ref = (con_from - con_to) / ((mat_from.sum(axis=0) - np.diag(mat_from)) - (mat_to.sum(axis=0) - np.diag(mat_to)))
    _clear_gpu_cache(clear_cache)
    return ref


def _amount_from_name(mat_from, mat_to, amount_name, ref_name, both_direction, pcfkwargs):
    if amount_name == "outflow":
        return (mat_from.sum(axis=1) - np.diag(mat_from)) - (mat_to.sum(axis=1) - np.diag(mat_to))
    if amount_name == "pop":
        return pcfkwargs["pop"].copy()
    if amount_name == "county":
        return None
    if amount_name == "flow":
        return mat_from.sum(axis=1) - mat_to.sum(axis=1)
    if amount_name == "flowmat":
        if both_direction:
            amount = mat_from - mat_to
            amount -= np.diag(np.diag(amount))
            return amount
        if ref_name in ["pchD", "pchDCross"]:
            return mat_from.sum(axis=0) - np.diag(mat_from) - (mat_to.sum(axis=0) - np.diag(mat_to))
        return mat_from.sum(axis=1) - mat_to.sum(axis=1)
    raise ValueError(f"unsupported amount mode {amount_name}")


def hotspot_county_transform(
    mat_from,
    mat_to=None,
    a=None,
    ref_name=None,
    both_direction=False,
    amount="outflow",
    inverse=False,
    **pcfkwargs,
):
    if mat_to is None:
        mat_to = np.eye(*mat_from.shape)
    if a is None:
        a = np.linspace(0, 1, 20)
    local_device, clear_cache, pop = _setup_gpu_inputs(pcfkwargs, enable_gpu=ref_name == "pchDextS")
    ref = _reference_from_name(mat_from, mat_to, ref_name, both_direction, pcfkwargs, local_device, clear_cache, pop)
    amount_value = _amount_from_name(mat_from, mat_to, amount, ref_name, both_direction, pcfkwargs)
    if hasattr(a, "__iter__"):
        return [
            _hotspot_county_transform(mat_from, mat_to, item, ref, both_direction, inverse, amount_value)
            for item in a
        ]
    internal_filename = pcfkwargs.get("_internal_used_filename")
    print(internal_filename, pcfkwargs.get("_internal_used_idx"))
    if internal_filename and pcfkwargs.get("_internal_used_idx"):
        internal_filename += f"_{pcfkwargs['_internal_used_idx']}"
    print("_internal_used_filename:", internal_filename)
    mat_this_round = _hotspot_county_transform(
        mat_from,
        mat_to,
        a,
        ref,
        both_direction,
        inverse,
        amount_value,
        _internal_used_filename=internal_filename,
    )
    return mat_this_round, None


def process_number_to_fns(process_number):
    if process_number in [31, 34]:
        return {
            # Fig5:
            "linear": linear_transform,
            "pop_both_inverse": partial(hotspot_county_transform, ref_name="pop", both_direction=True, inverse=True),
            "county_outflow_both": partial(hotspot_county_transform, both_direction=True, ref_name="flow"),
            
            # # Full Graph
            # "beta_both": partial(hotspot_county_transform, ref_name="beta", both_direction=True),
            # "beta_both_inverse": partial(hotspot_county_transform, ref_name="beta", both_direction=True, inverse=True),
            # "pop_both": partial(hotspot_county_transform, ref_name="pop", both_direction=True, inverse=False),
            # "county_outflow_both_inverse": partial(hotspot_county_transform, both_direction=True, ref_name="flow", inverse=True),
        }
    if process_number in [33, 36]:
        return {
            # Fig5:
            "realtime_random_both": partial(hotspot_county_transform, ref_name="random", both_direction=True),
            "realtime_county_pchDextS_both": partial(hotspot_county_transform, ref_name="pchDextS", both_direction=True),
            "realtime_county_rt_both": partial(hotspot_county_transform, ref_name="rt", both_direction=True),
            
            # Full Graph
            # "realtime_county_pchDextS_both_inverse": partial(hotspot_county_transform, ref_name="pchDextS", both_direction=True, inverse=True),
            # "realtime_county_rt_both_inverse": partial(hotspot_county_transform, ref_name="rt", both_direction=True, inverse=True),
        }
    if process_number in [40, 58, 67, 70]:
        return {"linear": linear_transform}
    raise ValueError(
        f"unsupported paper process_number {process_number}; "
        "expected one of 31, 33, 34, 36, 40, 58, 67, 70"
    )


def analysis_process_method(method):
    def get_criteria(method):
        for c, n in zip(criterias, criterias_name):
            if c in method:
                return n
        return False

    return {
        "criteria": get_criteria(method),
        "inverse": "inverse" in method,
        "both_direction": "both" in method,
    }
