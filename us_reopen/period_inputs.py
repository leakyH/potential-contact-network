"""Load graph and OD inputs for a US experiment period."""

from __future__ import annotations

import pickle as pkl

import networkx as nx
import numpy as np

from us_reopen.us_data import getInformation, buildUSNetwork


def load_period_inputs(period: str) -> tuple[tuple, np.ndarray]:
    if period in ["preCovid", "Alpha", "Delta", "AlphaRestrict", "preCovidlikeAlphaRestrict"]:
        graph_file = f"graphs/graph1/average_graph_full_daily_{period}_workday.pkl"
        with open(graph_file, "rb") as f:
            graph = pkl.load(f)
        csainfo = getInformation(graph.nodes(), ageMethod="death")
        sample_od = np.array(
            nx.linalg.graphmatrix.adjacency_matrix(
                graph, nodelist=csainfo[0].keys()
            ).todense()
        )
        return csainfo, sample_od
    if period in ["Omicron", "Omicron_lm"]:
        graph_file = f"graphs/graph1/average_graph_full_{period}.pkl"
        with open(graph_file, "rb") as f:
            graph = pkl.load(f)
        csainfo = getInformation(graph.nodes(), ageMethod="death")
        sample_od = np.array(
            nx.linalg.graphmatrix.adjacency_matrix(
                graph, nodelist=csainfo[0].keys()
            ).todense()
        )
        return csainfo, sample_od
    if period == "commuting":
        graph = buildUSNetwork("full", basedir="./ext-data/us-counties/", recompute=False)
        csainfo = getInformation(graph.nodes(), ageMethod="death")
        graph = graph.subgraph(list(csainfo[0].keys()))
        sample_od = np.array(
            nx.linalg.graphmatrix.adjacency_matrix(
                graph, nodelist=csainfo[0].keys()
            ).todense()
        )
        return csainfo, sample_od
    raise ValueError(f"unsupported period: {period}")


def attach_period_inputs(kwargs: dict) -> tuple:
    csainfo, sample_od = load_period_inputs(kwargs["period"])
    kwargs["sampleOD"] = sample_od
    kwargs["csainfo"] = csainfo
    return csainfo
