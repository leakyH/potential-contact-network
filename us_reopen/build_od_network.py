"""US county commuting network builder used by the formal model."""

from __future__ import annotations

import itertools
import os
import pickle as pkl

from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


def summarize_graph(G: nx.DiGraph, name, linear=False, basedir="./"):
    print(f"{name}:average out/in degree {sum(item[1] for item in G.out_degree()) / len(G.nodes)}")
    if nx.is_weakly_connected(G):
        print("weakly connected")
    if nx.is_strongly_connected(G):
        print("strongly connected")
    if nx.is_semiconnected(G):
        print("semiconnected")
    tempG = nx.Graph(G)
    print(f"{name}:undirected degree assortativity", nx.degree_assortativity_coefficient(tempG))

    dlist = nx.degree_histogram(G)
    width = max(max((v for _, v in G.in_degree())), max((v for _, v in G.in_degree()))) + 1
    mapping = {x: x for x in range(width + 1)}
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=[10, 10], dpi=200)
    for (x, y), ax in zip(itertools.product(["in", "out"], repeat=2), [ax1, ax2, ax3, ax4]):
        print(f"{name}:{x}-{y} degree_assortativity", nx.degree_assortativity_coefficient(G, x, y))
        tempMat = nx.degree_mixing_matrix(G, x=x, y=y, mapping=mapping)[0:50, 0:50]
        ax.imshow(tempMat)
        ax.set_title(f"{x}-{y} degree_assortativity:{nx.degree_assortativity_coefficient(G, x, y)}")
    plt.savefig(os.path.join(basedir, f"{name}_degree_cor_mat.jpg"))
    plt.close()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=[10, 10], dpi=200)
    for (x, y), ax in zip(itertools.product(["in", "out"], repeat=2), [ax1, ax2, ax3, ax4]):
        print(f"{name}:{x}-{y} degree_assortativity", nx.degree_assortativity_coefficient(G, x, y))
        tempDict = nx.degree_mixing_dict(G, x=x, y=y, normalized=False)
        degrees = []
        knn = []
        for k, v in tempDict.items():
            if len(v) == 0:
                continue
            psum = sum(k * _v for k, _v in v.items())
            psum /= sum(_v for _, _v in v.items())
            degrees.append(k)
            knn.append(psum)
        ax.scatter(degrees, knn, label="k-knn")
        ax.plot([1, 1000], [1, 1000], color="k", label="y=x")
        if y == "in":
            ax.axhline(np.mean([v for _, v in G.in_degree]), label="avg. in degree")
        else:
            ax.axhline(np.mean([v for _, v in G.out_degree]), label="avg. out degree")
        ax.legend()
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{x}-{y} degree_assortativity:{nx.degree_assortativity_coefficient(G, x, y)}")
    plt.savefig(os.path.join(basedir, f"{name}_degree_cor_mat.jpg"))
    plt.close()

    if linear:
        plt.scatter(range(len(dlist)), [item / len(G.nodes) for item in dlist])
        plt.grid()
        plt.xlabel("degree")
        plt.ylabel(r"$p_{degree}$")
        plt.xscale("log")
        plt.savefig(os.path.join(basedir, f"{name}_network_degree_hist.jpg"))
        plt.close()
    else:
        dlist_log = []
        dlist_cen = []
        left = 1
        while left < len(dlist):
            right = int(left * 1.5) + 1
            dlist_log.append(sum(dlist[left:right]) / (right - left) / len(G.nodes))
            dlist_cen.append((left + right) / 2)
            left = right
        startpointidx = np.argmax(dlist_log)
        startpointx = dlist_cen[startpointidx]
        startpointy = dlist_log[startpointidx]
        endpointx = startpointx * 1e2
        endpointy = startpointy * 1e-6
        endpointy2 = startpointy * 1e-5
        plt.scatter(dlist_cen, dlist_log, label=r"$p_k$")
        plt.plot([startpointx, endpointx], [startpointy, endpointy2], "--", color="k", label=r"$\lambda = -2.5$")
        plt.plot([startpointx, endpointx], [startpointy, endpointy], color="k", label=r"$\lambda = -3$")
        plt.grid()
        plt.legend()
        plt.xlabel("degree")
        plt.ylabel(r"$p_{degree}$")
        plt.xscale("log")
        plt.yscale("log")
        plt.savefig(os.path.join(basedir, f"{name}_network_degree_hist.jpg"))
        plt.close()

    out_degrees = G.out_degree()
    in_degrees = G.in_degree()
    neibours_in_degrees = []
    neibours_out_degrees = []
    for node in G.nodes:
        if out_degrees[node] == 10:
            for nb in G.neighbors(node):
                neibours_in_degrees.append(in_degrees[nb])
                neibours_out_degrees.append(out_degrees[nb])
    print("neighbors of out-degree-10 nodes average out degree:", np.mean(neibours_out_degrees))
    print("neighbors of out-degree-10 nodes average in degree:", np.mean(neibours_in_degrees))


def read_table(basedir="./"):
    odlist = pd.read_excel(
        os.path.join(basedir, "table1CommutingODAmerica.xlsx"),
        skiprows=[0, 1, 2, 3, 4, 5, 139440, 139441],
    )
    odlist.columns = [
        "F_S",
        "F_C",
        "State Name",
        "County Name",
        "T_S",
        "T_C",
        "State Name.1",
        "County Name.1",
        "Flow",
        "Margin",
    ]
    odlist.dropna(axis=0, how="any", subset=["T_S", "T_C"], inplace=True)
    odlist.loc[:, "F_FIPS"] = odlist.loc[:, "F_S"].astype(int).astype(str).str.zfill(2) + odlist.loc[:, "F_C"].astype(int).astype(str).str.zfill(3)
    odlist.loc[:, "T_FIPS"] = odlist.loc[:, "T_S"].astype(int).astype(str).str.zfill(2) + odlist.loc[:, "T_C"].astype(int).astype(str).str.zfill(3)
    return odlist.loc[:, ["F_FIPS", "T_FIPS", "Flow", "Margin"]]


def buildUSNetwork(method="trim", basedir="./", recompute=True):
    assert method in ["full", "no_circle", "trim", "trim_no_circle"]
    graph_path = os.path.join(basedir, f"{method}_commuting_graph.pkl")
    if recompute or not os.path.exists(graph_path):
        odlist = read_table(basedir=basedir)
        G = nx.DiGraph()
        if method == "full":
            G.add_weighted_edges_from(odlist.loc[:, ["F_FIPS", "T_FIPS", "Flow"]].values.tolist())
        elif method == "no_circle":
            G.add_weighted_edges_from(odlist.query("F_FIPS!=T_FIPS").loc[:, ["F_FIPS", "T_FIPS", "Flow"]].values.tolist())
        elif method == "trim":
            odlist.loc[:, "Trim"] = odlist["Flow"] - 3 * odlist["Margin"]
            odlist = odlist.query("Trim>0")
            allnodes = set(odlist["F_FIPS"]) | set(odlist["T_FIPS"])
            G.add_nodes_from(allnodes)
            G.add_weighted_edges_from(odlist.loc[:, ["F_FIPS", "T_FIPS", "Flow"]].values.tolist())
        elif method == "trim_no_circle":
            odlist.loc[:, "Trim"] = odlist["Flow"] - 3 * odlist["Margin"]
            odlist = odlist.query("Trim>0")
            odlist = odlist.query("F_FIPS!=T_FIPS")
            G.add_weighted_edges_from(odlist.loc[:, ["F_FIPS", "T_FIPS", "Flow"]].values.tolist())
        summarize_graph(G, method)
        with open(graph_path, "wb") as f:
            pkl.dump(G, f)
    else:
        with open(graph_path, "rb") as f:
            G = pkl.load(f)
    return G
