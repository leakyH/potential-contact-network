from paper_repro_paths import font_dir
test_exist = False
plot_curve_offset = -30

TestOver = 5
SHUTDOWN_NAME = '0p60'
REOPEN_NAME = '1'
# SHUTDOWN_NAME = 'fix0p20'
# REOPEN_NAME = '0p95'
REOPEN_DATE = 120
alphaoffset_name = '0'
alphaoffset = float(alphaoffset_name.replace("p",'.'))
RECOVER_RATE = float(REOPEN_NAME.replace("p",'.'))
SHUTDOWN_RATE = float(SHUTDOWN_NAME.removeprefix("fix").replace("p",'.'))
I_test = 1/TestOver
Ia_test = I_test * 1/2
process_period = 120
process_methods = [31,33,34,36]
# 10-39
# 30-37
# 120-42
import os
import pickle as pkl
import networkx as nx
import numpy as np
import pandas as pd
from functools import partial
import sys
sys.path.append("./")
from graphs.graph_common.figure_output import add_output_args, configure_from_args, configure_output_root, graph_output_dir, graph_output_path
from graphs.graph_common.plot_colors import create_hls_colormap
from us_reopen.network_processing import (
    analysis_process_method,
    criterias_name,
    linear_transform,
    hotspot_county_transform,
    process_number_to_fns,
)
from us_reopen.us_data import getInformation, buildUSNetwork
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import matplotlib as mpl
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from typing import Tuple,List
#计算每个参数对应矩阵的flow，
#读取pkl，找到每个在threshold后的peak和sum
# if plotQ:
#     if inf:
#         y_labels = ["peak_after"]
#     else:
#         y_labels = ["peak_in_60_days","peak_after_60_days",]
# else:
# if inf:
#     y_labels = ["peak_after","sum_after"]
# else:
#     y_labels = ["peak_in_60_days","sum_in_60_days","peak_after_60_days","sum_after_60_days"]
y_labels = [f"peak_in_{process_period}_days",f"sum_in_{process_period}_days",f"peak_after",f"sum_after"]
import matplotlib.font_manager as fm
plt.rcParams['svg.fonttype'] = 'none'
custom_font_dir = font_dir()
font_files = fm.findSystemFonts(fontpaths=[custom_font_dir])
for font_file in font_files:
    fm.fontManager.addfont(font_file)
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 6.0
plt.rcParams["pdf.fonttype"] = 42
DEFAULT_OUTPUT_ROOT = "graphs/Fig5/artifacts"
configure_output_root(default_root=DEFAULT_OUTPUT_ROOT)

markers = ['o', 's', '^', 'D', 'v', '*', 'p', 'X', '<', '>','+','d'] 
linestyles = ['--', '-', '-.', ":"]
linecolors = sns.color_palette("tab20")
linecolors = ["#aa3474","#3b5249","#845e48","#639aab","#b2a2ba","#fc8d62","#ff9d9f"]
simple_mid_point_idx = {
    "random_both":10,
    # "realtime_county_pchDextS_both":8,
    "realtime_county_pcfOverFlow_contact_compare_both":10,
    "realtime_county_rt_both_inverse":10,
    "realtime_county_rt_both":10,

}
def sanitize_output_label(label):
    return label.strip().replace("/", "_").replace(" ", "_")


def graph_curve_dir(kwargs):
    dirname = f"curve_{kwargs['suffix']}"
    output_label = kwargs.get("output_label", "")
    if output_label:
        dirname += f"_{output_label}"
    return graph_output_dir(f"graphs/graph_start_end/{kwargs['subdir']}/{dirname}")


def graph_file_suffix(kwargs):
    output_label = kwargs.get("output_label", "")
    if output_label:
        return f"{kwargs['suffix']}_{output_label}"
    return kwargs["suffix"]


def parse_process_methods(raw_methods, fallback_method):
    if not raw_methods:
        return [fallback_method]
    methods = []
    for raw_method in raw_methods.split(","):
        raw_method = raw_method.strip()
        if raw_method:
            methods.append(int(raw_method))
    if not methods:
        raise ValueError("--process_methods did not contain any process methods")
    return methods


def process_input_kwargs(**kwargs):
    subdir = kwargs['subdir']
    os.makedirs(graph_curve_dir(kwargs), exist_ok=True)
    
    print(subdir)
    if '-' in kwargs['period']:
        period = kwargs['period'].split('-')[0]
    else:
        period = kwargs['period']
    if period in ['preCovid','Alpha','Delta','AlphaRestrict','preCovidlikeAlphaRestrict']:
        mobile_phone_file1 = f"graphs/graph1/average_graph_full_daily_{period}_workday.pkl"
        mobile_phone_file2 = f"graphs/graph1/average_graph_full_daily_{period}_weekend.pkl"
        with open(mobile_phone_file1,'rb') as f:
            G1=pkl.load(f)
        # with open(mobile_phone_file2,'rb') as f:
        #     G2=pkl.load(f)
        # G = buildUSNetwork("trim_no_circle",basedir="./ext-data/us-counties/",recompute = False)
        csainfo = getInformation(G1.nodes())
        # subG = subG.subgraph(list(csainfo[0].keys()))
        # breakpoint()
        sampleOD = np.array(nx.linalg.graphmatrix.adjacency_matrix(G1,nodelist = csainfo[0].keys()).todense())
        # sampleOD_weekend = np.array(nx.linalg.graphmatrix.adjacency_matrix(G2,nodelist = csainfo[0].keys()).todense())
        # kwargs['sampleOD'] = (sampleOD_workday,sampleOD_weekend)
    elif period in ["Omicron","Omicron_lm"]:
        mobile_phone_file = f"graphs/graph1/average_graph_full_{period}.pkl"
        with open(mobile_phone_file,'rb') as f:
            G=pkl.load(f)
        # G = buildUSNetwork("trim_no_circle",basedir="./ext-data/us-counties/",recompute = False)
        csainfo = getInformation(G.nodes())
        sampleOD = np.array(nx.linalg.graphmatrix.adjacency_matrix(G,nodelist = csainfo[0].keys()).todense())
    elif period =='commuting':
        G = buildUSNetwork("full",basedir="./ext-data/us-counties/",recompute=False)
        # G = buildUSNetwork("trim_no_circle",basedir="./ext-data/us-counties/",recompute = False)
        csainfo = getInformation(G.nodes())
        G = G.subgraph(list(csainfo[0].keys()))
        sampleOD = np.array(nx.linalg.graphmatrix.adjacency_matrix(G,nodelist = csainfo[0].keys()).todense())
    return sampleOD,csainfo
def check_file(subdir,suffix,process_start_str):
    for process_method in process_methods:  
        fns = process_number_to_fns(process_method)
        for fnname , fnfunc in fns.items():
            for i,_ in enumerate(np.linspace(0,1,21)):
                _suffix = "_"+suffix.format("_prc"+str(process_method),process_start_str)+"_"+fnname+"_"+str(i+300)
                # kwcp['mat_process_func'] = partial(fnfunc,a = a_mat)
                subdir_name = f"{subdir}/{fnname}"
                if not os.path.exists(f"output/pkls/{subdir_name}/I_exist_count{_suffix}.npy"):
                    print(f"process_method{process_method} : {subdir_name} Inpy is not complete",)
                    print(f"missing:output/pkls/{subdir_name}/I_exist_count{_suffix}.npy")
                    break
                if not os.path.exists(f"output/pkls/{subdir_name}/S2E_ratio{_suffix}.npy"):
                    print(f"process_method{process_method} : {subdir_name} S2Enpy is not complete",)
                    print(f"missing:output/pkls/{subdir_name}/S2E_ratio{_suffix}.npy")
                    break
                # else:
                #     s2e = np.load(f"output/pkls/{subdir_name}/S2E_ratio{_suffix}.npy")
                #     if len(s2e.shape)!=2:
                #         print(f"process_method{process_method} : {subdir_name} need new s2e")
                #         break
                if not os.path.exists(f"output/csvs/{subdir_name}/final_flow{_suffix}.csv"):
                    print(f"process_method{process_method} : {subdir_name} flowcsv is not complete",)
                    break
def get_Q_each_county_datapoint(Qlist,**kwargs):
    subdir_name  = kwargs.get("subdir_name","")
    _suffix = '_'+kwargs['suffix']
    initpopulations = kwargs['initpopulations']
    with open(os.path.join("output/csvs/",subdir_name,f"final_flow{_suffix}.csv"),'r') as f:
        final_flow,sum_pop,flow_over_pop = f.readline().strip().split(',')
    # try:
    #     result_frame = np.load(f"output/pkls/{subdir_name}/result_frame_{kwargs['suffix']}.npy")
    # result_frame = np.load(f"output/pkls/{subdir_name}/I_exist_count{_suffix}.npy")
    s2e = np.load(f"output/pkls/{subdir_name}/S2E_ratio{_suffix}.npy")
    # print(f"{subdir_name}/result_frame{_suffix}.npy",np.argmax(result_frame[process_start:process_start+60,:].sum(axis = 1))+process_start,np.argmax(result_frame[process_start+60:,:].sum(axis = 1))+process_start+60)
    plot_start = max(0,process_start+plot_curve_offset)
    
    process_start_at_ts2 = process_start - plot_start
    for Q in Qlist:
        ts2 = s2e[process_start:,Q]*initpopulations[np.newaxis,Q]
        yield (
            float(flow_over_pop), 
            ts2[process_start_at_ts2:process_start_at_ts2+process_period,:].sum(axis = 0),# sum in process_period
            ts2[process_start_at_ts2:,:].sum(axis = 0),# sum in process_period
            
            )
def get_datapoint(**kwargs):
    subdir_name  = kwargs.get("subdir_name","")
    _suffix = '_'+kwargs['suffix']
    initpopulations = kwargs['initpopulations']
    with open(os.path.join("output/csvs/",subdir_name,f"final_flow{_suffix}.csv"),'r') as f:
        final_flow,sum_pop,flow_over_pop = f.readline().strip().split(',')
    # try:
    #     result_frame = np.load(f"output/pkls/{subdir_name}/result_frame_{kwargs['suffix']}.npy")
    result_frame = np.load(f"output/pkls/{subdir_name}/I_exist_count{_suffix}.npy")
    s2e = np.load(f"output/pkls/{subdir_name}/S2E_ratio{_suffix}.npy")

    print(f"{subdir_name}/result_frame{_suffix}.npy",np.argmax(result_frame[process_start:process_start+60,:].sum(axis = 1))+process_start,np.argmax(result_frame[process_start+60:,:].sum(axis = 1))+process_start+60)
    plot_start = max(0,process_start+plot_curve_offset)
    if len(s2e.shape)==1:
        ts2 = s2e[plot_start:]
    else:
        ts2 = (s2e[plot_start:,:]*initpopulations[np.newaxis,:]).sum(axis = 1)
    process_start_at_ts2 = process_start - plot_start
    # breakpoint()
    return (float(flow_over_pop), #x
            np.max(result_frame[process_start:process_start+process_period,:].sum(axis = 1)),#peak in process_period days
            ts2[process_start_at_ts2:process_start_at_ts2+process_period].sum(),# sum in process_period
            np.max(result_frame[process_start:,:].sum(axis = 1)),#peak after process_period days
            ts2[process_start_at_ts2:].sum(),# sum in process_period
            result_frame[plot_start:,:].sum(axis = 1),
            ts2,
)
    # if inf :
    #     return float(flow_over_pop), np.max(result_frame[125:,:].sum(axis = 1)),ts2.sum(),ts1,ts2
    # else:
    #     return float(flow_over_pop), np.max(result_frame[125:125+60,:].sum(axis = 1)),ts2[0:60].sum(),np.max(result_frame[125+60:,:].sum(axis = 1)),ts2[60:].sum(),ts1,ts2


def get_Q_datapoint(Qlist,**kwargs):
    subdir_name  = kwargs.get("subdir_name","")
    _suffix = '_'+kwargs['suffix']
    initpopulations = kwargs['initpopulations']
    with open(os.path.join("output/csvs/",subdir_name,f"final_flow{_suffix}.csv"),'r') as f:
        final_flow,sum_pop,flow_over_pop = f.readline().strip().split(',')
    # try:
    #     result_frame = np.load(f"output/pkls/{subdir_name}/result_frame_{kwargs['suffix']}.npy")
    result_frame = np.load(f"output/pkls/{subdir_name}/I_exist_count{_suffix}.npy")
    s2e = np.load(f"output/pkls/{subdir_name}/S2E_ratio{_suffix}.npy")
    print(f"{subdir_name}/result_frame{_suffix}.npy",np.argmax(result_frame[process_start:process_start+60,:].sum(axis = 1))+process_start,np.argmax(result_frame[process_start+60:,:].sum(axis = 1))+process_start+60)
    
    # if len(s2e.shape)==1:#没有s2e
    #     for Q in Qlist:
    #         if inf:
    #             yield float(flow_over_pop), np.max(result_frame[process_start:,Q].sum(axis = 1)),None,result_frame[process_start:,Q].sum(axis = 1),None
    #         else:
    #             yield float(flow_over_pop), np.max(result_frame[process_start:process_start+60,Q].sum(axis = 1)),np.max(result_frame[process_start+60:,Q].sum(axis = 1)),None,None,result_frame[process_start:,Q].sum(axis = 1),None

    # else:
    plot_start = max(0,process_start+plot_curve_offset)
    for Q in Qlist:
        yield (
            float(flow_over_pop), 
            np.max(result_frame[process_start:process_start+process_period,Q].sum(axis = 1)),
            np.sum(s2e[process_start:process_start+process_period,Q]*initpopulations[np.newaxis,Q]),
            np.max(result_frame[process_start:,Q].sum(axis = 1)),
            np.sum(s2e[process_start:,Q]*initpopulations[np.newaxis,Q]),
            result_frame[plot_start:,Q].sum(axis = 1),
            (s2e[plot_start:,Q]*initpopulations[np.newaxis,Q]).sum(axis = 1))
        # if inf:
        #     yield float(flow_over_pop), np.max(result_frame[process_start:,Q].sum(axis = 1)),np.sum(s2e[process_start:,Q]*initpopulations[np.newaxis,Q]),result_frame[process_start:,Q].sum(axis = 1),(s2e[process_start:,Q]*initpopulations[np.newaxis,Q]).sum(axis = 1)
        # else:
        #     yield float(flow_over_pop), np.max(result_frame[process_start:process_start+60,Q].sum(axis = 1)),np.max(result_frame[process_start+60:,Q].sum(axis = 1)),np.sum(s2e[process_start:process_start+60,Q]*initpopulations[np.newaxis,Q]),np.sum(s2e[process_start+60:,Q]*initpopulations[np.newaxis,Q]),result_frame[process_start:,Q].sum(axis = 1),(s2e[process_start:,Q]*initpopulations[np.newaxis,Q]).sum(axis = 1)

    
        # return float(flow_over_pop), np.max(result_frame[process_start:process_start+60,:].sum(axis = 1)),s2e[process_start:process_start+60].sum(),np.max(result_frame[process_start:,:].sum(axis = 1)),s2e[process_start:].sum()

def ODseir_compare(*args, **kwargs):
    sampleOD,csainfo = process_input_kwargs(**kwargs)
    kwargs['sampleOD'] = sampleOD
    kwargs['csainfo'] = csainfo#townIDs,town_population,town_area,max_ratios,csa_max_ratio,ageprops,REF_POP_DENSITY
    initpopulations = np.array(list(csainfo[1].values()))

    fns = process_number_to_fns(kwargs['process_inequal'][1])
    df = pd.DataFrame(columns = ["outflow","subdir","method","idx"]+y_labels)
    idx = 0
    for fnname , fnfunc in fns.items():
        method_ana = analysis_process_method(fnname)
        ls = linestyles[0] if method_ana['inverse'] else linestyles[1]
        marker = markers[0] if method_ana['both_direction'] else markers[1]
        for _criteria,_lc in zip(criterias_name,linecolors):
            if _criteria == method_ana['criteria']:
                lc = _lc
                break
        cmap = create_hls_colormap(color = lc,saturation_start= 0.0,saturation_end = None)
        fig, (ax_I,ax_s2e) = plt.subplots(2,1,figsize = [3,3],dpi = 300,sharex= False)
        ax_I.set_ylabel("Exist I")
        # bx_I = plt.twinx(ax_I)
        # bx_s2e = plt.twinx(ax_s2e)
        ax_s2e.set_ylabel("Daily S2E")
        tss = []
        xs = []
        for i in [0,simple_mid_point_idx.get(fnname,10),20]:
            kwcp = kwargs.copy()
            kwcp['suffix'] = kwargs['suffix']+"_"+fnname+"_"+str(i+300)
            # kwcp['mat_process_func'] = partial(fnfunc,a = a_mat)
            kwcp['initpopulations'] = initpopulations
            kwcp['subdir_name'] = f"{subdir}/{fnname}"
            datapoints = get_datapoint(**kwcp)
            df.loc[idx,["outflow"]+y_labels] = datapoints[0:(len(y_labels)+1)]
            ts_I,ts_s2e = datapoints[(len(y_labels)+1):]
            
            df.loc[idx,"subdir"] = subdir
            df.loc[idx,"method"] = fnname
            
            df.loc[idx,['criteria','inverse','both_direction']] = method_ana
            df.loc[idx,"idx"] = i
            idx+=1
            
            tss.append((ts_I,ts_s2e))
            xs.append(datapoints[0])
        norm = Normalize(vmin = np.min(xs),vmax = np.max(xs))

        base_ts_s2e_cumsum = np.cumsum(tss[0][1])
        base_ts_I = tss[0][0]
        # bx_I.set_ylabel(f"relative to x={xs[0]:.2f}")
        # bx_s2e.set_ylabel(f"cumsum relative to x={xs[0]:.2f}")
        for idx_exp,(x,(ts_I,ts_s2e)) in enumerate(zip(xs,tss)):
            if (method_ana['both_direction'] and idx_exp == 1) or (not method_ana['both_direction'] and idx_exp == 1):
                marker = None
                zorder = 2
                alpha = 1
                linewidth = 2
            else:
                marker = None
                zorder = 3
                alpha = 1
                linewidth = 1
            ax_I.plot(np.arange(len(ts_I))+max(0,process_start+plot_curve_offset),ts_I,
                    #   linestyle = ls,
                      marker = marker,
                      color = cmap(norm(x)),linewidth = linewidth,markersize = 4,zorder = zorder,markerfacecolor = 'white',alpha = alpha,label = f"Flow Prop.={x:.2f}")
            ax_s2e.plot(np.arange(len(ts_I))+max(0,process_start+plot_curve_offset),ts_s2e,
                        # linestyle = ls,
                        marker = marker,
                        color = cmap(norm(x)),linewidth = linewidth,markersize = 4,zorder = zorder,markerfacecolor = 'white',alpha = alpha,label = f"Flow Prop.={x:.2f}")
            # bx_I.plot(np.arange(len(ts_I))+process_start-10,ts_I/base_ts_I,color = cmap(norm(x)),linewidth = 2,linestyle = 'dashed')
            # bx_s2e.plot(np.arange(len(ts_I))+process_start-10,np.cumsum(ts_s2e)/base_ts_s2e_cumsum,color = cmap(norm(x)),linewidth = 2,linestyle = 'dashed')
            # bx_I.axhline(y=1,color = 'black',linewidth = 3)
            # bx_s2e.axhline(y=1,color = 'black',linewidth = 3)
        # ax_I.set_yscale("log")
        # ax_s2e.set_yscale("log")
        # fig.colorbar(mappable=ScalarMappable(norm = norm,cmap = cmap),ax = [ax_I,ax_s2e])
        ax_I.legend()
        ax_s2e.legend()
        ax_I.spines['top'].set_visible(False)
        ax_I.spines['right'].set_visible(False)
        ax_s2e.spines['top'].set_visible(False)
        ax_s2e.spines['right'].set_visible(False)
        ax_I.ticklabel_format(style='sci',axis='y',scilimits = (0,0),useOffset = True, useLocale=False, useMathText=True)
        ax_s2e.ticklabel_format(style='sci',axis='y',scilimits = (0,0),useOffset = True, useLocale=False, useMathText=True)
        plt.tight_layout()
        plt.savefig(f"{graph_curve_dir(kwargs)}/{fnname}.jpg")
        plt.savefig(f"{graph_curve_dir(kwargs)}/{fnname}.svg")
        plt.close(fig)
    
    return df
def ODseir_each_county(*args, **kwargs):
    sampleOD,csainfo = process_input_kwargs(**kwargs)
    kwargs['sampleOD'] = sampleOD
    kwargs['csainfo'] = csainfo#townIDs,town_population,town_area,max_ratios,csa_max_ratio,ageprops,REF_POP_DENSITY
    initpopulations = np.array(list(csainfo[1].values()))

    fns = process_number_to_fns(kwargs['process_inequal'][1])
    if 'log' in kwargs['beta_density']:
        density = np.array(list(csainfo[1].values()))/csainfo[2]
        Q_based_on = 'density'
    elif kwargs['beta_density'] in ['cfg','cfgFull']:
        alpha_csv = pd.read_csv("graphs/graphR/testR0.csv",dtype = {"fips":str})
        county_cfg = dict((line["fips"],line["r"]) for _,line in alpha_csv.iterrows())
        # with open("ext-data/us-counties/cases_coef_R0_local.json") as f:
        #     county_cfg = json.load(f)
        betamedian = np.median(list(county_cfg.values()))
        # countyalpha = (np.array([county_cfg.get(item,betamedian) for item in townIDs])+(1/8.85))/(0.896*alpha_ratio)
        # proper_slope = 1/(betamedian-1/12)
        density = np.array([county_cfg.get(item,betamedian) for item in csainfo[0]])
        Q_based_on = 'county_alpha'
    elif kwargs['beta_density'] == 'fit':
        alpha_csv = pd.read_csv("graphs/graphR/testR0_max_fitted_beta.csv",dtype = {"fips":str})
        alpha_csv['fips'] = alpha_csv['fips'].str.zfill(5)
        county_cfg = dict((line["fips"],line[f"TO{TestOver}_beta"]) for _,line in alpha_csv.iterrows())
        betamin = np.min(list(county_cfg.values()))
        density = np.array([county_cfg.get(item,betamin) for item in csainfo[0]])
        Q_based_on = 'county_alpha'


    Qlist = []
    ref  = np.argsort(density)
    amount = np.array(list(csainfo[1].values()))
    cumsum_amount = np.cumsum(amount[ref])/np.sum(amount)

    for qlevel in range(4):
        Qlist.append(ref[((qlevel)/4<=cumsum_amount) &(cumsum_amount<=(qlevel+1)/4)])
        # Qlist.append([int(len(density)/4*qlevel):int(len(density)/4*(qlevel+1))])
    df = pd.DataFrame(columns = ["outflow","subdir","method","idx"]+y_labels)
    idx = 0
    xs = []
    df = []
    fnname2color = {}
    fnname2ls = {}
    fnname2marker = {}
    collect_reopen = True
    fig, (ax,ax_bar) = plt.subplots(2,1,figsize = [5,5],dpi = 300)
    fig_inf, (ax_inf,ax_inf_bar) = plt.subplots(2,1,figsize = [5,5],dpi = 300)
    fig_Q, axs_Q = plt.subplots(2,2,figsize = [4,4],dpi = 300)
    fig_inf_Q, axs_inf_Q = plt.subplots(2,2,figsize = [4,4],dpi = 300)
    #for barplot
    barx = np.arange(4)  # the label locations
    width = 0.8/(len(fns)+1)  # the width of the bars, 0.2 Reopen
    blank = width*0.1
    multiplier = 0
    break_median = 1.1
    break_half_width = 0.1
    for fnsidx,(fnname , fnfunc) in enumerate(fns.items()):
        method_ana = analysis_process_method(fnname)
        
        ls = linestyles[0] if method_ana['inverse'] else linestyles[1]
        marker = markers[0] if method_ana['both_direction'] else markers[1]
        fnname2ls[fnname] = ls
        fnname2marker[fnname] = marker
        if (not method_ana['inverse']) and (not method_ana['both_direction']) and method_ana['criteria'] == 'pcf_compare':
                        linewidth_ratio = 2
        else:
            linewidth_ratio = 1
        
        for _criteria,_lc in zip(criterias_name,linecolors):
            if _criteria == method_ana['criteria']:
                lc = _lc
                fnname2color[fnname] = lc
                break
        cmap = create_hls_colormap(color = lc,saturation_start= 0.0,saturation_end = None)

        for i,_ in enumerate(np.linspace(0,1,21)):
            if i not in [0,10,20]:
                continue
            if i==20 :
                if collect_reopen & (fnsidx== len(fns)-1):
                    collect_reopen = False
                else:
                    continue
            kwcp = kwargs.copy()
            kwcp['suffix'] = kwargs['suffix']+"_"+fnname+"_"+str(i+300)
            kwcp['initpopulations'] = initpopulations
            # kwcp['mat_process_func'] = partial(fnfunc,a = a_mat)
            kwcp['subdir_name'] = f"{subdir}/{fnname}"
            counts = []
            counts_inf = []
            for datapoints in get_Q_each_county_datapoint(Qlist,**kwcp):
                x,count_process_Q, count_inf_Q = datapoints
                counts.append(count_process_Q)
                counts_inf.append(count_inf_Q)
            
            #现在还是按照Q来排列的
            if i == 0 :
                counts_base_Q = counts
                counts_inf_base_Q = counts_inf
            elif i == 20 :
                for idx,(ax_q,ax_inf_q) in enumerate(zip(axs_Q.flatten(),axs_inf_Q.flatten())):
                    ratio = counts[idx]/counts_base_Q[idx]
                    ratio_inf = counts_inf[idx]/counts_inf_base_Q[idx]
                    df= pd.DataFrame({'fnname':'Reopen',
                                    "ratio":ratio,
                                    "ratio_inf":ratio_inf})
                    
                    fnname2color['Reopen'] = 'black'
                    sns.kdeplot(df,x = 'ratio',ax = ax_q,legend = False,color = 'black',linestyle = "-",marker = "",label = 'Reopen',linewidth = 1*linewidth_ratio)
                    sns.kdeplot(df,x = 'ratio_inf',ax = ax_inf_q,legend = False,color = 'black',linestyle = "-",marker = "",label = 'Reopen',linewidth = 1*linewidth_ratio)
                    
                    
            else:
                for idx,(ax_q,ax_inf_q) in enumerate(zip(axs_Q.flatten(),axs_inf_Q.flatten())):
                    ratio = counts[idx]/counts_base_Q[idx]
                    ratio_inf = counts_inf[idx]/counts_inf_base_Q[idx]
                    df = pd.DataFrame({'fnname':fnname,
                                    "ratio":ratio,
                                    "ratio_inf":ratio_inf})

                    sns.kdeplot(df,x = 'ratio',ax = ax_q,legend = False,color = lc,linestyle = ls,marker = marker,label =fnname,linewidth = 1*linewidth_ratio)
                    sns.kdeplot(df,x = 'ratio_inf',ax = ax_inf_q,legend = False,color = lc,linestyle = ls,marker = marker,label =fnname,linewidth = 1*linewidth_ratio)

            
            
            
            xs.append(x)
            counts = np.concatenate(counts)
            counts_inf = np.concatenate(counts_inf)
            if i == 0 :
                counts_base = counts
                counts_inf_base = counts_inf
            elif i == 20:
                ratio = counts/counts_base        
                ratio_inf = counts_inf/counts_inf_base
                df= pd.DataFrame({'fnname':'Reopen',
                                "ratio":ratio,
                                "ratio_inf":ratio_inf})
                
                fnname2color['Reopen'] = 'black'
                df = pd.DataFrame({'fnname':fnname,
                                "ratio":ratio,
                                "ratio_inf":ratio_inf})
                fnname2color['Reopen'] = 'black'
                static_county_pop= [sum(ratio<break_median-break_half_width)/len(ratio),
                    sum(ratio>break_median+break_half_width)/len(ratio),
                   sum(amount[ratio<break_median-break_half_width])/sum(amount),
                    sum(amount[ratio>break_median+break_half_width])/sum(amount),
                ]
                static_inf_county_pop = [sum(ratio_inf<break_median-break_half_width)/len(ratio_inf),
                    sum(ratio_inf>break_median+break_half_width)/len(ratio_inf),
                   sum(amount[ratio_inf<break_median-break_half_width])/sum(amount),
                    sum(amount[ratio_inf>break_median+break_half_width])/sum(amount),
                ]
                sns.kdeplot(df,x = 'ratio',ax = ax,legend = False,color = 'black',linestyle = "-",marker = "",label = 'Reopen',linewidth = 2*linewidth_ratio)
                sns.kdeplot(df,x = 'ratio_inf',ax = ax_inf,legend = False,color = 'black',linestyle = "-",marker = "",label = 'Reopen',linewidth = 2*linewidth_ratio)
                offset = width * multiplier
                rects = ax_bar.bar(barx + offset, static_county_pop, width-2*blank, label='Reopen',color = 'black',hatch=marker,linestyle = '-',linewidth = 1*linewidth_ratio,edgecolor= 'black')
                ax_bar.bar_label(rects, padding=3,fmt = "%.2f")
                rects = ax_inf_bar.bar(barx + offset, static_inf_county_pop, width-2*blank, label='Reopen',color = 'black',hatch=marker,linestyle = '-',linewidth = 1*linewidth_ratio,edgecolor= 'black')
                ax_inf_bar.bar_label(rects, padding=3,fmt = "%.2f")
                multiplier+=1
            else:
                ratio = counts/counts_base        
                ratio_inf = counts_inf/counts_inf_base
                df = pd.DataFrame({'fnname':fnname,
                                "ratio":ratio,
                                "ratio_inf":ratio_inf})
                static_county_pop= [len(ratio[ratio<break_median-break_half_width])/len(ratio),
                    len(ratio[ratio>break_median+break_half_width])/len(ratio),
                   sum(amount[ratio<break_median-break_half_width])/sum(amount),
                    sum(amount[ratio>break_median+break_half_width])/sum(amount),
                ]
                static_inf_county_pop = [len(ratio_inf[ratio_inf<break_median-break_half_width])/len(ratio_inf),
                    len(ratio_inf[ratio_inf>break_median+break_half_width])/len(ratio_inf),
                   sum(amount[ratio_inf<break_median-break_half_width])/sum(amount),
                    sum(amount[ratio_inf>break_median+break_half_width])/sum(amount),
                ]
                sns.kdeplot(df,x = 'ratio',ax = ax,legend = False,color = lc,linestyle = ls,marker = marker,label =fnname,linewidth = 2*linewidth_ratio)
                sns.kdeplot(df,x = 'ratio_inf',ax = ax_inf,legend = False,color = lc,linestyle = ls,marker = marker,label =fnname,linewidth = 2*linewidth_ratio)
                offset = width * multiplier
                rects = ax_bar.bar(barx + offset, static_county_pop, width-2*blank, label=fnname,color = lc,hatch=marker,linestyle = ls,linewidth = 1*linewidth_ratio,edgecolor= 'black')
                ax_bar.bar_label(rects, padding=3,fmt = "%.2f")
                rects = ax_inf_bar.bar(barx + offset, static_inf_county_pop, width-2*blank, label=fnname,color = lc,hatch=marker,linestyle = ls,linewidth = 1*linewidth_ratio,edgecolor= 'black')
                ax_inf_bar.bar_label(rects, padding=3,fmt = "%.2f")
                multiplier+=1

    for idx,(ax_q,ax_inf_q) in enumerate(zip(axs_Q.flatten(),axs_inf_Q.flatten())):
        ax_q.set_xlim(0.5,1.5)
        ax_inf_q.set_xlim(0.5,1.5)
    figure_suffix = graph_file_suffix(kwargs)
    fig_Q.savefig(graph_output_path(f"graphs/graph_start_end/{subdir}/county_ratio_Q_{figure_suffix}.jpg"))
    fig_Q.savefig(graph_output_path(f"graphs/graph_start_end/{subdir}/county_ratio_Q_{figure_suffix}.svg"))
    fig_inf_Q.savefig(graph_output_path(f"graphs/graph_start_end/{subdir}/county_ratio_inf_Q_{figure_suffix}.jpg"))
    fig_inf_Q.savefig(graph_output_path(f"graphs/graph_start_end/{subdir}/county_ratio_inf_Q_{figure_suffix}.svg"))
    
    ax_bar.set_ylabel("Prop")
    ax_bar.set_ylim(0,1.5)
    ax_bar.set_xlabel("matrices")
    ax_bar.set_title("effects on new infections of different flow")
    ax_bar.set_xticks(barx + width*(len(fns))/2, [f"<{break_median-break_half_width:.2f} county"+r"$\uparrow$",f">{break_median+break_half_width:.2f} county"+r'$\downarrow$',f"<{break_median-break_half_width:.2f} pop"+r'$\uparrow$',f">{break_median+break_half_width:.2f} pop"+r"$\downarrow$"])
    ax_bar.legend(loc='upper left', ncols=2)
    ax_inf_bar.set_ylabel("Prop")
    ax_inf_bar.set_ylim(0,1.5)
    ax_inf_bar.set_xlabel("matrices")
    ax_inf_bar.set_title("effects on new infections of different flow")
    ax_inf_bar.set_xticks(barx + width*(len(fns))/2, [f"<{break_median-break_half_width:.2f} county",f">{break_median+break_half_width:.2f} county",f"<{break_median-break_half_width:.2f} pop",f">{break_median+break_half_width:.2f} pop"])
    ax_inf_bar.legend(loc='upper left', ncols=2)
    for _ax in [ax_bar,ax_inf_bar]:
        _color = 'gray'
        _alpha = 0.3
        for _barx in barx:
            _ax.axvspan(_barx+ width*(len(fns))/2-0.5,_barx+ width*(len(fns))/2+0.5,color = _color,alpha = _alpha,zorder = 0)
            _alpha = 1-_alpha
        _ax.set_xlim(barx[0]+ width*(len(fns))/2-0.5,barx[-1]+ width*(len(fns))/2+0.5)
    
    
    ax.legend()
    ax.set_xlim(0.5,1.5)
    # ax.axvline(1,color = 'gray',linewidth = 2,alpha = 0.5,linestyle = ':')
    ax.axvspan(xmin = 0.5,xmax = break_median-break_half_width,color = 'gray',alpha = 0.3,zorder = 0)
    ax.axvspan(xmin = break_median+break_half_width,xmax = 1.5,color = 'gray',alpha = 0.7,zorder = 0)
    fig.savefig(graph_output_path(f"graphs/graph_start_end/{subdir}/county_ratio_{figure_suffix}.jpg"))
    ax_inf.legend()
    ax_inf.axvspan(xmin = 0.5,xmax = break_median-break_half_width,color = 'gray',alpha = 0.3,zorder = 0)
    ax_inf.axvspan(xmin = break_median+break_half_width,xmax = 1.5,color = 'gray',alpha = 0.7,zorder = 0)
    # ax_inf.axvline(1,color = 'gray',linewidth = 2,alpha = 0.5,linestyle = ':')
    ax_inf.set_xlim(0.5,1.5)
    fig_inf.savefig(graph_output_path(f"graphs/graph_start_end/{subdir}/county_ratio_inf_{figure_suffix}.jpg"))
    
    
def ODseir_Q_compare(*args, **kwargs):
    sampleOD,csainfo = process_input_kwargs(**kwargs)
    kwargs['sampleOD'] = sampleOD
    kwargs['csainfo'] = csainfo#townIDs,town_population,town_area,max_ratios,csa_max_ratio,ageprops,REF_POP_DENSITY
    initpopulations = np.array(list(csainfo[1].values()))
    

    fns = process_number_to_fns(kwargs['process_inequal'][1])


    if 'log' in kwargs['beta_density']:
        density = np.array(list(csainfo[1].values()))/csainfo[2]
        Q_based_on = 'density'
    elif kwargs['beta_density'] in ['cfg','cfgFull']:
        alpha_csv = pd.read_csv("graphs/graphR/testR0.csv",dtype = {"fips":str})
        county_cfg = dict((line["fips"],line["r"]) for _,line in alpha_csv.iterrows())
        # with open("ext-data/us-counties/cases_coef_R0_local.json") as f:
        #     county_cfg = json.load(f)
        betamedian = np.median(list(county_cfg.values()))
        # countyalpha = (np.array([county_cfg.get(item,betamedian) for item in townIDs])+(1/8.85))/(0.896*alpha_ratio)
        # proper_slope = 1/(betamedian-1/12)
        density = np.array([county_cfg.get(item,betamedian) for item in csainfo[0]])
        Q_based_on = 'county_alpha'
    elif kwargs['beta_density'] == 'fit':
        alpha_csv = pd.read_csv("graphs/graphR/testR0_max_fitted_beta.csv",dtype = {"fips":str})
        alpha_csv['fips'] = alpha_csv['fips'].str.zfill(5)
        county_cfg = dict((line["fips"],line[f"TO{TestOver}_beta"]) for _,line in alpha_csv.iterrows())
        betamin = np.min(list(county_cfg.values()))
        density = np.array([county_cfg.get(item,betamin) for item in csainfo[0]])
        Q_based_on = 'county_alpha'
    elif kwargs['beta_density']=='fit_poisson':
        assert TestOver in [3,5,7,10]
        alpha_csv = pd.read_csv("graphs/graphR/testR0_max_fitted_beta_poisson.csv",dtype = {"fips":str})
        alpha_csv['fips'] = alpha_csv['fips'].str.zfill(5)
        county_cfg = dict((line["fips"],line[f"TO{TestOver}_beta"]) for _,line in alpha_csv.iterrows())
        betamin = np.min(list(county_cfg.values()))
        density = np.array([county_cfg.get(item,betamin) for item in csainfo[0]])
        Q_based_on = 'county_alpha'

    Qlist = []
    ref  = np.argsort(density)
    amount = np.array(list(csainfo[1].values()))
    cumsum_amount = np.cumsum(amount[ref])/np.sum(amount)

    for qlevel in range(4):
        Qlist.append(ref[((qlevel)/4<=cumsum_amount) &(cumsum_amount<=(qlevel+1)/4)])
        # Qlist.append([int(len(density)/4*qlevel):int(len(density)/4*(qlevel+1))])
    
    dfs = [pd.DataFrame(columns = ["outflow","subdir","method","idx"]+y_labels) for _ in Qlist]
    idx = 0
    for fnname , fnfunc in fns.items():
        method_ana = analysis_process_method(fnname)
        ls = linestyles[0] if method_ana['inverse'] else linestyles[1]
        marker = markers[0] if method_ana['both_direction'] else markers[1]
        for _criteria,_lc in zip(criterias_name,linecolors):
            if _criteria == method_ana['criteria']:
                lc = _lc
                break
        print(method_ana)
        cmap = create_hls_colormap(color = lc,saturation_start= 0.0,saturation_end = None)
        fig, axes = plt.subplots(2,len(Qlist),figsize = [6,3],dpi = 300,sharey = 'row')
        ax_Is = axes[0,:]
        ax_s2es = axes[1,:]
        ax_Is[0].set_ylabel("Exist I")
        ax_s2es[0].set_ylabel("daily S2E")
        # ax_Is[0].set_yscale("log")
        tss = [[] for _ in Qlist]
        xs = []
        for i in [0,simple_mid_point_idx.get(fnname,10),20]:
            kwcp = kwargs.copy()
            kwcp['suffix'] = kwargs['suffix']+"_"+fnname+"_"+str(i+300)
            kwcp['initpopulations'] = initpopulations
            # kwcp['mat_process_func'] = partial(fnfunc,a = a_mat)
            kwcp['subdir_name'] = f"{subdir}/{fnname}"
            for df,datapoints,tslist in zip(dfs,get_Q_datapoint(Qlist,**kwcp),tss):
                df.loc[idx,["outflow"]+y_labels] = datapoints[:len(y_labels)+1]#可能有None
                ts_I = datapoints[len(y_labels)+1]
                ts_s2e = datapoints[len(y_labels)+2]#可能是None
                df.loc[idx,"subdir"] = subdir
                df.loc[idx,"method"] = fnname
                df.loc[idx,['criteria','inverse','both_direction']] = method_ana
                df.loc[idx,"idx"] = i

                tslist.append((ts_I,ts_s2e))
            xs.append(datapoints[0])
            idx+=1
        norm = Normalize(vmin = np.min(xs),vmax = np.max(xs))
        for axidx,(ax_I,ax_s2e,tslist) in enumerate(zip(ax_Is,ax_s2es,tss)):
            ax_I.set_title(f"{Q_based_on}-Q{axidx}(pop weighted)")
            for ts_idx,x in enumerate(xs):
                if (method_ana['both_direction'] and ts_idx == 1) or (not method_ana['both_direction'] and ts_idx == 1):
                    marker = None
                    zorder = 2
                    alpha = 1
                    linewidth = 2
                else:
                    marker = None
                    zorder = 3
                    alpha = 1
                    linewidth = 1
                ts_I,ts_s2e = tslist[ts_idx]#ts_s2e可能是None
                ax_I.plot(np.arange(len(ts_I))+max(0,process_start+plot_curve_offset),ts_I,linestyle = ls,
                          marker = marker,
                          color = cmap(norm(x)),markersize = 4,zorder = zorder,markerfacecolor = 'white',alpha = alpha,linewidth = linewidth)
                if ts_s2e is not None:
                    ax_s2e.plot(np.arange(len(ts_I))+max(0,process_start+plot_curve_offset),ts_s2e,linestyle = ls,
                              marker = marker,
                            color = cmap(norm(x)),markersize = 4,zorder = zorder,markerfacecolor = 'white',alpha = alpha,linewidth = linewidth)
            ax_I.ticklabel_format(style='sci',axis='y',scilimits = (0,0),useOffset = True, useLocale=False, useMathText=True)
            ax_s2e.ticklabel_format(style='sci',axis='y',scilimits = (0,0),useOffset = True, useLocale=False, useMathText=True)
            ax_I.spines['top'].set_visible(False)
            ax_I.spines['right'].set_visible(False)
            ax_s2e.spines['top'].set_visible(False)
            ax_s2e.spines['right'].set_visible(False)
        # fig.colorbar(mappable=ScalarMappable(norm = norm,cmap = cmap),ax = axes)
        plt.tight_layout()
        plt.savefig(f"{graph_curve_dir(kwargs)}/{fnname}_Q.jpg")
        plt.savefig(f"{graph_curve_dir(kwargs)}/{fnname}_Q.svg")
        plt.close(fig)

    
    
    
    return df

if __name__=="__main__":
    # profile = lp.LineProfiler(ODseir_simple)
    # # # # 自己制作一个profile工具，并且传入要分析的代码
    # profile.enable()
    # # # 起始分析
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha_ratio_name",type = str,required=True,help="float, use p to replace . in a number")
    parser.add_argument("--flow_ratio_name",type = str,required=True,help="float, use p to replace . in a number")
    parser.add_argument("--period",type = str,required=True,help="commuting, Omicron preCovid, Delta or Alpha ")
    parser.add_argument("--init_method",type = str,default = "airport50k",help="pop10t10, airport50k,cfg,everywhere",choices=["fitpoisson","pop10t10","airport50k","airport5k","airport10k","airport100k","cfg","everywhere","pop100k",'pop1k','random1k'])
    parser.add_argument("--beta_density",type = str,default = "cfg",help="cfg or False",choices=["fit_poisson","fit","cfg",'cfgFull',"False","log2","log10","ln","log2R1","log2R1p5"])
    parser.add_argument("--process_threshold",type = int,default = None,help="process_threshold")
    parser.add_argument("--simmode",type = str,default = 'aftershutdown',help="simmode")
    parser.add_argument("--process_method",type = int,default = 0,help="process_method, 0 for nothing")
    parser.add_argument("--process_methods",type = str,default = "",help="comma-separated process methods to plot; overrides --process_method")
    parser.add_argument("--output_label",type = str,default = "",help="optional graph-output suffix to avoid overwriting figure variants")
    parser.add_argument("--run_label",type = str,default = "",help="optional experiment run label included in input suffix/subdir")
    add_output_args(parser, default_figure="Fig5", default_root=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--process_inf",action = 'store_true',help="process_inf store True")
    parser.add_argument("--Rtype",type = str,default = "R4m",help="Rtype")
    parser.add_argument("--Pratio",type = str,default = "0p3",help="Pratio - 0p3 - 0.3")
    args = parser.parse_args()
    configure_from_args(args)

    if args.process_threshold is None:
        process_start_str = ""
        process_start = 125
    else:
        process_start = args.process_threshold
        process_start_str = "_{}".format(process_start)


    process_methods = parse_process_methods(args.process_methods, args.process_method)
    output_label = sanitize_output_label(args.output_label)
    run_label = sanitize_output_label(args.run_label)

    alpha_ratio =  float(args.alpha_ratio_name.replace('p','.'))
    flow_ratio =  float(args.flow_ratio_name.replace('p','.'))
    s2eversion = 'v1'  #接下来只保留v1
    beta_density = args.beta_density #cfg 和False都要
    init_method = args.init_method
    df_list = []
    strategy_type = {
        31:'static_info',
        32:'epidemic_included',
        33:'weekly_epidemic_included',
        34:'static_info',
        35:'epidemic_included',
        36:'weekly_epidemic_included',
        37:'static_info',
        38:'epidemic_included',
        39:'weekly_epidemic_included',
        40:'static_info',
        41:'epidemic_included',
        42:'weekly_epidemic_included',
        43:'static_info',
        44:'epidemic_included',
        45:'weekly_epidemic_included',
    }
    strategy_type_list = [
        'static_info','epidemic_included','weekly_epidemic_included'
    ]
    suffix = f"{args.period}_I{init_method}_{args.Rtype}_P{args.Pratio}_a{args.alpha_ratio_name}{'_'+beta_density if beta_density else ''}_f{args.flow_ratio_name}_{args.simmode}_{s2eversion}{{}}_ar{SHUTDOWN_NAME}{REOPEN_DATE}_{REOPEN_NAME}_TO{TestOver}_fitted500kIa0p9initPF120{{}}"
    if args.process_inf:
        suffix += '_inf'
    if run_label:
        suffix += f"_{run_label}"
    subdir = f"us_{args.period}_{str(beta_density)}"
    subdir += ("_"+"_".join(suffix.split('_')[2:7]))
    if run_label:
        subdir += f"_{run_label}"
    graph_output_dir(f"graphs/graph_start_end/{subdir}")
    if test_exist:
        check_file(subdir = subdir,suffix = suffix,process_start_str = process_start_str)
        exit()

    for process_method in process_methods:
        process_inequal = (args.process_threshold,process_method)
        formatted_suffix = suffix.format("_prc"+str(process_method),process_start_str)
        result_df = ODseir_Q_compare(dryrun = False,gif = False,scalar = True,
            beta_density=beta_density,
            period = args.period,
            suffix = formatted_suffix,
            alpha_ratio = alpha_ratio,
            flow_ratio = flow_ratio,
            s2eversion=s2eversion,
            init_method = init_method,
            process_inequal=process_inequal,
            subdir = subdir,
            Rtype = args.Rtype,
            Pratio = float(args.Pratio.replace("p",".")),
            output_label = output_label)
        result_df = ODseir_compare(dryrun = False,gif = False,scalar = True,
            beta_density=beta_density,
            period = args.period,
            suffix = formatted_suffix,
            alpha_ratio = alpha_ratio,
            flow_ratio = flow_ratio,
            s2eversion=s2eversion,
            init_method = init_method,
            process_inequal=process_inequal,
            subdir = subdir,
            Rtype = args.Rtype,
            Pratio = float(args.Pratio.replace("p",".")),
            output_label = output_label)
    
    # ODseir_each_county(dryrun = False,gif = False,scalar = True,
    #     beta_density=beta_density,
    #     period = args.period,
    #     suffix = suffix.format("_prc"+str(args.process_method)+process_start_str),
    #     alpha_ratio = alpha_ratio,
    #     flow_ratio = flow_ratio,
    #     s2eversion=s2eversion,
    #     init_method = init_method,
    #     process_inequal=process_inequal,
    #     subdir = subdir,
    #     Rtype = args.Rtype,
    #     Pratio = float(args.Pratio.replace("p",".")))
            
