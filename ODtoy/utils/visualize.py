import networkx as nx
import numpy as np
from typing import Dict,List,Tuple
import matplotlib
from matplotlib import cm,colors
from matplotlib import pyplot as plt
import itertools
def mat2list(mat:np.ndarray,namelst:List[str]):
    lst = []
    for i in  range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i,j]!=0:
                lst.append((namelst[i],namelst[j],8+2*np.log10(mat[i,j])))
    return lst
def drawNetwork(list_size_of_node:List[Tuple[str,int]],adjmat:np.ndarray,filename,ODtype,ax = None):  
    G = nx.DiGraph()
    for node in list_size_of_node:
        G.add_node(node[0],size = node[1])
    adjmat/=np.max(adjmat)
    edgelist = mat2list(adjmat,[item[0] for item in list_size_of_node])
    edge_shrink = 1.0
    G.add_weighted_edges_from(edgelist)
    if ax is None:
        fig,ax = plt.subplots(1,1,dpi=300)
    if ODtype in ['cascade','full','circle']:
        pos = nx.shell_layout(G)
    elif ODtype in ['central','bi-central','web']:
        pos = nx.spring_layout(G)
    elif ODtype in ['bi-node']:
        pos = nx.spiral_layout(G)
    elif ODtype in ['grid']:
        pos = {}
        row_number = int(np.sqrt(len(list_size_of_node)))
        while row_number>=1:
            if  len(list_size_of_node)%row_number ==0:
                break
            row_number -=1
        col_number = len(list_size_of_node)//row_number
        network_shape = [row_number,col_number]
        for idx,(nodename,_) in enumerate(list_size_of_node):
            pos[nodename] = np.array([idx%col_number,idx//col_number])
        edge_shrink = 9/len(list_size_of_node)

    if max(list_size_of_node,key=lambda x:x[1])[1]-min(list_size_of_node,key=lambda x:x[1])[1] >0: 
        nodenorm = colors.Normalize(min(list_size_of_node,key=lambda x:x[1])[1],max(list_size_of_node,key=lambda x:x[1])[1])
        nx.draw_networkx_nodes(G, pos,ax = ax,node_color=[node[1] for node in list_size_of_node],cmap  = 'coolwarm',node_size=4000/len(list_size_of_node))
    else:
        nx.draw_networkx_nodes(G, pos,ax = ax,node_size=4000/len(list_size_of_node))
    nx.draw_networkx_edges(G, pos, ax=ax,edgelist=G.edges(), edge_color='black', width=[edge_shrink*item[2] for item in edgelist], arrowstyle='->', arrowsize=20 if ODtype!='web' else 10, connectionstyle='arc3,rad=0.3')
    nx.draw_networkx_labels(G, pos,ax = ax,font_color='white')
    if ax is None:
        plt.savefig(filename)

def drawNetwork_color_by_sequence(list_size_of_node:List[Tuple[str,int]],adjmat:np.ndarray,filename,ODtype,ax = None):  
    G = nx.DiGraph()
    for node in list_size_of_node:
        G.add_node(node[0],size = node[1])
    adjmat/=np.max(adjmat)
    edgelist = mat2list(adjmat,[item[0] for item in list_size_of_node])
    edge_shrink = 1.0
    G.add_weighted_edges_from(edgelist)
    if ax is None:
        fig,ax = plt.subplots(1,1,dpi=300,figsize=[5,5])
        save_inside = True
    if ODtype in ['cascade','full','circle']:
        pos = nx.shell_layout(G)
    elif ODtype in ['central','bi-central','web']:
        pos = nx.spring_layout(G)
    elif ODtype in ['bi-node','inequal_node']:
        pos = nx.spiral_layout(G)
    elif ODtype in ['grid']:
        pos = {}
        row_number = int(np.sqrt(len(list_size_of_node)))
        while row_number>=1:
            if  len(list_size_of_node)%row_number ==0:
                break
            row_number -=1
        col_number = len(list_size_of_node)//row_number
        network_shape = [row_number,col_number]
        for idx,(nodename,_) in enumerate(list_size_of_node):
            pos[nodename] = np.array([idx%col_number,idx//col_number])
        edge_shrink = np.log10(9)/np.log10(len(list_size_of_node))
    default_mpl_sequence = itertools.cycle(matplotlib.colormaps["tab10"].colors)
    nxnodesize = [4000/len(list_size_of_node) for _ in list_size_of_node]
    if ODtype =='bi-central':
        nxnodesize[0] *=2
        nxnodesize[-1] *=2
    if ODtype =='central':
        nxnodesize[0] *=2
    nx.draw_networkx_nodes(G, pos,ax = ax,node_size=nxnodesize,node_color = [next(default_mpl_sequence) for _ in range(len(list_size_of_node)) ])
    nx.draw_networkx_edges(G, pos, ax=ax,edgelist=G.edges(), edge_color='black', width=[item[2]*edge_shrink for item in edgelist], arrowstyle='->', arrowsize=20 if ODtype!='web' else 10, connectionstyle='arc3,rad=0.3')
    nx.draw_networkx_labels(G, pos,ax = ax,font_color='white')
    if save_inside:
        plt.savefig(filename)
        print('savefig')
        plt.close()




def draw_flow_pcf_mat(initpopulations,town_area,adjmat:np.ndarray,filename,formats,vmin = None,vmax = None,):  
    plt.rcParams['svg.fonttype'] = 'none'
    adjmat+=np.diag(initpopulations-np.sum(adjmat,axis = 1))
    adjmat_plot = adjmat/initpopulations[:,np.newaxis] 


    pop_work = adjmat.sum(axis = 0)
    pcf = adjmat@np.diag(1/pop_work)@adjmat.T
    pcf/=np.sum(pcf,axis = 1,keepdims=True)

    
    REF_POP_DENSITY = initpopulations[0]/town_area[0]
    beta = np.log2(1+pop_work/town_area/REF_POP_DENSITY)
    pcf_beta = adjmat@np.diag(beta/pop_work)@adjmat.T
    pcf_beta/=initpopulations[:,np.newaxis]



    fig,axs = plt.subplots(1,3,figsize = (6,2),dpi = 200)
    plt.tight_layout()
    

    adjmat_plot_masked = np.where(adjmat_plot>0,adjmat_plot,np.nan)
    if np.min(adjmat_plot[adjmat_plot>0])<0.1 or np.max(adjmat_plot[adjmat_plot>0])>1:
        print("warning: adjmat_plot_masked[0.1,1]:",np.min(adjmat_plot[adjmat_plot>0]),np.max(adjmat_plot[adjmat_plot>0]),)

    plt.sca(axs[0])
    pcm = plt.pcolormesh(adjmat_plot_masked, cmap='flare_r', edgecolors='k', linewidth=0.01,
                      norm = colors.LogNorm(0.1,1))
    axs[0].set_aspect('equal', 'box')
    axs[0].set_title("Flow Matrix")
    axs[0].axis("off")
    fig.colorbar(pcm, ax=axs[0], label='Transition Probability')
    fig.text(.2, .9, "{:.2f}".format(np.dot(np.sqrt(adjmat_plot[0,:]),np.sqrt(adjmat_plot[1,:]))), fontsize=7, ha='center')
    axs[0].invert_yaxis()


    pcf_plot = pcf
    adjmat_plot_masked = np.where(pcf_plot>0,pcf_plot,np.nan)
    if np.min(pcf_plot[pcf_plot>0])<1e-3 or np.min(pcf_plot[pcf_plot>0])>1:
        print("pcf_plot_masked:",np.min(pcf_plot[pcf_plot>0]),np.max(pcf_plot[pcf_plot>0]),)
    plt.sca(axs[1])
    pcm = plt.pcolormesh(adjmat_plot_masked, cmap='flare_r', edgecolors='k', linewidth=0.01,
                      norm = colors.LogNorm(1e-3,1))
    axs[1].set_aspect('equal', 'box')
    axs[1].set_title(r"PCF Matrix w/o $\beta$")
    axs[1].axis("off")
    fig.colorbar(pcm, ax=axs[1], label='Normalized PCF')
    plt.axis()
    axs[1].invert_yaxis()
    fig.text(.5, .9, "{:.2f}".format(np.dot(np.sqrt(pcf_plot[0,:]),np.sqrt(pcf_plot[1,:]))), fontsize=7, ha='center')

    pcf_beta_plot = pcf_beta
    adjmat_plot_masked = np.where(pcf_beta_plot>0,pcf_beta_plot,np.nan)
    if np.min(pcf_beta_plot[pcf_beta_plot>0])<0.001 or np.max(pcf_beta_plot[pcf_beta_plot>0])>1:
        print("pcf_beta_plot_masked:",np.min(pcf_beta_plot[pcf_beta_plot>0]),np.max(pcf_beta_plot[pcf_beta_plot>0]),)
    plt.sca(axs[2])
    pcm = plt.pcolormesh(adjmat_plot_masked, cmap='flare_r', edgecolors='k', linewidth=0.01,
                      norm = colors.LogNorm(0.001,1))
    axs[2].set_aspect('equal', 'box')
    axs[2].set_title(r"PCF Matrix w/ $\beta$")
    axs[2].axis("off")
    axs[2].invert_yaxis()
    fig.text(.8, .9, "{:.2f}".format(np.dot(np.sqrt(pcf_beta_plot[0,:]),np.sqrt(pcf_beta_plot[1,:]))), fontsize=7, ha='center')
    fig.colorbar(pcm, ax=axs[2], label='Normalized PCF')


    if not isinstance(formats,list):
        formats = [formats]
    for fmt in formats:
        plt.savefig(filename+"."+fmt)  
    plt.close()