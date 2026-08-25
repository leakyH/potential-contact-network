import networkx as nx
import numpy  as np
from typing import List,Union



def preprocess_OD_Mobile(spod:np.ndarray,pop=None,)->np.ndarray:
    nowOsum = spod.sum(axis=1)
    extendratio = pop/nowOsum
    extendratio[np.isinf(extendratio)] = 0 
    normalizedOD = spod*np.reshape(extendratio,(-1,1))
    eyemask = np.eye(*normalizedOD.shape,dtype = bool)
    result = np.where(eyemask,0,normalizedOD)

    result[result<5]  = 0

    result[eyemask] = pop - result.sum(axis = 1)
    return result
def preprocess_OD_commuting(spod:np.ndarray,pop=None,)->np.ndarray:
    spod = spod.astype(float)
    eyemask = np.eye(*spod.shape,dtype = bool)
    result = np.where(eyemask,0,spod)
    result[result<5]  = 0
    result[eyemask] = pop - result.sum(axis = 1)
    
    return result
def process_adjmat_directed_amount(adjmat:np.ndarray,pop=None,distance = None)->np.ndarray:
    """处理一个adjmat，输出上三角矩阵。

    Args:
        adjmat (np.ndarray): n*n方阵

    Returns:
        np.ndarray: 列向量，为处理后adj的上三角矩阵，不包含对称轴
    """
    if pop is None:
        pop = adjmat.sum(axis = 1)
    adjmat_bk = adjmat.copy()
    adjmat_bk*= 1-np.eye(*adjmat_bk.shape)
    nowOsum = adjmat_bk.sum(axis=1)
    ODandLeft = np.diag(pop-nowOsum)+adjmat_bk
    pop_work = ODandLeft.sum(axis = 0)
    ODcorr = ODandLeft@np.diag(1/pop_work)@ODandLeft.T
    return adjmat,ODcorr


def process_adjmat(adjmat:np.ndarray,beta = None):

    ODandLeft = adjmat
    pop_work = ODandLeft.sum(axis = 0)
    if beta is not None:
        ODcorr = ODandLeft@np.diag(beta/pop_work)@ODandLeft.T
    else:
        ODcorr = ODandLeft@np.diag(1/pop_work)@ODandLeft.T

    adjmat +=adjmat.T
    adjmat /=2
    
    return adjmat,ODcorr

def normalize_adjmat(adjmat:np.ndarray,beta = None):
    population:np.ndarray = adjmat.sum(axis = 1)
    _,ODcorr = process_adjmat(adjmat,beta)
    adjmat_norm  = adjmat/adjmat.sum(axis = 1,keepdims=True)
    ODcorr_norm = ODcorr/population.reshape((-1,1))
    return adjmat_norm,ODcorr_norm

def get_direct_distance_adjmat(adjmat,beta = None):
    adjmat_norm,ODcorr_norm = normalize_adjmat(adjmat,beta)
    adjmat_distance  = np.where(adjmat_norm>0,1-np.log(adjmat_norm),0)
    ODcorr_distance  = np.where(ODcorr_norm>0,1-np.log(ODcorr_norm),0)
    return adjmat_distance,ODcorr_distance


def get_adjmat_random_walk(adjmat,sources,dest,beta = None):
    adjmat_norm,ODcorr_norm = normalize_adjmat(adjmat,beta)
    for source in sources:
        adjmat_norm_cp = adjmat_norm.T.copy()
        adjmat_norm_cp[source,:] = 0 
        adjmat_norm_cp[:,dest] = 0 
        colsum = adjmat_norm_cp.sum(axis = 0)
        adjmat_norm_cp = np.diag(-colsum)+adjmat_norm_cp
        adjmat_rw = np.linalg.solve(adjmat_norm_cp,np.zeros(adjmat_norm_cp.shape[0]))

    adjmat_rw = np.linalg.solve(adjmat_norm.T-np.eye(*adjmat_norm.shape),0)



    colsum = ODcorr_norm.sum(axis = 0)
    ODcorr_norm = np.diag(-colsum)+ODcorr_norm
    ODcorr_rw = np.linalg.solve(ODcorr_norm,0)

    return adjmat_rw,ODcorr_rw

def graph_flow2force(G:Union[nx.Graph,np.ndarray],nodelist,sources,dest,beta=None):
    if isinstance(G,np.ndarray):
        weight_mat = G
    elif isinstance(G,(nx.Graph,nx.DiGraph)):
        weight_mat  = np.array(nx.adjacency_matrix(G,nodelist).todense()).astype(float)
    adjmat_rw,ODcorr_rw = get_adjmat_random_walk(weight_mat,sources,dest,beta = beta)


    return adjmat_rw,ODcorr_rw


def graph_flow2dist(G:Union[nx.Graph,np.ndarray],nodelist,beta=None):
    if isinstance(G,np.ndarray):
        weight_mat = G
    elif isinstance(G,(nx.Graph,nx.DiGraph)):
        weight_mat  = np.array(nx.adjacency_matrix(G,nodelist).todense()).astype(float)
    adjmat_distance,ODcorr_distance = get_direct_distance_adjmat(weight_mat,beta)
    inverseG_adj = nx.from_numpy_array(adjmat_distance,create_using=nx.DiGraph)
    inverseG_corr = nx.from_numpy_array(ODcorr_distance,create_using=nx.DiGraph)

    relabel = dict(zip(range(len(nodelist)),nodelist))
    nx.relabel_nodes(inverseG_adj,relabel,copy = False)
    nx.relabel_nodes(inverseG_corr,relabel,copy = False)
    return inverseG_adj,inverseG_corr



def graph_flow2shortest_dist(G:Union[nx.Graph,np.ndarray],nodelist,target,dests:List,beta = None):
    inverseG_adj,inverseG_corr = graph_flow2dist(G,nodelist,beta = beta)
    dist,_ = nx.multi_source_dijkstra(inverseG_adj,sources=dests,target = target)
    dist_corr,_ = nx.multi_source_dijkstra(inverseG_corr,sources=dests,target = target)
    return dist,dist_corr

def graph_flow2multi_wave_indicator(Gold:Union[nx.Graph,np.ndarray],Gnew:Union[nx.Graph,np.ndarray,None]=None,nodelist=None,targets=None,sources=None,sinks=None,beta_old = None,beta_new=None):
    inverseG_adj_old,inverseG_corr_old = graph_flow2dist(Gold,nodelist,beta = beta_old)
    if Gnew is None:
        if beta_new is None:
            inverseG_adj_new,inverseG_corr_new = inverseG_adj_old,inverseG_corr_old
        else:
            inverseG_adj_new,inverseG_corr_new = graph_flow2dist(Gold,nodelist,beta=beta_new)
    else:
        inverseG_adj_new,inverseG_corr_new = graph_flow2dist(Gnew,nodelist,beta=beta_new)

    indicator_new = np.zeros(len(targets))
    indicator_old = np.zeros(len(targets))
    indicator_corr_new = np.zeros(len(targets))
    indicator_corr_old = np.zeros(len(targets))

    indicator = np.zeros((len(targets),len(sinks)))
    for idx,target in enumerate(targets):
        for jdx, sink in enumerate(sinks):
                indicator[idx,jdx] = np.divide(
                                      nx.multi_source_dijkstra(inverseG_adj_new,sources=[sink],target = target)[0],
                                      nx.multi_source_dijkstra(inverseG_adj_old,sources=sources,target = sink)[0])

        
        
    return indicator


