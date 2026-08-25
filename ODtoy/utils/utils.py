from paper_repro_paths import font_dir

from numpy import random
import pandas as pd
import imageio
import matplotlib
import matplotlib.cm
from matplotlib import animation,container,axes,colorbar
from matplotlib import pyplot as plt
import seaborn as sns
from copy import copy,deepcopy
import os
import itertools
import sys
from typing import Union,List
from collections.abc import Iterable
sys.path.append(".")
from .analysis_curves import local_get_second,CompareGridParaScalar,check_get_second,observe_line
import scienceplots

CMAP = copy(matplotlib.cm.get_cmap("inferno"))
CMAP.set_under('gray')
import numpy as np
from scipy.signal import savgol_filter
linestyle_tuple = [
     ('solid', 'solid'),
     ('densely dotted',        (0, (1, 1))),
     ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),

     ('densely dashdotted',    (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]

def init_towns(node_count,node_population=100000,ODtype=None,direction=None):
    if node_count<=26:
        townIDs=dict([(chr(i+ord("A")),i+1) for i in range(node_count)])
    else:
        townIDs=dict([(chr((i%26)+ord("A"))+str(i//26),i+1) for i in range(node_count)])
        
    if ODtype =='central':
        assert direction in ['single','both','inequal']
        if direction =='both':
            town_population = dict(zip(townIDs.keys(),[node_population]*node_count))
            initpopulations = np.ones(node_count)*node_population
            town_area = np.ones(node_count)
        elif direction in ['inequal','single']:
            town_population = dict(zip(townIDs.keys(),[node_population]+[node_population/5]*(node_count-1)))
            initpopulations = np.array([node_population]+[node_population/5]*(node_count-1))
            town_area = np.ones(node_count)
    elif ODtype=='cascade':
        assert direction in ['single','both','inequal','inverse','both-async','both-iso']
        town_population = dict(zip(townIDs.keys(),[node_population]*node_count))
        initpopulations = np.ones(node_count)*node_population
        town_area = np.ones(node_count)
    elif ODtype=='web':
        assert direction in ['city-equal','equal','city']
        if direction in ['city','city-equal']:
            town_population = dict(zip(townIDs.keys(),[node_population]+[node_population/5]*(node_count-1)))
            initpopulations = np.array([node_population]+[node_population/5]*(node_count-1))
            town_area = np.ones(node_count)
        elif direction == 'equal':
            town_population = dict(zip(townIDs.keys(),[node_population]*node_count))
            initpopulations = np.ones(node_count)*node_population
            town_area = np.ones(node_count)
    elif ODtype=='full':
        town_population = dict(zip(townIDs.keys(),[node_population]*node_count))
        initpopulations = np.ones(node_count)*node_population
        town_area = np.ones(node_count)
    elif ODtype=='circle':
        assert direction in ['single','both','both-async','both-iso','both-dist-async','both-dist-iso']
        town_population = dict(zip(townIDs.keys(),[node_population]*node_count))
        initpopulations = np.ones(node_count)*node_population
        town_area = np.ones(node_count)
    elif ODtype=='bi-central':
        assert direction in ['single','both','inequal','inverse']
        assert node_count%2==0,'must be odd'
        town_population = dict(zip(townIDs.keys(),[node_population]+[node_population/5]*(node_count-2)+[node_population]))
        initpopulations = np.array([node_population]+[node_population/5]*(node_count-2)+[node_population])
        town_area = np.ones(node_count)

    elif ODtype=='bi-node':
        assert direction in ['single','both','inequal']
        assert node_count==2
        town_population = dict(zip(townIDs.keys(),[node_population]*node_count))
        initpopulations = np.ones(node_count)*node_population
        town_area = np.ones(node_count)
    elif ODtype =='inequal_node':
        assert direction in ['single','both']
        assert node_count==2
        town_population = dict(zip(townIDs.keys(),[node_population,node_population/5]))
        initpopulations = np.array([node_population,node_population/5])
        town_area = np.ones(node_count)
    elif ODtype =='grid':
        assert direction in ['single','both','both-async','both-iso','both-async-dist']
        town_population = dict(zip(townIDs.keys(),[node_population]*node_count))
        initpopulations = np.ones(node_count)*node_population
        town_area = np.ones(node_count)
    elif ODtype == 'PCFoverFlow':
        assert direction in ['singleFull','singleCBD','singleIndustry','singleIso0','singleIso1','direct','bothFull','bothCBD','bothIndustry','bothIso0','bothIso1',]
        assert node_count == 4 
        town_population = dict(zip(townIDs.keys(),[node_population,node_population,3*node_population,node_population/10]))
        initpopulations = np.array([node_population,node_population,3*node_population,node_population/10])
        town_area = np.array([1,1.5,6,0.2])
    else:
        raise AttributeError(f"ODtype {ODtype} is invalid")
    REF_POP_DENSITY = node_population
    return townIDs,town_population,initpopulations,town_area,REF_POP_DENSITY
def init_i(towns,_from,_to,local_symptom_rate=None,init_town_index = None,count=None):
    probs = [town.array.sum() for town in towns]
    probs = np.array(probs)/sum(probs)
    selectedtowns = []
    assert (init_town_index is None) ^ (count is None)
    if init_town_index is not None:
        for _id in init_town_index:
            randomtown = towns[_id]
            selectedtowns.append((randomtown.townid,randomtown.townname))
            if local_symptom_rate is None:
                randomtown.init_i(_from,_to)
            else:
                assert len(_to)==2,"当local_symptom_rate 非None时，需要有两个可选的statusid"
                rand = random.random()
                if rand<local_symptom_rate:
                    randomtown.init_i(_from,_to[0])
                else:
                    randomtown.init_i(_from,_to[1])
    else: 
        for _ in range(count):
            randomtown = random.choice(towns,p=probs)
            selectedtowns.append((randomtown.townid,randomtown.townname))
            if local_symptom_rate is None:
                randomtown.init_i(_from,_to)
            else:
                assert len(_to)==2,"当local_symptom_rate 非None时，需要有两个可选的statusid"
                rand = random.random()
                if rand<local_symptom_rate:
                    randomtown.init_i(_from,_to[0])
                else:
                    randomtown.init_i(_from,_to[1])
    return selectedtowns



def generate_toy_OD(node_count = 6,ODtype = 'central',direction='both',flow=10000):
    '''parameters:
    
    ODtype                                  direction
    
    central                                 ['single','both','inequal']
    cascade                                 ['single','both','inequal','inverse','both-async','both-iso']
    web                                     ['city-equal','equal','city']
    full                                    -
    circle                                  ['single','both','both-async','both-iso']
    bi-central                              ['single','both','inequal']
    bi-node                                 ['single','both','inequal']
    grid                                    ['single','both','both-async','both-iso','both-async-dist']
    PCFoverFlow                             ['singleFull','singleCBD','singleIndustry','singleIso0','singleIso1','direct','bothFull','bothCBD','bothIndustry','bothIso0','bothIso1',]
    
    '''
    ODmat = np.zeros([node_count,node_count])
    if ODtype =='central':
        assert direction in ['single','both','inequal']
        for f_n  in range(1,node_count):
            ODmat[f_n,0] = flow
            if direction=='both':
                ODmat[0,f_n] = flow
            elif direction=='inequal':
                ODmat[0,f_n] = flow/10
    elif ODtype=='cascade':
        assert direction in ['single','both','inequal','inverse','both-async','both-iso']
        if direction == 'inverse':
            for f_n  in range(1,node_count):
                    ODmat[f_n,f_n-1] = flow
        else:
            for f_n  in range(1,node_count):
                ODmat[f_n-1,f_n] = flow
                if 'both' in direction:
                    ODmat[f_n,f_n-1] = flow
                elif direction=='inequal':
                    ODmat[f_n,f_n-1] = flow/10
        if 'async' in direction or 'iso' in direction:
            ODmat[-1,:]/=100
            ODmat[:,-1]/=100
    elif ODtype=='web':
        assert direction in ['city-equal','equal','city']
        if direction =='city':
            outflow = flow/10
        elif direction == 'equal':
            outflow  = flow
        elif direction == 'city-equal':
            outflow  = flow/10
        for f_n  in range(1,node_count):
            if f_n ==node_count-1:
                ODmat[f_n,1] = outflow
                ODmat[1,f_n] = outflow
            else:
                ODmat[f_n,f_n+1] = outflow
                ODmat[f_n+1,f_n] = outflow
            ODmat[f_n,0] = flow
            if direction =='city':
                ODmat[0,f_n] = int(outflow/node_count-1)
            else:
                ODmat[0,f_n] =int(flow/node_count-1)
    elif ODtype=='full':
        ODmat = (np.ones([node_count,node_count])-np.eye(node_count))*flow
    elif ODtype=='circle':
        assert direction in ['single','both','both-async','both-iso','both-dist-async','both-dist-iso']
        for i in range(node_count):
            ODmat[i,(i+1)%node_count] = flow
            if 'both' in direction:
                ODmat[(i+1)%node_count,i] = flow
            if 'dist' in direction:
                ODmat[-1,0]/=10
                ODmat[0,-1]/=10
        if 'async' in direction or 'iso' in direction:
            ODmat[-1,:]/=100
            ODmat[:,-1]/=100
    elif ODtype=='bi-central':
        assert direction in ['single','both','inequal','inverse']
        assert node_count%2==0,'must be odd'
        for i in range(1,int(node_count/2)):
            ODmat[0,i]=flow/1
            ODmat[i,0]=flow/1
            ODmat[node_count-1,node_count-1-i]=flow/1
            ODmat[node_count-1-i,node_count-1]=flow/1
        if direction == 'single':
            ODmat[0,node_count-1] = flow/100
        elif direction =='inverse':
            ODmat[node_count-1,0] = flow/100
        elif direction =='both':
            ODmat[0,node_count-1] = flow/100
            ODmat[node_count-1,0] = flow/100
        elif direction =='inequal':
            ODmat[0,node_count-1] = flow/100
            ODmat[node_count-1,0] = flow/1000

    elif ODtype=='bi-node':
        assert direction in ['single','both','inequal']
        assert node_count==2
        ODmat[0,1]=flow
        if direction=='both':
            ODmat[1,0]=flow
        elif direction=='inequal':
            ODmat[1,0]=flow/10
    elif ODtype =='inequal_node':
        assert direction in ['single','both']
        assert node_count==2
        ODmat[1,0]=flow
        if direction=='both':
            ODmat[0,1]=flow*5  
    elif ODtype =='grid':
        assert direction in ['single','both','both-async','both-iso','both-async-dist']
        row_number = int(np.sqrt(node_count))
        while row_number>=1:
            if  node_count%row_number ==0:
                break
            row_number -=1
        col_number = node_count//row_number
        assert row_number!=1
        for item in range(node_count):
            col_id,row_id = item%col_number,item//col_number
            if col_id+1<col_number:
                ODmat[item,item+1]=flow
            if row_id+1<row_number:
                ODmat[item,item+col_number] = flow
            
            if 'both' in direction:
                if col_id-1>=0:
                    ODmat[item,item-1]=flow
                if row_id-1>=0:
                    ODmat[item,item-col_number] = flow

        if "dist" in direction:
            ODmat[-1,0] = flow/10
            ODmat[0,-1] = flow/10
        if 'async' in direction or 'iso' in direction:
            ODmat[-1,:] /=100
            ODmat[:,-1] /=100
    elif ODtype == 'PCFoverFlow':
        assert direction in ['singleFull','singleCBD','singleIndustry','singleIso0','singleIso1','direct','bothFull','bothCBD','bothIndustry','bothIso0','bothIso1',]
        assert node_count == 4 
        ODmat = np.zeros([node_count,node_count])
        if direction in ['singleFull','singleCBD','singleIso0','singleIso1']:
            ODmat[0,2] = flow
            ODmat[1,2] = flow
        if direction in ['singleFull','singleIndustry','singleIso0','singleIso1']:
            ODmat[0,3] = flow
            ODmat[1,3] = flow
        if direction== 'singleIso0':
            ODmat[0,:] = 0
        if direction =='singleIso1':
            ODmat[1,:] = 0
        if direction == 'direct':
            ODmat[0,1] = flow
            ODmat[1,0] = flow
        if direction in ['bothFull','bothCBD','bothIso0','bothIso1']:
            ODmat[0,2] = flow
            ODmat[1,2] = flow
            ODmat[2,0] = flow*3
            ODmat[2,1] = flow*3
        if direction in ['bothFull','bothIndustry','bothIso0','bothIso1']:
            ODmat[0,3] = flow
            ODmat[1,3] = flow
            ODmat[3,0] = flow/10
            ODmat[3,1] = flow/10
        if direction== 'bothIso0':
            ODmat[0,:] = 0
            ODmat[:,0] = 0
        if direction =='bothIso1':
            ODmat[1,:] = 0
            ODmat[:,1] = 0

    else:
        raise AttributeError(f"ODtype {ODtype} is invalid")
    return ODmat
def initMapGif(savepath):
    fig,ax = plt.subplots(1,1,figsize=[6,5],dpi = 200,facecolor='white')
    fig.subplots_adjust(right=0.8)
    position = fig.add_axes([0.85, 0.2, 0.015, .6 ])
    norm= matplotlib.colors.Normalize(vmin=0.0, vmax=1.0 ,clip=True)
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm,cmap=CMAP),cax = position)
    ax.axis('off')
    gifwriter = animation.FFMpegWriter()
    gifwriter.setup(fig,savepath,dpi=200)
    return gifwriter,fig,ax,cb



def DrawMapGif(gifwriter:animation.FFMpegFileWriter,ax:axes.Axes,cb:colorbar.Colorbar,data,map_flag,title,vmin = 0.0,vmax = 1.0,log_norm=False):
    ax.clear()
    if log_norm:
        norm = matplotlib.colors.LogNorm(vmin =vmin+1e-6,vmax=vmax,clip=True)
        
    else:
        norm= matplotlib.colors.Normalize(vmin=vmin, vmax=vmax ,clip=True)
    cb.update_normal(matplotlib.cm.ScalarMappable(norm=norm,cmap=CMAP))
    map_flag_copy = deepcopy(map_flag)
    for i in range(len(data)):
        map_flag_copy[map_flag==i+1] = data[i]
    alpha = deepcopy(map_flag)
    alpha[map_flag>=0] = 1.0
    alpha[map_flag<0] = 0.0
    ax.imshow(map_flag_copy,alpha=alpha, norm=norm,cmap = CMAP)
    ax.set_title(title)
    gifwriter.grab_frame()

def saveMapGif_mpa(gifwriter:animation.FFMpegFileWriter):
    gifwriter.finish()

def saveMapPng(data,map_flag,title,savepath,vmin = 0.0,vmax = 1.0,log_norm=False):
    fig,ax = plt.subplots(1,1,figsize=[6,5],dpi = 200,facecolor='white')
    fig.subplots_adjust(right=0.8)
    if log_norm:
        norm = matplotlib.colors.LogNorm(vmin =vmin+1e-6,vmax=vmax,clip=True)
        
    else:
        norm= matplotlib.colors.Normalize(vmin=vmin, vmax=vmax ,clip=True)
    position = fig.add_axes([0.85, 0.2, 0.015, .6 ])
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm,cmap=CMAP),cax = position)
    ax.axis('off')
    map_flag_copy = deepcopy(map_flag)
    for i in range(len(data)):
        map_flag_copy[map_flag==i+1] = data[i]
    alpha = deepcopy(map_flag)
    alpha[map_flag>=0] = 1.0
    alpha[map_flag<0] = 0.0
    ax.imshow(map_flag_copy,alpha=alpha, norm=norm,cmap = CMAP)
    ax.set_title(title)
    fig.savefig(savepath)
    plt.close(fig=fig)

def saveMapGif(pngfilepaths,savefigpath):
    imglist = []
    for file in pngfilepaths:
        imglist.append(imageio.imread(file))
    imageio.mimsave(savefigpath,imglist,duration=0.3)


def PlotSelectedScalar(selectedtowns,Indexinselected,title,filename,logscale=False,extraInfo = None,peak = False,sense_thres = None):
    for i,(s,item) in enumerate(zip(selectedtowns,Indexinselected)):
        _id,_name,_ = s
        l = plt.plot(item,label=_name)
        if sense_thres is not None:
            plt.axhline(y=sense_thres[i],color = l[0].get_color(),alpha = 0.5,linestyle = '-.')
        if peak:
            idx = get_peaks(item)
            plt.scatter(idx,np.array(item)[idx],facecolors = 'white',edgecolors=l[0].get_color())
        
    if extraInfo is not None:
        l = plt.plot(extraInfo,color='black',label='Total')
        if peak:
            idx = get_peaks(extraInfo)
            plt.scatter(idx,np.array(extraInfo)[idx],facecolors = 'white',edgecolors='black')
    plt.legend()
    plt.title(title)
    plt.yscale("log" if logscale else "linear")
    plt.savefig(filename)
    plt.close()

def PlotPhasePlane(Iratio,Sratio,title,filename,):
    plt.plot(Sratio,Iratio)
    plt.title(title)
    plt.xlabel("S ratio")
    plt.ylabel("I ratio")
    plt.ylim(0.02,)
    plt.yscale("log")
    plt.xlim(0,1)
    plt.savefig(filename)
    plt.close()
def PlotSelectedPhasePlane(selectedtowns,Iinselected,Sinselected,title,filename,extraInfo = None):
    for s,_I,_S in zip(selectedtowns,Iinselected,Sinselected):
        _id,_name,_= s
        plt.plot(_S,_I,label=_name)
    if extraInfo is not None:
        Iratio,Sratio = extraInfo
        plt.plot(Sratio,Iratio,color='black',label='Total')
    plt.title(title)
    plt.xlabel("S ratio")
    plt.ylabel("I ratio")
    plt.ylim(1e-7,)
    plt.yscale("log")
    plt.xlim(0,1)
    plt.savefig(filename)
    plt.close()


def PlotParaScalar(multits,paratuple,title,filename,ylog = True,cmap = 'jet',xlabel='days',ylabel='I ratio',para_x = None,x_line = None,y_line=None):
    fig,ax = plt.subplots(1,1,figsize=[6,5],dpi = 200,facecolor='white')
    if 'log' in paratuple[1]:
        norm= matplotlib.colors.LogNorm(vmin=paratuple[2][0], vmax=paratuple[2][1] ,clip=True)
        different_p = np.logspace(np.log10(paratuple[2][0]),np.log10(paratuple[2][1]),len(multits))
    else:
        norm= matplotlib.colors.Normalize(vmin=paratuple[2][0], vmax=paratuple[2][1] ,clip=True)
        different_p = np.linspace(paratuple[2][0],paratuple[2][1],len(multits))
    tocolor = matplotlib.cm.get_cmap(cmap)
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap),ax = ax)
    cb.set_label(paratuple[0])
    if para_x is not None:
        different_p = para_x
    
    if para_x is not None:
        p2ts = {}
        for ts,p in zip(multits,different_p):
            if p not in p2ts:
                p2ts[p] = [ts]
            else:
                p2ts[p].append(ts)
        for p,tsl in p2ts.items():
            tsnp = np.array(tsl)
            ax.plot(np.median(tsnp,axis = 0),color = tocolor(norm(p)))
            ax.fill_between(x=np.arange(tsnp.shape[1]),
                            y1=np.quantile(tsnp,axis = 0,q = 0.25),y2=np.quantile(tsnp,axis = 0,q = 0.75),
                            color = tocolor(norm(p)),alpha = 0.1)

    else:
        for ts,p in zip(multits,different_p):
            ax.plot(ts,color = tocolor(norm(p)),alpha = 0.1)
    if ylog:
        ax.set_yscale('log')
    if x_line is not None:
        if not isinstance(x_line,Iterable):
            x_line = [x_line]
        for xl in x_line:
            ax.axvline(x = xl,color = 'black')
    if y_line is not None:
        if not isinstance(y_line,Iterable):
            y_line = [y_line]
        for yl in y_line:
            ax.axhline(y = yl,color = 'black')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.savefig(filename)
    plt.close(fig)
def saveParaScalar(multits,paratuple,filename,para_x = None):
    if 'log' in paratuple[1]:
        different_p = np.logspace(np.log10(paratuple[2][0]),np.log10(paratuple[2][1]),len(multits))
    else:
        different_p = np.linspace(paratuple[2][0],paratuple[2][1],len(multits))
    if para_x is not None:
        different_p = para_x
    df = pd.DataFrame(np.array(multits).T,columns=different_p)
    df.to_csv(filename)
    
def CompareParaScalar(multits,paratuple,title,filename,ylog = False,cmap = 'jet',xlabel='parameter',ylabel='max I ratio after first wave',time_thres = 500,para_x = None,compare_method = 'second_max',local_get_second_idx = 1):
    fig,ax = plt.subplots(1,1,figsize=[6,5],dpi = 200,facecolor='white')
    if 'log' in paratuple[1]:
        norm= matplotlib.colors.LogNorm(vmin=paratuple[2][0], vmax=paratuple[2][1] ,clip=True)
        different_p = np.logspace(np.log10(paratuple[2][0]),np.log10(paratuple[2][1]),len(multits))
        ax.set_xscale('log')
    else:
        norm= matplotlib.colors.Normalize(vmin=paratuple[2][0], vmax=paratuple[2][1] ,clip=True)
        different_p = np.linspace(paratuple[2][0],paratuple[2][1],len(multits))
    if para_x is not None:
        different_p = para_x
    tocolor = matplotlib.cm.get_cmap(cmap)
    buildy = []
    for item in multits:
        buildy.append(local_get_second(np.array(item),version = 'simulation',time_thres = time_thres,compare_method=compare_method)[local_get_second_idx])
    if para_x is None:
        ax.scatter(different_p,buildy,color = tocolor(norm(different_p)),alpha = 0.1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    else:
        df = pd.DataFrame({xlabel:different_p,ylabel:buildy})
        if filename is None:
            plt.close(fig)
            return df
        else:
            sns.boxplot(df,x=xlabel,y = ylabel,ax = ax)
    
    if ylog:
        ax.set_yscale('log')
    
    ax.set_title(title)
    fig.savefig(filename)
    plt.close(fig)
    if para_x is not None:
        return df

def PlotSelectedMultiScalar(selectedtowns,multiTownFrames,title,filename,logscale=False,agg=1,extrax:dict[str,float] = None,extray:dict[str,float] = None,extraInfo= None,ref = None,legend = True,ylim = None,linewidth = 2):
    import matplotlib.font_manager as fm
    plt.rcParams['svg.fonttype'] = 'none'
    custom_font_dir = font_dir()
    font_files = fm.findSystemFonts(fontpaths=[custom_font_dir])
    for font_file in font_files:
        fm.fontManager.addfont(font_file)
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 7.0
    plt.rcParams["pdf.fonttype"] = 42

    overallzorder = 10
    fig = plt.figure(figsize = (3.6,1.8))
    plt.tight_layout()
    for idx,s in enumerate(selectedtowns):
        _id,_name,_ = s
        item:np.ndarray = multiTownFrames[:,idx,:]
        if agg>1:
            if np.max(item)<1:
                item = np.lib.stride_tricks.sliding_window_view(item,agg,1)[:,::agg,:].mean(axis = 2)
            else:
                item = np.lib.stride_tricks.sliding_window_view(item,agg,1)[:,::agg,:].sum(axis = 2)
        plt.plot(np.median(item,axis = 0),label=_name,zorder = overallzorder,linewidth = linewidth)
        overallzorder -=1
    if extrax is not None:
        for (k,v),(_,ls) in zip(extrax.items(),linestyle_tuple):
            plt.axvline(x = v,label = k,color = 'k',linestyle = ls,linewidth = linewidth)
    if extray is not None:
        for k,v in extray.items():
            plt.axhline(y = v/agg,label = k,linewidth = linewidth)

    if extraInfo is not None:
        if isinstance(extraInfo,dict):
            for (k,v),(_,ls) in zip(extraInfo.items(),linestyle_tuple):
                plt.plot(np.median(v,axis = 0),label=k,linewidth = linewidth)
                plt.fill_between(range(v.shape[1]),np.quantile(v,0.25,axis=0),np.quantile(v,0.75,axis=0),alpha = 0.5)
        else:
            plt.plot(extraInfo,color='black',label='overall',marker = 's',linewidth = linewidth)
    plt.fill_between([],[],[],label = 'Q1-Q3 range',color = 'gray',alpha = 0.3)
    if legend:
        plt.legend()
    plt.yscale("log" if logscale else "linear")

    if ref is not None:
        bx = plt.twinx()
        if isinstance(ref,dict):
            for (k,v),(_,ls) in zip(ref.items(),linestyle_tuple):
                bx.plot(v,color='chocolate',label=k,linestyle = ls,marker = 'o',linewidth = linewidth)
        else:
            bx.plot(ref,color='chocolate',label='reference',marker = 'o',linewidth = linewidth)
        if legend:
            plt.legend(loc = 0)
        plt.yscale("log" if logscale else "linear")
    if ylim is not None:
        plt.ylim(ylim)
        
    plt.ylabel(title)
    if agg ==1:
        xlabel = 'day'
    elif agg == 7:
        xlabel = 'week'
    else:
        xlabel = 'timestep'
    plt.xlabel(xlabel)
    plt.xlim(0,None)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.savefig(filename,transparent = True)
    plt.close()

def PlotCompareSelectedMultiScalar(selectedtowns,multiTownFrames,parainfo,title,filename,logscale=False,agg=1,extrax:dict[str,float] = None,extray:dict[str,float] = None):
    linestyle_tuple = [
     ('solid', 'solid'),
     ('densely dotted',        (0, (1, 1))),
     ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),

     ('densely dashdotted',    (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]
    zorder = multiTownFrames.shape[0]+5
    colors = matplotlib.colors.TABLEAU_COLORS
    for (townidx,(s,c)),(paraidx) in itertools.product(enumerate(zip(selectedtowns,colors)),range(multiTownFrames.shape[0])):
        _id,_name,_ = s
        item:np.ndarray = multiTownFrames[paraidx,:,townidx,:]
        if agg>1:
            if np.max(item)<1:
                item = np.lib.stride_tricks.sliding_window_view(item,agg,1)[:,::agg,:].mean(axis = 2)
            else:
                item = np.lib.stride_tricks.sliding_window_view(item,agg,1)[:,::agg,:].sum(axis = 2)
        if paraidx >0:
            linelabel = str(_id)+"|"+str(parainfo[paraidx])
        else:
            linelabel = str(_id)+"|"+'baseline'
        plt.plot(np.median(item,axis = 0),label=linelabel,linestyle=linestyle_tuple[paraidx][1],color = c,zorder = zorder-paraidx)
        if paraidx == 0:
            plt.fill_between(range(item.shape[1]),np.quantile(item,0.25,axis=0),np.quantile(item,0.75,axis=0),alpha = 0.5,label = linelabel,zorder = 4)
    
    if extrax is not None:
        for k,v in extrax.items():
            plt.axvline(x = v,label = k)
    if extray is not None:
        for k,v in extray.items():
            plt.axhline(y = v,label = k)
    plt.legend(loc='lower right')
    plt.title(title)
    plt.yscale("log" if logscale else "linear")
    plt.savefig(filename)
    plt.close()
def wrapupdateDaysGif(bc1:container.BarContainer,bc2:container.BarContainer,bc3:container.BarContainer,bc1n:container.BarContainer,bc2n:container.BarContainer,bc3n:container.BarContainer):
    def updateDaysGif(count,eb,ib,rb,ebn,ibn,rbn):
        _eb,_ib,_rb,_ebn,_ibn,_rbn = eb[count+1],ib[count+1],rb[count+1],ebn[count+1],ibn[count+1],rbn[count+1]
        for c,b in enumerate(bc1.patches):
            b.set_height(_eb[c])
        for c,b in enumerate(bc2.patches):
            b.set_height(_ib[c])
        for c,b in enumerate(bc3.patches):
            b.set_height(_rb[c])
        for c,b in enumerate(bc1n.patches):
            b.set_height(_ebn[c])
        for c,b in enumerate(bc2n.patches):
            b.set_height(_ibn[c])
        for c,b in enumerate(bc3n.patches):
            b.set_height(_rbn[c])
    return updateDaysGif


def drawDaysGif(E_bydate,I_bydate,R_bydate,E_datenew,I_datenew,R_datenew,log = True):
    fig, ax = plt.subplots(figsize=[10,6],dpi=150)
    bx = ax.twinx()
    bc1 = ax.bar(range(len(E_bydate[0])),E_bydate[0],color = 'green',label='E')
    bc1n = ax.bar(range(len(E_bydate[0])),E_datenew[0],color = 'lime',label='newE')
    bc2 = ax.bar(range(len(E_bydate[0]),len(E_bydate[0])+len(I_bydate[0])),I_bydate[0],color='darkred',label='I')
    bc2n = ax.bar(range(len(E_bydate[0]),len(E_bydate[0])+len(I_bydate[0])),I_datenew[0],color='red',label='newI')
    bc3 = bx.bar(range(len(E_bydate[0])+len(I_bydate[0]),len(E_bydate[0])+len(I_bydate[0])+len(R_bydate[0])),R_bydate[0],color='blue',label='R')
    bc3n = bx.bar(range(len(E_bydate[0])+len(I_bydate[0]),len(E_bydate[0])+len(I_bydate[0])+len(R_bydate[0])),R_datenew[0],color='cyan',label='newR')
    ax.set_ylabel("E and I")
    bx.set_ylabel("R")
    if log:
        ax.set_ylim(1e-6,0.1)
        bx.set_ylim(1e-7,0.01)
        ax.set_yscale('log')
        bx.set_yscale('log')
    else:
        ax.set_ylim(0,0.1)
        bx.set_ylim(0,0.01)
    ax.legend([bc1,bc1n,bc2,bc2n,bc3,bc3n],['E','newE','I','newI','R','newR'])
    update_figure = wrapupdateDaysGif(bc1,bc2,bc3,bc1n,bc2n,bc3n)
    gifwriter = animation.FFMpegWriter()
    with gifwriter.saving(fig, 'output/scalars/DaysInStatus.mp4', dpi=150):
        gifwriter.grab_frame()
        for j in range(len(E_bydate)-1):
            update_figure(j,E_bydate,I_bydate,R_bydate,E_datenew,I_datenew,R_datenew)
            gifwriter.grab_frame()

def process_kwargs(**kwargs):
    flow_ratio_local = kwargs.get("flow_ratio",None)
    
    dryrun ,scalar ,suffix ,beta_density,node_count,ODtype,ODdirection= kwargs.get("dryrun",False),kwargs.get("scalar",False),kwargs.get("suffix",None),kwargs.get("beta_density",False),kwargs.get("node_count",6),kwargs.get("ODtype",None),kwargs.get("ODdirection",None),
    if suffix is None:
        _suffix = ''
    else:
        _suffix = '_'+suffix
    return flow_ratio_local,dryrun ,scalar ,_suffix ,beta_density,node_count,ODtype,ODdirection

def MapInit():
    if not os.path.exists("output/pngs"):
        os.makedirs("output/pngs")
    if not os.path.exists("output/gifs"):
        os.makedirs("output/gifs")
    _vmax = 0.001
    _vmax2 = 1
    map_flag = None
    return _vmax,_vmax2,map_flag


def addSelected(selected:list,towns:list,node_list,initpopulations,town_area):
    for item in node_list:
        selected.append((towns[item].townid,towns[item].townname,initpopulations[item]/town_area[item]))
def getS2Eratio(beta_density,population_work,population,worktown,towns,infectables,statusbeta,alphamat,beta,town_area=None,REF_POP_DENSITY=None,):
    beta_iovern_work = div_consider_zero(np.dot(worktown[:,:,infectables],statusbeta[infectables]),worktown.sum(axis = 2))
    beta_iovern_home = div_consider_zero(np.dot(towns[:,:,infectables],statusbeta[infectables]),towns.sum(axis = 2))
    beta_iovern_work = np.repeat(np.expand_dims(beta_iovern_work,1),len(beta),1)*beta.reshape(1,-1,1)
    beta_iovern_home = np.repeat(np.expand_dims(beta_iovern_home,1),len(beta),1)*beta.reshape(1,-1,1)
    
    
    if beta_density=='linear':
        assert town_area is not None,"town area is required"
        assert REF_POP_DENSITY is not None,"REF_POP_DENSITY is required"
        alphamat_work  = np.expand_dims(alphamat,0)*0.8*(population_work/town_area/REF_POP_DENSITY).reshape(-1,1,1)
        alphamat_home  = np.expand_dims(alphamat,0)*0.2*(population/town_area/REF_POP_DENSITY).reshape(-1,1,1)

    elif beta_density == 'log':
        alphamat_work  = np.expand_dims(alphamat,0)*1.0*np.log2(1+population_work/town_area/REF_POP_DENSITY).reshape(-1,1,1)
        alphamat_home  = np.expand_dims(alphamat,0)*0.0*np.log2(1+population/town_area/REF_POP_DENSITY).reshape(-1,1,1)
    elif beta_density == 'cfg':
        alphamat_work  = alphamat * 0.8
        alphamat_home  = alphamat * 0.2
    elif beta_density == 'cfgFull':
        alphamat_work  = alphamat * 1
        alphamat_home  = alphamat * 0
    elif beta_density == False or  beta_density=='False':
        alphamat_work  = np.expand_dims(alphamat,0)*0.8
        alphamat_home  = np.expand_dims(alphamat,0)*0.2
    
    notinfect_work = np.prod(np.power(1-beta_iovern_work,alphamat_work),axis = 2)
    notinfect_home = np.prod(np.power(1-beta_iovern_home,alphamat_home),axis = 2)

    return notinfect_work,notinfect_home
   
def getS2Eratio_v0(beta_density,population_work,population,worktown,towns,infectables,statusbeta,alphamat,beta,town_area=None,REF_POP_DENSITY=None,):

    if beta_density=='log':
        assert town_area is not None,"town area is required"
        assert REF_POP_DENSITY is not None,"REF_POP_DENSITY is required"


        S2Erate_work =np.repeat(np.expand_dims(np.where(population_work.reshape(-1,1)>0,
                            ((np.matmul(np.dot(worktown[:,:,infectables],statusbeta[infectables]),alphamat*0.8))*beta.reshape(1,-1)*(np.log2(population_work/town_area/REF_POP_DENSITY+1)/population).reshape(-1,1)),0),0),len(population),0)
        S2Erate_home =np.where(population.reshape(-1,1)>0,
                ((np.matmul(np.dot(towns[:,:,infectables],statusbeta[infectables]),alphamat*0.2))*beta.reshape(1,-1)*(np.log2(population/town_area/REF_POP_DENSITY+1)/population).reshape(-1,1)),0)
    elif beta_density=='linear':
        assert town_area is not None,"town area is required"
        assert REF_POP_DENSITY is not None,"REF_POP_DENSITY is required"
        S2Erate_work =np.repeat(np.expand_dims(np.where(population_work.reshape(-1,1)>0,
                            ((np.matmul(np.dot(worktown[:,:,infectables],statusbeta[infectables]),alphamat*0.8))*beta.reshape(1,-1)/(town_area*REF_POP_DENSITY).reshape(-1,1)),0),0),len(population),0)
        S2Erate_home =np.where(population.reshape(-1,1)>0,
                ((np.matmul(np.dot(towns[:,:,infectables],statusbeta[infectables]),alphamat*0.2))*beta.reshape(1,-1)/(town_area*REF_POP_DENSITY).reshape(-1,1)),0)
    elif beta_density == False:
        S2Erate_work =np.repeat(
                        np.expand_dims(
                            np.where(population_work.reshape(-1,1)>0,
                                    (np.matmul(
                                        np.dot(worktown[:,:,infectables],statusbeta[infectables]),
                                        alphamat*0.8)
                                    ) * beta.reshape(1,-1) / (population_work).reshape(-1,1)
                                    ,0)
                                    ,0)
                                ,len(population),0)
        S2Erate_home =np.where(population.reshape(-1,1)>0,
                        ((np.matmul(np.dot(towns[:,:,infectables],statusbeta[infectables]),alphamat*0.2))*beta.reshape(1,-1)/population.reshape(-1,1)),0)
    else:
        raise NotImplementedError("Specify log or linear or BOOL False")
    return S2Erate_work,S2Erate_home

def getS2Eratio_v0f(beta_density,population_work,population,worktown,towns,infectables,statusbeta,alphamat,beta,town_area=None,REF_POP_DENSITY=None,):

    if beta_density=='log':
        assert town_area is not None,"town area is required"
        assert REF_POP_DENSITY is not None,"REF_POP_DENSITY is required"


        S2Erate_work =np.repeat(np.expand_dims(np.where(population_work.reshape(-1,1)>0,
                            ((np.matmul(np.dot(worktown[:,:,infectables],statusbeta[infectables]),alphamat*0.8))*beta.reshape(1,-1)*(np.log2(population_work/town_area/REF_POP_DENSITY+1)/population).reshape(-1,1)),0),0),len(population),0)
        S2Erate_home =np.where(population.reshape(-1,1)>0,
                ((np.matmul(np.dot(towns[:,:,infectables],statusbeta[infectables]),alphamat*0.2))*beta.reshape(1,-1)*(np.log2(population/town_area/REF_POP_DENSITY+1)/population).reshape(-1,1)),0)
    elif beta_density=='linear':
        assert town_area is not None,"town area is required"
        assert REF_POP_DENSITY is not None,"REF_POP_DENSITY is required"
        S2Erate_work =np.repeat(np.expand_dims(np.where(population_work.reshape(-1,1)>0,
                            ((np.matmul(np.dot(worktown[:,:,infectables],statusbeta[infectables]),alphamat*0.8))*beta.reshape(1,-1)/(town_area*REF_POP_DENSITY).reshape(-1,1)),0),0),len(population),0)
        S2Erate_home =np.where(population.reshape(-1,1)>0,
                ((np.matmul(np.dot(towns[:,:,infectables],statusbeta[infectables]),alphamat*0.2))*beta.reshape(1,-1)/(town_area*REF_POP_DENSITY).reshape(-1,1)),0)
    elif beta_density == False:
        S2Erate_work =np.repeat(
                        np.expand_dims(
                                np.matmul(
                                    div_consider_zero(np.dot(worktown[:,:,infectables],statusbeta[infectables])
                                                        ,worktown.sum(axis = 2)),
                                    alphamat*0.8)
                                * beta.reshape(1,-1) 
                                ,0)
                                ,len(population),0)
        S2Erate_home =np.matmul(
                        div_consider_zero(np.dot(towns[:,:,infectables],statusbeta[infectables])
                                            ,towns.sum(axis = 2)),
                        alphamat*0.2) * beta.reshape(1,-1) 
    elif beta_density =='cfg':
        S2Erate_work =np.repeat(
                        np.expand_dims(
                                np.matmul(
                                    div_consider_zero(np.dot(worktown[:,:,infectables],statusbeta[infectables])
                                                        ,worktown.sum(axis = 2)),
                                    alphamat*0.8)
                                * beta 
                                ,0)
                                ,len(population),0)
        S2Erate_home =np.matmul(
                        div_consider_zero(np.dot(towns[:,:,infectables],statusbeta[infectables])
                                            ,towns.sum(axis = 2)),
                        alphamat*0.2) * beta 
    else:
        raise NotImplementedError("Specify log or linear or BOOL False")
    return S2Erate_work,S2Erate_home
def div_consider_zero(numerator:Union[np.ndarray,float],denominator:Union[np.ndarray,float],_allow_nonzero_over_zero = False):
    """
    0/0: nan -> 0
    if _allow_nonezero_over_zero:
        5/0: inf
        -5/0: -inf
    else:
        warn
    """

    numerator,denominator = np.broadcast_arrays(numerator,denominator)
    denominatoriszero = denominator==0
    if (not _allow_nonzero_over_zero) and (numerator[denominatoriszero]!=0).any():
        raise Exception("in div_consider_zero, numerator[denominatoriszero] includes 0, \nwhile nonzero/0 is not allowed here")
    broadcast_shape = np.broadcast_shapes(np.array(numerator).shape,np.array(denominator).shape)
    result = np.zeros(broadcast_shape)
    np.divide(numerator,denominator,out = result,where = ~denominatoriszero)
    return result
def get_peaks(ts:np.ndarray,smooth = True):
    if smooth:
        ts = np.convolve(ts,np.hanning(7),'same')/(np.hanning(7).sum())
    peaks = []
    tsmd = np.median(ts)
    for idx in range(3,len(ts)-2):
        if ts[idx]>ts[idx-1] and ts[idx]>ts[idx+1] and ts[idx]>tsmd:
            peaks.append(idx)
    return peaks


def get_START_WEEK_period(result_frame,target:List[int],population=None ):
    if len(result_frame.shape)==3:
        result_frame = np.median(result_frame,axis = 0)
    result_frame_target = result_frame[:,target].sum(axis = 1)
    result_frame_others = result_frame.sum(axis =1)-result_frame_target
    if population is not None:
        population_target = population[target].sum()
        population_others = population.sum()-population_target

        result_frame_target /= population_target
        result_frame_others /= population_others
    else:
        result_frame_others = result_frame_others/(result_frame.shape[1]-len(target))
        result_frame_target =result_frame_target/len(target)
    idx = np.arange(len(result_frame_others))
    
    if len(idx[result_frame_target>result_frame_others])==0:
        plt.plot(result_frame_others)
        plt.savefig("tmp/tmp.jpg")
        goingdown = np.argmin(np.diff(result_frame_others))
        temp = (np.diff(result_frame_others)[1:]*np.diff(result_frame_others)[:-1])[goingdown:]
        tempidx = np.arange(len(temp))
        argmin = tempidx[temp<0][0]+goingdown
        plt.scatter(argmin,result_frame_others[argmin])
        plt.savefig("tmp/tmp.jpg")
        print("取其他下降最快的地方")
        return argmin
    else:
        print("取交叉点")
        return idx[result_frame_target>result_frame_others][0]
        
        
