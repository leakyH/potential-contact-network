from seir.AgePopulation import AgePopulation
from ODtools.area import Town,buildnp
from utils.utils import generate_toy_OD,process_kwargs,init_towns,addSelected,PlotSelectedScalar,PlotSelectedMultiScalar,PlotSelectedPhasePlane,getS2Eratio_v0f,getS2Eratio,init_i,get_peaks,PlotParaScalar,CompareParaScalar,saveParaScalar,CompareGridParaScalar,PlotCompareSelectedMultiScalar
from utils.visualize import drawNetwork_color_by_sequence,draw_flow_pcf_mat
from cfg._toy_configs import *
from utils.analysis_curves import local_get_second
from differentR import getR

from scipy import stats
import numpy as np
import itertools
from tqdm import tqdm,trange
import logging
import os
import pickle as pkl
from matplotlib import pyplot as plt
import seaborn as sns
import multiprocessing as mp
from graphs.graph_common.figure_output import configure_output_root, graph_output_dir, graph_output_path

import line_profiler as lp
import psutil
import sys
import gc
logging.basicConfig(filename="checktoyOD.log",filemode="w")
weekcount= 52*2
configure_output_root(default_root="graphs/Fig2/artifacts")

STATUS  = ["S","E","I","R",'P']
status2id = {}
for idx,s in enumerate(STATUS):
    status2id[s]=idx
moveable = [0,1,2,3,4]
E_beta = 1/10 
I_beta = 0.25 
statusbeta = np.array([0,E_beta,I_beta,0,0])

plotsaveformat = 'svg'

E_days_prob = [1/4]*4
I_days_prob = [1/8]*8

R_days_prob = getR('R3mGeom')
START_AT = [0]*100+[1]*100+[2]*1000+[3]*5

def ODseir_simple(*args,**kwargs):
    np.random.seed()
    flow_ratio_local,dryrun ,scalar ,_suffix, beta_density,node_count,ODtype,ODdirection  = process_kwargs(**kwargs)
    if flow_ratio_local is None:
        flow_ratio_local = flow_ratio
    alpha_ratio = kwargs.get("alpha_ratio",1.0)
    reopen_date = kwargs.get("REOPEN_WEEK",-1)*7 if kwargs.get("REOPEN_WEEK",-1)>0 else kwargs.get("REOPEN_DATE",-1)
    epi_date = kwargs.get("EPI_WEEK",-1)*7 if kwargs.get("EPI_WEEK",-1)>0 else kwargs.get("EPI_DATE",-1)
    
    
    
    
    
    print(_suffix,alpha_ratio,flush = True)
    beta = np.array(agebeta)
    infectables = [status2id["E"],status2id["I"]]
    result_frame = []
    daily_new = []
    towns = []
    townIDs,_,initpopulations,town_area,REF_POP_DENSITY=init_towns(node_count,10_0000,ODtype,ODdirection)
    REF_POP_DENSITY = initpopulations[0]/town_area[0]
    E_date = np.zeros([node_count,len(ages),5],int)
    I_date = np.zeros([node_count,len(ages),12],int)
    R_date = np.zeros([node_count,len(ages),365],int)
    for town,townid in townIDs.items():
        if town !="-1":
            population = initpopulations[townid-1]
            townseir = AgePopulation(population,len(ages),len(STATUS),ageprop)
            townseir.array = townseir.array
            towns.append(Town(town,townid,townseir))
    selected = init_i(towns,status2id["S"],status2id["I"],init_town_index=START_AT)
    for i,_ in selected:
        I_date[i-1,1,0] +=1
    
    sampleOD :np.ndarray =generate_toy_OD(node_count=node_count,ODtype=ODtype,direction=ODdirection,flow=1_0000)*flow_ratio_local
    selected = []
    if scalar:
        addSelected(selected,towns,[0,1,2,3],initpopulations,town_area)
    else:
        addSelected(selected,towns,[1],initpopulations,town_area)




    IinSelected = [[] for _ in selected]
    IinAll = []
    IratioinAll = []
    SinAll = []
    SratioinAll = []
    SinSelected = [[] for _ in selected]
    IratioinSelected = [[] for _ in selected]
    SratioinSelected = [[] for _ in selected]
    S2EinSelected = [[] for _ in selected]
    S2EratioinSelected = [[] for _ in selected]
    I2RinSelected = [[] for _ in selected]
    R2SinSelected = [[] for _ in selected]
    I2PinSelected = [[] for _ in selected]
    populationinSelected = [[] for _ in selected]
    E_bydate = []
    I_bydate = []
    R_bydate = []
    E_bydatenew = []
    I_bydatenew = []
    R_bydatenew = []
    
    towns:np.ndarray = buildnp(towns,len(ages),len(STATUS)).astype(int)
    
    
    

    population = initpopulations.copy()

    for week, day, in itertools.product(range(weekcount), range(7)):
        if dryrun:
            print(f"Week{week}Day{day} S:{towns[:,:,status2id['S']].sum():.2f} E:{towns[:,:,status2id['E']].sum():.2f} I:{towns[:,:,status2id['I']].sum():.2f} R:{towns[:,:,status2id['R']].sum():.2f} P:{towns[:,:,status2id['P']].sum():.2f}")            
            continue
        result_frame.append(towns[:,:,status2id['I']].sum(axis = 1))      
        
        for s,item,rt,sitem,srt,pop in zip(selected,IinSelected,IratioinSelected,SinSelected,SratioinSelected,populationinSelected):
            _id = s[0]
            item.append(towns[_id-1,:,[status2id['I']]].sum())
            sitem.append(towns[_id-1,:,[status2id['S']]].sum())
            rt.append(towns[_id-1,:,[status2id['I']]].sum()/towns[_id-1,:,:].sum())
            srt.append(towns[_id-1,:,[status2id['S']]].sum()/towns[_id-1,:,:].sum())
        IratioinAll.append(towns[:-1,:,[status2id['I']]].sum()/population[:-1].sum())
        IinAll.append(towns[:,:,[status2id['I']]].sum())
        SratioinAll.append(towns[:,:,[status2id['S']]].sum()/population.sum())
        SinAll.append(towns[:,:,[status2id['S']]].sum())
        
        for s,pop in zip(selected,populationinSelected):
            _id = s[0]
            pop.append(population[_id-1])


        if week*7+day>=reopen_date :
            nowOD = sampleOD
            global_infect_decline = 1
        else:
            if week*7+day>=epi_date:
                nowOD = sampleOD*0
                global_infect_decline = 1
            else:
                global_infect_decline = 1
                nowOD = sampleOD


        nowOsum = nowOD.sum(axis=1)
        ODandLeft = (np.diag(population-nowOsum)+nowOD)/np.expand_dims(population,1)  
        if (nowOsum>population).any():
            items= []
            for item,(n,p) in enumerate(zip(nowOsum,population)):
                if n>p:
                    items.append(item)
            logging.error(f"suffix{_suffix}-{week}-{day} population_scale:{1} idx:{items}")
        

        worktown = np.zeros_like(towns)
        Sh2Sw = np.zeros([node_count,node_count,len(ages)],int)
        for _t,_a,_s in itertools.product(range(towns.shape[0]),range(towns.shape[1]),range(towns.shape[2])):
            if towns[_t,_a,_s]>0:
                if _s == status2id["S"]:
                    home2work = np.random.multinomial(towns[_t,_a,_s],ODandLeft[_t,:])
                    Sh2Sw[_t,:,_a]+=home2work
                    worktown[:,_a,_s] += home2work
                else:
                    worktown[:,_a,_s] += np.random.multinomial(towns[_t,_a,_s],ODandLeft[_t,:])

        population_work = worktown.sum(axis = (1,2))
        if kwargs['s2eversion'] == 'v0':
            S2Erate_work, S2Erate_home = getS2Eratio_v0(beta_density,population_work,population,worktown,towns,infectables,statusbeta,alphamat,beta*alpha_ratio,town_area,REF_POP_DENSITY)
            S2Erate =S2Erate_work+ np.expand_dims(S2Erate_home,1)
            S2Erate[S2Erate>1]=1
        elif  kwargs['s2eversion'] =='v0f':
            S2Erate_work, S2Erate_home = getS2Eratio_v0f(beta_density,population_work,population,worktown,towns,infectables,statusbeta,alphamat,beta*alpha_ratio,town_area,REF_POP_DENSITY)
            S2Erate =S2Erate_work+ np.expand_dims(S2Erate_home,1)
        elif kwargs['s2eversion'] == 'v1':
            noS2Erate_work, noS2Erate_home = getS2Eratio(beta_density,population_work,population,worktown,towns,infectables,statusbeta,alphamat*alpha_ratio*global_infect_decline,beta,town_area,REF_POP_DENSITY)
            S2Erate =1-(noS2Erate_work* np.expand_dims(noS2Erate_home,1))
        R2S = R_date[:,:,-1]
        S2E = np.random.binomial(Sh2Sw,S2Erate).sum(axis = 1)
        E2I = E_date[:,:,-1]
        I2P = np.random.binomial(I_date[:,:,-1],0.5)
        I2R = I_date[:,:,-1]-I2P
        deltaS = R2S-S2E
        deltaE = S2E-E2I
        deltaI = E2I-I2R-I2P
        deltaR = I2R-R2S
        deltaP = I2P
        daily_new.append(S2E.sum(axis = 1))  

        I_date_new = np.zeros_like(I_date)
        E_date_new = np.zeros_like(E_date)
        R_date_new = np.zeros_like(R_date)
        for idx ,_age in itertools.product(range(node_count),range(len(ages))):
            if E2I[idx,_age]>0:
                I_days_index_count = np.random.multinomial(E2I[idx,_age],I_days_prob)
                I_date_new[idx,_age,len(I_days_prob)-1::-1] += I_days_index_count
            if S2E[idx,_age]>0:
                E_days_index_count = np.random.multinomial(S2E[idx,_age],E_days_prob)
                E_date_new[idx,_age,len(E_days_prob)-1::-1] += E_days_index_count
            if I2R[idx,_age]>0:
                R_days_index_count = np.random.multinomial(I2R[idx,_age],R_days_prob)
                R_date_new[idx,_age,len(R_days_prob)-1::-1] += R_days_index_count
        
        I_date[:,:,1:] = I_date[:,:,0:-1]
        I_date[:,:,0] = 0
        E_date[:,:,1:] = E_date[:,:,0:-1]
        E_date[:,:,0] = 0
        R_date[:,:,1:] = R_date[:,:,0:-1]
        R_date[:,:,0] = 0
        I_date+=I_date_new
        E_date+=E_date_new
        R_date+=R_date_new
            
        

        for s,s2e_s,s2e_rs,i2r_s,r2s_s,i2p_s in zip(selected,S2EinSelected,S2EratioinSelected,I2RinSelected,R2SinSelected,I2PinSelected):
            _id = s[0]
            s2e_s.append(S2E[_id-1,:].sum())
            s2e_rs.append(S2Erate[_id-1,_id-1,:].mean())
            i2r_s.append(I2R[_id-1,:].sum())
            r2s_s.append(R2S[_id-1,:].sum())
            i2p_s.append(I2P[_id-1,:].sum())


        towns[:,:,status2id['S']]+=deltaS
        towns[:,:,status2id['E']]+=deltaE
        towns[:,:,status2id['I']]+=deltaI
        towns[:,:,status2id['R']]+=deltaR
        towns[:,:,status2id['P']]+=deltaP


        E_bydate.append(E_date.sum(axis = (0,1))/population.sum())
        I_bydate.append(I_date.sum(axis = (0,1))/population.sum())
        R_bydate.append(R_date.sum(axis = (0,1))/population.sum())
        E_bydatenew.append(E_date_new.sum(axis = (0,1))/population.sum())
        I_bydatenew.append(I_date_new.sum(axis = (0,1))/population.sum())
        R_bydatenew.append(R_date_new.sum(axis = (0,1))/population.sum())
        



    if scalar:
        plotLog = False


        PlotSelectedScalar(selected,IinSelected,"I_count",f"toyoutput/scalars/I_count_init{_suffix}.png",plotLog )    
        PlotSelectedScalar(selected,IratioinSelected,"I_ratio",f"toyoutput/scalars/I_ratio_init{_suffix}.png",plotLog )    
        PlotSelectedScalar(selected,SinSelected,"S_count",f"toyoutput/scalars/S_count_init{_suffix}.png",plotLog)    
        PlotSelectedScalar(selected,SratioinSelected,"S_ratio",f"toyoutput/scalars/S_ratio_init{_suffix}.png",plotLog)
        PlotSelectedScalar(selected,S2EinSelected,"S2E",f"toyoutput/scalars/S2E{_suffix}.png",plotLog)
        PlotSelectedScalar(selected,I2RinSelected,"I2R",f"toyoutput/scalars/I2R{_suffix}.png",plotLog)
        PlotSelectedScalar(selected,R2SinSelected,"R2S",f"toyoutput/scalars/R2S{_suffix}.png",plotLog)    
        PlotSelectedScalar(selected,S2EratioinSelected,"S2E Ratio",f"toyoutput/scalars/S2Eratio{_suffix}.png",False) 
        PlotSelectedScalar(selected,I2PinSelected,"I2P",f"toyoutput/scalars/I2P{_suffix}.png",plotLog)    
        PlotSelectedPhasePlane(selected,IratioinSelected,SratioinSelected,"Phase Plane",f"toyoutput/scalars/SIphase{_suffix}.png" )   

    return np.array(result_frame),daily_new,selected,IinSelected,IratioinSelected,S2EinSelected,SinSelected,SratioinSelected,START_AT

def compare_para_multi(count_each = 20,para=None,*args,**kwargs):
    paraname = para[0]
    POOLSIZE = 50

    def error_handler(someerror):
        print(someerror,flush = True)
        return 
    allresult = []
    for p in para[1:]:
        pool = mp.Pool(POOLSIZE)
        expbatch = []
        for j in range(count_each):
            kwcp = kwargs.copy()
            kwcp[paraname]=p
            kwcp['suffix'] = str(p)+"_"+str(j)
            expbatch.append((ODseir_simple,args,kwcp))
        
        mif = psutil.Process(os.getppid()).memory_full_info()
        print(f"在计算之前,主进程 {os.getpid()} ",mif.rss/1024/1024/1024,mif.pss/1024/1024/1024,mif.uss/1024/1024/1024,flush = True)
        list_of_frames = pool.starmap_async(apply_args_and_kwargs,expbatch,error_callback=error_handler).get()
        pool.close()
        pool.join()
        allresult.append(list_of_frames)
    with open(f"toyoutput/pkls/compare/compare_{paraname}.pkl",'wb') as f:
        pkl.dump(allresult,f)
    with open(f"toyoutput/pkls/compare/compare_{paraname}.txt",'w') as f:
        for p in para[1:]:
            f.write(str(p)+'\n')

    multiIratioinSelected = []
    for p,list_of_frames in zip(para[1:],allresult):
        partialresult = []
        for item in list_of_frames:
            totalI,daily_new,selected,IinSelected,IratioinSelected,SinSelected,SratioinSelected,_ = item
            partialresult.append(IratioinSelected)
        multiIratioinSelected.append(partialresult)
    multiIratioinSelected = np.array(multiIratioinSelected)
    PlotCompareSelectedMultiScalar(selected,multiIratioinSelected,para[1:],title=r"I_ratio for different"+paraname,filename="toyoutput/scalars/compare/I_ratio_multi.png",logscale=True)

def ODseir_main(*args, **kwargs):
    kwargs['s2eversion']='v1'

    print(args,kwargs)
    result_frame = ODseir_simple(*args,**kwargs)
    if not os.path.exists("tmp/"):
        os.mkdir('tmp')
    suffix = kwargs.get("suffix",None)
    if suffix is None:
        np.save("tmp/result_frame.npy",result_frame[0])
    else:
        np.save(f"tmp/result_frame_{suffix}.npy",result_frame[0])
    return result_frame
def apply_args_and_kwargs(fn, args, kwargs):
    return fn(*args, **kwargs)
def ODseri_grid(replicate = 5,para1=None,para2=None,*args,**kwargs):
    assert ~kwargs['scalar'], "scalar should be False"
    kwargs['s2eversion']='v1'
    _,town_population,_,_,_ = init_towns(kwargs['node_count'],node_population=10_0000,ODtype=kwargs['ODtype'],direction=kwargs['ODdirection'])
    ODtype = kwargs.get('ODtype')
    ODdirection = kwargs.get('ODdirection')
    drawNetwork_color_by_sequence([(k,v) for k,v in town_population.items()],generate_toy_OD(node_count = kwargs['node_count'],ODtype = kwargs['ODtype'],direction=kwargs['ODdirection'],flow=10000),filename = f'tmp/{ODtype}_{ODdirection}.jpg',ODtype = kwargs['ODtype'])
    if kwargs.get('beta_density',False) or "async" in kwargs['ODdirection']:
        if kwargs.get("REOPEN_WEEK",-1)>0:
            subdir = f"{kwargs['ODtype']}_{kwargs['beta_density']}_SW{kwargs['REOPEN_WEEK']}/{kwargs['ODdirection']}/{kwargs['node_count']}"
        else:
            subdir = f"{kwargs['ODtype']}_{kwargs['beta_density']}/{kwargs['ODdirection']}/{kwargs['node_count']}"
    else:
        subdir = f"{kwargs['ODtype']}/{kwargs['ODdirection']}/{kwargs['node_count']}"
    recompute = True
    if recompute:
        print(f'Recomputing outputs in {subdir}')
    if 'log' in para1[1]:
        different_p1 = np.logspace(np.log10(para1[2][0]),np.log10(para1[2][1]),para1[3])
    else:
        different_p1 = np.linspace(para1[2][0],para1[2][1],para1[3])
    if 'log' in para2[1]:
        different_p2 = np.logspace(np.log10(para2[2][0]),np.log10(para2[2][1]),para2[3])
    else:
        different_p2 = np.linspace(para2[2][0],para2[2][1],para2[3])
    keyword1 = para1[0]
    keyword2 = para2[0]

    if recompute:
        POOLSIZE = 50
        pool = mp.Pool(POOLSIZE)
        expbatch = []
        list_of_frames = []
        for i,p1 in tqdm(enumerate(different_p1)): 
            for j,p2 in enumerate(different_p2):
                kwcp = kwargs.copy()
                kwcp['suffix'] = str(i)+'_'+str(j)
                kwcp[keyword1] = p1
                kwcp[keyword2] = p2
                expbatch.append((ODseir_simple,args,kwcp))
        print(len(expbatch))
        def error_handler(someerror):
            print(someerror,flush = True)
            return 
        mif = psutil.Process(os.getppid()).memory_full_info()
        print(f"在计算之前,主进程 {os.getpid()} ",mif.rss/1024/1024/1024,mif.pss/1024/1024/1024,mif.uss/1024/1024/1024,flush = True)

        list_of_frames = pool.starmap_async(apply_args_and_kwargs,expbatch,error_callback=error_handler).get()
        pool.close()
        pool.join()

        if not os.path.exists("toyoutput/scalars/{}".format(subdir)):
            os.makedirs("toyoutput/scalars/{}".format(subdir))
        if not os.path.exists("toyoutput/pkls/{}".format(subdir)):
            os.makedirs("toyoutput/pkls/{}".format(subdir))
        multitotalI = []
        multinewI = []
        grid_IinSelected = []
        for item in list_of_frames:
            totalI,daily_new,selected,IinSelected,IratioinSelected,SinSelected,SratioinSelected,_ = item
            multitotalI.append(totalI)
            multinewI.append(daily_new)
            grid_IinSelected.append(IinSelected)
        with open('toyoutput/pkls/{}/multitotalI.pkl'.format(subdir),'wb')as f:
            pkl.dump((multitotalI,different_p1,different_p2,para1,para2),f)
            print('toyoutput/pkls/{}/multitotalI.pkl'.format(subdir))
        with open('toyoutput/pkls/{}/multinewI.pkl'.format(subdir),'wb')as f:
            pkl.dump((multinewI,different_p1,different_p2,para1,para2),f)
            print('toyoutput/pkls/{}/multinewI.pkl'.format(subdir))
        with open('toyoutput/pkls/{}/gridIinSelected.pkl'.format(subdir),'wb')as f:
            pkl.dump((selected,grid_IinSelected,different_p1,different_p2,para1,para2),f)
            print('toyoutput/pkls/{}/gridIinSelected.pkl'.format(subdir))
    else:
        if os.path.exists('toyoutput/pkls/{}/multitotalI.pkl'.format(subdir)):
            print('toyoutput/pkls/{}/multitotalI.pkl'.format(subdir))
            with open('toyoutput/pkls/{}/multitotalI.pkl'.format(subdir),'rb') as f:
                item = pkl.load(f)
                multitotalI,different_p1,different_p2,para1,para2 = item
        else:
            return
        if os.path.exists('toyoutput/pkls/{}/multinewI.pkl'.format(subdir)):
            print('toyoutput/pkls/{}/multinewI.pkl'.format(subdir))
            with open('toyoutput/pkls/{}/multinewI.pkl'.format(subdir),'rb') as f:
                item = pkl.load(f)
                multinewI,different_p1,different_p2,para1,para2 = item
        if os.path.exists('toyoutput/pkls/{}/gridIinSelected.pkl'.format(subdir)):
            with open('toyoutput/pkls/{}/gridIinSelected.pkl'.format(subdir),'rb')as f:
                selected,grid_IinSelected,different_p1,different_p2,para1,para2 = pkl.load(f)
        else:
            return

    
    CompareGridParaScalar(np.array(multitotalI).sum(axis = 1),different_p1,different_p2,"second wave size",'toyoutput/scalars/{}/I_ratio_second_max_grid.jpg'.format(subdir),cmap = 'jet',xlabel=keyword1,ylabel=keyword2,time_thres = 500,xlog = True if 'log' in para1[1] else False,ylog = True if 'log' in para2[1] else False,compare_method='sum_after_first',version = 'v1')
    CompareGridParaScalar(multinewI,different_p1,different_p2,"second wave size",'toyoutput/scalars/{}/newI_ratio_second_max_grid.jpg'.format(subdir),cmap = 'jet',xlabel=keyword1,ylabel=keyword2,time_thres = 500,xlog = True if 'log' in para1[1] else False,ylog = True if 'log' in para2[1] else False,compare_method='sum_after_first',version = 'v1')
    count = gc.collect()
    print(f"函数返回前，gc{count}")
    mif = psutil.Process(os.getppid()).memory_full_info()
    print(f"第一次清理后,主进程 {os.getpid()} ",mif.rss/1024/1024/1024,mif.pss/1024/1024/1024,mif.uss/1024/1024/1024)

    

def ODseir_multi(count = 20,paratuple=None,*args, **kwargs):
    _,town_population,initpopulations,town_area,_ = init_towns(kwargs['node_count'],node_population=100_0000,ODtype=kwargs['ODtype'],direction=kwargs['ODdirection'])
    ODtype = kwargs.get('ODtype')
    ODdirection = kwargs.get('ODdirection')
    kwargs['s2eversion']='v1'
    recompute = True
    same_para = paratuple is None
    import multiprocessing as mp
    POOLSIZE = 50
    expbatch = []
    para_x  = []
    reopen_date = kwargs.get("REOPEN_WEEK",-1)*7 if kwargs.get("REOPEN_WEEK",-1)>0 else kwargs.get("REOPEN_DATE",-1)
    epi_date = kwargs.get("EPI_WEEK",-1)*7 if kwargs.get("EPI_WEEK",-1)>0 else kwargs.get("EPI_DATE",-1)
    
    if same_para:

        if kwargs.get('beta_density',False) or "async" in kwargs['ODdirection']:
            if reopen_date<0 and epi_date<0:
                subdir = f"{kwargs['ODtype']}_{kwargs['beta_density']}/{kwargs['ODdirection']}/{kwargs['node_count']}"
            else:
                subdir = f"{kwargs['ODtype']}_{kwargs['beta_density']}_ED{epi_date}_RD{reopen_date}/{kwargs['ODdirection']}/{kwargs['node_count']}"
        else:
            subdir = f"{kwargs['ODtype']}/{kwargs['ODdirection']}/{kwargs['node_count']}"
    else:
        keyword = paratuple[0]
        if kwargs.get('beta_density',False) or "async" in kwargs['ODdirection']:
            subdir = f"{kwargs['ODtype']}_{kwargs['beta_density']}/{kwargs['ODdirection']}/{kwargs['node_count']}/{keyword}"
        else:
            subdir = f"{kwargs['ODtype']}/{kwargs['ODdirection']}/{kwargs['node_count']}/{keyword}"
    
    if not os.path.exists("toyoutput/scalars/{}".format(subdir)):
        os.makedirs("toyoutput/scalars/{}".format(subdir))

    graph_output_dir("graphs/graph_toy/networks/{}".format(subdir))

    
    draw_flow_pcf_mat(initpopulations,town_area,kwargs['flow_ratio']*generate_toy_OD(node_count = kwargs['node_count'],ODtype = kwargs['ODtype'],direction=kwargs['ODdirection'],flow=10_0000),filename = graph_output_path(f'graphs/graph_toy/networks/{subdir}/network_matrix'),formats = ['png','svg'],vmin = 1e-3,vmax = 3)
    if same_para:
        if recompute:
            pool = mp.Pool(POOLSIZE)

            for i in range(count):
                kwcp = kwargs.copy()
                kwcp['suffix'] = str(i)
                expbatch.append((ODseir_simple,args,kwcp))
            list_of_frames = pool.starmap(apply_args_and_kwargs,expbatch)
            pool.close()
            pool.join()

            multitotalI = []
            multi_daily_new = []
            multiIinSelected = []
            multiIratioinSelected = []
            multiS2EinSelected = []
            multiSinSelected = []
            multiSratioinSelected = []
            for item in list_of_frames:
                totalI,daily_new,selected,IinSelected,IratioinSelected,S2EinSelected,SinSelected,SratioinSelected,_ = item
                multitotalI.append(totalI)
                multi_daily_new.append(daily_new)
                multiIinSelected.append(IinSelected)
                multiIratioinSelected.append(IratioinSelected)
                multiS2EinSelected.append(S2EinSelected)
                multiSinSelected.append(SinSelected)
                multiSratioinSelected.append(SratioinSelected)
            multitotalI=np.array(multitotalI)
            multi_daily_new=np.array(multi_daily_new)
            multiIratioinSelected = np.array(multiIratioinSelected)
            multiIinSelected = np.array(multiIinSelected)
            multiS2EinSelected = np.array(multiS2EinSelected)
            multiSratioinSelected = np.array(multiSratioinSelected)
            multiSinSelected = np.array(multiSinSelected)
            with open('toyoutput/scalars/{}/result_frame_same_para.pkl'.format(subdir) ,'wb') as fb:
                pkl.dump((multitotalI,selected),fb)
            with open('toyoutput/scalars/{}/daily_new_same_para.pkl'.format(subdir) ,'wb') as fb:
                pkl.dump((multi_daily_new,selected),fb)

        else:
            try:
                with open('toyoutput/scalars/{}/result_frame_same_para.pkl'.format(subdir) ,'rb') as fb:
                    multitotalI,selected = pkl.load(fb)
                with open('toyoutput/scalars/{}/daily_new_same_para.pkl'.format(subdir) ,'rb') as fb:
                    multi_daily_new,selected = pkl.load(fb)
                    
            except Exception:
                with open('toyoutput/scalars/{}/result_frame_same_para.pkl'.format(subdir) ,'rb') as fb:
                    multitotalI = pkl.load(fb)
        xlabels = {}
        if reopen_date>0:
            xlabels[f'reopen:{reopen_date}'] = reopen_date
        if epi_date>0:
            xlabels[f'lockdown:{epi_date}'] = epi_date

        PlotSelectedMultiScalar(selected,multiIinSelected,title="Active cases",filename=f"toyoutput/scalars/{subdir}/I_count_multi.{plotsaveformat}",logscale=True,extrax = xlabels,legend = False,ylim = (1,None),linewidth = 2,agg = 1,
                                )
        PlotSelectedMultiScalar(selected,multiIratioinSelected,title="Prevalence",filename=f"toyoutput/scalars/{subdir}/I_ratio_multi.{plotsaveformat}",logscale=True,extrax = xlabels,legend = False,ylim = (1e-4,None),linewidth = 2,agg = 1,
                                )
        PlotSelectedMultiScalar(selected,multiS2EinSelected,title="# of newly infected cases",filename=f"toyoutput/scalars/{subdir}/daily_new_multi.{plotsaveformat}",logscale=True,extrax = xlabels,legend = False,ylim = (1,None),linewidth = 2,agg = 1,
                                )
        PlotSelectedMultiScalar(selected,multiS2EinSelected/initpopulations[np.newaxis,:,np.newaxis]*1e4,title="# of newly infected cases per 10k population",filename=f"toyoutput/scalars/{subdir}/daily_new_ratio_multi.{plotsaveformat}",logscale=True,extrax = xlabels,legend = False,ylim = (1e-2,None),linewidth = 2,agg = 1,
                                )
        
        PlotSelectedMultiScalar(selected,multiSinSelected,title="S_count",filename=f"toyoutput/scalars/{subdir}/S_count_multi.{plotsaveformat}",logscale=True,extrax = xlabels)
        PlotSelectedMultiScalar(selected,multiSratioinSelected,title="S_ratio",filename=f"toyoutput/scalars/{subdir}/S_ratio_multi.{plotsaveformat}",logscale=True,extrax = xlabels)
        

        
        

    else:
        if recompute:
            pool = mp.Pool(POOLSIZE)

            if 'log' in paratuple[1]:
                different_p = np.logspace(np.log10(paratuple[2][0]),np.log10(paratuple[2][1]),paratuple[3])
            else:
                different_p = np.linspace(paratuple[2][0],paratuple[2][1],paratuple[3])
            if "START" in paratuple[0]:
                different_p = different_p.astype(int)
                if 0 not in different_p:
                    different_p = np.concatenate(([0],different_p))
            list_of_frames = []
            for _ in trange(count):
                for i,p in enumerate(different_p):
                    kwcp = kwargs.copy()
                    kwcp['suffix'] = str(i)+"_"+str(count)
                    kwcp[keyword] = p
                    expbatch.append((ODseir_simple,args,kwcp))
                    para_x.append(p)
            list_of_frames += pool.starmap(apply_args_and_kwargs,expbatch)
            print(len(list_of_frames))
            pool.close()
            pool.join()
            
            multitotalI = []
            multiselectedI = []
            multi_daily_new = []
            for item in list_of_frames:
                totalI,daily_new,selected,IinSelected,IratioinSelected,SinSelected,SratioinSelected,_ = item
                multitotalI.append(np.array(totalI))
                multiselectedI.append(np.sum(IinSelected,axis = 0))
                multi_daily_new.append(np.array(daily_new))
            controlpara = ('flow_ratio',kwargs['flow_ratio']) if 'alpha_ratio' == paratuple[0] else ('alpha_ratio',kwargs['alpha_ratio'])
            multitotalI = np.array(multitotalI)
            multi_daily_new = np.array(multi_daily_new)
            with open('toyoutput/scalars/{}/result_frame_para{}.pkl'.format(subdir,keyword) ,'wb') as fb:
                pkl.dump(multitotalI,fb)
            with open('toyoutput/scalars/{}/multi_daily_new_para{}.pkl'.format(subdir,keyword) ,'wb') as fb:
                pkl.dump(multi_daily_new,fb)

        else:
            keyword = paratuple[0]
            if 'log' in paratuple[1]:
                different_p = np.logspace(np.log10(paratuple[2][0]),np.log10(paratuple[2][1]),paratuple[3])
            else:
                different_p = np.linspace(paratuple[2][0],paratuple[2][1],paratuple[3])
            if "START" in paratuple[0]:
                different_p = different_p.astype(int)
                if 0 not in different_p:
                    different_p = np.concatenate(([0],different_p))
            for _ in trange(count):
                for i,p in enumerate(different_p):
                    para_x.append(p)
            with open('toyoutput/scalars/{}/result_frame_para{}.pkl'.format(subdir,keyword) ,'rb') as fb:
                multitotalI = pkl.load(fb)
            with open('toyoutput/scalars/{}/multi_daily_new_para{}.pkl'.format(subdir,keyword) ,'rb') as fb:
                multi_daily_new = pkl.load(fb)
        PlotParaScalar(multitotalI.sum(axis = 2),paratuple,"total cases for different {}".format(keyword),'toyoutput/scalars/{}/totalI_ratio_para{}.jpg'.format(subdir,keyword),
                        ylog = False,cmap = 'jet',para_x = para_x)
        if multitotalI.shape[2]<=40:
            nodeidxes = np.linspace(0,multitotalI.shape[2]-1,multitotalI.shape[2]).astype(int)
        else:
            nodeidxes = np.linspace(0,multitotalI.shape[2]-1,30).astype(int)
        for nodeidx in nodeidxes:
            PlotParaScalar(np.cumsum(multi_daily_new[:,:paratuple[2][1],nodeidx]/initpopulations[nodeidx],axis = 1),paratuple,"node{} for different {}".format(nodeidx,keyword),
                            'toyoutput/scalars/{}/cumulateE_ratio_para{}_node{}.jpg'.format(subdir,keyword,nodeidx),
                            ylog = False,cmap = 'jet',para_x = para_x,ylabel = 'cumulate # of Infected',
                            x_line = [paratuple[2][0],paratuple[2][1]])
            PlotParaScalar(multitotalI[:,:paratuple[2][1],nodeidx]/initpopulations[nodeidx],paratuple,"node{} for different {}".format(nodeidx,keyword),
                            'toyoutput/scalars/{}/selectedI_ratio_para{}_node{}.jpg'.format(subdir,keyword,nodeidx),
                            ylog = True,cmap = 'jet',para_x = para_x,ylabel = '# of Infected',
                            x_line = [paratuple[2][0],paratuple[2][1]])
            CompareParaScalar(multi_daily_new[:,:,nodeidx]/initpopulations[nodeidx],paratuple,"node{} for different {}".format(nodeidx,keyword),'toyoutput/scalars/{}/selectednewE_period_para{}_node{}.jpg'.format(subdir,keyword,nodeidx),
                                ylog = False,cmap = 'jet',xlabel=keyword,ylabel='sum of cases during Day{} and {}'.format(paratuple[2][0],paratuple[2][1]),
                                time_thres = (paratuple[2][0],paratuple[2][1]),para_x = para_x,compare_method = 'sum_between_thres')
            CompareParaScalar(multitotalI[:,:,nodeidx]/initpopulations[nodeidx],paratuple,"node{} for different {}".format(nodeidx,keyword),'toyoutput/scalars/{}/selectedI_ratio_second_max_para{}_node{}.jpg'.format(subdir,keyword,nodeidx),
                                ylog = False,cmap = 'jet',xlabel=keyword,
                                ylabel='first max I ratio during period',time_thres = paratuple[2][0],para_x = para_x,compare_method = 'first_max_after_thres')
            CompareParaScalar(multitotalI[:,:,nodeidx],paratuple,"node{} for different {}".format(nodeidx,keyword),'toyoutput/scalars/{}/selectedI_ratio_second_arise_para{}_node{}.jpg'.format(subdir,keyword,nodeidx),
                                ylog = False,cmap = 'jet',xlabel=keyword,
                                ylabel='second arise date',time_thres = paratuple[2][0],para_x = para_x,compare_method = 'first_arise_after_thres',local_get_second_idx = 0)
        
        para_x_np = np.array(para_x)
        df1s = []
        df2s = []
        for px in different_p:
            paraidx = para_x_np==px
            df2 = CompareParaScalar(np.concatenate(multitotalI[paraidx,:,:],axis = 1).T,
                                paratuple,"{}{} for different node".format(keyword,px),
                                None,
                                ylog = False,cmap = 'jet',xlabel='node',
                                ylabel='date',time_thres = paratuple[2][0],para_x = list(range(multitotalI.shape[2]))*count,compare_method = 'first_arise_after_thres',local_get_second_idx = 0)
            df1:pd.DataFrame = CompareParaScalar(np.concatenate(multitotalI[paraidx,:,:],axis = 1).T,
                                paratuple,"{}{} for different node".format(keyword,px),
                                None,
                                ylog = False,cmap = 'jet',xlabel='node',
                                ylabel='date',time_thres = paratuple[2][0],para_x = list(range(multitotalI.shape[2]))*count,compare_method = 'first_max',local_get_second_idx = 0)
            df2.loc[:,keyword] = px
            df1.loc[:,keyword] = px
            df1.loc[:,"feature"] = "first_peak_date"
            df2.loc[:,"feature"] = "second_arise_date"
            df1.loc[:,"group"] = sum([[item]*multitotalI.shape[2] for item in range(count)],start = [])
            df2.loc[:,"group"] = sum([[item]*multitotalI.shape[2] for item in range(count)],start = [])

            df1s.append(df1)
            df2s.append(df2)

            
            fig,ax = plt.subplots(1,1,figsize=[10,5],dpi = 200,facecolor='white')
            sns.boxplot(pd.concat([df1.query("node in @nodeidxes"),df2.query("node in @nodeidxes")]),x='node',y = 'date',ax = ax,hue = 'feature')
            ax.set_title("{}{} for different node".format(keyword,px))
            fig.savefig('toyoutput/scalars/{}/selectedI_ratio_date_{}{}.jpg'.format(subdir,keyword,px))
            plt.close(fig)

            fig,(ax,bx) = plt.subplots(1,2,figsize=[12,5],dpi = 200,facecolor='white')
            df12 = pd.concat([df1,df2],axis = 0)
            sns.histplot(df12,x='date',hue = 'feature',ax = ax)
            
            sns.scatterplot(pd.pivot(df12,index = ["node","group"],columns = "feature",values = "date").groupby(by = "node").median().loc[:,['first_peak_date','second_arise_date']]
                            ,x='first_peak_date',y = 'second_arise_date',ax = bx)

            plt.savefig('toyoutput/scalars/{}/date_dist{}{}.jpg'.format(subdir,keyword,px))
            plt.close(fig)

        
    return multitotalI


    

        
        




if __name__=="__main__":
    time_to_changes = [200]
    for date in time_to_changes:
        ODseir_multi(count = 100,dryrun = False,scalar = True,node_count=4,ODtype='PCFoverFlow',ODdirection='direct',flow_ratio=1,alpha_ratio = 0.2,beta_density = 'log',s2eversion = 'v1',suffix = 'testPCF',REOPEN_DATE = date,EPI_WEEK= -1)
    
    for date in time_to_changes:
        ODseir_multi(count = 100,dryrun = False,scalar = True,node_count=4,ODtype='PCFoverFlow',ODdirection='singleFull',flow_ratio=1,alpha_ratio = 0.2,beta_density = 'log',s2eversion = 'v1',suffix = 'testPCF',REOPEN_DATE = date,EPI_WEEK= -1)
    
    for date in time_to_changes:
        ODseir_multi(count = 100,dryrun = False,scalar = True,node_count=4,ODtype='PCFoverFlow',ODdirection='singleCBD',flow_ratio=1,alpha_ratio = 0.2,beta_density = 'log',s2eversion = 'v1',suffix = 'testPCF',REOPEN_DATE = date,EPI_WEEK= -1)
    
    for date in time_to_changes:
        ODseir_multi(count = 100,dryrun = False,scalar = True,node_count=4,ODtype='PCFoverFlow',ODdirection='singleIndustry',flow_ratio=1,alpha_ratio = 0.2,beta_density = 'log',s2eversion = 'v1',suffix = 'testPCF',REOPEN_DATE = date,EPI_WEEK= -1)
    
    
    
