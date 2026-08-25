import numpy as np
import itertools
from scipy.signal import savgol_filter,find_peaks
from scipy.ndimage.filters import generic_filter,uniform_filter
from matplotlib import pyplot as plt
from matplotlib import axes
import matplotlib
def local_get_second(*args,**kwargs):
    version = kwargs.pop("version")
    if version =='v0':
        return local_get_second_v0(*args,**kwargs)
    elif version =='v1':
        return local_get_second_v1(*args,**kwargs)
    elif version =='v2':
        return local_get_second_v2(*args,**kwargs)
    elif version =='ampd':
        return local_get_second_ampd(*args,**kwargs)
    elif version =='simple_curve':
        return local_get_second_simple_curve(*args,**kwargs)
    elif version =='simulation':
        return local_get_second_simulation(*args,**kwargs)

def local_get_second_v0(ts,compare_method,time_thres=500):
    assert compare_method in ['sum','max','sum_after_first','second_max','sum_after_thres','max_after_thres','sum_between_thres','max_between_thres']
    if compare_method=='sum':
        return 0,sum(ts)
    if compare_method=='max':
        return np.argmax(ts),max(ts)
    if compare_method=='sum_after_first':
        fts = savgol_filter(np.log(np.array(ts)+1e-8),51,2,mode='nearest')
        dts = np.diff(fts)
        dts_m = dts[1:]*dts[:-1]
        negtiveplace = dts_m<=0
        peakplace = negtiveplace.copy()
        peakplace[dts[:-1]<0]=False
        peakplace[dts[1:]>0]=False

        poolplace = negtiveplace.copy()
        poolplace[dts[:-1]>0]=False
        poolplace[dts[1:]<0]=False


        idx = np.arange(len(dts_m))

        if np.any(peakplace):
            firstMax = idx[peakplace][0]
        else:
            print(idx)
            firstMax = idx[0]

        poolplace[(fts>fts[firstMax+1]*0.1)[1:-1]] = False
        poolplace[ts[1:-1]==0] = True
        poolplace[0:10] = False

        if np.any(poolplace[firstMax:]):
            firstMin = idx[firstMax:][poolplace[firstMax:]][0]
        else:
            firstMin = firstMax+1+1 
        secondMax = np.argmax(np.array(ts)[1+firstMin:])+1+firstMin
        firstMax +=1 

        argmin = np.argmin(ts[firstMax:secondMax])

        return  argmin+firstMax,np.sum(ts[argmin+firstMax:])
    if compare_method=='second_max':
        fts = savgol_filter(np.log(np.array(ts)+1e-8),51,2,mode='nearest')
        dts = np.diff(fts)
        dts_m = dts[1:]*dts[:-1]
        negtiveplace = dts_m<=0
        peakplace = negtiveplace.copy()
        peakplace[dts[:-1]<0]=False
        peakplace[dts[1:]>0]=False

        poolplace = negtiveplace.copy()
        poolplace[dts[:-1]>0]=False
        poolplace[dts[1:]<0]=False


        idx = np.arange(len(dts_m))

        if np.any(peakplace):
            firstMax = idx[peakplace][0]
        else:
            firstMax = idx[0]



        poolplace[(fts>fts[firstMax+1]*0.1)[1:-1]] = False
        poolplace[ts[1:-1]==0] = True
        poolplace[0:10] = False

        if np.any(poolplace[firstMax:]):
            firstMin = idx[firstMax:][poolplace[firstMax:]][0]
        else:
            firstMin = firstMax+1+1 
        secondMax = np.argmax(np.array(ts)[1+firstMin:])+1+firstMin

        return  secondMax,ts[secondMax]
    if compare_method=='sum_after_thres':
        return time_thres,sum(ts[time_thres:])
    if compare_method=='max_after_thres':
        return time_thres,max(ts[time_thres:])
    if compare_method == 'sum_between_thres':
        return time_thres,sum(ts[time_thres[0]:time_thres[1]])
    if compare_method == 'max_between_thres':
        return np.argmax(max(ts[time_thres[0]:time_thres[1]]))+time_thres[0],max(ts[time_thres[0]:time_thres[1]])
    

    

def local_get_second_v1(ts,compare_method,time_thres=500):
    assert compare_method in ['sum','max','sum_after_first','second_max','sum_after_thres','max_after_thres','sum_between_thres','max_between_thres']
    if compare_method=='sum':
        return 0,sum(ts)
    if compare_method=='max':
        return np.argmax(ts),max(ts)
    if compare_method=='sum_after_first':
        fts = savgol_filter(np.log(np.array(ts)+1e-7),51,2,mode='nearest')
        dts = np.diff(fts)
        dts_m = np.sign(dts[1:]*dts[:-1])
        negtiveplace = dts_m<=0
        poolplace = negtiveplace.copy()
        poolplace[dts[:-1]>0]=False
        if sum(poolplace)>=1:
            ts_place = np.arange(len(dts_m))[poolplace][0]+1
        else:
            ts_place = len(ts)-2
        return ts_place,sum(ts[ts_place:])
    if compare_method=='second_max':
        fts = savgol_filter(np.log(np.array(ts)+1e-7),51,2,mode='nearest')
        dts = np.diff(fts)
        dts_m = np.sign(dts[1:]*dts[:-1])
        negtiveplace = dts_m<=0
        peakplace = negtiveplace.copy()
        peakplace[dts[:-1]<0]=False
        if sum(peakplace)>1:
            ts_place = np.arange(len(dts_m))[peakplace][1]+1
        else:
            ts_place = len(ts)-1
        return ts_place,np.array(ts)[ts_place]
    if compare_method=='sum_after_thres':
        return time_thres,sum(ts[time_thres:])
    if compare_method=='max_after_thres':
        maxplace = np.argmax(ts[time_thres:])
        return time_thres+maxplace,max(ts[time_thres:])
    if compare_method == 'sum_between_thres':
        return time_thres,sum(ts[time_thres[0]:time_thres[1]])
    if compare_method == 'max_between_thres':
        return np.argmax(max(ts[time_thres[0]:time_thres[1]]))+time_thres[0],max(ts[time_thres[0]:time_thres[1]])
    

    

def local_get_second_v2(ts,compare_method,time_thres=500):
    assert compare_method in ['sum','max','sum_after_first','second_max','sum_after_thres','max_after_thres','sum_between_thres','max_between_thres']
    if compare_method=='sum':
        return 0,sum(ts)
    if compare_method=='max':
        return np.argmax(ts),max(ts)
    if compare_method=='sum_after_first':
        maxplace = np.argmax(ts)
        minplace = np.argmin(ts[maxplace:])
        ts_place = maxplace+minplace
        return ts_place,sum(ts[ts_place:])
    if compare_method=='second_max':
        maxplace = np.argmax(ts)
        minplace = np.argmin(ts[maxplace:])
        secondmax = np.argmax(ts[maxplace+minplace:])
        ts_place = maxplace+minplace+secondmax
        return ts_place,np.array(ts)[ts_place]
    if compare_method=='sum_after_thres':
        return time_thres,sum(ts[time_thres:])
    if compare_method=='max_after_thres':
        maxplace = np.argmax(ts[time_thres:])
        return time_thres+maxplace,max(ts[time_thres:])
def compute_local_mean_std(image,kernel_r = 5):
    std_image = generic_filter(image, np.std, size=kernel_r*2+1)
    mean_image = uniform_filter(image, size=kernel_r*2+1)
    return mean_image,std_image
def observe_line(multi_ts,different_p1,different_p2,imgdir,x= None,y = None,version = 'v0'):
    assert (x is None and y is not None) or (x is not None and y is None)
    fig = plt.figure(figsize = [15,10],dpi = 200)
    axes = []
    axes.append(plt.subplot2grid((3, 2), (0, 0), rowspan=1, colspan=2,fig=fig))
    axes.append(plt.subplot2grid((3, 2), (1, 0), rowspan=1, colspan=1,fig=fig))
    axes.append(plt.subplot2grid((3, 2), (1, 1), rowspan=1, colspan=1,fig=fig))
    axes.append(plt.subplot2grid((3, 2), (2, 0), rowspan=1, colspan=1,fig=fig))
    axes.append(plt.subplot2grid((3, 2), (2, 1), rowspan=1, colspan=1,fig=fig))
    axes = np.array(axes)
    

    cmap = 'jet'
    wanted_ts = []
    if x is not None:
        xidx=np.argmin(abs(np.array(different_p1)-x))
        norm= matplotlib.colors.Normalize(vmin=min(different_p2), vmax=max(different_p2) ,clip=True)
        tocolor = matplotlib.cm.get_cmap(cmap)
        cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap),ax = axes[1:])
        cb.set_label('alpha_ratio')
        axes[0].set_title(f"flow_ratio = {x}")
        for ax_count , method in enumerate(['sum','max','sum_after_first','second_max']):
            axes[ax_count+1].set_ylabel(method)
            axes[ax_count+1].set_xlabel('alpha_ratio')
    if y is not None:
        yidx=np.argmin(abs(np.array(different_p2)-y))
        norm= matplotlib.colors.LogNorm(vmin=min(different_p1), vmax=max(different_p1) ,clip=True)
        tocolor = matplotlib.cm.get_cmap(cmap)
        cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap),ax = axes)
        cb.set_label('flow_ratio')
        axes[0].set_title(f"alpha_ratio = {y}")
        for ax_count , method in enumerate(['sum','max','sum_after_first','second_max']):
            axes[ax_count+1].set_ylabel(method)
            axes[ax_count+1].set_xlabel('flow_ratio')
            axes[ax_count+1].set_xscale('log')


    

    for (p1,p2),item in zip(itertools.product(enumerate(different_p1),enumerate(different_p2)),multi_ts):
        if (x is not None and xidx == p1[0] ):
            different_p = p2[1]
        elif (y is not None and yidx == p2[0]):
            different_p = p1[1]
        else:
            continue
        axes[0].plot(item,color = tocolor(norm(different_p)),alpha = 0.2,zorder = 1)
        
        
        for ax_count , method in enumerate(['sum','max','sum_after_first','second_max']):
            place,result = local_get_second(item,method,version = version)
            axes[ax_count+1].scatter(different_p,result,color = tocolor(norm(different_p)))
            if method =='sum_after_first':
                axes[0].scatter(place,item[place],edgecolor= tocolor(norm(different_p)),color='black',marker = "v",zorder=4)
            if method == 'second_max':
                axes[0].scatter(place,item[place],edgecolor= tocolor(norm(different_p)),color='black',marker = '^',zorder=4)



    fig.savefig('{}/I_ratio_ts_observe.jpg'.format(imgdir))
    plt.close(fig)

def check_get_second(multi_ts,compare_method,imgdir,time_thres=500,sample_rate = 1,version = 'v0',ts_name = "I_ratio"):
    assert compare_method in ['sum','max','sum_after_first','second_max','sum_after_thres','max_after_thres','sum_between_thres','max_between_thres']
    if compare_method=='sum':
        return 
    if compare_method=='max':
        fig,ax = plt.subplots(1,1)
        ax.set_yscale('log')
        for i,ts in enumerate(multi_ts):
            if (i+50)%sample_rate==0:
                line = ax.plot(ts,zorder = 1)
                place,value = local_get_second(ts,compare_method,time_thres,version = version)
                if value>0:
                    ax.scatter(place,ts[place],edgecolor= line[0].get_color(),color='white',zorder=2)
                else:
                    ax.axvline(x= place,color= line[0].get_color(),zorder=2)
        fig.savefig('{}/{}_ts_check_maxplace.jpg'.format(imgdir,ts_name))
    if compare_method=='sum_after_first':
        fig,ax = plt.subplots(1,1)
        ax.set_yscale('log')
        for i,ts in enumerate(multi_ts):
            if (i+50)%sample_rate==0:
                line = ax.plot(np.array(ts)[1:-1],zorder = 1)
                place,value = local_get_second(ts,compare_method,time_thres,version = version)
                if ts[place]>0:
                    ax.scatter(place,ts[place],edgecolor= line[0].get_color(),color='white',zorder=2)
                else:
                    ax.axvline(x= place,color= line[0].get_color(),zorder=2)                
        fig.savefig('{}/{}_ts_check_firstmin.jpg'.format(imgdir,ts_name))
    if compare_method=='second_max':
        fig,ax = plt.subplots(1,1)
        ax.set_yscale('log')
        for i,ts in enumerate(multi_ts):
            if (i+50)%sample_rate==0:
                line = ax.plot(np.array(ts)[1:-1],zorder = 1)
                place,value = local_get_second(ts,compare_method,time_thres,version = version)
                if ts[place]>0:
                    ax.scatter(place,ts[place],edgecolor= line[0].get_color(),color='white',zorder=2)
                else:
                    ax.axvline(x= place,color= line[0].get_color(),zorder=2)
        fig.savefig('{}/{}_ts_check_secondmax.jpg'.format(imgdir,ts_name))
    if compare_method in ['sum_after_thres','sum_between_thres']:
        return 
    if compare_method=='max_after_thres':
        fig,ax = plt.subplots(1,1)
        ax.set_yscale('log')
        for i,ts in enumerate(multi_ts):
            if (i+50)%sample_rate==0:
                line = ax.plot(ts,zorder = 1)
                place,value = local_get_second(ts,compare_method,time_thres,version = version)
                if ts[place]>0:
                    ax.scatter(place,value,edgecolor= line[0].get_color(),color='white',zorder=2)
                else:
                    ax.axvline(x= place,color= line[0].get_color(),zorder=2)
        fig.savefig('{}/{}_ts_check_thresmax.jpg'.format(imgdir,ts_name))
    if compare_method == 'max_between_thres':
        fig,ax = plt.subplots(1,1)
        ax.set_yscale('log')
        for i,ts in enumerate(multi_ts):
            if (i+50)%sample_rate==0:
                line = ax.plot(ts,zorder = 1)
                place,value = local_get_second(ts,compare_method,time_thres,version = version)
                if ts[place]>0:
                    ax.scatter(place,value,edgecolor= line[0].get_color(),color='white',zorder=2)
                else:
                    ax.axvline(x= place,color= line[0].get_color(),zorder=2)
        ax.axvline(x= time_thres[0],color= 'black',zorder=2)
        ax.axvline(x= time_thres[1],color= 'black',zorder=2)
        fig.savefig('{}/{}_ts_check_betweenthresmax.jpg'.format(imgdir,ts_name))
    plt.close(fig)
def CompareGridParaScalar(multitotalI,different_p1,different_p2,title,filename,cmap ,xlabel,ylabel,time_thres,xlog,ylog,compare_method='sum',version = 'v0'):
    fig,axs = plt.subplots(1,3,figsize = [15,4],dpi = 200)
    Cmesh = np.zeros((len(different_p2),len(different_p1)))
    for (p1,p2),item in zip(itertools.product(enumerate(different_p1),enumerate(different_p2)),multitotalI):
        temp = local_get_second(item,compare_method,time_thres,version = version)[1]
        Cmesh[p2[0],p1[0]] = temp
    Xmesh , Ymesh = np.meshgrid(different_p1, different_p2)
    norm= matplotlib.colors.Normalize(vmin=np.min(Cmesh), vmax=np.max(Cmesh) ,clip=True)
    axs[0].pcolormesh(Xmesh,Ymesh,Cmesh,cmap = cmap,shading = 'nearest',norm = norm)
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap),ax  = axs[0])
    axs[0].set_title(title)
    for i in range(3):
        axs[0].set_xlabel(xlabel)
    axs[0].set_ylabel(ylabel)


    mean_image,std_image = compute_local_mean_std(Cmesh,5)
    axs[1].set_title('local mean')
    axs[2].set_title('local std')
    norm= matplotlib.colors.Normalize(vmin=np.min(mean_image), vmax=np.max(mean_image) ,clip=True)
    axs[1].pcolormesh(Xmesh,Ymesh,mean_image,cmap = cmap,shading = 'nearest',norm = norm)
    contours = axs[1].contour(Xmesh,Ymesh,mean_image,colors = 'black',linestyles = 'dashed')
    axs[1].clabel(contours,colors='black',fontsize = 10)
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap),ax  = axs[1])
    norm= matplotlib.colors.Normalize(vmin=np.min(std_image), vmax=np.max(std_image) ,clip=True)
    axs[2].pcolormesh(Xmesh,Ymesh,std_image,cmap = cmap,shading = 'nearest',norm = norm)

    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap),ax  = axs[2])
    
    if xlog:
        for i in range(3):
            axs[i].set_xscale("log")
    if ylog:
        for i in range(3):
            axs[i].set_yscale("log")


    plt.savefig(filename)
    plt.close()


def AMPD(data):
    """
    实现AMPD算法
    :param data: 1-D numpy.ndarray 
    :return: 波峰所在索引值的列表
    """
    p_data = np.zeros_like(data, dtype=np.int32)
    count = data.shape[0]
    arr_rowsum = []
    for k in range(1, count // 2 + 1):
        row_sum = 0
        for i in range(k, count - k):
            if data[i] > data[i - k] and data[i] > data[i + k]:
                row_sum -= 1
        arr_rowsum.append(row_sum)
    min_index = np.argmin(arr_rowsum)
    max_window_length = min_index
    for k in range(1, max_window_length + 1):
        for i in range(k, count - k):
            if data[i] > data[i - k] and data[i] > data[i + k]:
                p_data[i] += 1
    result = np.where(p_data == max_window_length)[0]
    if len(result)==0:
        result = [np.argmax(data),np.argmax(data)]
    if len(result)==1:
        result = [result[0],result[0]]
    return result 
def local_get_second_ampd(ts,compare_method,time_thres=500):
    ts = np.array(ts)
    assert compare_method in ['sum','max','sum_after_first','second_max','sum_after_thres','max_after_thres','sum_between_thres','max_between_thres']
    if compare_method=='sum':
        return 0,sum(ts)
    if compare_method=='max':
        idxs = AMPD(ts)
        return idxs[0],ts[idxs[0]]
    if compare_method=='sum_after_first':
        pos_idxs = AMPD(ts)
        neg_idxs = AMPD(-ts[pos_idxs[0]:])
        return neg_idxs[0]+pos_idxs[0],sum(ts[neg_idxs[0]+pos_idxs[0]:])
    if compare_method=='second_max':
        idxs = AMPD(ts)
        return idxs[1],ts[idxs[1]]
    if compare_method=='sum_after_thres':
        return time_thres,sum(ts[time_thres:])
    if compare_method=='max_after_thres':
        maxplace = np.argmax(ts[time_thres:])
        return time_thres+maxplace,max(ts[time_thres:])
    if compare_method == 'sum_between_thres':
        return time_thres,sum(ts[time_thres[0]:time_thres[1]])
    if compare_method == 'max_between_thres':
        return np.argmax(max(ts[time_thres[0]:time_thres[1]]))+time_thres[0],max(ts[time_thres[0]:time_thres[1]])
    

    


def local_get_second_simple_curve(ts,compare_method,time_thres=500):
    if (ts==0).all():
        logts = ts
        peakthres = 1
    else:
        non0min = np.min(ts[ts!=0])
        logts = np.log(ts+non0min)
        peakthres = np.max(logts)-np.log(1e2)
    assert compare_method in ['sum','max','sum_after_first','second_max','sum_after_thres','max_after_thres',
                              'sum_between_thres','max_between_thres','first_max_after_thres','first_max',
                              'first_arise_after_thres','first_arise','second_arise']
    if compare_method=='sum':
        return 0,sum(ts)
    if compare_method=='max':
        return np.argmax(ts),max(ts)
    if compare_method=='first_max':
        peaks,peak_prop = find_peaks(logts,width = (15, None),height= peakthres)
        if len(peaks)>=1:
            first_max = peaks[0]
        elif len(peaks) == 0:
            first_max = np.argmax(ts)
        return first_max,ts[first_max]
    if compare_method=='sum_after_first':
        peaks,peak_prop = find_peaks(logts,width = (15, None),height= peakthres)
        if len(peaks)==1:
            argmin = np.argmin(ts[peaks[0]:])+peaks[0]
        elif len(peaks) == 0:
            argmin = len(ts)-1
        else:
            argmin = np.argmin(ts[peaks[0]:peaks[1]])+peaks[0]
        return  argmin,np.sum(ts[argmin:])
    if compare_method=='second_max':
        peaks,peak_prop = find_peaks(logts,width = (15, None),height= peakthres)
        if len(peaks)==1:
            argmin = np.argmin(ts[peaks[0]:])+peaks[0]
            secondMax = np.argmax(ts[argmin:])
            return  secondMax+argmin,ts[secondMax+argmin]
        elif len(peaks) == 0:
            secondMax = len(ts)-1

            return  secondMax,0

        else:
            secondMax = peaks[1]
            return  secondMax,ts[secondMax]
    if compare_method=='sum_after_thres':
        return time_thres,sum(ts[time_thres:])
    if compare_method=='max_after_thres':
        return np.argmax(ts[time_thres:])+time_thres,max(ts[time_thres:])
    if compare_method == 'sum_between_thres':
        return time_thres,sum(ts[time_thres[0]:time_thres[1]])
    if compare_method == 'max_between_thres':
        return np.argmax(max(ts[time_thres[0]:time_thres[1]]))+time_thres[0],max(ts[time_thres[0]:time_thres[1]])
    if compare_method == 'first_max_after_thres':
        temp = local_get_second_simple_curve(ts[time_thres:],"first_max")
        return temp[0]+time_thres,ts[time_thres+temp[0]]
    if compare_method == 'first_arise_after_thres':
        temp = local_get_second_simple_curve(ts[time_thres:],"first_arise")
        return temp[0]+time_thres,ts[time_thres+temp[0]]
    if compare_method == 'first_arise':
        peaks,peak_prop = find_peaks(logts,width = (15, None))
        if len(peaks)>=1:
            left_min = np.min(ts[:peaks[0]])
            midheight = (left_min+ts[peaks[0]])/2
            idx = np.arange(peaks[0])
            result = idx[ts[idx]<midheight][-1]
            return  result,ts[result]
        elif len(peaks) == 0:
            argmax = np.argmax(ts)
            if argmax ==0:
                return argmax,ts[argmax]
            left_min = np.min(ts[:argmax])
            midheight = (left_min+ts[argmax])/2
            idx = np.arange(argmax)
            result = idx[ts[idx]<midheight][-1]
            return  result,ts[result]
    if compare_method == 'second_arise':
        argmin,_ = local_get_second_simple_curve(ts,"sum_after_first")
        return local_get_second_simple_curve(ts,"first_arise_after_thres",time_thres = argmin,)

def local_get_second_simulation(ts,compare_method,time_thres=500):
    if (ts==0).all():
        logts = ts
        peakthres = 1
    else:
        non0min = np.min(ts[ts!=0])
        logts = np.log(ts+non0min)
        peakthres = np.max(logts)-np.log(1e2)
    assert compare_method in ['sum','max','sum_after_first','second_max','sum_after_thres','max_after_thres',
                              'sum_between_thres','max_between_thres','first_max_after_thres','first_max',
                              'first_arise_after_thres','first_arise','second_arise']
    if compare_method=='sum':
        return 0,sum(ts)
    if compare_method=='max':
        return np.argmax(ts),max(ts)
    if compare_method=='first_max':
        peaks,peak_prop = find_peaks(logts,width = (15, None),height= peakthres)
        if len(peaks)>=1:
            first_max = peaks[0]
        elif len(peaks) == 0:
            first_max = np.argmax(ts)
        return first_max,ts[first_max]
    if compare_method=='sum_after_first':
        peaks,peak_prop = find_peaks(logts,width = (15, None),height= peakthres)
        if len(peaks)==1:
            if ts[-1] == 0:
                argmin = len(ts)-1
            else:
                argmin = np.argmin(ts[peaks[0]:])+peaks[0]
        elif len(peaks) == 0:
            argmin = len(ts)-1
        else:
            argmin = np.argmin(ts[peaks[0]:peaks[1]])+peaks[0]
        return  argmin,np.sum(ts[argmin:])
    if compare_method=='second_max':
        peaks,peak_prop = find_peaks(logts,width = (15, None),height= peakthres)
        if len(peaks)<=1:
            secondMax = len(ts)-1
            return  secondMax,0

        else:
            for peakidx in range(1,len(peaks)):
                if logts[peaks[peakidx]] == logts[peaks[0]]:
                    continue
                return  peaks[peakidx],ts[peaks[peakidx]]
    if compare_method=='sum_after_thres':
        return time_thres,sum(ts[time_thres:])
    if compare_method=='max_after_thres':
        return np.argmax(ts[time_thres:])+time_thres,max(ts[time_thres:])
    if compare_method == 'sum_between_thres':
        return time_thres,sum(ts[time_thres[0]:time_thres[1]])
    if compare_method == 'max_between_thres':
        return np.argmax(max(ts[time_thres[0]:time_thres[1]]))+time_thres[0],max(ts[time_thres[0]:time_thres[1]])
    if compare_method == 'first_max_after_thres':
        temp = local_get_second_simulation(ts[time_thres:],"first_max")
        return temp[0]+time_thres,ts[time_thres+temp[0]]
    if compare_method == 'first_arise_after_thres':
        temp = local_get_second_simulation(ts[time_thres:],"first_arise")
        return temp[0]+time_thres,ts[time_thres+temp[0]]
    if compare_method == 'first_arise':
        peaks,peak_prop = find_peaks(logts,width = (15, None))
        if len(peaks)>=1:
            left_min = np.min(ts[:peaks[0]])
            midheight = (left_min+ts[peaks[0]])/2
            idx = np.arange(peaks[0])
            result = idx[ts[idx]<midheight][-1]
            return  result,ts[result]
        elif len(peaks) == 0:
            argmax = np.argmax(ts)
            if argmax ==0:
                return argmax,ts[argmax]
            left_min = np.min(ts[:argmax])
            midheight = (left_min+ts[argmax])/2
            idx = np.arange(argmax)
            result = idx[ts[idx]<midheight][-1]
            return  result,ts[result]
    if compare_method == 'second_arise':
        peaks,peak_prop = find_peaks(logts,width = (15, None),height= peakthres)
        if len(peaks)<=1:
            secondMax = len(ts)-1
            return  secondMax,0

        else:
            for peakidx in range(1,len(peaks)):
                if logts[peaks[peakidx]] == logts[peaks[0]]:
                    continue
                left_min = np.min(ts[peaks[0]:peaks[peakidx]])
                argmin = np.argmin(ts[peaks[0]:peaks[peakidx]])+peaks[0]
                midheight = (left_min+ts[peaks[peakidx]])/2
                idx = np.arange(argmin,peaks[peakidx])
                result = idx[ts[idx]<midheight][-1]
                return  result,ts[result]