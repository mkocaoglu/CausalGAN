import pandas as pd
import numpy as np
from scipy import stats
import os
import matplotlib.pyplot as plt
from models import GeneratorTypes,DataTypes


def makeplots(x_iter,tvd_datastore,show=False,save=False,save_name=None):
    #Make plots
    dtypes=tvd_datastore.keys()
    fig,axes=plt.subplots(1,len(dtypes))

    #fig.subplots_adjust(hspace=0.5,wspace=0.025)
    fig.subplots_adjust(hspace=0.75,wspace=0.05)

    x_iter=x_iter.astype('float')/1000


    for ax,dtype in zip(axes,dtypes):
        if ax in axes[:-1]:
            use_legend = False
        else:
            use_legend = True

        if ax==axes[0]:
            prefix='Synthetic Data Graph:    '
            posfix='                                    '
        else:
            prefix=''
            posfix=''
        axtitle=prefix+dtype+posfix

        #df=pd.DataFrame.from_dict(tvd_datastore[dtype])


        df_tvd=pd.DataFrame(data={gtype:tvd_datastore[dtype][gtype]['tvd'] for gtype in gtypes})
        df_sem=pd.DataFrame(data={gtype:tvd_datastore[dtype][gtype]['sem'] for gtype in gtypes})
        df_tvd.index=x_iter;df_sem.index=x_iter


        df_tvd.plot.line(ax=ax,sharey=True,use_index=True,yerr=df_sem,legend=use_legend,capsize=5,capthick=3,elinewidth=1,errorevery=100)


        ax.set_title(axtitle.title(),fontsize=18)
        ax.set_ylabel('Total Variational Distance',fontsize=18)
        if ax is axes[1]:
            ax.set_xlabel('iter(thousands)',fontsize=18)

    t='Graph Structured Generator tvd Convergence on Synthetic Data with Known Causal Graph'
    plt.suptitle(t,fontsize=20)

    fig.set_figwidth(15,forward=True)
    fig.set_figheight(7,forward=True)

    #TODO
    #Make text larger
    #error bars

    if save:
        save_name=save_name or 'synth_tvd_vs_time.pdf'
        save_path=os.path.join('assets',save_name)

        plt.savefig(save_path,bbox_inches='tight')
        #plt.savefig(save_path)

    if show:
        plt.show(block=False)
    return fig,axes


if __name__=='__main__':
    dtypes=DataTypes.keys()
    gtypes=GeneratorTypes.keys()

    logdir='logs/figure_logs'

    #init
    #Create a dictionary for each dataset, of dictionaries for each gen_type
    tvd_all_datastore={dt:{gt:[] for gt in gtypes} for dt in dtypes}
    tvd_datastore={dt:{} for dt in dtypes}
    runs=os.listdir(logdir)

    for dtype in dtypes:
        print ''
        print 'Collecting data for datatype ',dtype,'...'

        typed_runs=filter(lambda x:x.endswith(dtype),runs)

        for gtype in gtypes:
            n_runs=0

            #Go through all runs for each (dtype,gtype) pair
            for run in typed_runs:
                #tvd_csv={gt:os.path.join(logdir,run,gt,'tvd.csv') for gt in gtypes}
                tvd_csv=os.path.join(logdir,run,gtype,'tvd.csv')

                #cols=['step','tvd','mvd']
                dat=pd.read_csv(tvd_csv,sep=' ')

                if len(dat)!=1001:
                    print 'WARN: file',tvd_csv,'was of length:',len(dat),
                    print 'it may be in the process of optimizing.. not using'
                    continue

                #tvd_all_datastore[dtype][gtype]+=dat['tvd']
                tvd_all_datastore[dtype][gtype].append(dat['tvd'])
                n_runs+=1


            #after (dtype,gtype) collection
            if n_runs==0:
                #remove key since no matching gtype for dtype
                print 'Warning: for dtype',dtype,' no runs of gtype ',gtype
                #tvd_all_datastore[dtype].pop(gtype)
            else:
                df_concat=pd.concat(tvd_all_datastore[dtype][gtype],axis=1)
                gb=df_concat.groupby(by=df_concat.columns,axis=1)
                mean=gb.mean()
                sem=gb.sem().rename(columns={'tvd':'sem'})
                tvd_datastore[dtype][gtype]=pd.concat([mean,sem],axis=1)

                #tvd_all_datastore[dtype][gtype]/=n_runs

                #concat
                #groupby

        #after dtype collection
        if len(tvd_datastore[dtype])==0:
            print 'Warning: no runs of dtype ',dtype
            tvd_datastore.pop(dtype)


        print '...There were ',n_runs,' runs of ',dtype


    x_iter=dat['iter'].values



    fig,axes=makeplots(x_iter,tvd_datastore,show=False,save=True)




