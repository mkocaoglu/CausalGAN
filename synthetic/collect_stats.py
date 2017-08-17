import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from models import GeneratorTypes,DataTypes


def makeplots(x_iter,tvd_datastore,show=False):
    #Make plots
    dtypes=tvd_datastore.keys()
    fig,axes=plt.subplots(1,len(dtypes))

    fig.subplots_adjust(hspace=0.5,wspace=0.025)
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

        df=pd.DataFrame.from_dict(tvd_datastore[dtype])
        df.plot(x=x_iter,ax=ax,sharey=True,legend=use_legend)
        ax.set_title(axtitle)
        #ax.set_title(dtype+'synthetic data')
        ax.set_xlabel('iter')
        ax.set_ylabel('tvd')

    t='Graph Structured Generator tvd Convergence on Synthetic Data with Known Causal Graph'
    plt.suptitle(t)

    if show:
        plt.show(block=False)
    return fig,axes


if __name__=='__main__':
    dtypes=DataTypes.keys()
    gtypes=GeneratorTypes.keys()

    logdir='logs'

    #init
    #Create a dictionary for each dataset, of dictionaries for each gen_type
    tvd_datastore={dt:{gt:0. for gt in gtypes} for dt in dtypes}
    runs=os.listdir(logdir)

    for dtype in dtypes:
        print 'Collecting data for datatype ',dtype
        typed_runs=filter(lambda x:x.endswith(dtype),runs)

        for gtype in gtypes:
            n_runs=0
            for run in typed_runs:
                #tvd_csv={gt:os.path.join(logdir,run,gt,'tvd.csv') for gt in gtypes}
                tvd_csv=os.path.join(logdir,run,gtype,'tvd.csv')

                #cols=['step','tvd','mvd']
                dat=pd.read_csv(tvd_csv,sep=' ')

                if len(dat)!=1001:
                    print 'WARN: file',tvd_csv,'was of length:',len(dat),
                    print 'it may be in the process of optimizing.. not using'
                    continue

                tvd_datastore[dtype][gtype]+=dat['tvd']
                n_runs+=1

            #end runs loop
            if n_runs==0:
                print 'Warning: no runs of type ',dtype
                tvd_datastore.pop(dtype)#remove key
                break#break inner loop?
            else:
                tvd_datastore[dtype][gtype]/=n_runs
        print 'There were ',n_runs,' runs of ',dtype


    x_iter=dat['iter'].values







