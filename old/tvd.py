
import pandas as pd
import numpy as np





if __name__=='__main__':

    attr=0.5*(1+pd.read_csv('./data/list_attr_celeba.txt',delim_whitespace=True))
    names=['Young','Male','Smiling','Narrow_Eyes']

    print 'Warning debug!'
    attr=attr[names].iloc[:500]

    df2=attr.drop_duplicates()
    df2 = df2.reset_index(drop = True).reset_index()
    df2=df2.rename(columns = {'index':'ID'})

    print 'warning random'
    dat = np.random.randint(0,2,(70,4))#0's and 1's

    df_dat=pd.DataFrame(columns=names,data=dat)
    dat_id=pd.merge(df_dat,df2,on=names,how='left')
    dat_counts=pd.value_counts(dat_id['ID'])
    dat_pdf = dat_counts / dat_counts.sum()

    real_data_id=pd.merge(attr,df2)
    real_counts = pd.value_counts(real_data_id['ID'])
    real_pdf=real_counts/len(attr)

    diff=real_pdf-dat_pdf
    absdiff = np.abs(diff.values)
    tvd=0.5*np.mean(absdiff)





