import pandas as pd
import numpy as np

def drop_na_cols(df,percent=0.1):
    #Drop columns with >pearcent% NAs
    nas=df.isnull().sum()
    nas=nas[1:]

    for i in nas:
        if i>=len(df.index)*percent:
            try:
                df=df.drop(nas.keys()[i], axis=1)
            except KeyError:
                pass
    return df

def mean_impute(data):
    nas=data.isnull().sum()    
    col_means=data.mean(axis=0)
    na_cols=[]
    na_cols_means=[]

    for i in range(len(nas)):
        if nas[i]!=0:        
            na_cols.append(nas.keys()[i])
            na_cols_means.append(col_means[i])
        
    ids=list(data.index)
    for i in ids:
        for j in range(len(na_cols)):
            if str(data.loc[i][na_cols[j]])=="nan":
                data.loc[i][na_cols[j]]=na_cols_means[j]
    
    
    return data
