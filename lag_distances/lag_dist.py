# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 11:45:34 2018

@author: Will
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
from statsmodels.stats.anova import anova_lm


df = pd.read_csv(r'C:\Users\Will\Desktop\Will\maxlag_07_06_2017.txt', sep=" ", skiprows=[0], header=None)
data = df.values

scan_id = data[:,1]
distance = data[:,6]
lagmax = data[:,9]
lagsign = data[:,10]
maxAbsCC = data[:,8]

d = {'id': scan_id, 'dist': distance, 'max_lag': lagmax, 'max_abs_cc': maxAbsCC}
df = pd.DataFrame(data=d)
df = df[df.max_lag != -999]

df0 = df[df.id == 0]
df1 = df[df.id == 1]
df2 = df[df.id == 2]
df3 = df[df.id == 3]
df4 = df[df.id == 4]
df5 = df[df.id == 5]
df6 = df[df.id == 6]
df100 = df[df.id == 100]

def get_res(dataframe):
    sns.lmplot(x='dist', y='max_lag', data=dataframe, legend=False, fit_reg=True, palette='binary')
    model = sm.OLS(dataframe['max_lag'],np.vander(dataframe['dist'],2))
    results = model.fit()
    dataframe['residuals']=results.resid
    print(results.summary())
    
def anova_2way(dataframe):
    formula = 'max_lag ~ dist'
    model = smf.ols(formula, data=dataframe).fit()
    aov_table = anova_lm(model, typ=2)
    print(aov_table)


anova_2way(df)