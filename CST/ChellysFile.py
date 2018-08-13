# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 09:40:04 2017

@author: Chelley
"""

fname = r'E:\Chelley\1 PROJECTS\WILL\CST_Project\output\2017-07-27\data2to100.txt'

import pickle
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import os
import seaborn as sns
todaystr_apg = datetime.datetime.today().strftime('%d%b%y').lower()
todaystr = datetime.datetime.today().strftime('%Y_%m_%d')

############## set file names #################################################
fname_results = r'E:\Users\Will\Desktop\Will\data8to248products.txt'
fname_matrixkeyval =  r'E:\Users\Will\Desktop\Will\ids8to248products'

fname_outFile = r'E:\Users\Will\Desktop\Will\merged_files\\posneg'+fname_results.split('\\')[-1].split('.')[0][:-3]+'_'+todaystr_apg+'.txt'

##### read files, merge, and format output file ###############################

results_df = pd.read_table(fname_results, header=None, delim_whitespace=True,
                           names= ['nTime','nTotalChannels','nSelected',
                                   'corMatID', 'actualMeanNetCor','actualMeanNetAbsCor','corInOut', 'rmsInOut'],
                           na_values=-999, index_col=0)

# Read the pickle file of dictionary of matrix indexes
with open(fname_matrixkeyval,'rb') as f:
    allMatrixIndices_dict =  pickle.load(f,encoding="bytes")
print(allMatrixIndices_dict)
allMatrixIndices = [ [k,v] for k, v in allMatrixIndices_dict.items() ]

# make network matric CC mean column
def get_mean_mat_cor(mat):
    inums = np.triu_indices(len(mat),1)
    mean = np.mean(np.array(mat)[inums])
    return mean
#def get_mean_mat_cor(mat):
#    mean = np.abs(np.array(mat)[0,1])
#    return mean
allMatrixMeans =  [  [row[0], get_mean_mat_cor(row[1])] for row in allMatrixIndices ] 

matrixVals_df = pd.DataFrame(data = allMatrixMeans,
                             columns = ['corMatID','Network_Cor'])
                             
# merge network correlation matrix with 
df = pd.merge(results_df, matrixVals_df, on='corMatID', how='left')

# save
df.to_csv(fname_outFile, sep=' ', na_rep=-999, index=False)





###### Plotting #############################
# Fischer z-transform
df['zcorInOut'] = np.arctanh(df['corInOut'])
## plot some data (optional)

#df[df.networkSign==1].plot(kind='scatter',x='Network_Cor',y='corInOut')
#df[df.networkSign==-1].plot(kind='scatter',x='Network_Cor',y='corInOut')
#
#import scipy
#
#testVal = 'rmsInOut'
#print(scipy.stats.ttest_ind(df[df.networkSign==1][testVal],df[df.networkSign==-1][testVal]))
#
#testVal = 'zcorInOut'
#print(scipy.stats.ttest_ind(df[df.networkSign==1][testVal],df[df.networkSign==-1][testVal]))

#
#df[df.actualMeanNetCor<0.02].groupby(by='nSelected', as_index=False).mean().plot(kind='scatter',x='nSelected',y='corInOut')

#df.groupby(by='Mean_Network_Cor', as_index=False).mean().plot(kind='scatter',x='Mean_Network_Cor',y='rmsInOut')
#
## Plot with separate plots for number os channels
#splitby="nSelected"#"nTotalChannels"
#g = sns.FacetGrid(df, col=splitby, col_wrap=5)
#g = g.map(plt.scatter,'Mean_Network_Cor','zcorInOut', marker='.')
#g = sns.FacetGrid(df, col=splitby, col_wrap=5)
#g = g.map(plt.scatter,'Mean_Network_Cor','rmsInOut', marker='.')
#
## Plot CCInOutwith actual cor
#g = sns.FacetGrid(df, col=splitby, col_wrap=5)
#g = g.map(plt.scatter,'actualMeanNetCor','zcorInOut', marker='.')
#g = sns.FacetGrid(df, col=splitby, col_wrap=5)
#g = g.map(plt.scatter,'actualMeanNetCor','rmsInOut', marker='.')
#
#
#g = sns.FacetGrid(df, col='Mean_Network_Cor', col_wrap=5)
#g = g.map(plt.scatter,'nSelected','zcorInOut', marker='.')
#

#
## plot accuracy of network correlations
#g = sns.FacetGrid(df, col=splitby, col_wrap=5)
#g = g.map(plt.scatter,'Mean_Network_Cor','actualMeanNetCor', marker='.')
#
## Plot RMS with actual corr
#g = sns.FacetGrid(df, col=splitby, col_wrap=5, sharey=False)
#g = g.map(plt.axvline, color='k')
#g = g.map(plt.scatter,'actualMeanNetCor','rmsInOut', marker='.')
#
## Plot with regression
#g = sns.FacetGrid(df, col=splitby, col_wrap=5, sharey=False)
#g = g.map(plt.axvline, color='k')
#g = g.map(sns.regplot, 'actualMeanNetCor', 'rmsInOut', 
#                 scatter_kws={"s": 5},
#                 order=2, ci=None)
##  Results: plotting a regression with the sctual mean correlation results in a 
##  quadratic curve with minima near zero. We may need to increase the precision of the 
##  simulation
#               
## Plot with regression
#g = sns.FacetGrid(df, col=splitby, col_wrap=5, sharey=False)
#g = g.map(plt.axvline, color='k')
#g = g.map(sns.regplot, 'Mean_Network_Cor', 'rmsInOut', 
#                 scatter_kws={"s": 5},
#                 order=2, ci=None)
## Results: target correlation for the network results in a straight line. Odd.
#                 
#                 
#g = sns.FacetGrid(df[df[splitby]%5==0]
#                .groupby(['Mean_Network_Cor',splitby],as_index=False)
#                .mean(), col=splitby, col_wrap=5, sharey=False)
#g = g.map(plt.axvline, color='k')
#g = g.map(plt.scatter, 'Mean_Network_Cor', 'rmsInOut')
#
#g = sns.FacetGrid(df[df[splitby]%5==0], col=splitby, col_wrap=5, sharey=False)
#g = g.map(plt.axvline, color='k')
#g = g.map(plt.scatter, 'Mean_Network_Cor', 'rmsInOut')
#
#
#meanpercor_df = (df[df[splitby]%5==0]
#                .groupby(['Mean_Network_Cor',splitby],as_index=False)
#                .mean()
#)
#(meanpercor_df.iloc[meanpercor_df
#                .groupby([splitby],as_index=False)
#                .idxmin()['rmsInOut']]).plot(kind='scatter',x='Mean_Network_Cor', y='rmsInOut')
#meanpercor_df.groupby([splitby],as_index=False).min()['rmsInOut']
#
#meanpercor_df = (df
#                .groupby(['Mean_Network_Cor',splitby],as_index=False)
#                .mean()
#)
#meanpercor_df['rms_rank'] = meanpercor_df.groupby(["nSelected"]).rank()['rmsInOut']
#plt.figure();meanpercor_df[meanpercor_df.rms_rank==1].plot(kind='scatter',x="nSelected",y='rmsInOut')
#
#g = sns.FacetGrid(df[df[splitby]%5==0], col=splitby, col_wrap=5, sharey=False)
#g = g.map(plt.axvline, color='k')
#g = g.map(plt.scatter, 'Mean_Network_Cor', 'rms_rank')
#
#meanpercor_df = (df[df[splitby]%5==0]
#                .groupby(['Mean_Network_Cor',splitby],as_index=False)
#                .mean()
#)
#g = sns.FacetGrid(meanpercor_df, col=splitby, col_wrap=5, sharey=False)
#g = g.map(plt.axvline, color='k')
#g = g.map(plt.scatter, 'Mean_Network_Cor', 'rms_rank')
#
#meanpercor_df = (df[df[splitby]<=5]
#                .groupby(['Mean_Network_Cor',splitby],as_index=False)
#                .mean()
#)
#g = sns.FacetGrid(df[df[splitby]<=6], col=splitby, col_wrap=5, sharey=False)
#g = g.map(plt.scatter, 'rmsInOut', 'corInOut')
#
#########################################################
## make dataframe with all cor values included for 3x3
##flattened_matrix_data = [[row[0]] + list(np.array(row[1])[np.triu_indices(len(row[1]),1)]) for row in allMatrixIndices ]
##if len(flattened_matrix_data[int(df.corMatID[0])]) > 2:
##    
##    flattened_matrix_df = pd.DataFrame(data = flattened_matrix_data,
##                                 columns = ['corMatID','CC_1_2','CC_1_3','CC_2_3'])
##    df = pd.merge(results_df, flattened_matrix_df, on='corMatID', how='left')
##    allMatrixMeans =  [  [row[0], get_mean_mat_cor(row[1])] for row in allMatrixIndices ] 
##    
##    matrixVals_df = pd.DataFrame(data = allMatrixMeans,
##                                 columns = ['corMatID','Mean_Network_Cor'])
##    df = pd.merge(df, matrixVals_df, on='corMatID', how='left')
##    # save
##    fname_outFile_allCC = '\\'.join(fname_outFile.split('\\')[:-2])+'\\allCC_'+fname_outFile.split('\\')[-1]
##    df[['nTime', 'nTotalChannels', 'nSelected','corInOut',
##           'rmsInOut', 'CC_1_2', 'CC_1_3', 'CC_2_3', 'Mean_Network_Cor']].to_csv(fname_outFile_allCC, sep=' ', na_rep=-999, index=False)
##    
##    # Fischer z-transform
##    df['zcorInOut'] = np.arctanh(df['corInOut'])
##    # plot some data (optional)
##    df.plot(kind='scatter',x='CC_1_2',y='zcorInOut')
##    df.plot(kind='scatter',x='CC_1_2',y='rmsInOut')
##    
##    df.groupby(by='CC_1_2', as_index=False).mean().plot(kind='scatter',x='CC_1_2',y='zcorInOut')
##    df.groupby(by='CC_1_3', as_index=False).mean().plot(kind='scatter',x='CC_1_3',y='zcorInOut')
##    df.groupby(by='CC_2_3', as_index=False).mean().plot(kind='scatter',x='CC_2_3',y='zcorInOut')




#with open(r'E:\Chelley\1 PROJECTS\WILL\CST_Project\output\merged_files\posneg_09aug17.txt', 'w') as outfile:
#    for f in os.listdir(r'E:\Chelley\1 PROJECTS\WILL\CST_Project\output\merged_files\09aug17'):
#        fname = os.path.join(r'E:\Chelley\1 PROJECTS\WILL\CST_Project\output\merged_files\09aug17',f)
#        with open(fname) as infile:
#            s = infile.read()
#            outfile.write(s[s.index('\n')+1:])