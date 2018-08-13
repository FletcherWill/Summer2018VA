# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 13:45:14 2018

@author: Will
"""

import numpy as np
import pandas as pd
import math as m
import sys
sys.path.append(r'E:\Users\Will\Desktop\Will')
from random import randint
import IDtracker as idt

ids = idt.IDtracker()

def frange(start, stop, step):
     """range function that accepts floats"""
     i = start
     while i < stop:
         yield i
         i += step

def cor_nxn(mat):
    """
    input: matrix
    output: boolean value that is True if the input matrix can be used as a nxn correlation matrix
    """
    flag = True
    if mat.shape[1] != mat.shape[0]:
        return False
    for i in range(0, mat.shape[0]):
        for j in range(i, mat.shape[0]):
            if i == j:
                if mat.item((i,j)) != 1:
                    flag = False
            else:
                if mat.item((i,j)) != mat.item((j,i)):
                    flag = False
    return flag

def rand_neg(cor_mat, percent = .5):
    size = (cor_mat.shape[0] ** 2 - cor_mat.shape[0]) / 2
    num_neg = int(size * percent)
    (rows, cols) = np.triu_indices(cor_mat.shape[0], 1)
    while num_neg > 0:
        index = randint(0, len(rows) - 1)
        row = rows[index]
        col = cols[index]
        if cor_mat[row,col] >= 0:
            cor_mat[row,col] = -cor_mat[row,col]
            cor_mat[col,row] = -cor_mat[col,row]
            num_neg = num_neg - 1
    return cor_mat

def all_pos(mat):
    for a in range(mat.shape[0]):
        for b in range(mat.shape[0]):
            if mat[a,b] < 0:
                mat[a,b] = -mat[a,b]
    return mat

def method_2(cor_mat, out_len, ntries=0):
    """Input is correlation matrix and output is correlated white noise"""
    if cor_nxn(cor_mat):
        (w,v) = np.linalg.eigh(cor_mat)
#        w = w.real; v=v.real
        V = np.dot(v,np.diag(np.sqrt(w)))
        # now np.dot(V,V.T).flatten() = np.array(C).flatten()
        U = V.T
        
#        random_noise = np.random.normal(0,.1,(out_len,cor_mat.shape[0]))
        random_noise = np.random.normal(0,1,(out_len,cor_mat.shape[0]))
        output = np.dot(random_noise, U).T
#        output = np.multiply(output.T,1/np.max(np.abs(output), axis=1)).T
    else:
        raise ValueError('Not a correlation Matrix')

    print('number in network > 1: %s ; ntries: %s'%(np.sum((output > 1) |
                                              (output < -1)), ntries))
    hasnan = np.isnan(output.flatten()).any()
    if hasnan:
        try:
            if ntries<10:
                percent_neg = np.sum(cor_mat[np.triu_indices(cor_mat.shape[0],1)] < 0)/(cor_mat.shape[0]*(cor_mat.shape[0]-1)/2)
                out = method_2(rand_neg(all_pos(cor_mat),percent_neg),out_len,ntries=ntries+1)
            else:
                out = output
        except RecursionError as e:
            out = np.ones(output.shape)*np.nan
        return out
        #raise ValueError('The group of generated time series contains a nan value.')
    else:
        return output
    
def pos_eigs(mat):
    vals = np.linalg.eigvals(mat)
    flag = True
    for item in vals:
        if item < 0:
            flag = False
    return flag

def get_vars(wn_series):
    var_vals = []
    for series_num in range(wn_series.shape[0]):
        var_vals.append(np.var(wn_series[series_num]))
    return var_vals

def flat_vals(covars):
    return covars.flatten()

def cor_mats_2x2(inc = .1, end = 1):
    cor_mats = []
    for a in frange(0, end, inc):
        cor_mat = np.matrix([[1, a],[a, 1]])
        cor_mats.append(cor_mat)
    return np.asarray(cor_mats)

def cor_mats_3x3():
    cor_mats = []
    for a in frange(0, .3, .1):
        for b in frange(0, .3, .1):
            for c in frange(0, .3, .1):
                cor_mat = np.matrix([ [1.0, a, b],  
                                     [a, 1.0, c],  
                                     [ b, c, 1.0]])
                if pos_eigs(cor_mat):
                    cor_mats.append(cor_mat)
    return np.asarray(cor_mats)

def cor_mats_nxn(size, inc = .01, end = 1):
    cor_mats = []
    for value in frange(0, end, inc):
        cor_mat = np.zeros((size,size))
        for a in range(size):
            for b in range(size):
                if a == b:
                    cor_mat[a,b] = 1
                else:
                    cor_mat[a,b] = value
        cor_mats.append(cor_mat)
    return np.asarray(cor_mats)

def cor_mats_nxn_split(size, inc = .01, end = 1):
    cor_mats = []
    for value1 in frange(0, end, inc):
        for value2 in frange(value1, end, inc):
            cor_mat = np.zeros((size,size))
            flag = 0
            for a in range(size):
                for b in range(a, size):
                    if a == b:
                        cor_mat[a,b] = 1
                    elif flag % 2 == 0:
                        cor_mat[a,b] = value1
                        cor_mat[b,a] = value1
                        flag = flag + 1
                    else:
                        cor_mat[a,b] = value2
                        cor_mat[b,a] = value2
                        flag = flag + 1
            cor_mats.append(cor_mat)
    return np.asarray(cor_mats)

def label_split(inc = .01, end = 1, dup = 10):
    labels = []
    index = 1
    mat_id = 1
    for value1 in frange(0, end, inc):
        for value2 in frange(value1, end, inc):
            for time in range(dup):
                labels.append([index, mat_id, value1, value2])
                index = index + 1
            mat_id = mat_id + 1
    return np.asarray(labels)
    

def duplicate(series, n=10):
    new_series = []
    for thing in series:
        for a in range(n):
            new_series.append(thing)
    return np.asarray(new_series)

def output(cor_mats, lengs, trials = 1):
    length = cor_mats.shape[0]
    data = np.zeros((length,cor_mats[-1].shape[0]**2+4))
    index = 1
    for value in range(length):
        data[value] = get_row(cor_mats[value], lengs[value], index, trials)
        index += 1
    return data

def make_label(length):
    labels = ["index", "matrixID", "seriesLength", "numSeries"]
    for a in range(length):
        for b in range(length):
            labels.append("cov_" +str(a+1)+"_"+str(b+1))
    return np.asarray(labels)

def make_output(data):
    label = make_label(int(round(m.sqrt(data.shape[1]-4))))
    return pd.DataFrame(data,columns=label)

def get_row(cor_mat, leng, index, trials):
    data = np.zeros(cor_mat.shape[0]**2)
    corMatID = ids.index(cor_mat)
    for trial in range(trials):
        series = method_2(cor_mat, leng)
        data = np.add(data, np.cov(series).flatten())
    data = np.true_divide(data, trials)
    info = np.array([index, corMatID, leng, cor_mat.shape[0]])
    return np.append(info, data)
    
D = np.matrix([ [1.0, 0.6, 0.3, 0.5],  
               [0.6, 1.0, 0.5,0.8],  
               [ 0.3, 0.5, 1.0,0.4],
               [0.5,0.8,0.4,1.0]])
    


#covar2x2 = output(duplicate(cor_mats_2x2(),2),duplicate(np.ones(11, dtype=np.int) * 100000,2))
#covar10x10 = output(duplicate(cor_mats_nxn(10,.01,1)),duplicate(np.ones(100, dtype=np.int) * 100000))
#data3x3 = output(cor_mats_3x3(),np.ones(27, dtype=np.int) * 100000, np.arange(1, 28))
#series_1 = method_2(D, 100000)
#covar10x10split = output(duplicate(cor_mats_nxn_split(10, inc = .01, end = 1),10), duplicate(np.ones(5050, dtype=np.int) * 100000,10))

#np.savetxt(r'C:\Users\Will\Desktop\Will\covar10x10splitLabel.txt', label_split(), delimiter=" ")
#make_output(covar10x10split).to_csv(r'C:\Users\Will\Desktop\Will\covar10x10split.txt', sep=" ")

data = pd.read_csv(r'C:\Users\Will\Desktop\Will\covar10x10split.txt', sep=" ", header=None)
data.fillna(-999, inplace = True)
#print(data)
values = label_split()[:,2:]
values_label = np.array(['value1','value2'])
insert = pd.DataFrame(values, columns=values_label)
result = pd.concat([insert,data], axis = 1)
result.to_csv(r'C:\Users\Will\Desktop\Will\covar10x10split2.txt', sep=" ")
