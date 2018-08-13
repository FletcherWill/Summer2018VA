"""
Spyder Editor

This is a temporary script file.
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 10:42:58 2017
 
@author: eric
"""
 
import numpy as np
import sys
sys.path.append(r'E:\Users\Will\Desktop\Will')
import Test_Correlated_Noise as tcn
import IDtracker as idt
from random import randint, seed, uniform
import math as m
import pickle
import time as t
import matplotlib.pyplot as plt

#import statsmodels.api as sm
#import matplotlib.pyplot as plt
#from statsmodels.tsa.stattools import acf, pacf

t1= t.time()
seed(1234567)
np.random.seed(1234567)
ids = idt.IDtracker()
 
def new_array(size):
    """
    input: intended length of array
    output: array of randomly generated data with mean 0 and standard deviation 1
    """
    x = np.random.normal(0,1,size)
    return np.asarray(x)
    
def cor_2x2(mat):
    """
    input: matrix
    output: boolean value that is True if the input matrix can be used as a 2x2 correlation matrix
    """
    length = 2
    flag = True
    if mat.shape[0] != 2:
        flag = False
    if mat.shape[1] != 2:
        flag = False
    for i in range(0, length):
        for j in range(i, length):
            if i == j:
                if mat.item((i,j)) != 1:
                    flag = False
            else:
                if mat.item((i,j)) != mat.item((j,i)):
                    flag = False
    return flag
    
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
    
    
 
 
def method_1(cor_mat, out_len):
    """Input is correlation matrix and output is correlated white noise"""
    if cor_2x2(cor_mat):
        x_one = new_array(out_len)
        x_two = new_array(out_len)
        c = cor_mat.item((0,1))
        y = c * x_one + ((1 - c**2)**0.5) * x_two
        return np.matrix([x_one,y]).T
    else:
        raise ValueError('Not a 2x2 correlation Matrix')
     
     
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
    hasnan = any(np.isnan(output.flatten()))
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
        
correlation = np.matrix([[1, 0.8],[0.8, 1]])
 
C = np.matrix([ [1.0, -0.6, 0.9],  
              [-0.6, 1.0, 0.9],  
              [ 0.9, 0.9, 1.0]])

B = np.matrix([ [1.0, 0.6, 0.3],  
              [0.6, 1.0, 0.5],  
              [ 0.3, 0.5, 1.0]])    
 
#print(method_1(correlation,10))      
#print(method_2(C,5))
 
def frange(start, stop, step):
     """range function that accepts floats"""
     i = start
     while i < stop:
         yield i
         i += step
 
def search_2x2(length):
    """
    input: length of time series to be tested
    output: list of correlation values that failed to generate white noise when plugged into method 2
    """
    fail_test = []
    for a in frange(-1, 1, .1):
        cor = np.matrix([[1, a],[a, 1]])
        if test(method_2(cor, length)) == 1:
            fail_test.append(a)
    return fail_test
    
def search_3x3(length):
    """
    input: length of time series to be tested
    output: list of correlation matrices that failed to generate white noise when plugged into method 2
    """
    fail_test = []
    cause_error = []
    pas = 0
    fail = 0
    error = 0
    for a in frange(-1, 1, .1):
        for b in frange(-1, 1, .1):
            for c in frange(-1, 1, .1):
                cor = np.matrix([ [1.0, a, b],  
                [a, 1.0, c],  
                [ b, c, 1.0]])
                try:    
                    if test(method_2(cor, length)) == 1:
                        fail = fail + 1
                        fail_test.append((a,b,c))
                    else:
                        #print('pass')
                        pas = pas + 1
                except ValueError:
                    cause_error.append((a,b,c))
                    error = error + 1
    print('Pass: ' +str(pas)+'  Fail: ' +str(fail) + ' Error: ' + str(error))
    print(float(pas) / (pas + fail + error))
    return (fail_test, cause_error)
            
def test(mat):
    """
    input: matrix
    output: returns 0 if input matrix is white noise and 1 if any series in it isn't
    """
    size = mat.shape[0]
    names = []
    ars = []
    for value in range(size):
        names.append(value + 1)
        ars.append(5)
    (whiteTestPassFail, iswhite) = tcn.residualWhiteNoiseTest(mat,np.asarray(ars), np.asarray(names))
    flag = 0
    for value in iswhite:
        if value == False:
            #print(whiteTestPassFail)
            flag = 1
    return flag

def trim_mat(mat, leng = 0):
    """
    input: matrix of time series and number (n) of time series that should remain (if second argument is missing, the input matrix will be returned)
    output: matrix with n time series selected from input matrix
    """
    if leng > mat.shape[0] or leng < 0:
        raise ValueError('Can not create a matrix of given length')
    matrix = mat
    if leng != 0 and matrix.shape[0] > leng:
        while matrix.shape[0] > leng:
            gone = randint(0, matrix.shape[0]-1)
            matrix = np.delete(matrix, gone, 0)
    return matrix

def sum_products(mat, leng = 0):
    """
    input: matrix of time series and number (n) of time series that should remain (if second argument is missing, all time series will remain)
    output: time series where each point is the sum of products of all pairs of values at the given time for all remaining time series
    """
    if leng == 1:
        raise ValueError('Can not sum a matrix with one row')
    matrix = trim_mat(mat, leng)
#    sums = np.zeros(mat.shape[1])
#    for a in range(matrix.shape[0]-1):
#        for b in range(a+1,matrix.shape[0]):
#            array_a = np.squeeze(np.asarray(matrix[a]))
#            array_b = np.squeeze(np.asarray(matrix[b]))
#            sums = sums + (array_a * array_b)
#    for col in range(len(sums)):
#        temp = 1
#        for row in range(matrix.shape[0]):
#            temp = temp * matrix[row,col]
#        sums[col] = temp
    sums = np.prod(matrix, axis=0)
    print(sums.shape)
    return (sums, np.corrcoef(matrix))
            
def cst1(mat, leng=0):
    """
    input: matrix of time series and number (n) of time series that should remain
    (if second argument is missing, all time series will remain) before matrix entered into sum_products()
    output: correlation between white noise and product of white noise and time series from sum_products()
    """
    (series, corr) = sum_products(mat, leng)
    wn = np.random.uniform(-1,1,len(series))
    return np.corrcoef(wn,wn * series)[1,0]

def cst2(series, inputFreq):
    """
    input: series that will be used as network
    output: correlation between white noise and product of white noise and time series
    """
    if inputFreq is None: # input is white noise
        wn = np.random.uniform(-1,1,len(series))
    else:
        t = np.linspace(0, len(series)/1000, len(series)) # time in seconds where each point is a ms
        w = 2.0*np.pi * inputFreq
        wn = np.sin(w * t)
    return (np.corrcoef(wn,wn * series)[1,0], wn)


def get_rms(series):
    """
    input: series
    output: root mean square
    """
    sum_sq = 0
    for value in series:
        sum_sq = sum_sq + value ** 2
    return m.sqrt(sum_sq / len(series))

def rand_neg(cor_mat, percent = .5):
    if percent == 0:
        return cor_mat
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
        

    
def get_col(cor_mat, leng, index, num_series=0, inputFreq=None, neg_fraction=0, std=None):
    corMatID = ids.index(cor_mat)
    if std is not None:
        corvals = np.random.normal(cor_mat[0][1], std,int(cor_mat.shape[0]*(cor_mat.shape[0]-1)/2)).round(5)
        corvals[corvals>1] = 1;corvals[corvals<-1] = -1;
        cor_mat = np.ones(cor_mat.shape)
        cor_mat[np.triu_indices(len(cor_mat),1)] = corvals
        cor_mat[np.triu_indices(len(cor_mat),1)[1],np.triu_indices(len(cor_mat),1)[0]] = corvals
    print(cor_mat.shape)

    series = method_2(rand_neg(cor_mat,neg_fraction), leng)
    
    multiplier = 0.5#1.0
    series = series * multiplier
    
    (net_series, mat_cor) = sum_products(series, num_series)
#    plt.plot(net_series);plt.show()
    plt.figure();plt.matshow(cor_mat)
    ns = num_series
    if ns == 0:
        ns = cor_mat.shape[0]
    
    (ioc, wn) = cst2(net_series, inputFreq)
#    multiplier = np.max(np.abs(wn)) / np.max(np.abs(net_series))
#    multiplier = np.std(wn) / np.std(net_series)
    multiplier = 1.0
    rms = get_rms(wn - (net_series * multiplier * wn))
#    print(rms, end=', ')
    ids.track_rms(corMatID, rms)
#    return np.array([index, len(net_series), cor_mat.shape[0], ns, corMatID, np.corrcoef(series)[1,0], ioc, rms])
    real_cor = np.corrcoef(series)
    mean_real_cor = real_cor[np.triu_indices(len(real_cor),1)].mean()
    mean_selected_cor = mat_cor[np.triu_indices(len(mat_cor),1)].mean()
    pos_cor = all_pos(real_cor)
    mean_pos_cor = pos_cor[np.triu_indices(len(pos_cor),1)].mean()
    col = np.array([index, len(net_series), cor_mat.shape[0], ns, corMatID, cor_mat[0,1], mean_real_cor, mean_pos_cor, mean_selected_cor, ioc, rms, np.sqrt(np.mean(np.square((wn))))])
    print('index: %s'%index)
    print('col: %s'%col)
    print('mean_selected_cor',mean_selected_cor)
    print('target_cor',cor_mat[0,1])
    return col
    
def output(cor_mats, lengs, num_series, indices, inputFreq=None, neg_fraction=0, std=None):
    data = np.zeros((len(indices),12))
    for value in range(len(indices)):
        data[value] = get_col(cor_mats[value], lengs[value], indices[value], num_series[value], inputFreq, neg_fraction,std=std)
    return data


        
def create_cor_mat(size, high=1, low=-1):
    cor_mat = np.zeros((size,size))
    for a in range(0,cor_mat.shape[0]):
        for b in range(a,cor_mat.shape[1]):
            if a == b:
                cor_mat[a,b] = 1
            else:
                temp=uniform(low,high)
                cor_mat[a,b] = temp
                cor_mat[b,a] = temp
    return cor_mat

def cor_mats_2x2(inc = .1, end = 1):
    cor_mats = []
    for a in frange(0, end, inc):
        cor_mat = np.matrix([[1, a],[a, 1]])
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

def cor_mats_nxn_rand(size, inc = .01, end = 1, sd = .01):
    cor_mats = []
    for value in frange(0, end, inc):
        cor_mat = np.zeros((size,size))
        for a in range(size):
            for b in range(a,size):
                if a == b:
                    cor_mat[a,b] = 1
                else:
                    cor_mat[a,b] = np.random.normal(value, sd)
                    cor_mat[b,a] = cor_mat[a,b]
        cor_mats.append(cor_mat)
    return np.asarray(cor_mats)

def combine_cor_mats_nxn(start = 3, end = 100, inc = 3):
    cor_mats = []
    c2x2 = cor_mats_nxn(2)
    for value in range(len(c2x2)):
        cor_mats.append(c2x2[value])
    for size in frange(start, end, inc):
        cms = cor_mats_nxn(size)
        for value in range(len(cms)):
            cor_mats.append(cms[value])
    c100x100 = cor_mats_nxn(100)
    for value in range(len(c100x100)):
        cor_mats.append(c100x100[value])
    return cor_mats
    

def cor_mats_3x3():
    cor_mats = []
    for a in frange(0, 1, .1):
        for b in frange(0, 1, .1):
            for c in frange(0, 1, .1):
                cor_mat = np.matrix([ [1.0, a, b],  
                                     [a, 1.0, c],  
                                     [ b, c, 1.0]])
                if pos_eigs(cor_mat):
                    cor_mats.append(cor_mat)
    return np.asarray(cor_mats)

def cor_mats_3x3l():
    cor_mats = []
    for a in frange(0, .3, .001):
                cor_mat = np.matrix([ [1.0, a, a],  
                                     [a, 1.0, a],  
                                     [ a, a, 1.0]])
                if pos_eigs(cor_mat):
                    cor_mats.append(cor_mat)
    return np.asarray(cor_mats)

def cor_mats_4x4():
    cor_mats = []
    for a in frange(0, 1, .1):
        for b in frange(0, 1, .1):
            for c in frange(0, 1, .1):
                for d in frange(0, 1, .1):
                    for e in frange(0, 1, .1):
                        for f in frange(0, 1, .1):
                            cor_mat = np.matrix([ [1.0, a, b, d],  
                                     [a, 1.0, c, e],  
                                     [b, c, 1.0, f],
                                     [d, e, f, 1.0]])
                            if pos_eigs(cor_mat):
                                cor_mats.append(cor_mat)
    return np.asarray(cor_mats)

def cor_mats_248():
    cor_mats = np.array(np.zeros(10), dtype = object)
    cor_mat = np.zeros((248,248))
    for i in frange(0, 10, 1):
        for a in range(0,cor_mat.shape[0]):
            for b in range(0,cor_mat.shape[1]):
                if a == b:
                    cor_mat[a,b] = 1
                else:
                    cor_mat[a,b] = i 
        cor_mats[i] = cor_mat.copy()
    return np.asarray(cor_mats)
   

def pos_eigs(mat):
    vals = np.linalg.eigvals(mat)
    flag = True
    for item in vals:
        if item < 0:
            flag = False
    return flag

def x10(series, n=31):
    new_series = []
    for thing in series:
        for a in range(n):
            new_series.append(thing)
    return np.asarray(new_series)

def y10(series, n=100):
    new_series = []
    for a in range(n):
        for thing in series:
            new_series.append(thing)
    return np.asarray(new_series)

def find_mins(num = 1000):
    for n in range(num):
        output(x10(cor_mats_2x2(.001, 0.2)), x10(np.ones(200, dtype=np.int) * 1000), x10(np.zeros(200)), np.arange(1, 20001))
        rms_means = ids.get_rms_mean()
        mv = 100.0
        mp = 1.0
        place = 0.0
        for value in rms_means:
            if value < mv:
                mv = value
                mp = place
            place = place + 0.001
        ids.track_min(mp,mv)
        place =  0
        ids.clean()
    return ids.get_mins()

def find_means(num = 1000):
    data = np.zeros((num, 200))
    for n in range(num):
        output(x10(cor_mats_2x2(.001, 0.2)), x10(np.ones(200, dtype=np.int) * 1000), x10(np.zeros(200)), np.arange(1, 20001))
        data[n] = ids.get_rms_mean()
    return data

def page_check():
    table = np.zeros((100000,8))
    b = 0
    for a in frange(0,1,.01):
        cor_mat = np.matrix([[1, a],[a, 1]])
        temp = np.zeros((1000,8))
        series = method_2(cor_mat, 1000)
        temp[:,1] = series[0]
        temp[:,2] = series[1]
        temp[:,0] = np.arange(1,1001)
        temp[:,3] = np.ones(1000) * abs(np.corrcoef(series[0],series[1])[1,0])
        temp[:,4] = np.ones(1000) * a
        wn = np.random.normal(0,1,1000)
        temp[:,5] = wn
        network = sum_products(series)
        output = network * wn
        temp[:,6] = output
        temp[:,7] = get_rms((output) - wn)
        table[b*1000:(b+1)*1000,:] = temp
        b = b + 1
    return table
    

D = np.matrix([ [1.0, 0.6, 0.3, 0.5],  
               [0.6, 1.0, 0.5,0.8],  
               [ 0.3, 0.5, 1.0,0.4],
               [0.5,0.8,0.4,1.0]])
    
E = np.matrix([ [1.0, .500001, .500001,.500001,.500001],  
              [.500001, 1.0, 0, 0,0],  
              [ .500001, 0, 1.0, 0,0],
              [.500001, 0, 0, 1,0],
              [.500001,0,0,0,1]])

#data = find_means()    
 
#########################################################################################################################################################################  
#output(a,b,c,d)
#
#input:
#a: array of correlation matrices that will be used by method 2
#b: array of integers, ith integer will be the length of the (multiple) series created by method 2 and the ith correlation matirx
#c: array of integers, ith integer will be the number of series (created by method 2) randomly selected that will be used to make network (value of 0 uses all series)
#d: array of integers, ith integer will be the index of the ith trial

#x10(series): duplicates each element 10 (or whatever it is set to) times
#example: x10((1,2,3),4) => (1,1,1,1,2,2,2,2,3,3,3,3)
#note: default may not be set to ten, check method
#
#y10(series): duplicates full series 10 (or whatever it is set to) times
#example: x10((1,2,3),4) => (1,2,3,1,2,3,1,2,3,1,2,3)
#note: default may not be set to ten, check method
#
#output: ix11 matrix, ith row corresponds to ith trial
#col1: index
#col2: length of (multiple) series created by method 2
#col3: number of series created by method 2
#col4: number of series used to create network
#col5: corMatID (trial using same corMat will have same ID)
#col6: cormat[1,0]
#col7: mean of correlations between all pairs of series created by method 2
#col8: mean of absolute value of correlations between all pairs of series created by method 2
#col9: mean of correlatoins between all pairs of series used in creating network series
#col10: input-output correlation
#col11: rms
#
#########################################################################################################################################################################

#E = create_cor_mat(248, 0, .1)
#data_min = output(x10(cor_mats_2x2(.001, 0.2)), x10(np.ones(200, dtype=np.int) * 1000), x10(np.zeros(200)), np.arange(1, 20001))
#data2x2 = output(x10(cor_mats_2x2()), x10(np.ones(11, dtype=np.int) * 1000), x10(np.zeros(11)), np.arange(1, 111))    
#data3x3 = output(x10(cor_mats_3x3()), x10(np.ones(916, dtype=np.int) * 1000), x10(np.zeros(916)), np.arange(1, 91601))
#data4x4 = output(x10(cor_mats_4x4()), x10(np.ones(555828, dtype=np.int) * 1000), x10(np.zeros(555828)), np.arange(1, 55582801))
#data = output(np.array([B,B,B]), np.array([100,100,100]), np.array([0,0,0]),np.array([1,2,3]))
#data2x2HD = output(x10(cor_mats_2x2(.001, 0.3)), x10(np.ones(300, dtype=np.int)*60000), x10(np.zeros(300)), np.arange(1, 300001))
#data3x3l = output(x10(cor_mats_3x3l()), x10(np.ones(300, dtype=np.int)*1000), x10(np.zeros(300)), np.arange(1, 30001))
#data248l = output(x10(cor_mats_248()), x10(np.ones(300, dtype=np.int)*1000), x10(np.zeros(300)), np.arange(1, 30001))
#data2to100 = output(x10(combine_cor_mats_nxn()), x10(np.ones(3500, dtype=np.int) * 1000), x10(np.zeros(3500)), np.arange(1, 35001))
#data100 = output(x10(x10(cor_mats_nxn(100,.01,.2)),50), x10(x10(np.ones(20, dtype=np.int) * 1000), 50), x10(y10(np.asarray(range(2,101)),20),50), np.arange(1, 99001))
#data3x3_neg = output(x10(cor_mats_nxn(3,0.001,0.2),10), x10(np.ones(200, dtype=np.int) * 1000, 10), x10(np.zeros(200), 10), np.arange(2000))

#print(data2x2)
#np.savetxt(r'C:\Users\Will\Desktop\Will\index2x2.txt', data2x2, delimiter=" ")
#np.savetxt(r'C:\Users\Will\Desktop\Will\index3x3.txt', data3x3, delimiter=" ")
#np.savetxt(r'C:\Users\Will\Desktop\Will\data_min.txt', data_min, delimiter=" ")
#np.savetxt(r'C:\Users\Will\Desktop\Will\data3x3l.txt', data3x3l, delimiter=" ")
#np.savetxt(r'C:\Users\Will\Desktop\Will\data4x4.txt', data4x4, delimiter=" ")
#np.savetxt(r'C:\Users\Will\Desktop\Will\index248l.txt', data248l, delimiter=" ")
#np.savetxt(r'C:\Users\Will\Desktop\Will\rms_means.txt', meansrms, delimiter=" ")
#np.savetxt(r'C:\Users\Will\Desktop\Will\rms_means_mat.txt', data, delimiter=" ")
#np.savetxt(r'C:\Users\Will\Desktop\Will\data2to100p-2.txt', data2to100, delimiter=" ") 

#np.savetxt(r'C:\Users\Will\Desktop\Will\data3x3_neg.txt', data3x3_neg, delimiter=" ")
  
#with open(r"C:\Users\Will\Desktop\Will\ids248to248_HD", "wb") as f:
#    pickle.dump(ids.ids, f)
 
#with open(r"C:\Users\Will\Desktop\Will\ids100", "wb") as f:
    #pickle.dump(ids.ids, f)

#with open(r"C:\Users\Will\Desktop\Will\ids248to248_HD", "rb") as f:
#    x = pickle.load(f)

#page_check = page_check()

#np.savetxt(r'C:\Users\Will\Desktop\Will\page_check.txt', page_check, delimiter=" ")
for irun in np.arange(2,50):
    ntrials = 1
    ntypes=1
    cby = 0.005 # correlation interval to explore
    cmax = 1.0
    nchans = 248
    inputFreq = irun*500/50
#    data248to248_HD = output(x10(x10(cor_mats_nxn(nchans,cby,cmax),ntypes),ntrials),
#                             x10(x10(np.ones(100, dtype=np.int) * 1000), ntrials),
#                             x10(y10(nchans*np.ones(ntypes),100),ntrials), 
#                             np.arange(1,(cmax/cby)*ntypes*ntrials+1),
#                             inputFreq=None, neg_fraction=0, std=None)
    inputFreq=None; neg_fraction=0; std=None
    data = np.zeros((int(cmax/cby*ntypes*ntrials),12))
    allcormats = x10(x10(cor_mats_nxn(nchans,cby,cmax),ntypes),ntrials)
    for value in range(data.shape[0]):
        data[value] = get_col(allcormats[value],
                            1000, value, 0, inputFreq, neg_fraction,std=std)
    
#    np.savetxt(r'C:\Users\Will\Desktop\Will\morefreq\data248to248_'+str(int(inputFreq))+'.txt', data248to248_HD, delimiter=" ")
#    np.savetxt(r'\\NOVA\1xchange\Will\CST_Project\output\2017-08-29\individualfrequencies\data248to248_'+str(int(inputFreq))+'.txt', data248to248_HD, delimiter=" ")
#    np.savetxt(r'\\NOVA\1xchange\Will\CST_Project\output\2017-08-30\samestd_halfneg\data248to248_'+str(int(irun))+'.txt', data248to248_HD, delimiter=" ")
#    np.savetxt(r'\\NOVA\1xchange\Will\CST_Project\output\2017-08-30\samestd_freq\data248to248_'+str(int(irun))+'.txt', data248to248_HD, delimiter=" ")
#    np.savetxt(r'\\NOVA\1xchange\Will\CST_Project\output\2017-08-30\samestd_distribution\data248to248_'+str(int(irun))+'.txt', data248to248_HD, delimiter=" ")
    np.savetxt(r'\\NOVA\1xchange\Will\CST_Project\output\2017-08-30\randomfull0to1_allDivby2\data248to248_'+str(int(irun))+'.txt', data, delimiter=" ")

    
t2 = t.time()
print(t2 - t1)   