# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 09:54:43 2017

@author: Chelley
"""
import numpy as np
from scipy import stats
import pandas as pd

def residualWhiteNoiseTest(residuals, arModelOrder, channelNums,
                  logger=None):
    """
    After applying ARIMA modeling to raw data, the innovations need to be
    tested to be virtually indistinguishable from white noise.  To test this,
    we check the innovation series for (1) stationarity, and (2)
    nonautocorrelation.  Innovation series that fail either of these two
    tests will be replaced with NaN.

    As a result of this white noise test, for MEG data, sensors that are dead
    and/or flat will be found with the nonautocorrelation check.  Also,
    sensors with artifacts will be found using the stationarity check.

    Inputs:
    -------
    residuals : float np.array
        Residual data array where rows = channels, columns = samples
    arModelOrder : int np.array
        AR model orders where each value is a channel's max autocorrelation
        lag to test. 
        For example, If the residuals were prewhitened with ARIMA(50,1,3), 
        use arModelOrder=50
    channelNums : int np.array
        Array of channel label numbers. (The first channel in the data file
        would begin at 1 not 0)
    logger : logging.Logger
        Log from pipeline. Default is a null logger.

    Outputs:
    -------
    whiteTestMatrixPassFail : float np.array
        The first column labels the channel numbers starting from index 1.
        Subsequent columns describe the pass/fail quality of each stage 
        of the white noise test, where 0 = passed and 1 = failed:
        1) ARIMA modeling failure 
           (There was an error in obtaining the residuals. Values for all other
           tests will be zero)
        2) Extreme values 
           (More than 5 channels exceeded the upper and lower fence)
        3) KS Test
           (Kolmogorov–Smirnov test (alpha=0.01))
        4) ACF Counts 
           (The number of significant ACF lags (alpha=0.01/AR) up to
              AR is greater than x, where x = round(AR * 0.1))
        5) PACF Counts 
           (The number of significant PACF lags (alpha=0.01/AR) up 
              to AR is greater than x, where x = round(AR * 0.1))
        6) Ten Part t-test 
           (Split the residuals into 10 parts, count n = number of
              parts that have a nonzero mean (alpha=0.001), if n > 1, 
              then fail)
        7) Ten Part Bartlett's test 
           (Split the residuals into 10 parts, count n = number of 
              parts that have a variance different from the whole
              series (alpha=0.001), if n > 1, then fail)
    residuals : float np.array
        Residual data array where  rows = channels, columns = samples
        Failed channels will be NaN
    stationarityFail : int np.array
        Array of channel numbers that failed the stationarity portion of this
        test. Channel numbers are indexed starting at 1, not 0
    nonautocorrelationFail : int np.array
        Array of channel numbers that failed the nonautocorrelation portion
        of this test. Channel numbers are indexed starting at 1, not 0
    acfCount : int np.array
        The number of significant ACF lags (alpha=0.01/AR) up to AR is 
        greater than x, where x = round(AR * 0.1)
    pacfCount : int np.array
        The number of significant PACF lags (alpha=0.01/AR) up to AR is 
        greater than x, where x = round(AR * 0.1)
    stationarityFailCount : int np.array
        Count of extreme values exceeding the upper and lower fence
    ks : int np.array
        KS Test
	  Kolmogorov–Smirnov test for normal distribution at alpha=0.01
    manyTtest : int np.array
        Split the residuals into 10 parts, count n = number of parts that 
        have a nonzero mean (alpha=0.001), if n > 1, then fail)
    bartlettCount : int np.array
        Split the residuals into 10 parts, count n = number of parts that have 
        a variance different from the whole series (alpha=0.001)
    References:
    -------
    Bartlett, M. S. 1946. On the theoretical specification of sampling
    properties of autocorrelated time series. Journal of Royal Statistical
    Society, Series B, 8, 27-27.
    """
    import numpy as np
    from scipy.stats import t
    from statsmodels.tsa.stattools import pacf
    from statsmodels.tsa.stattools import acf as acfProcedure
    from scipy.stats import kstest

    # Handle default logger
    if logger is None:
        import logging
        logger = logging.getLogger('nullLog')
        if logger.handlers == []:
            logger.addHandler(logging.NullHandler())

    # Ensure data is the correct type, shape, and is row major order
    residuals = np.asarray(residuals, dtype=np.float64, order='C')
    if len(residuals.shape) == 1:
        logger.debug('Data must be an np.array of 2 dimensions. ' +
                     'Converting data into a one channel 2 dimensional array.')
        residuals = np.array([residuals], order='C')
    (nChannels, nSamples) = residuals.shape

    # Initialize lists of failed channels
    stationarityFail = np.array([], dtype='i4')
    nonautocorrelationFail = np.array([], dtype='i4')
    acfCount = np.ones((nChannels))*np.nan
    pacfCount = np.ones((nChannels))*np.nan
    stationarityFailCount = np.ones((nChannels))*np.nan
    ks = np.ones((nChannels))*np.nan
    manyTtest = np.ones((nChannels))*np.nan
    bartlettCount = np.ones((nChannels))*np.nan

    # Stationarity check
    for jj in range(nChannels):
        if any(np.isnan(residuals[jj, :])):
            # If statsmodels failed to solve, some samples will be NaN
            logger.info('Innovations for sensor %s NaN.', channelNums[jj])
            residuals[jj, :] = np.nan
            continue
        if all(residuals[jj, :] == 0):
            logger.info('Innovations for sensor %s Zeroed. Converting to NaN.',
                        channelNums[jj])
            # Sensor labels indexed starting at 1
            residuals[jj, :] = np.nan
            continue
        # (1) Take one innovation series distribution and find the 25th &
        #     75th percentiles, along with the inter quartile range.  Then
        #     using these, find the upper and lower fence.
        # Y = np.percentile(X, P) returns the P percentile of the values in X

        seventyfifthpercentile = np.percentile(residuals[jj, :], 75)
        twentyfifthpercentile = np.percentile(residuals[jj, :], 25)
        interquartilerange = seventyfifthpercentile-twentyfifthpercentile
        lowerfence = twentyfifthpercentile-3*interquartilerange
        upperfence = seventyfifthpercentile+3*interquartilerange

        # (2) Next, count the innovations below the lower fence and above the
        #     upper fence.
        count = 0
        for ii in range(nSamples):
            innovation = residuals[jj, ii]
            if innovation > upperfence or innovation < lowerfence:
                count = count + 1

        #  (3) Finally, if the count is greater than 5 (MEG),
        #      the innovation series failes the stationarity test and is
        #      replaced with NaN.  This is based on heuristics completed
        #      by Apostolos Georgopoulos July 2014 (~7/16/2014 to 7/25/2014).

        if count > 5:
            logger.debug('Innovations for sensor %s failed stationarity test.',
                        channelNums[jj])
            stationarityFail = np.append(stationarityFail, channelNums[jj])
            #continue
            logger.debug('Innovations for sensor %s failed ' +
                        'stationarity test: count = %s', channelNums[jj], count)
        stationarityFailCount[jj] = count
        
            
        # KS Test
        kstat,pStat = kstest(residuals[jj, :],'norm')
        ks[jj] = int(pStat<0.01)
        
        # Constant mean test
        tenthRes = np.split(residuals[jj, :],range(int(len(residuals[jj, :])/10),
                                len(residuals[jj, :]),
                                int(np.ceil(len(residuals[jj, :])/10))))
        nonZeroCount = 0
        for sub in tenthRes:
            tStat, pStat = stats.ttest_1samp(sub, 0)
            nonZeroCount += int(pStat<=0.001)
        manyTtest[jj] = nonZeroCount
        
        # Constant variance Test
        nonZeroCount = 0
        for ii in range(10):
            thisSub = tenthRes[ii]
            tStat, pStat = stats.bartlett(residuals[jj, :], thisSub)
            nonZeroCount += int(pStat<=0.001)
        bartlettCount[jj] = nonZeroCount
        

        # Nonautocorrelation check
        if all(np.isnan(residuals[jj, :])):
            logger.info('Innovations for sensor %s NaN.', channelNums[jj])
            continue

        # (1) Take one innovation series and calculate the autocorrelation
        #     for AR lags.
        #nLags = arModelOrder[jj]
        #d = nSamples*np.ones(2. * nSamples - 1.)
        #fullAvf = np.correlate(residuals[jj, :],
        #                       residuals[jj, :],
        #                       'full') / d
        #avf = fullAvf[nSamples-1:nSamples+nLags]
        #acfArray = avf/avf[0]  # normalize for lag 0 = 1
        
        acfArray = acfProcedure(residuals[jj,:], nlags=arModelOrder[jj])
        
        # (2) Find the standard error of the sample autocorrelation.  This is
        #     computted using Bartlett's formula (see reference).

        # Under the assumption that the true MA order of the process is k-1,
        # the approximate variance of r(k) is:
        # SE( r(k) ) ~= sqrt( (1/n) * (1+2*sum((from l = 1 to k-1) r(l)^2))
        # The following was tested against SPSS Version 21 on Windows 7.
        standard_error = np.nan*np.ones(len(acfArray))
        # Calculate for each lag, other than zero-lag
        for kk in range(1, len(acfArray)):
            kkm1 = kk-1
            r = 0.
            # At lag 1, there are no previous autocorrelations so r = 0.
            if kkm1 == 0:
                standard_error[kk] = np.sqrt(1./nSamples)
            else:
                for lk in range(1, kk):
                    r = r + acfArray[lk]*acfArray[lk]
                standard_error[kk] = np.sqrt((1. + 2. * r) / nSamples)

        # (3) Calculate the statistic at each lag and count those above 2
        count = 0
        alpha = 0.01/arModelOrder[jj]
        degrees_of_freedom = nSamples-1
        for ii in range(1, len(acfArray)):  # Except zero-lag
            tval = (abs(acfArray[ii]))/standard_error[ii]

            # Compute the correct p-value for the two-tailed test
            p = 2.0 * t.cdf(-abs(tval), degrees_of_freedom)

            # Determine if the actual significance exceeds the
            # desired significance
            if p <= alpha:
                count = count + 1
                
        # (4) Finally, if the count is greater than 10% of the AR order, the
        #     innovation series fails the nonautocorrelation test and is
        #     replaced with NaN.
        if count > np.round(0.1*arModelOrder[jj]) or any(np.isnan(acfArray)):
            logger.debug('Innovations for sensor %s failed ' +
                        'nonautocorrelation test: count=%s acf? %s', channelNums[jj], count, acfArray)
            # Sensor labels indexed starting at 1
            nonautocorrelationFail = np.append(nonautocorrelationFail,
                                               channelNums[jj])
                                               
                                               
        acfCount[jj] = count
        
        
        # PACF Count
        pacfArray = pacf(residuals[jj, :],nlags=int(arModelOrder[jj]))
        # (2) Find the standard error of the sample autocorrelation.  This is
        #     computted using Bartlett's formula (see reference).

        # Under the assumption that the true MA order of the process is k-1,
        # the approximate variance of r(k) is:
        # SE( r(k) ) ~= sqrt( (1/n) * (1+2*sum((from l = 1 to k-1) r(l)^2))
        # The following was tested against SPSS Version 21 on Windows 7.
        standard_error = np.nan*np.ones(len(pacfArray))
        # Calculate for each lag, other than zero-lag
        for kk in range(1, len(pacfArray)):
            kkm1 = kk-1
            r = 0.
            # At lag 1, there are no previous autocorrelations so r = 0.
            if kkm1 == 0:
                standard_error[kk] = np.sqrt(1./nSamples)
            else:
                for lk in range(1, kk):
                    r = r + pacfArray[lk]*pacfArray[lk]
                standard_error[kk] = np.sqrt((1. + 2. * r) / nSamples)

        # (3) Calculate the statistic at each lag and count those above 2
        count = 0
        alpha = 0.01/arModelOrder[jj]
        degrees_of_freedom = nSamples-1
        for ii in range(1, len(pacfArray)):  # Except zero-lag
            tval = (abs(pacfArray[ii]))/standard_error[ii]

            # Compute the correct p-value for the two-tailed test
            p = 2.0 * t.cdf(-abs(tval), degrees_of_freedom)

            # Determine if the actual significance exceeds the
            # desired significance
            if p <= alpha:
                count = count + 1
        pacfCount[jj] = count
        
    logger.info('White noise test completed.')
    
#    Make matrix of qualities that passed or failed the test
    whiteTestMatrixPassFail = np.nan*np.ones((nChannels,8))
    whiteTestMatrixPassFail[:,0] = range(1,nChannels+1)
    for ch in range(nChannels):
        # pre-pipeline failure
        r = residuals[ch]
        # ARIMA modeling error
        whiteTestMatrixPassFail[ch,1] = int((np.isnan(r[0]) or all(r==0)))
    # Extreme values
    whiteTestMatrixPassFail[:,2] = [int(np.isnan(x) or x>5) for x in stationarityFailCount]
    # KS test for gaussian distribution
    whiteTestMatrixPassFail[:,3] = ks
    # ACF threshold test
    whiteTestMatrixPassFail[:,4] = [int(np.isnan(x) or x>np.round(0.1*arModelOrder[0])) for x in acfCount]
    # PACF threshold test
    whiteTestMatrixPassFail[:,5] = [int(np.isnan(x) or x>np.round(0.1*arModelOrder[0])) for x in pacfCount]
    # T-test for constant zero mean
    whiteTestMatrixPassFail[:,6] = [int(np.isnan(x) or x>1) for x in manyTtest]
    # Bartlett's test for constant variance
    whiteTestMatrixPassFail[:,7] = [int(np.isnan(x) or x>1) for x in bartlettCount]
    
    # Convert those channels which were removed and those which failed
    # to only fail the first 2 tests
    whiteTestMatrixPassFail[np.where(whiteTestMatrixPassFail[:,1] == 1)[0],2:] = 0.
    whiteTestMatrixPassFail[np.where(whiteTestMatrixPassFail[:,2] == 1)[0],3:] = 0.
    
    whiteTestMatrixLabels = ['Channel#',
        'Modeling_Failure','Extreme_Values','KS_Test',
        'ACF_Counts','PACF_Counts','Ten_Part_T_Test',
        'Ten_Part_Bartlett_Test']
    whiteTestPassFail_df = pd.DataFrame(data = whiteTestMatrixPassFail,
                                              columns=whiteTestMatrixLabels)
                                              
    iswhite = whiteTestMatrixPassFail[:,1:].sum(1)<=0

    return (whiteTestPassFail_df, iswhite)
            
# Test random noise:
nChannels = 3 # number of channels

# As a demonstration, this is the white noise test on some random noise
# generate random input: 
mu, sigma = 0, 1 # mean and standard deviation
s = np.random.normal(mu, sigma, (nChannels, 60000))

# run white noise test
(whiteTestPassFail, iswhite) = residualWhiteNoiseTest(s, 
                           np.tile([50],nChannels), 
                           np.arange(nChannels)+1)
                           
                           
                           
#print(whiteTestPassFail)


### test linear data
s = np.array([np.arange(100),np.arange(1,101),np.arange(2,102)])

# run white noise test
(whiteTestPassFail, iswhite) = residualWhiteNoiseTest(s, 
                           np.tile([5],nChannels), 
                           np.arange(nChannels)+1)
                           
#print(whiteTestPassFail)