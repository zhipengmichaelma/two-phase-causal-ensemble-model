# Sub-model_4: Granger Causality (GC)


import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error
import scipy.stats
import numpy as np
import pandas as pd
import math


def adf_test(X, signif=0.05):
    """
    Check the stationarity of time series. If not, difference them.

    Parameters
    ----------
    X: dataframe
        Multivariable time series dataset
    
    Returns
    -------
    X_stationary: dataframe
        Differenced multivariable time series dataset
    """
    
    # Check the stationarity
    non_stationary = False
    for _, column in X.iteritems():
        r = adfuller(column, autolag='AIC')
        p_value = round(r[1], 4)
        if p_value > signif:
            non_stationary = True
    
    # if the time series are not stationary, difference them repeatedly
    # until they are stationary
    while (non_stationary):
        X = X.diff().dropna()
        for _, column in X.iteritems():
            r = adfuller(column, autolag='AIC')
            p_value = round(r[1], 4)
            if p_value <= signif:
                non_stationary = False
            else:
                non_stationary = True
                break
    
    X_stationary = X
    return X_stationary


def VecAutoReg(X, max_lag=6):
    """
    Vector Autoregression to get causality strengths for time series.

    Parameters
    ----------
    X: 2d array
        Time series with shape (num_features, 2)
    max_lag: int
        max time lag periods to consider
    
    Returns
    -------
    corr_1: float
        relative coefficient forecasting time series 1 with time series 2
    corr_2: float
        relative coefficient forecasting time series 2 with time series 1
    """

    # split the dataset into training and test data in line with max_lag
    nobs = max_lag
    X_train, X_test = X[0:-nobs], X[-nobs:]

    # train the VAR Model of selected order
    model = VAR(X_train)
    res = model.fit(maxlags=max_lag)
    lag_order = max_lag
    # forcast the VAR model
    input_data = X_train.values[-lag_order:]
    pred = res.forecast(y=input_data, steps=nobs)
    X_pred = (pd.DataFrame(pred, index=X_test.index, 
                           columns=X_test.columns + '_pred'))
    
    # calculate the relative coefficent as causality strength
    r_1 = scipy.stats.pearsonr(X_test.iloc[:,0].values, X_pred.iloc[:,0].values)
    corr_1 = r_1[0]
    r_2 = scipy.stats.pearsonr(X_test.iloc[:,1].values, X_pred.iloc[:,1].values)
    corr_2 = r_2[0]

    return corr_1, corr_2


def model_granger_causality(X, feature_names, max_lags=1, test='ssr_chi2test', 
                      verbose=False, signif=0.05, threshold = 0.3):    
    """
    Apply Granger causality test.

    Parameters
    ----------
    X: 2d array
        Time series of shape (num_samples, num_features)
    feature_names: list of str
        Names of all features
    max_lag: int
        Max time lag periods to consider
    test: object
        Significance test for Granger causality
    verbose: bool
        If True, results will be printed
    signif: float
        threshold for significance test
    threshold: folat
        The boundry value to filter the causation relationships

    Returns
    -------
    causality strength: 2d array
        causality relationsips with shape (num_features, num_features)
    """
    
    # df_gc to store the p_value of granger causality test,
    # df_corr to store the relative coefficient of VAR model
    df_gc = pd.DataFrame(np.zeros((len(feature_names), len(feature_names))), 
                         columns=feature_names, index=feature_names)
    df_corr = pd.DataFrame(np.zeros((len(feature_names), len(feature_names))), 
                           columns=feature_names, index=feature_names)

    # apply granger causality test for every two time series
    for c in df_gc.columns:
        for r in df_gc.index:
            
            # remove the influence of constant values
            q_max_partition_1, q_min_partition_1 = np.percentile(X[c], [98,2])
            q_max_partition_2, q_min_partition_2 = np.percentile(X[r], [98,2])

            if (q_max_partition_1-q_min_partition_1<1e-8) or (q_max_partition_2-q_min_partition_2<1e-8):
                print(c)
                print(r)
                print(min((q_max_partition_1-q_min_partition_1), (q_max_partition_2-q_min_partition_2)))
                df_gc.loc[r, c] = 1.
                df_gc.loc[c, r] = 1.
                df_corr.loc[r, c] = 0.
                df_corr.loc[c, r] = 0.
            
            else:
                gc_result = grangercausalitytests(X[[c, r]], 
                                                maxlag=max_lags, 
                                                verbose=verbose)
                p_values = [round(gc_result[i+1][0][test][1], 4) for i in range(max_lags)]
                if verbose: print(f'X = {r}, Y = {c}, P Values = {p_values}')
                min_p_value = np.min(p_values)
                df_gc.loc[r, c] = min_p_value
            
                # insert values of causality strengths to the strength matrix
                if list(df_gc.columns).index(c) >= list(df_gc.index).index(r) and r != c:
                    corr_1, corr_2 = VecAutoReg(X=X[[c,r]])
                    df_corr.loc[r, c] = round(abs(corr_1), 4) if abs(corr_1) > threshold else 0.
                    df_corr.loc[c, r] = round(abs(corr_2), 4) if abs(corr_2) > threshold else 0.

    for c in df_gc.columns:
        for r in df_gc.index:
            if df_gc.loc[r, c] >= signif:
                df_corr.loc[r, c] = 0.
    
    # if A->B and B->A are both causal, pick the better one
    for c in df_corr.columns:
        for r in df_corr.index:
            if c > r:
                if df_corr.loc[r, c] > df_corr.loc[c, r] > 0:
                    df_corr.loc[c, r] = 0. 
                if df_corr.loc[c, r] > df_corr.loc[r, c] > 0:
                    df_corr.loc[r, c] = 0.
    
    causality_strength = df_corr.values
    return causality_strength



def nan2zero(X):
    """
    Transfer nan values to zero values

    parameters
    ----------
    X: 2d array
    
    Returns
    -------
    X: 2d array
    """

    n,m  = X.shape
    for i in range(n):
        for j in range(m):
            X[i,j] = 0. if math.isnan(X[i,j]) else X[i,j]
            
    return X


def gc_partitions(X, num_parts, feature_names, max_lag=1, test='ssr_chi2test', verbose=False, signif=0.05, threshold=0.3):
    """
    Detect causality with GC for all data partitions.

    Parameters
    ----------
    X: 2d array
        Time series of shape (num_samples, num_features)
    num_parts: int
        The number of data partitions
    feature_names: list of str
        Names of all features
    max_lag: int
        Max time lag periods to consider
    test: object
        Significance test for Granger causality
    verbose: bool
        If True, results will be printed
    signif: float
        threshold for significance test
    threshold: folat
        The boundry value to filter the causation relationships

    Returns
    -------
    causal_gc_parts: 3d array
        causality relationsips with shape 
        (num_partitions, num_features, num_features)
    """
    
    n, m = X.shape
    causal_gc_parts = []
    for i in range(num_parts):
        # split the dataset into partitions
        X_temp = X[int(n/num_parts*i): int(n/num_parts*(i+1))]
        # utilize the function above to detect causality
        causality_strength = model_granger_causality(X=X_temp, 
                                               feature_names=feature_names, 
                                               test=test, 
                                               max_lags=max_lag, 
                                               verbose=verbose, 
                                               signif=signif,
                                               threshold=threshold)
        causality_strength = nan2zero(causality_strength)
        causal_gc_parts.append(causality_strength.tolist())
    
    causal_gc_parts = np.around(np.array(causal_gc_parts), 4)
    
    return causal_gc_parts
    







