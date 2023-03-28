# Sub-model_1: Convergent Cross Mapping (CCM)
#
# Paper:
# Detecting Causality in Complex Ecosystems. George Sugihara et al. 2012


import warnings
warnings.filterwarnings('ignore')

import skccm as ccm
import numpy as np
import pandas as pd
from skccm.utilities import train_test_split
from itertools import combinations
import math


def error(a, b):
    """
    Relative error to determine convergence
    
    Parameters
    ----------
    a: float
        The first of the two values
    b: float
        The second of the two values

    Return
    ------
    err: float
        The relative error
    """

    err = abs((a-b)/a)

    return err


def model_ccm(X, lag, embed, split_percent, num_iter=40, error_num=8, threshold=0.03):
    """
    Apply the libraty 'skccm' to detect causality with CCM
    for every two embedded time series.

    Parameters
    ----------
    X: 2d array
        Time series of shape (num_samples, num_features)
    lag : int
        Lag value as calculated from the first minimum of the mutual info
    embed : int
        Embedding dimension. How many lag values to take.
    split_percent: float
        Percent to use for training set
    num_iter: int
        Number of iterations of predictions
    error_num: int
        Number of values to determine convergence
    threshold: float
        The boundry to determine the convergence
    
    Returns
    -------    
    causality_strength: dict
        Keys: tuple
            The indexes of features with causality relationship, 
            e.g., (1,2) represents feature 1 is the cause of feature 2
        Values: float
            The causality strengths for corresponding features

    Example
    -------
    >>> X = [0,1,2,3,4,5,6,7,8]
    em = 3
    lag = 2
    >>> embed_vectors_1d
    features = [[0,2,4], [1,3,5], [2,4,6], [3,5,7], [4,6,8]]
    """

    num_features = X.shape[1]
    X_embed = []
    causality_strength = dict()
    
    # embed both time series
    for i in range(num_features):
        e = ccm.Embed(X[:,i])
        X_embed.append(e.embed_vectors_1d(lag,embed))
    
    # the list of no-repeated 2-length combinations of all features
    feature_combin = list(combinations(np.arange(0, num_features,1), 2))
    
    for i in feature_combin:
        
        # split the embedded time series
        x1tr, x1te, x2tr, x2te = train_test_split(X_embed[i[0]], X_embed[i[1]], 
                                                  percent=split_percent)
        # initiate the class
        CCM = ccm.CCM()  
        len_tr = len(x1tr)
        # library lengths to test
        lib_lens = np.arange(20, len_tr, len_tr/num_iter, dtype='int')  
        # test causation
        CCM.fit(x1tr,x2tr)
        x1p, x2p = CCM.predict(x1te, x2te, lib_lengths=lib_lens)
        sc1,sc2 = CCM.score()
        
        # compute the relative error
        
        error_1 = []
        error_2 = []
        for j in range(error_num-1):
            error_1.append(error(sc1[-(j+1)], sc1[-(j+2)]))
            error_2.append(error(sc2[-(j+1)], sc2[-(j+2)]))
        
        error_1 = [10. if math.isnan(x) else x for x in error_1]
        error_2 = [10. if math.isnan(x) else x for x in error_2]

        # determine the convergence
        if np.max(error_1)<threshold and sc1[-1]>=1e-4:
            sc_1 = (sc1[-1] + sc1[-2] + sc1[-3]) / 3
        else:
            sc_1 = 0.
        if np.max(error_2)<threshold and sc2[-1]>=1e-4:
            sc_2 = (sc2[-1] + sc2[-2] + sc2[-3]) / 3
        else:
            sc_2 = 0.
        
        # add the causality strength values to the dictionary
        if sc_1>sc_2 and sc_1>0:
            causality_strength[(i[0], i[1])] = round(sc_1, 4)
        if sc_2>sc_1 and sc_2>0:
            causality_strength[(i[1], i[0])] = round(sc_2, 4)

    return causality_strength


def ccm_partitions(X, num_parts=10, lag=1, embed=1, 
                   split_percent=0.75, num_iter=40, error_num=8,
                   convergence_threshold=0.03, threshold=0.3):
    """
    Detect causality with CCM for every two embedded time series 
    of all data partitions.

    Parameters
    ----------
    X: 2d array
        Time series of shape (num_samples, num_features)
    num_parts: int
        The number of data partitions
    lag : int
        Lag value as calculated from the first minimum of the mutual info
    embed : int
        Embedding dimension. How many lag values to take
    split_percent: float
        Percent to use for training set
    num_iter: int
        Number of iterations of predictions
    error_num: int
        Number of values to determine convergence
    convergence_threshold: float
        The boundry to determine the convergence
    threshold: float
        The boundry value to filter the causation relationships

    Returns
    -------    
    causal_ccm_parts: 3d array
        causality relationsips with shape 
        (num_partitions, num_features, num_features)
    """
    
    if type(X) == pd.core.frame.DataFrame:
        X = X.values
    n,m = X.shape
    causal_ccm_parts = []

    for i in range(num_parts):
        # split the dataset into partitions
        X_partition = X[int(n/num_parts*i):int(n/num_parts*(i+1)), :]

        # utilize the function above to detect causality
        causality_strength = model_ccm(X_partition, lag=lag, embed=embed, 
                                       split_percent=split_percent, 
                                       num_iter=num_iter, 
                                       error_num=error_num,
                                       threshold=convergence_threshold)
        indexes = list(causality_strength.keys())
        strengths = list(causality_strength.values())

        causal_strength_part = np.zeros([m,m])
        # insert values of causality strengths to the strength matrix
        for j in range(len(indexes)):
            i_0, i_1 = indexes[j]
            causal_strength_part[i_0][i_1] = strengths[j]
        # filter the causality strength
        causal_strength_part[causal_strength_part<threshold] = 0.
        causal_ccm_parts.append(causal_strength_part.tolist())

    return np.array(causal_ccm_parts)


