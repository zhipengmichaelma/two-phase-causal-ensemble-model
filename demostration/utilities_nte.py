# Sub-model_2: Normalized Transfer Entropy (NTE)
#
# TE can be written as:
# TE(X->Y) = H(Yt|Yt-1:t-L) - H(Yt|Yt-1:t-L, Xt-1:t-L)
#
# NTE can be written as:
# NTE(X->Y) = (TE(X->Y) - TE_shuffle(X->Y)) / H(Yt|Yt-1:t-L)
#
# The computation of TE references the method in PyIF library
# 


import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from itertools import combinations
from PyIF import te_compute as te
from statsmodels.stats.weightstats import ztest as ztest
from sklearn.neighbors import KDTree 
from nose.tools import assert_true
from random import shuffle
import CPU_TE
import GPU_TE
from PyIF import helper
from PyIF.helper import make_spaces


def te_mapping(X):
    """
    Curve fitting of NTE values so that they can be compared with 
    the coefficents of other submodels.

    Parameters
    ----------
    X: 2d array
        The original causality strength matrix of TE

    Returns
    -------
    output: 2d array
        The mapped causality strength matrix of TE
    """

    a = 0.19604753  
    b = 6.01585935  
    c = 0.00793923 
    d = -0.29553602 
    e = 0.94236698
    output = a * np.log(b*X + c) + d*X + e
    # make sure the range is [0,1]
    output[output<=0.] = 0.
    output[output>=1.] = 1.

    return output


def nte_param_compute(X, Y, k=1, embedding=1, safetyCheck=False, GPU=False):
    """
    Parameters
    ----------
    X: 1d array
        The first time series of shape (num_samples,)
    Y: 1d array
        The second time series of shape (num_samples,)
    k: int
        The number of nearest neighbors
    embedding: int 
        The number of lag periods to consider
    safetyCheck: bool
        Boolean value when True, will check for unique values 
        and abort estimation if duplicate values are found
    GPU: bool
        Boolean value that when set to true will use CUDA compatiable GPUs

    Returns
    -------
    HY: float
        Shannon's entropy H(Yt|Yt-1:t-L)
    HYX: float
        Shannon's entropy H(Yt|Yt-1:t-L, Xt-1:t-L)
    TE: float
        The estumated transfer entropy
    """

    assert_true(k>=1, msg="K should be greater than or equal to 1")
    assert_true(embedding >= 1, msg='The embedding must be greater than or equal to 1')
    assert_true(type(X) == np.ndarray, msg='X should be a numpy array')
    assert_true(type(Y) == np.ndarray, msg='Y should be a numpy array')
    assert_true(len(X) == len(Y), msg='The length of X & Y are not equal')

    if safetyCheck and (not helper.safetyCheck(X,Y)):
        print("Safety check failed. There are duplicates in the data.")
        return None
    
    # Make Spaces
    xkyPts, kyPts, xkPts, kPts, nPts = make_spaces(X, Y, embedding=embedding)
    # Make Trees
    xkykdTree = KDTree(xkyPts, metric="chebyshev")
    kykdTree = KDTree(kyPts, metric="chebyshev")
    xkkdTree = KDTree(xkPts, metric="chebyshev")
    kkdTree = KDTree(kPts, metric="chebyshev")

    if GPU:
        HY, HYX, TE = GPU_TE.compute(xkykdTree, kykdTree, xkkdTree, kkdTree,
        xkyPts, kyPts, xkPts, kPts, nPts, X, embedding=embedding, k=k)
    else:
        HY, HYX, TE = CPU_TE.compute(xkykdTree, kykdTree, xkkdTree, kkdTree,
        xkyPts, kyPts, xkPts, kPts, nPts, X, embedding=embedding, k=k)

    return HY, HYX, TE


def model_nte(X, k=1, max_lag=1, safetyCheck=False, GPU=False):
    """
    Cumpute the normalized transfer entropy.
    
    Parameters
    ----------
    X: 2d array
        Time series of shape (num_samples, num_features)
    k: int
        The number of nearest neighbors
    max_lag: int 
        The number of lag periods to consider
    safetyCheck: bool
        Boolean value when True, will check for unique values 
        and abort estimation if duplicate values are found
    GPU: bool
        Boolean value that when set to true will use CUDA compatiable GPUs

    Returns
    -------
    causality_strength: dict
        Keys: tuple
            The indexes of features with causality relationship, 
            e.g., (1,2) represents feature 1 is the cause of feature 2.
        Values: float
            The causality strengths for corresponding features.
    """

    m, n = X.shape
    causality_strength = dict()
    # the list of no-repeated 2-length combinations of all features
    feature_combinations = list(combinations(np.arange(0,n,1), 2))

    # initialization for z-test
    s = [0]
    for i in range(10):
        s.append(int(m/10*(i+1)))
    
    for i in feature_combinations:
        # prepare the data for NTE computation
        x_1 = X[:,i[0]]
        xs_1 = x_1.copy()
        shuffle(xs_1)
        x_2 = X[:,i[1]]
        xs_2 = x_2.copy()
        shuffle(xs_2)
        
        # parameters for NTE computation
        HY_1, _, TE_1 = nte_param_compute(x_1, x_2, k=k, embedding=max_lag, 
                                          safetyCheck=safetyCheck, GPU=GPU)
        HY_2, _, TE_2 = nte_param_compute(x_2, x_1, k=k, embedding=max_lag, 
                                          safetyCheck=safetyCheck, GPU=GPU)
        _, _, TES_1 = nte_param_compute(xs_1, x_2, k=k, embedding=max_lag, 
                                        safetyCheck=safetyCheck, GPU=GPU)
        _, _, TES_2 = nte_param_compute(xs_2, x_1, k=k, embedding=max_lag, 
                                        safetyCheck=safetyCheck, GPU=GPU)

        NTE_1 = max(((TE_1 - TES_1) / HY_1), 0)
        NTE_2 = max(((TE_2 - TES_2) / HY_2), 0)

        # hyperthesis significance testing: z-test
        # split the sub_dataset into 10 partitions 
        # and compute the NTE separately
        # then test the two NTE-lists containing 10 NTE values

        test_1 = []
        test_2 = []
        for j in range(9):
            
            # initialization
            xtest_1 = x_1[s[j]:s[j+2]]
            xtest_2 = x_2[s[j]:s[j+2]]
            xs_test_1 = xtest_1.copy()
            shuffle(xs_test_1)
            xs_test_2 = xtest_2.copy()
            shuffle(xs_test_2)
            
            # parameters
            HY_test_1, _, TE_test_1 = nte_param_compute(x_1, x_2, k=k, 
                                                        embedding=max_lag, 
                                                        safetyCheck=safetyCheck, 
                                                        GPU=GPU)
            HY_test_2, _, TE_test_2 = nte_param_compute(x_2, x_1, k=k, 
                                                        embedding=max_lag, 
                                                        safetyCheck=safetyCheck, 
                                                        GPU=GPU)
            _, _, TES_test_1 = nte_param_compute(xs_1, x_2, k=k,
                                                 embedding=max_lag, 
                                                 safetyCheck=safetyCheck, 
                                                 GPU=GPU)
            _, _, TES_test_2 = nte_param_compute(xs_2, x_1, k=k, 
                                                 embedding=max_lag, 
                                                 safetyCheck=safetyCheck, 
                                                 GPU=GPU)

            NTE_test_1 = max(((TE_test_1 - TES_test_1) / HY_test_1), 0)
            NTE_test_2 = max(((TE_test_2 - TES_test_2) / HY_test_2), 0)
            test_1.append(NTE_test_1)
            test_2.append(NTE_test_2)
        
        # z_test, significance threshold is 0.05
        p_z = ztest(test_1, test_2, value=0)
        ToF_z = True if p_z[1] < 0.05 else False
        
        # add the causality strength values to the dictionary
        if ToF_z and  NTE_1>NTE_2 and NTE_1>0:
            causality_strength[(i[1],i[0])] = round(NTE_1, 4)
        if ToF_z and  NTE_2>NTE_1 and NTE_2>0:
            causality_strength[(i[0],i[1])] = round(NTE_2, 4)
           
    return causality_strength


def nte_partitions(X, num_parts=10, k=1, max_lag=1, 
                   safetyCheck=False, GPU=False, threshold=0.3):
    """
    Detect causality with NTE for every two embedded time series 
    of all data partitions.

    Parameters
    ----------
    X: 2d array
        Time series of shape (num_samples, num_features)
    num_parts: int
        The number of data partitions
    k: int
        The number of nearest neighbors
    maxlag: int 
        The number of lag periods to consider
    safetyCheck: bool
        Boolean value when True, will check for unique values 
        and abort estimation if duplicate values are found
    GPU: bool
        Boolean value that when set to true will use CUDA compatiable GPUs
    threshold: float
        The boundry value to filter the causation relationships.

    Returns
    -------    
    causal_nte_parts: 3d array
        causality relationsips in matrix form with shape 
        (num_partitions, num_features, num_features)
    """

    if type(X) == pd.core.frame.DataFrame:
        X = X.values
    n,m = X.shape
    causal_nte_parts = []

    for i in range(num_parts):
        # split the dataset into partitions
        X_temp = X[int(n/num_parts*i):int(n/num_parts*(i+1)),:]
        # utilize the function above to detect causality.
        causality_strength = model_nte(X=X_temp, k=k, max_lag=max_lag, 
                                         safetyCheck=safetyCheck, GPU=GPU)
        indexes = list(causality_strength.keys())
        strengths = list(causality_strength.values())
        causal_strength_part = np.zeros([m,m])
        
        # insert values of causality strengths to the strength matrix
        for j in range(len(indexes)):
            i_0, i_1 = indexes[j]
            causal_strength_part[i_0][i_1] = strengths[j]
        
        # pick the useful NTE values
        causal_strength_part[causal_strength_part <= 0.] = 0.
        # map and filter the causality strength
        causal_strength_part = te_mapping(causal_strength_part)
        causal_strength_part = np.around(causal_strength_part, 4)
        causal_strength_part[causal_strength_part < threshold] = 0.
        causal_nte_parts.append(causal_strength_part) 

    return causal_nte_parts


