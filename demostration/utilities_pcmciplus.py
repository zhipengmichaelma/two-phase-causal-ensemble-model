## Sub-model_3: PCMCI+
#
# Paper:
# 1. Detecting and quantifying causal associations in larg nonlinear 
# time series datasets. Jakob Runge et al. 2019
# 2. Discovering contemporaneous and lagged causal relations in 
# autocorrelated nonlinear time series datasets. Jakob Runge et al. 2020


import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr, GPDC, CMIknn, CMIsymb


def model_pcmciplus(X, feature_names, tau_max, pc_alpha, cond_ind_test):
    """
    Apply the libraty 'tigramite' to detect causality with PCMCI+.

    Parameters
    ----------
    X: 2d array
        Time series of shape (num_samples, num_features)
    feature_names: list of str
        Names of all features
    tau_max: int
        Maximum time lag. Must be larger or equal to 0.
    pc_alpha: float or list of floats
        Significance level in algorithm. If a list or None is passed, the
        pc_alpha level is optimized for every graph across the given
        values ([0.001, 0.005, 0.01, 0.025, 0.05] for None) using the score 
        computed in pcmci.cond_ind_test.get_model_selection_criterion().
    cond_ind_test: conditional independence test object
        This can be ParCorr or other classes from
        "tigramite.independence_tests"

    Returns
    -------
    results: 
        return values of "pcmci.run_pcmciplus" including
        graph : array of shape [N, N, tau_max+1]
            Resulting causal graph.
        val_matrix : array of shape [N, N, tau_max+1]
            Estimated matrix of test statistic values regarding adjacencies.
        p_matrix : array of shape [N, N, tau_max+1]
            Estimated matrix of p-values regarding adjacencies.
        sepset : dictionary
            Separating sets. See paper for details.
        ambiguous_triples : list
            List of ambiguous triples, only relevant for 'majority' and
            'conservative' rules, see paper for details.
    """
    
    # determine the method of condition independent test
    if cond_ind_test == 'parCorr':
        cit = ParCorr()
    elif cond_ind_test == 'GPDC':
        cit = GPDC()
    elif cond_ind_test == 'CMIknn':
        cit = CMIknn()
    elif cond_ind_test == 'CMIsymb':
        cit = CMIsymb()
    else:
        print("Error: cond_ind_test should be ParCorr, GPDC, CMIknn or CMIsymb")
    
    # reset the dataset for pcmci+
    dataframe = pp.DataFrame(X, 
                             datatime = np.arange(len(X)), 
                             var_names=feature_names)
    # run the library
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cit, verbosity=2)
    results = pcmci.run_pcmciplus(tau_min=0, tau_max=tau_max, pc_alpha=pc_alpha)

    return results


def causality_strength_pcmciplus(results, threshold=0.3):
    """
    Schedule the matrix of causality strength according to the results of 
    "pcmci.run_pcmciplus".
    
    Parameters
    ----------
    results:
        return values of the function "model_pcmciplus"
    threshold: float
        The boundry value to filter the causation relationships

    Returns
    -------
    causal_pcmciplus: 2d array
        causality relationsips with shape (num_features, num_features)
    """
    
    # symbol matrix showing the causal relationships
    link_matrix = results['graph']
    # results of conditional independence test
    val_matrix=results['val_matrix']
    n,m,tau = val_matrix.shape
    
    # pick the causal relationships from the results of pcmciplus
    val = np.zeros([n,m])
    link = np.array(['']*(n*m), dtype='<U3').reshape((n,m))
    for u in range(n):
        for v in range(m):
            argmax = np.abs(val_matrix[u, v][1:]).argmax() + 1
            val[u,v] = np.abs(np.around(val_matrix[u, v][argmax],4))
            link[u,v] = link_matrix[u, v, argmax]
    
    causal_pcmciplus = np.zeros((n,m))
    # contemporaneous causal link matrix
    link_c = link_matrix[:,:,0]
    
    # insert values of causality strengths to the strength matrix
    for u in range(n):
        for v in range(m):
            if u!=v and link[u,v]=='-->':
                causal_pcmciplus[u,v] = np.abs(val[u,v])

            #"graph[i,j,0]=-->" (and "graph[j,i,0]=<--") denotes a directed, 
            # contemporaneous causal link from 'i' to 'j'
            if link_c[u,v]=='-->' and link_c[v,u]=='<--':
                causal_pcmciplus[u,v] = np.abs(round(val_matrix[u,v,0], 4))
    
    # filter the causality strength
    causal_pcmciplus[causal_pcmciplus<threshold] = 0.

    return causal_pcmciplus


def pcmciplus_partitions(X, num_parts, feature_names, 
                         pc_alpha, cond_ind_test, max_lag=1, threshold=0.3):
    """
    Detect causality with PCMCI+ for all data partitions.

    Parameters
    ----------
    X: 2d array
        Time series of shape (num_samples, num_features)
    num_parts: int
        The number of data partitions
    feature_names: list of str
        Names of all features
    pc_alpha: float or list of floats
        Significance level in algorithm. If a list or None is passed, the
        pc_alpha level is optimized for every graph across the given
        values ([0.001, 0.005, 0.01, 0.025, 0.05] for None) using the score 
        computed in pcmci.cond_ind_test.get_model_selection_criterion().
    cond_ind_test: conditional independence test object
        This can be ParCorr or other classes from
        "tigramite.independence_tests"
    max_lag: int
        Maximum time lag. Must be larger or equal to 0.
    threshold: float
        The boundry value to filter the causation relationships

    Returns
    -------    
    causal_pcmciplus_parts: 3d array
        causality relationsips with shape 
        (num_partitions, num_features, num_features)
    """

    if type(X) == pd.core.frame.DataFrame:
        X = X.values
    n,_ = X.shape

    causal_pcmciplus_parts = []
    for i in range(num_parts):
        # split the dataset into partitions
        X_temp = X[int(n/num_parts*i):int(n/num_parts*(i+1)), :]
        # utilize the function above to detect causality
        results = model_pcmciplus(X_temp, feature_names=feature_names, 
                                  tau_max=max_lag, pc_alpha=pc_alpha, 
                                  cond_ind_test=cond_ind_test)
        causal_strength_part = causality_strength_pcmciplus(results, 
                                                            threshold=threshold)
        causal_pcmciplus_parts.append(causal_strength_part.tolist())

    return np.array(causal_pcmciplus_parts)


