# 2-Layer Causal Ensemble Model

import pandas as pd
import numpy as np
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from utilities_ccm import *
from utilities_nte import *
from utilities_pcmciplus import *
from utilities_gc import *
from utilities_ensemble import *
from utilities_plotting import *


class causal_ensemble:
    """
    Causal Ensemble Model for multivariable time series.
    
    Parameters
    ----------
    X: 2d array
        Time series of shape (num_samples, num_features)
    feature_names: list of str
        Names of all features
    num_parts: int
        The number of data partitions
    threshold: folat
        The boundry value to filter the causation relationships
    """

    def __init__(self, X, feature_names, num_parts, threshold=0.3):

        self.X = X
        self.feature_names = feature_names
        self.num_parts = num_parts
        self.threshold = threshold

        if type(X) == pd.core.frame.DataFrame:
            X = X.values
        n,m = X.shape

        X_partitions = []
        t = n/num_parts*0.5/(num_parts-1)
        for i in range(num_parts):
            # split the dataset into partitions
            X_partition = X[int(n/num_parts*i-t*i):int(n/num_parts*(i+1.5)-t*i), :]
            X_partitions.append(X_partition)

        self.X_partitions = X_partitions


    def ccm(self, lag=1, embed=1, split_percent=0.75, num_iter=40, 
            error_num=8, convergence_threshold=0.01):
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

        Returns
        -------    
        causal_ccm_parts: 3d array
            causality relationsips with shape 
            (num_partitions, num_features, num_features)
        """
        
        X = self.X
        num_parts = self.num_parts
        threshold = self.threshold

        if type(X) == pd.core.frame.DataFrame:
            X = X.values
        n,m = X.shape
        causal_ccm_parts = []

        X_partitions = self.X_partitions

        def func(data):
            # utilize the function above to detect causality
            causality_strength = model_ccm(data, lag=lag, embed=embed, 
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

            return causal_strength_part.tolist()
            
        # multithreaded running
        result = []
        threadpool = ThreadPoolExecutor(num_parts)
        
        for i in X_partitions:
            future = threadpool.submit(func, i)
            result.append(future)
        
        for i in result:
            causal_ccm_parts.append(i.result())

        threadpool.shutdown()
        
        self.ccm_parts = np.array(causal_ccm_parts)
        return np.array(causal_ccm_parts)


    def nte(self, k=1, max_lag=1, safetyCheck=False, GPU=False):
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

        Returns
        -------    
        causal_nte_parts: 3d array
            causality relationsips in matrix form with shape 
            (num_partitions, num_features, num_features)
        """

        X = self.X
        num_parts = self.num_parts
        threshold = self.threshold

        if type(X) == pd.core.frame.DataFrame:
            X = X.values
        n,m = X.shape
        causal_nte_parts = []

        X_partitions = self.X_partitions

        def func(data):
            # utilize the function to detect causality.
            causality_strength = model_nte(X=data, k=k, max_lag=max_lag, 
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

            return causal_strength_part

            #causal_nte_parts.append(causal_strength_part) 

        # multithreaded running
        result = []
        threadpool = ThreadPoolExecutor(num_parts)
        
        for i in X_partitions:
            future = threadpool.submit(func, i)
            result.append(future)
        
        for i in result:
            causal_nte_parts.append(i.result())

        threadpool.shutdown()

        self.nte_parts = np.array(causal_nte_parts)
        return np.array(causal_nte_parts)


    def pcmciplus(self, pc_alpha, cond_ind_test, max_lag=1):
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

        Returns
        -------    
        causal_pcmciplus_parts: 3d array
            causality relationsips with shape 
            (num_partitions, num_features, num_features)
        """

        X = self.X
        num_parts = self.num_parts
        feature_names=self.feature_names
        threshold = self.threshold

        if type(X) == pd.core.frame.DataFrame:
            X = X.values
        n,_ = X.shape
        causal_pcmciplus_parts = []

        X_partitions = self.X_partitions

        def func(data):
            # utilize the function to detect causality
            results = model_pcmciplus(data, feature_names=feature_names, 
                                      tau_max=max_lag, pc_alpha=pc_alpha, 
                                      cond_ind_test=cond_ind_test)
            causal_strength_part = causality_strength_pcmciplus(results, 
                                                                threshold=threshold)
            return causal_strength_part.tolist()
            #causal_pcmciplus_parts.append(causal_strength_part.tolist())

        
        # multithreaded running
        result = []
        threadpool = ThreadPoolExecutor(num_parts)
        
        for i in X_partitions:
            future = threadpool.submit(func, i)
            result.append(future)
        
        for i in result:
            causal_pcmciplus_parts.append(i.result())

        threadpool.shutdown()
        
        self.pcmciplus_parts = np.array(causal_pcmciplus_parts)
        return np.array(causal_pcmciplus_parts)


    def gc(self, max_lag=1, test='ssr_chi2test', verbose=False, signif=0.05):
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

        Returns
        -------
        causal_gc_parts: 3d array
            causality relationsips with shape 
            (num_partitions, num_features, num_features)
        """
        
        X = self.X
        # make time series stationary
        X = adf_test(X, signif=signif)

        num_parts = self.num_parts
        feature_names=self.feature_names
        threshold = self.threshold

        n,m = X.values.shape

        X_partitions = []
        t = n/num_parts*0.5/(num_parts-1)
        for i in range(num_parts):
            # split the dataset into partitions
            X_partition = X.loc[int(n/num_parts*i-t*i):int(n/num_parts*(i+1.5)-t*i)]
            X_partitions.append(X_partition)

        causal_gc_parts = []

        def func(data):
            # utilize the function to detect causality
            causality_strength = model_granger_causality(X=data, 
                                                         feature_names=feature_names, 
                                                         test=test, 
                                                         max_lags=max_lag, 
                                                         verbose=verbose, 
                                                         signif=signif,
                                                         threshold=threshold)
            causality_strength = nan2zero(causality_strength)
            return causality_strength.tolist()
            #causal_gc_parts.append(causality_strength.tolist())
    
        # multithreaded running
        result = []
        threadpool = ThreadPoolExecutor(num_parts)
        
        for i in X_partitions:
            future = threadpool.submit(func, i)
            result.append(future)
        
        for i in result:
            causal_gc_parts.append(i.result())

        threadpool.shutdown()
        
        causal_gc_parts = np.around(np.array(causal_gc_parts), 4)
        self.gc_parts = causal_gc_parts
        return causal_gc_parts


    def ensemble(self, n_init=1, means_init=False, means_2=0.4, 
                 threshold=1.9, boundry=0.4):
        """
        The ensemble model with some comprehensive evaluation methods.

        Parameters
        ----------
        n_init: int
            The number of initializations to perform. The best results are kept.
        means_init: bool
            if True, initial means of the 2 clusters will be provided. 
            The first means will be an 1d-array of zereos while
            the second one depends on the parameter "means_2".
        means_2: float
            The value for the array of the means of the second GMMs cluster.
        threshold: float
            The trustness boundry of the evaluation of l1_ensemble.
        boundry: float
            The trustness boundry of the causality strength of l2_ensemble. 

        Returns
        -------
        strength_ensemble: 2d array
            The ensembled causality strangth matrix with the shape 
            (num_features, num_features)
        """
        
        X_1 = l1_ensemble(self.ccm_parts, n_init=n_init, 
                          means_init=means_init, means_2=means_2)
        X_2 = l1_ensemble(self.nte_parts, n_init=n_init, 
                          means_init=means_init, means_2=means_2)
        X_3 = l1_ensemble(self.pcmciplus_parts, n_init=n_init, 
                          means_init=means_init, means_2=means_2)
        X_4 = l1_ensemble(self.gc_parts, n_init=n_init, 
                          means_init=means_init, means_2=means_2)
        self.ensemble_ccm = X_1
        self.ensemble_nte = X_2
        self.ensemble_pcmciplus = X_3
        self.ensemble_gc = X_4
        
        # The evalutions of the first layer ensemble
        acc_1, acc_2, acc_3, acc_4 = accuracy_index(self.ccm_parts, 
                                                    self.nte_parts, 
                                                    self.pcmciplus_parts, 
                                                    self.gc_parts)  
        
        strength_ensemble = l2_ensemble(X_1, X_2, X_3, X_4, 
                                        acc_1, acc_2, acc_3, acc_4,
                                        threshold=threshold, boundry=boundry)

        # remove the indirect relationships                          
        strength_ensemble = optimization(strength_ensemble)

        self.strength_ensemble = strength_ensemble
        return strength_ensemble


    def evaluation(self):
        """
        Evaluation the ensemble model and get the trustness score
        
        Returns
        -------
        score: float
            The evaluation score for the ensemble model.
        """
        
        X_1 = self.ensemble_ccm
        X_2 = self.ensemble_nte
        X_3 = self.ensemble_pcmciplus
        X_4 = self.ensemble_gc
        X_0 = self.strength_ensemble
        
        n,m = X_0.shape
        score_matrix = np.zeros((n,m))

        # reshape the l1_ensemble results for evaluation 
        temp = np.vstack((X_1, X_2, X_3, X_4)).reshape(4,n,m)
        for i in range(n):
            for j in range(m):
                list_1 = temp[:,i,j]
                num = np.count_nonzero(list_1)
                if X_0[i,j] > 0:
                    score_matrix[i,j] = 0.25 * num
                else:
                    score_matrix[i,j] = 0.25 * (4 - num)
        score = (score_matrix.sum() - n)/(n * (n-1))

        # normalize the score
        score = (score - 0.5) * 2
        return score 


    
















