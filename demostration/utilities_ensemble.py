# A 2-layer causal ensemble model is proposed. 
# The first layer is to ensemble each causality detector 
# for diffrent data partitions with GMMs.
# The second layer is to ensemble the results from the first layer 
# with a comprehensive selection procedure.


import warnings
warnings.filterwarnings('ignore')

import numpy as np 
from sklearn.mixture import GaussianMixture
from itertools import combinations


def direction_choosing(X):
    """
    Determine the direction of causality after ensemble steps.
    
    Parameters
    ----------
    X: 2d array
        Causality strength matrix of shape (num_features, num_features)
    
    Returns
    -------
    X: 2d array
        Processed causality strength matrix 
        of shape (num_features, num_features)
    """

    n,m  = X.shape
    for i in range(n):
        for j in range(m):
            if i > j:
                if X[i,j] > X[j,i] > 0:
                    X[i,j] = 0. 
                if X[j,i] > X[i,j] > 0:
                    X[j,i] = 0.
    return X


def accuracy_index(X_1, X_2, X_3, X_4):
    """
    evalutions fro l1_ensmeble.

    Parameters
    ----------
    X_1: 3d array
        Integrated causality strength matrix for CCM
        of shape (num_partitions, num_features, num_features)
    X_2: 3d array
        Integrated causality strength matrix for NTE
        of shape (num_partitions, num_features, num_features)
    X_3: 3d array
        Integrated causality strength matrix for PCMCI+
        of shape (num_partitions, num_features, num_features)
    X_4: 3d array
        Integrated causality strength matrix for GC
        of shape (num_partitions, num_features, num_features)

    Returns
    -------
    acc_1: 2d array
        The evaluation score matrix of l1_ensemble on CCM 
        with the shape (num_features, num_features)
    acc_2: 2d array
        The evaluation score matrix of l1_ensemble on NTE
        with the shape (num_features, num_features)
    acc_3: 2d array
        The evaluation score matrix of l1_ensemble on PCMCI+
        with the shape (num_features, num_features)
    acc_4: 2d array
        The evaluation score matrix of l1_ensemble on GC
        with the shape (num_features, num_features)
    """
    
    n,m,l = X_1.shape
    acc_1 = np.zeros((m,l))
    acc_2 = np.zeros((m,l))
    acc_3 = np.zeros((m,l))
    acc_4 = np.zeros((m,l))
    
    def score(X):
        """
        Calculate the evaluation score
        for each relationship of each detector

        Parameters
        ----------
        X: 1d array
            The array containing values of the same relationship
            position of different partitions of shape (num_partitions,).
        
        Returns
        -------
        acc: float
            The score to evaluate the performance of l1_ensmeble 
            of the same relationship position of different partitions.
        """

        X = X[X>0.]
        num = X.shape[0]
        #
        if num < 2:
            acc = 0.
        else:
            #
            std = np.around(np.std(X),4)
            mean = np.around(np.mean(X),4)
            acc = np.round((mean / (std + 1e-8))*num/n,4)
        return acc
    
    for u in range(m):
        for v in range(l):
            acc_1[u,v] = score(X_1[:,u,v])
            acc_2[u,v] = score(X_2[:,u,v])
            acc_3[u,v] = score(X_3[:,u,v])
            acc_4[u,v] = score(X_4[:,u,v])
            
    return acc_1, acc_2, acc_3, acc_4


def l1_ensemble(X, n_init=1, means_init=False, means_2=0.4):
    """
    The first ensemble layer with GMMs clustering.
    
    Parameters
    ----------
    X: 3d array
        Integrated causality strength matrix 
        of shape (num_partitions, num_features, num_features)
    n_init: int
        The number of initializations to perform. The best results are kept.
    means_init: bool
        if True, initial means of the 2 clusters will be provided. 
        The first means will be an 1d-array of zereos while
        the second one depends on the parameter "means_2".

    means_2: float
        The value for the array of the means of the second GMMs cluster.
        
    Returns
    -------
        ensemble_1: 2d array
            The ensembled causality strangth matrix for certain detector 
            with the shape (num_features, num_features)
    """

    n,m,l = X.shape
    ensemble_1 = np.zeros((m,l))
    data_reshape = []

    # if the means should be initialized, set the values.
    if means_init is True:
        mean_init = []
        mean_init.append(([0.]*n))
        mean_init.append(([means_2]*n))
    
    # reshape the input data to (num_feature^2, num_partitions)
    for u in range(m):
        for v in range(l):
            data_reshape.append(X[:,u,v].tolist())
    data_reshape = np.array(data_reshape)
    
    # utilize GMMs to split the relationships into 2 groups.
    # group 0 represents non-causality while grouo 1 is causality
    if means_init is True:
        gmm = GaussianMixture(n_components=2, random_state=0, n_init=n_init,
                              means_init = np.array(mean_init))
    else:
        gmm = GaussianMixture(n_components=2, random_state=0, n_init=n_init)
    r = gmm.fit_predict(data_reshape).reshape((m,l))
    gmm_means = gmm.means_
    gmm_non = np.argmin(gmm_means.mean(axis=1))
    
    # choose the strength value for causality relationship
    for u in range(m):
        for v in range(l):
            a = X[:,u,v]
            if r[u,v] != gmm_non:
                a = a[a != 0]
                ensemble_1[u,v] = np.median(a)
    ensemble_1 = np.around(direction_choosing(ensemble_1), 4)

    return ensemble_1



def l2_ensemble(X_1, X_2, X_3, X_4, 
                acc_1, acc_2, acc_3, acc_4, 
                threshold=1.9, boundry=0.3):
    """
    The second ensemble layer with some comprehensive evaluation methods.

    Parameters
    ----------
    X_1: 2d array
        The output of l1_ensemble on CCM with the shape
        (num_features, num_features).
    X_2: 2d array
        The output of l1_ensemble on NTE with the shape
        (num_features, num_features).
    X_3: 2d array
        The output of l1_ensemble on PCMCI+ with the shape
        (num_features, num_features).
    X_4: 2d array
        The output of l1_ensemble on GC with the shape
        (num_features, num_features).
    acc_1: 2d array
        The evaluation score matrix of l1_ensemble on CCM 
        with the shape (num_features, num_features)
    acc_2: 2d array
        The evaluation score matrix of l1_ensemble on NTE
        with the shape (num_features, num_features)
    acc_3: 2d array
        The evaluation score matrix of l1_ensemble on PCMCI+
        with the shape (num_features, num_features)
    acc_4: 2d array
        The evaluation score matrix of l1_ensemble on GC
        with the shape (num_features, num_features)
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
    
    # The first filter for l1_ensmeble results.
    X_1[acc_1<0.9]=0.
    X_2[acc_2<0.9]=0.
    X_3[acc_3<0.9]=0.
    X_4[acc_4<0.9]=0.

    n, m = X_1.shape
    strength_ensemble = np.zeros((n,m))
    
    # ensemble procedure
    for u in range(n):
        for v in range(m):
            
            # reshape the data for further process
            ensemble_temp = np.hstack((X_1[u,v], X_2[u,v], X_3[u,v], X_4[u,v]))
            weight = np.hstack((acc_1[u,v], acc_2[u,v], acc_3[u,v], acc_4[u,v]))

            # normalizded the weight
            weight_temp = weight.copy()
            weight_temp[ensemble_temp<=0.3] = 0.
            normalized_weight = weight_temp / weight_temp.sum()
            
            # the causal strength if the causality relationship is true
            target = (ensemble_temp * normalized_weight).sum()
            
            # the precedure to determine the ensemble causality relationship
            if np.count_nonzero(ensemble_temp) > 2:   
                strength_ensemble[u,v] = target
            if np.count_nonzero(ensemble_temp) < 2:
                strength_ensemble[u,v] = 0.
            if np.count_nonzero(ensemble_temp) == 2:
                temp  = weight.copy()
                temp.sort()
                if temp[-1] > 9.9:
                    strength_ensemble[u,v] = target
                elif temp[-2] > threshold:
                    strength_ensemble[u,v] = target
                else:
                    strength_ensemble[u,v] = 0.
                #if ensemble_temp[1] == 0. and ensemble_temp[2] == 0.:
                    #strength_ensemble[u,v] = 0.
    
    # filter the final results
    strength_ensemble = direction_choosing(strength_ensemble)
    strength_ensemble = np.around(strength_ensemble,4)
    strength_ensemble[strength_ensemble<boundry] = 0.

    return strength_ensemble


def optimization(X):
    """
    remove the indirect causal relationships.

    parameters
    ----------
    X: 2d array
        The ensembled causal strength of shape (num_features, num_features)
    
    Returns
    -------
    X: 2d array
         The filtered ensembled causal strength 
         of shape (num_features, num_features)
    """
    n,_ = X.shape
    list_1 = np.linspace(0,n-1,n).astype(int)

    for i in list_1:
        list_rest = list_1[list_1!=i]
        combi = list(combinations(list_rest,2))
        for j in combi:
            if (X[i,j[0]]>0.) & (X[i,j[1]]>0.):

                if (X[j[0]][j[1]]) > 0:
                    list_3 = np.array([X[i][j[0]], X[i][j[1]], X[j[0]][j[1]]])
                    list_3[np.argmin(list_3)] = 0
                    X[i][j[0]], X[i][j[1]], X[j[0]][j[1]] = list_3

                if (X[j[1]][j[0]]) > 0:
                    list_3 = np.array([X[i][j[0]], X[i][j[1]], X[j[1]][j[0]]])
                    list_3[np.argmin(list_3)] = 0
                    X[i][j[0]], X[i][j[1]], X[j[1]][j[0]] = list_3
    return X
