# Evaluations

import warnings
warnings.filterwarnings('ignore')
import numpy as np 
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
from scipy import stats


def evaluation_l2(m_1, m_2, m_3, m_4, m_0):
    n,m = m_1.shape
    score_matrix = np.zeros((n,m))
    temp = np.vstack((m_1, m_2, m_3, m_4)).reshape(4,n,m)
    for i in range(n):
        for j in range(m):
            list_1 = temp[:,i,j]
            num = np.count_nonzero(list_1)
            if m_0[i,j] > 0:
                score_matrix[i,j] = 0.25 * num
            else:
                score_matrix[i,j] = 0.25 * (4 - num)
    score = (score_matrix.sum() - n)/(n * (n-1))
    score = (score - 0.5) * 2
    return score_matrix, score 




