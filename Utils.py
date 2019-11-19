#functions for getting clusters
import numpy as np
import pandas as pd
import re
import copy
from collections import namedtuple

#for getting the fisher exact test
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr
rpy2.robjects.numpy2ri.activate()


from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.base import ClusterMixin, BaseEstimator

def l1(x1, x2):
    return np.sum(np.abs(x1-x2))

def tanimoto_dist(x1, x2):
    if l1(x1, x2) == 0:
        return 0
    tanimoto = x1.dot(x2)/(x1.dot(x1) + x2.dot(x2) - x1.dot(x2))
    #guadalupe used 1 - similarity for her clustering
    return 1 - tanimoto

def l2(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

def pdist(x, dist_func):
    distance = []
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            distance.append(dist_func(x[i], x[j]))
    return np.array(distance)

def get_contingency_table(x, y):
    #assumes x and y are two equal length vectors, creates a mxn contigency table from them
    cols = sorted(list(np.unique(y)))
    rows = sorted(list(np.unique(x)))
    tabel = np.zeros((len(rows), len(cols)))
    for row_index in range(len(rows)):
        row_var = rows[row_index]
        for col_index in range(len(cols)):
            rowset = set(np.argwhere(x == row_var).ravel())
            colset = set(np.argwhere(y == cols[col_index]).ravel())
            tabel[row_index, col_index] = len(rowset & colset)
    return tabel

def fisher_exact_test(c_labels, y):
    if len(np.unique(y)) == 1:
        print('fisher test run with no positive class')
        return 0
    #call fishers test from r
    contingency = get_contingency_table(c_labels, y)
    stats = importr('stats')
    pval = stats.fisher_test(contingency,workspace=2e8)[0][0]
    return pval

class FClusterer(ClusterMixin, BaseEstimator):

    def __init__(self, n_clusters, dist_func = l1, link = 'weighted', criterion = 'maxclust'):
        self.link = link
        self.dist_func = dist_func if link not in ['median', 'ward', 'centroid'] else 'euclidean'
        self.t = n_clusters
        self.criterion = criterion

    def fit_predict(self, x, y = None):
        clusters = linkage(x, method = self.link, metric = self.dist_func)
        return fcluster(clusters, self.t, criterion = self.criterion)
   