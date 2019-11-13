#functions for getting clusters
import numpy as np
import pandas as pd
import re
import copy
from collections import namedtuple
from multiprocessing import Pool, Process, Lock

#for getting the fisher exact test
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr
rpy2.robjects.numpy2ri.activate()


from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import adjusted_rand_score, f1_score, roc_auc_score
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.base import ClusterMixin, BaseEstimator

def l1(x1, x2):
    return np.sum(np.abs(x1-x2))

def tanimoto_dist(x1, x2):
    if l1(x1, x2) == 0:
        return 0
    tanimoto = x1.dot(x2)/(x1.dot(x1) + x2.dot(x2) - x1.dot(x2))
    return 1/(1+tanimoto)

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
    
def best_clusterer_score(clusterer,curr_clusterer, x, y):
    return clusterer.score(curr_clusterer,x,y)
    
class BestClusterer(ClusterMixin, BaseEstimator):
    
    def __init__(self, clusterers = None, min_clusters = 2, max_clusters = 5, metric = None, n_jobs = 1):
        if clusterers is None:
            c_range = range(min_clusters, max_clusters + 1)
            clusterers = [FClusterer(c) for c in c_range]
            clusterers +=  [FClusterer(c, dist_func = l2) for c in c_range]
#             clusterers += [KMeans(c) for c in c_range]
#             clusterers += [AgglomerativeClustering(c) for c in c_range]
        self.clusterers = clusterers
#         if metric is None:
#             #default score is 1 - correlation p value
#             metric = lambda c,y: 1-fisher_exact_test(c,y)
#         self.metric = metric #should take cluster_labes, classes and return a goodness score
        self.n_jobs = n_jobs
        
    def metric(self, c, y):
        return 1 - fisher_exact_test(c,y)
    
    def score(self, clusterer_idx, x, y):
        clusterer = self.clusterers[clusterer_idx]
        cluster_labels = clusterer.fit_predict(x).ravel()
        return (clusterer_idx, self.metric(cluster_labels, y.ravel()))
        
    def fit(self, x, y):
        if self.n_jobs == 1:
            self.fit_singlecore(x,y)
        else:
            self.fit_multicore(x,y)
    
    def fit_singlecore(self, x, y):
        score_tuple = [self.score(i, x, y) for i in range(len(self.clusterers))]
        scores = sorted(score_tuple, key = lambda x: -x[1]) #sort in descending order
        self.best_clusterer_idx = scores[0][0]
        self.best_score = scores[0][1]
        
    def fit_multicore(self, x,y):
#         scores = np.zeros((len(self.clusterers),))
#         def update_scores(i):
#             scores[i] = self.score(i, x, y)
#         for i, clusterer in enumerate(self.clusterers):
#             NoDaemonProcess(target = update_scores, args = (i,)).start()
#         self.best_clusterer_idx = np.argmax(scores)
#         self.best_score = np.max(scores)
        with Pool(self.n_jobs) as cpool:
            results = [cpool.apply_async(best_clusterer_score, args = (self,c,x,y)) for c in range(len(self.clusterers))]
            score_tuple = [res.get() for res in results]
            cpool.close()
            cpool.join()
            print(score_tuple)
        scores = sorted(score_tuple, key = lambda x: -x[1]) #sort in descending order
        self.best_clusterer_idx = scores[0][0]
        self.best_score = scores[0][1]
        
    def get_best_clusterer(self):
        return self.clusterers[self.best_clusterer_idx]
    
    def predict(self, x):
        return self.get_best_clusterer().fit_predict(x)
                
    def fit_predict(self, x, y = None):
        self.fit(x,y)
        return self.predict(x)
            
    
class FeatureClusterer(ClusterMixin, BaseEstimator):
    #clusters features together
    #like sklearn feature agglomeration, but can work on dataframes and tracks names of the features

    def __init__(self, base_model = 'default', scale = False):
        if base_model is None or base_model == 'default':
            self.base_model = AffinityPropagation()
        else:
            self.base_model = base_model
        self.scale = scale
        assert(hasattr(self.base_model, 'fit_predict'))

    def fit(self, x, y = None):
        if self.scale:
            x = StandardScaler().fit_transform(x)
        x = x.transpose()
        self.labels = self.base_model.fit_predict(x)
        self.labels = self.map_to_zero(self.labels)

    def map_to_zero(self, labels):
        labels -= labels.min()
        unique_labels = set(labels)
        n_labels = len(unique_labels)
        if n_labels == labels.max():
            return labels
        for i in range(n_labels):
            if i not in set(labels):
                args = np.argwhere(labels > i)
                labels[args] -= 1
        return labels

    def predict(self, x, y = None):
        index = list(x.index)
        x = x.transpose()
        is_df = isinstance(x, pd.DataFrame)

        groups = [[] for x in range(len(set(self.labels)))]
        group_names = [[] for x in range(len(set(self.labels)))]
        for pos, groupnum in enumerate(self.labels):
            if is_df:
                feature = x.iloc[pos]
                groups[groupnum].append(feature.values)
                group_names[groupnum].append(feature.name)
            else:
                groups[groupnum].append(x[pos])

        f_out = np.zeros((len(set(self.labels)), x.shape[1]))
        for row, vals in enumerate(groups):
            f_out[row] = np.mean(vals, axis = 0)
        x_out = f_out.transpose()
        group_names = [','.join(gn) for gn in group_names]
        if is_df:
            x_out = pd.DataFrame(x_out, index=index, columns = group_names)
        return x_out

    def fit_predict(self, x, y = None):
        self.fit(x)
        return self.predict(x)
    
    
def score_multithreaded(feature_selector, fset, y):
    x = fset[1].copy()
    name = fset[0]
    return (name, feature_selector.bootstrap_score(x, y.copy()))

class FeatureSelector(BaseEstimator):

    def __init__(self, model = None, n_samples = 1, rescale = True, threshold = 0, print_out = True, n_jobs = None):
        if model is None:
            model = self.default_model()
        self.model = model
        self.n_samples = n_samples
        self.threshold = threshold
        self.rescale = rescale
        self.n_jobs = n_jobs
        self.print_out = print_out
        self.isfit = False
        
    def default_model(self):
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(C = 100, solver = 'lbfgs', max_iter = 10000)

    def get_importances(self, x, y, baseline = None, as_df = True):
        base_set = set(baseline) if baseline is not None else set([])
#         importances = np.zeros((self.n_samples, x.shape[1]))
        if baseline is not None:
            getcols = lambda x: [x] + baseline
        else:
            getcols = lambda x: [x]
        fsets = [(col, x.loc[:, getcols(col)]) for col in x.columns if col not in base_set]
        importances = []
        with Pool(self.n_jobs) as pool:
            results = [pool.apply_async(score_multithreaded, args=(self,fset,y)) for fset in fsets]
            result_vals = list([res.get(timeout=10000) for res in results])
            importances = [i[1].reshape(-1,1) for i in result_vals]
            importances = np.hstack(importances)
        if as_df:
            titles = [r[0] for r in result_vals]
            return pd.DataFrame(importances, columns = titles)
        return importances

    def bootstrap_score(self, x, y, metric = roc_auc_score):
        if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
            x = x.copy().values
        x = x.astype('float64')
        score = []
        for dummy in range(self.n_samples):
            if self.n_samples > 1:
                xtemp, ytemp = resample(x, y)
            else:
                xtemp, ytemp = x, y
            if xtemp.ndim == 1:
                xtemp = xtemp.reshape(-1,1)
            ypred = self.cv_predict(xtemp, ytemp)
            score.append(metric(ytemp.ravel(), ypred.ravel()))
        return np.array(score)

    def cv_predict(self, x, y):
        ypred = np.zeros(y.shape)
        for d in range(y.shape[0]):
            xtrain = np.delete(x, d, axis = 0)
            ytrain = np.delete(y, d, axis = 0)
            xtest = x[d].reshape(1, -1)
            if self.rescale:
                xtrain, xtest = Metrics.rescale(xtrain, xtest)
            self.model.fit(xtrain, ytrain)
            ypred[d] = self.model.predict_proba(xtest)[:,1]
        return ypred

    def get_most_important(self, x, y, baseline = None):#         for pos, col in enumerate(x.columns):
        importances = self.get_importances(x,y,baseline)
        names = importances.columns
        important = importances.mean(axis = 0)
        important = important.values.ravel()
        fname = names[np.argmax(important)]
        return fname, important.max()

    def fit(self, x, y):
        x = x.copy()
        top_feature, best_score = self.get_most_important(x, y)
        features_to_keep = [top_feature]
        while len(features_to_keep) < x.shape[1]:
            next_best_feature, new_score = self.get_most_important(x,y,baseline = features_to_keep)
            if new_score < best_score + self.threshold:
                break
            best_score = new_score
            features_to_keep = features_to_keep + [next_best_feature]
            self.print_updates(features_to_keep, best_score)
     
        self.features_to_keep = features_to_keep
        self.feature_inds = [np.argwhere(x.columns == f)[0][0] for f in features_to_keep]
        self.isfit = True
        return self
    
    def print_updates(self, features_to_keep, best_score):
        if self.print_out == False:
            return
        print(features_to_keep)
        print(best_score)
        print()

    def transform(self, x, y = None):
        assert(self.isfit)
        return x.iloc[:, self.feature_inds]

    def fit_transform(self, x, y):
        self.fit(x,y)
        return self.transform(x,y)

class FeatureClusterSelector(FeatureSelector):

    def default_model(self):
        return FClusterer(4)

    def bootstrap_score(self, x, y):
        if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
            x = x.copy().values
        x = x.astype('float64')
        score = []
        for d in range(self.n_samples):
            if self.n_samples > 1:
                xtemp, ytemp = resample(x, y)
            else:
                xtemp, ytemp = x, y
            if xtemp.ndim == 1:
                xtemp = xtemp.reshape(-1,1)
            clusters = self.model.fit_predict(xtemp, ytemp).ravel()
            score.append(1-fisher_exact_test(clusters, ytemp))
        return np.array(score)
    
    def predict_labels(self, x, y=None):
        assert(self.isfit)
        x = self.transform(x)
        return self.model.fit_predict(x.values, y)
        
    def fit_predict_labels(self, x):
        x = self.fit_transform(x,y)
        return self.model.fit_predict(x.values, y)
    
    def print_updates(self, features_to_keep, best_score):
        if self.print_out == False:
            return
        print(features_to_keep)
        print(1-best_score)
        print()
