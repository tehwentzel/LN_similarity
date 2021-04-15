import pandas as pd
import numpy as np
import re

#for getting the fisher exact test
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr
rpy2.robjects.numpy2ri.activate()

from scipy.cluster.hierarchy import fcluster, linkage, dendrogram
from sklearn.base import ClusterMixin, BaseEstimator
from scipy.spatial.distance import pdist, squareform
from scipy.stats import chisquare
from scipy.spatial.distance import pdist
from scipy.stats import chi2_contingency
from sklearn.model_selection import StratifiedKFold
from scipy.spatial.distance import canberra
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import roc_curve, roc_auc_score

class LNDataset():
    
    data_columns = ['Dummy ID', 
               'Affected Lymph node UPPER',
                'Aspiration rate Pre-therapy',
                'Aspiration rate Post-therapy',
               'T-category',
#                'N-category',
                   ]
    
    clean_col_names = {
        'Feeding tube 6m': 'FT',
        'Affected Lymph node UPPER': 'affected_nodes',
        'Aspiration rate(Y/N)': 'AS',
        'Neck boost (Y/N)': 'neck_boost',
        'Gender': 'gender',
        'Tm Laterality (R/L)': 'laterality',
        'AJCC 8th edition': 'ajcc8',
        'N_category_full': 'N-category',
        'HPV/P16 status': 'hpv',
        'Tumor subsite (BOT/Tonsil/Soft Palate/Pharyngeal wall/GPS/NOS)': 'subsite',
        'Total dose': 'total_dose',
        'Therapeutic combination': 'treatment',
        'Smoking status at Diagnosis (Never/Former/Current)': 'smoking_status'
    }
    
    ambiguous_nodes = set(['2/3','3/4','2/3/4','/3','2/','-R4'])
    ln_features = ['id','nodes','position','gender','scores', 'similarity',
               'Aspiration_rate_Post-therapy','Aspiration_rate_Pre-therapy',
               'Feeding_tube_6m','Neck_boost']
    
    js_name_dict = {'Gender': 'gender', 
                 'Tm Laterality (R/L)': 'position', 
                'Feeding tube 6m': "Feeding_tube_6m",
                'Aspiration rate Pre-therapy': 'Aspiration_rate_Pre-therapy',
                'Aspiration rate Post-therapy': 'Aspiration_rate_Post-therapy',
                'Neck boost (Y/N)': 'Neck_boost'}
    ids_to_remove = set([2002, 5118,10001,10111,10148,10179,10184,10200])
    #maximum distance between sets of nodes.  Limited to current cohort valid options (2A and 2B always co-occur)
    spread_dict = {
        '1A1B': 1,
        '1A2A': 1,
        '1A2B': 3,
        '1A3': 2,
        '1A4': 2,
        '1A5A': 3,
        '1A5B': 3,
        '1A1B': 1,
        '1B2A': 1,
        '1B2B': 2,
        '1B4': 3,
        '1B3': 2,
        '1B5A': 3,
        '1B5B': 4,
        '2A2B': 1,
        '2A3': 1,
        '2A5A': 2, #2B is only closer to 5A than 2A so we can skip everything else
        '2B2A': 1,
        '2B3': 2,
        '2B4': 3,
        '2B5A': 1,
        '2B5B': 2, 
        '34': 1,
        '35A':1,
        '35B': 2,
        '45A': 2,
        '45B': 1
    }
    
    def __init__(self, cohort_data, adjacency_path = './connectivity_646.csv', drop_tween = None, 
                 drop_static_ids = None, validation = False):
        self.validation = validation
        self.drop_static_ids = drop_static_ids if drop_static_ids is not None else (not validation)
        #if true, drop patients with '/' in the node list, which are in-between nodes
        self.drop_tween = drop_tween if drop_tween is not None else (not validation)
        self.adjacency = pd.read_csv(adjacency_path, index_col = 0)
        self.node_list = sorted(self.adjacency.columns)
        self.get_data(cohort_data)
        
        self.setup_grams()
        self.input_full_ncat()
        
    def get_data(self, cohort_data):
        cols = LNDataset.data_columns + list(LNDataset.clean_col_names.keys())
        if isinstance(cohort_data, pd.DataFrame):
            self.data = cohort_data                   
        else:
            
            extension = cohort_data.split('.')[-1]
            if extension == 'tsv':
                sep = '\t'
                read_func = pd.read_csv
                engine = 'python'
            elif extension in ['xlsx','xls']:
                read_func = pd.read_excel
                sep = ','
                engine = None
            else:
                sep = ','
                read_func = pd.read_csv
                engine = 'python'
            try:
                self.data = read_func(cohort_data, 
                           sep = sep,
                           index_col=0, 
                           usecols = cols,
                           engine = engine,
                           dtype = {'Affected Lymph node UPPER': str})
            except Exception as e:
                print(e)
        self.data = self.data.rename(columns = LNDataset.clean_col_names)
        
        base_count = self.data.shape[0]
        if not self.validation:
            self.data = self.data.dropna(subset=['affected_nodes'])
        else:
            self.data = self.data.fillna({'affected_nodes': ''})
        if self.data.shape[0] < base_count:
            print( base_count - self.data.shape[0], 'patients removed due to missing nodes')
        if self.drop_tween:
            self.data = self.data[self.data['affected_nodes'].apply(lambda x: '/' not in x)]

        self.data.index.rename('id',inplace = True)
        if self.drop_static_ids:
            self.data = self.data.drop(LNDataset.ids_to_remove,errors='ignore')

        self.data.sort_index(inplace=True)
        #clean T-category
        self.data.loc[:,'T-category'] = self.data.loc[:,'T-category'].apply(lambda x: 'T1' if x in ['Tis','Tx'] else x)
        self.data.loc[:,['FT','AS']] = self.data.loc[:,['FT','AS']].apply(lambda x: x == 'Y').astype('int')
        self.data['TOX'] = (self.data.loc[:,'FT'] + self.data.loc[:,'AS']) > 0
        
        
    def setup_grams(self):
        #extract all the node names and such
        self.left_nodes = sorted(['L'+n for n in self.adjacency.columns])
        self.right_nodes = ['R'+n for n in self.adjacency.columns]
        self.rpln = ['RRPLN', 'LRPLN']
        self.nodes = self.left_nodes + self.right_nodes
        self.all_nodes = set(self.nodes)
        self.node_to_index = {word: position for position, word in enumerate(self.nodes)}
        
        self.clean_ln_data()
        self.index = self.data.copy().index
        self.ids = self.index.values
        self.monograms = self.get_monograms(self.data.copy())
        self.dual_bigrams = self.setup_bigrams()
        self.dual_monograms = self.get_dual_monograms()
        self.bilateral_bonus = self.get_bilateral_bonus()
        self.spread = node_spread(self.data)
        self.nonspatial = lambda : pd.concat(
            [self.dual_monograms, self.bilateral_bonus], 
            axis = 1)
     
    def bigrams(self, 
                partial_points = False, 
                bilateral_count = False, 
                ln_spread = True
               ):
        bigrams = self.dual_bigrams.copy()
        if not partial_points:
            bigrams = bigrams.apply(lambda x: np.floor(x/2))
        if bilateral_count:
            bigrams = pd.concat([bigrams, self.bilateral_bonus],axis=1)
        if ln_spread:
            bigrams = bigrams.merge(self.spread, on='id')
        return pd.concat([bigrams, self.dual_monograms], axis = 1)
    
    def get_bilateral_bonus(self):
        bonus = self.dual_monograms.copy().apply(lambda l: np.sum([x for x in l if x > 1]), axis = 1)
        bonus.rename('bilateral',inplace=True)
        return bonus
        
    def clean_names(self,x):
        return [re.sub('^[LR]\s*','',x) for x in x.columns]
        
    def clean_names_string(self, x):
        return ''.join(self.clean_names(x))
        
    def get_dual_monograms(self):
        lgrams = self.monograms.loc[:,self.left_nodes]
        rgrams = self.monograms.loc[:,self.right_nodes]
        lnames = self.clean_names_string(lgrams)
        rnames = self.clean_names_string(rgrams)
        assert(lnames == rnames)
        df_values = lgrams.values + rgrams.values
        dual_monograms = pd.DataFrame(df_values, columns = self.clean_names(lgrams), index = lgrams.index)
        return dual_monograms
        
    def setup_bigrams(self):
        l_bigrams = self.bigramize(self.monograms.loc[:, self.left_nodes], 'L')
        r_bigrams = self.bigramize(self.monograms.loc[:, self.right_nodes], 'R')
        rnames = self.clean_names_string(l_bigrams)
        lnames = self.clean_names_string(r_bigrams)
        assert(rnames == lnames)
        assert(np.all(l_bigrams.index == r_bigrams.index))
        dual_bigrams = pd.DataFrame(l_bigrams.values +r_bigrams.values, 
                               columns = self.clean_names(l_bigrams),
                               index = self.index)
        return dual_bigrams
        
    #helper functions
    def parse_lymph_nodes(self, node_string):
        if self.validation:
            node_string = re.sub('2/3', '2A', node_string)
        #the data apparently has just '2' when theres a '2A' and '2B'
        node_string = re.sub('([L,R])([2,5]),+','\g<1>\g<2>A, \g<1>\g<2>B,', node_string)
        node_string = re.sub('([L,R])([2,5])$','\g<1>\g<2>A, \g<1>\g<2>B', node_string)
        node_string = re.sub('R RPLN', 'RRPLN', node_string)
        node_string = re.sub('L RPLN', 'LRPLN', node_string)
        nodes = [n.strip().upper() for n in node_string.split(',')]
        #remove the node with 'in-between' labeled nodes?
        if not self.validation:
            for n in nodes:
                if n in LNDataset.ambiguous_nodes:
                    return np.NaN
        nodes = [n for n in nodes if n in self.all_nodes]
        if not self.validation and len(nodes) <= 0:
            return np.NaN
        return nodes
    
    def clean_ln_data(self):
        self.data['nodes'] = (self.data.copy())['affected_nodes'].apply(self.parse_lymph_nodes).values
        self.data = self.data.dropna(subset=['nodes'])
        self.data['nodes'] = self.data['nodes'].apply(lambda x: sorted(x))

    def get_monograms(self, data):
        monograms = pd.DataFrame(index = self.index, columns = self.nodes, dtype = np.int32).fillna(0)
        for pos, p in enumerate(data['nodes']):
            index = self.index[pos]
            for lymph_node in p:
                monograms.loc[index, lymph_node] = 1
        return monograms

    def get_bigram_names(self):
        bigram_set = set([])

        for i, name in enumerate(self.node_list):
            for i2 in range(i+1, len(self.node_list)):
                if self.adjacency.iloc[i,i2] > 0:
                    bigram_set.add(name + self.node_list[i2])
        ' '.join(sorted(bigram_set))
        bigram_names = (sorted(bigram_set))
        return bigram_names, bigram_set

    def bigramize(self, v, side):
        #shoudl take a unilateral (left or right) matrix of affected lypmh nnodes
        assert(v.shape[1] == self.adjacency.shape[1])
        col_names = list(v.columns)
        clean = lambda x:  re.sub('^[LR]\s*','', x)
        bigrams = []
        names = []
        _, bigram_set = self.get_bigram_names()
        for i, colname in enumerate(col_names):
            nodename = clean(colname)
            for i2 in range(i+1, v.shape[1]):
                colname2 = col_names[i2]
                bigram_name = nodename + clean(colname2)
                if bigram_name in bigram_set:
                    if bigram_name not in names:
                        names.append(side + bigram_name)
                    bigram_vector = (v[colname].values + v[colname2].values)
                    bigrams.append(bigram_vector.reshape(-1,1))
        return pd.DataFrame(np.hstack(bigrams), columns = names, index = self.data.index)
    
    def formatted_features(self):
        datacopy = self.data.copy()
        datacopy.columns = [LNDataset.js_name_dict.get(x, x) for x in datacopy.columns]
        datacopy.nodes = rename_nodes(datacopy.nodes)
        return datacopy
    
    def input_full_ncat(self):
        for i,row in self.data.iterrows():
            ncat = row['N-category']
            if ncat == 'N2':
                if len(row['nodes']) <= 1:
                    ncat = 'N2a'
                else:
                    nodestring = ''.join(row['nodes'])
                    if 'R' in nodestring and 'L' in nodestring:
                        ncat = 'N2c'
                    else:
                        ncat = 'N2c'
                self.data.loc[i,'N-category'] = ncat

def rename_nodes(nodes):
    name_dict = {'RRPLN':'RRP', 'LRPLN':'LRP'}
    rename_node = lambda nodelist: [name_dict.get(x,x) for x in nodelist]
    return nodes.apply(lambda x: sorted(rename_node(x)))



def ln_spread(laterality, nodes):
    if len(nodes) == 0:
        return 0, 0
    pattern = '^[LR]\s*'
    is_bilateral = False
    if laterality == 'Bilateral':
        is_bilateral = True
        laterality = 'L' #bilateral should be treated as a special case were both part sare contralateral
    ipsilateral = [re.sub(pattern,'',n) for n in nodes if n[0] in laterality]
    contralateral = [re.sub(pattern,'',n) for n in nodes if n[0] not in laterality]
    ips_spread = lateral_spread(ipsilateral,False)
    contra_spread = lateral_spread(contralateral,False)
    if is_bilateral:
        max_spread = np.max([ips_spread, contra_spread])
        min_spread = np.min([ips_spread, contra_spread])
        return max_spread, min_spread
    return ips_spread, contra_spread
    
def lateral_spread(nodes, clean = True):
    max_dist = 0
    if clean:
        nodes = [re.sub('^[LR]\s*','',n) for n in nodes]
    if len(nodes) == 1:
        return 0
    nodes = sorted(nodes)
    spread_dict = LNDataset.spread_dict
    max_value = max(spread_dict.values())
    for i in range(len(nodes) - 1):
        for ii in range(i, len(nodes)):
            pair = nodes[i] + nodes[ii]
            if pair in spread_dict:
                new_dist = spread_dict[pair]
                if new_dist > max_dist:
                    max_dist = new_dist
                    if max_dist == max_value:
                        break
    if 'RPLN' in nodes: #idk add more spread if this is here?
        max_dist += 1
    return max_dist

def node_spread(df, to_dataframe = True, max_ips = 2, max_contra = 4):
    node_df = df.loc[:,['laterality','nodes']]
    spread = node_df.apply(lambda x: ln_spread(x[0],x[1]),axis=1).values
    spread = np.array(list(spread))
    spread = np.array([max_ips, max_contra])*spread/spread.max(axis=0)
    if to_dataframe:
        spread = pd.DataFrame(spread, columns = ['ips_spread','contra_spread'], index = df.index.values)
        spread.index.name = df.index.name
    return spread

def l1(x1, x2):
    return np.sum(np.abs(x1-x2))

def tanimoto_dist(x1, x2):
    if l1(x1, x2) == 0:
        return 0
    tanimoto = x1.dot(x2)/(x1.dot(x1) + x2.dot(x2) - x1.dot(x2))
    #guadalupe used 1 - similarity for her clustering
    return 1/(1+tanimoto)

def tanimoto_dist_alt(x1, x2):
    if l1(x1, x2) == 0:
        return 0
    tanimoto = x1.dot(x2)/(x1.dot(x1) + x2.dot(x2) - x1.dot(x2))
    #guadalupe used 1 - similarity for her clustering
    return (1 - tanimoto)

def l2(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

def dist_matrix(x, dist_func, scale = False):
    n = x.shape[0]
    distance = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1, n):
            distance[i,j] = (dist_func(x[i], x[j]))
    distance += distance.transpose()
    if scale:
        distance = (distance - distance.min())/(distance.max() - distance.min())
    return distance

def sim_matrix(x, dist_func, zero_axis = True):
    dist = dist_matrix(x, dist_func, scale = True)
    sim = 1 - dist
    if zero_axis:
        sim[np.arange(x.shape[0]), np.arange(x.shape[0])] = 0
    return sim


def LN_similarity_result_matrix(ln_dataset):
    x = ln_dataset.spatial().values
    ids = ln_dataset.index.values.ravel()
    similarity = sim_matrix(x, dist_func = tanimoto_dist)
    most_similar_ids = []
    similarity_scores = []
    for pos, sim_row in enumerate(similarity):
        sorted_args = np.argsort(-sim_row).ravel()
        sorted_sims = sim_row[sorted_args]
        sorted_ids = ids[sorted_args]
        nonzero = np.argwhere(sorted_sims > 0).ravel()
        sorted_sim = np.insert(sorted_sims[nonzero], 0, 1)
        sorted_ids = np.insert(sorted_ids[nonzero], 0, ids[pos])
        most_similar_ids.append(list(sorted_ids))
        similarity_scores.append(list(sorted_sim))
    db = pd.DataFrame({'similarity':most_similar_ids, 'scores':similarity_scores},
                     index = ln_dataset.index)
    return(db)
    
    
class AClusterer(ClusterMixin, BaseEstimator):

    def __init__(self, n_clusters, dist_func = tanimoto_dist, link = 'weighted', criterion = 'maxclust', square_dist = False):
        self.link = link
        self.dist_func = dist_func 
        self.t = n_clusters
        self.square_d = square_dist
        self.criterion = criterion
        
    def get_leaves(self, x):
        clusters = linkage(x, method = self.link, metric = self.dist_func, optimal_ordering = True)
        dendro = dendrogram(clusters)
        return np.array(dendro['leaves']).astype('int32')
    
    def fit_predict(self, x, y = None):
        dists = pdist(x,metric = self.dist_func)
        if self.square_d:
            dists = dists**2
        clusters = linkage(dists, method = self.link, metric = self.dist_func, optimal_ordering = True)
        return fcluster(clusters, self.t, criterion = self.criterion)
    
class FClusterer(ClusterMixin, BaseEstimator):

    def __init__(self, n_clusters, dist_func = tanimoto_dist, link = 'weighted', criterion = 'maxclust'):
        self.link = link
        self.dist_func = dist_func if link not in ['median', 'ward', 'centroid'] else 'euclidean'
        self.t = n_clusters
        self.criterion = criterion
        
    def get_leaves(self, x):
        clusters = linkage(x, method = self.link, metric = self.dist_func, optimal_ordering = True)
        dendro = dendrogram(clusters)
        return np.array(dendro['leaves']).astype('int32')
    
    def fit_predict(self, x, y = None):
        clusters = linkage(x, method = self.link, metric = self.dist_func, optimal_ordering = True)
        return fcluster(clusters, self.t, criterion = self.criterion)
    
def ordinalize(array):
    vals = np.unique(array)
    mapkey = {val: pos+1 for pos,val in enumerate(vals)}
    array = np.array([mapkey[a] for a in array])
    return array 

def get_clustering(x, dist_func = tanimoto_dist, k=6, max_frac=.5, as_df = True, link = 'ward'):
    fcluster = FClusterer(0, criterion = 'distance', dist_func = dist_func, link=link)
    gcluster = FClusterer(k, dist_func = dist_func, link= link)
    clusters = fcluster.fit_predict(x)
    groups = gcluster.fit_predict(x)
    c, g = ordinalize(clusters), ordinalize(groups)
    leaves = fcluster.get_leaves(x)
    return c, g, leaves

def save_clustering(dataset, target_file, k =6, g = None, bigrams = True):
    x = dataset.spatial().values if bigrams else dataset.nonspatial().values
    if g is None:
        g = x.shape[0]
    clusters, groups, leaves = get_clustering(x, k, g)
    #leaves is  something, groups is big clusters, clusters is small clusters
    df_dict = {'clusterId': leaves, 'groupId': groups, 'dendrogramId': clusters}
    df = pd.DataFrame(df_dict, index = dataset.index)
    df.index.rename('patientId', inplace = True)
    df.to_csv(target_file)
    return df
    
def cohort_sample_weights(df, var_name):
    if var_name == 'affected_nodes':
        try:
            weights = df['affected_nodes'].apply(lambda x: len(x.split(','))).values
            return weights
        except:
            pass
    return np.ones((df.shape[0],))
    
def randomized_cohort_data(df, frac = .6):
    if isinstance(df, LNDataset):
        df = df.data
    newdata = df.copy()
    for col in newdata.columns:
        weights = cohort_sample_weights(df, col)
        newdata[col] = newdata[col].sample(frac=1, replace=True, weights = weights).values
    newdata = newdata.sample(frac=frac,replace=False)
    newdata.index = np.arange(newdata.shape[0]) + 1
    newdata.index.rename('Dummy ID',inplace=True)
    return newdata.dropna()

def json_ready_dataset(dataset):
    data = dataset.formatted_features()
    similarity_data = LN_similarity_result_matrix(dataset)
    data = pd.concat([data, similarity_data],axis=1)
    data['id'] = data.index.values.astype('int32')
    return data
    
def dataset_to_json(dataset, path = 'example_dataset.json'):
    #todo: get scores
    data = json_ready_dataset(dataset)
    data.to_json(path, orient='records')
    
def export_dataset(dataset,
                   target_json = 'example_patients.json',
                   target_bigram_cluster_csv = 'example_bigram_clusters.csv',
                   target_unigram_cluster_csv = 'example_unigram_clusters.csv'):
    #can be passed a LNDataset or a valid csv string with cohort data, and exports the js frontend files
    if isinstance(dataset, str):
        dataset = LNDataset(dataset)
    dataset_to_json(dataset, target_json)
    save_clustering(dataset, target_bigram_cluster_csv, bigrams = True)
    save_clustering(dataset, target_unigram_cluster_csv, bigrams = False)
    
def gen_random_cohort(patient_path, 
                     target_csv ='example_patients.csv',
                     target_json ='example_patients.json',
                     target_bigram_cluster_csv = 'example_bigram_clusters.csv',
                     target_unigram_cluster_csv = 'example_unigram_clusters.csv'):
    dataset = LNDataset(patient_path)
    data = randomized_cohort_data(dataset.data)
    dataset = LNDataset(data.sort_index())
    if target_csv is not None:
        try:
            example_data = dataset.data.drop(['nodes'],axis=1)
            example_data.index.rename('Dummy ID', inplace = True)
            example_data.to_csv(target_csv)
            print('saved synthetic data csv to', target_csv)
        except:
            print('error saving synthetic data as csv')
    export_dataset(dataset, target_json, target_bigram_cluster_csv, target_unigram_cluster_csv)
    return dataset

def get_cluster_percentages(dataset, clusters, 
                        outcomes = ['FT', 'AS']):
    data = dataset.data.copy()
    data['clusters'] = clusters
    result_table = {}
    for c in np.unique(clusters):
        subset = data[data.clusters == c]
        result_row = {}
        n_patients = subset.shape[0]
        result_row['Total Count'] = n_patients
        for o in outcomes:
            n_positive = subset[o].values.sum()
            percent = np.round(100*n_positive/n_patients, 1)
            result_row[o + ' count'] = n_positive
            result_row['%'+o] = percent
        result_table[c] = result_row
    return pd.DataFrame(result_table).T

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

def lg():
    return LogisticRegression(C = 100, solver = 'lbfgs', max_iter = 5000)

def stratified_lg_AUC(model, x, y):
    if isinstance(x, pd.DataFrame):
        x = x.values
    if isinstance(y, pd.DataFrame):
        y = y.values.ravel()
    if x.ndim == 1:
        x = x.reshape(-1,1)
    cv = StratifiedKFold(n_splits = 10)
    aucs = []
    for i, (train, test) in enumerate(cv.split(x,y)):
        model.fit(x[train],y[train])
        y_pred = model.predict_proba(x[test])[:,1]
        aucs.append(roc_auc_score(y[test],y_pred))
    return np.mean(aucs)
        
def get_cluster_correlations(dataset, clusters, 
                        outcomes = ['FT', 'AS']):
    data = dataset.data.copy()
    result_table = {}
    if isinstance(clusters, pd.DataFrame):
        clusters = clusters.values
    clusters = clusters.ravel()
    for o in outcomes:
        y = data[o].values
        pval = fisher_exact_test(clusters, y)
        chi2_pval = chi2_contingency(get_contingency_table(clusters, y))[1]
        strat_AUC = stratified_lg_AUC(lg(), clusters, y)
        result_table[o] = {'fisher_exact': pval,  'chi2': chi2_pval,'LgCvAUC': strat_AUC}
    return pd.DataFrame(result_table).T