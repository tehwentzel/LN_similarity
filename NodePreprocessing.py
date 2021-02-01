import pandas as pd
import numpy as np
import re
from Clustering import *

#helper functions
def parse_lymph_nodes(node_string):
    #the data apparently has just '2' when theres a '2A' and '2B'
    node_string = re.sub('L2,*','L2A, L2B,', node_string)
    node_string = re.sub('R2,*','R2A, R2B,', node_string)
    node_string = re.sub('R RPLN', 'RRPLN', node_string)
    node_string = re.sub('L RPLN', 'LRPLN', node_string)
    nodes = [n.strip() for n in node_string.split(',')]
    #remove the node with 'in-between' labeled nodes?
    for n in nodes:
        if n in ambiguous_nodes:
            return np.NaN
    nodes = [n for n in nodes if n in all_nodes]
    return nodes if len(nodes) > 0 else np.NaN

def bigramize(v, side):
    #shoudl take a unilateral (left or right) matrix of affected lypmh nnodes
    assert(v.shape[1] == adjacency.shape[1])
    col_names = list(v.columns)
    clean = lambda x:  re.sub('^[LR]\s*','', x)
    bigrams = []
    names = []
    for i, colname in enumerate(col_names):
        nodename = clean(colname)
        for i2 in range(i+1, v.shape[1]):
            colname2 = col_names[i2]
            bigram_name = nodename + clean(colname2)
            if bigram_name in bigram_set:
                if bigram_name not in names:
                    names.append(side + bigram_name)
                bigram_vector = v[colname].values * v[colname2].values
                bigrams.append(bigram_vector.reshape(-1,1))
    print(names)
    return pd.DataFrame(np.hstack(bigrams), columns = names, index = data.index)