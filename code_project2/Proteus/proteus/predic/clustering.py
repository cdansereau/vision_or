__author__ = 'Christian Dansereau'

import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import fcluster
import numpy as np

def hclustering(data, t):
    row_dist = pd.DataFrame(squareform(pdist(data, metric='euclidean')))
    row_clusters = linkage(row_dist, method='ward')
    ind = fcluster(row_clusters, t, criterion='maxclust')
    return ind

def part(m,ind):
    n_cluster = np.max(ind)
    a = pd.DataFrame(m)
    new_m = np.zeros((n_cluster, n_cluster))
    for i1 in range(0, n_cluster):
        for i2 in range(0, n_cluster):
            new_m[i1, i2] = a.loc[ind == i1+1, ind == i2+1].mean().mean()
    return new_m

def ind2matrix(ind):
    part = pd.DataFrame(np.zeros([len(ind), len(ind)]))  # init partition
    for i in range(0,max(ind)):
        l = (ind == i+1)
        part.loc[l, l] = i+1
    return part

def order(ind):
    order_idx = []
    for i in range(0,max(ind)):
        l = (ind == i+1)
        order_idx = order_idx + np.where(l)[0].tolist()
    return order_idx

def ordermat(m,ind):
    order_idx = order(ind)
    new_m = pd.DataFrame(m)  # init
    return new_m.loc[order_idx, order_idx].values

# Test functions
def test_ind2matrix():
    ind = np.array([1,2,3,1])
    assert np.all(ind2matrix(ind) == np.array([[ 1.,  0.,  0.,  1.],
       [ 0.,  2.,  0.,  0.],
       [ 0.,  0.,  3.,  0.],
       [ 1.,  0.,  0.,  1.]]))
