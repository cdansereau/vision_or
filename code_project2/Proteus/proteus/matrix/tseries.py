__author__ = 'Christian Dansereau'

import numpy as np
import math

def normalize_data(x):
    x1 = x.copy()
    x1 = (x1 - x1.mean(axis=1)[0])#/x1.std(axis=1)[0]
    return x1

def vol2vec(vol):
    a = vol.shape
    vec_vol = np.reshape(vol,a[0]*a[1]*a[2])
    return vec_vol


def mat2vec(m):
    inddown = np.tril_indices_from(m,-1)
    return m[inddown]


def vec2mat(vec, val_diag=1):
    if vec.ndim > 1:
        vec = vec[:,0]
    M = len(vec)
    N = int(round((1+math.sqrt(1+8*M))/2))
    m = np.identity(N)*val_diag
    inddown = np.tril_indices_from(m,-1)
    indup = np.triu_indices_from(m,1)
    m[inddown]= vec
    m = m.T
    m[inddown]= vec
    return m

def ts2vol(vec,part):
    pass 

def vec2vol(vec,part):
    vol = np.zeros(part.shape)
    for idx in range(0,len(vec)):
        idxs = part == (idx+1)
        vol[idxs] = vec[idx]
    return vol

def get_ts(vol,part):
    # create a NxT (partitions x time points)
    idx = np.unique(part)
    idx = idx[idx>0] #exclude the index 0
    for i in idx:
        mask_parcel = (part == i)
        # compute the average of the time series defined in a partition
        ts_new = np.matrix(vol[mask_parcel].mean(axis=0))
        #print ts_new.shape
        if 'ts' in locals():
            ts = np.vstack((ts,ts_new))
        else:
            ts = ts_new
    return ts


def get_connectome(vol,part):
    # convert the vol in time series
    ts = get_ts(vol,part)
    return np.corrcoef(ts)


def test_mat2vec():
    a = np.array([[1,2,3],[2,1,4],[3,4,1]])
    va = mat2vec(a)
    # [2, 3, 4]
    assert np.all(va == [2, 3, 4])
    new_a = vec2mat(va)
    assert np.all(new_a == a)


def test_vec2mat():
    a = np.array([[1,2,3],[2,1,4],[3,4,1]])
    # [2, 3, 4]
    va == [2, 3, 4]
    new_a = vec2mat(va)
    assert np.all(new_a == a)
