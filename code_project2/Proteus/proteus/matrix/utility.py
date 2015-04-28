__author__ = 'Christian Dansereau'

import numpy as np
import math

def ismember(a, b):
    bind = {}
    for i, elt in enumerate(b):
        if elt not in bind:
            bind[elt] = i
    return [bind.get(itm, None) for itm in a]

def test_ismember():
    a = np.array([[1,2,3,3,9,8]])
    b = np.array([[2,3,3,8]])
    res = ismember(a,b)
    assert np.all(new_a == a)
