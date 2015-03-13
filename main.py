__author__ = "Christian Dansereau"
__copyright__ = "Copyright 2015, Christian Dansereau"

import manip
from manip import show as show
from manip import showbw as showbw
import fextraction as fe
import numpy as np
from skimage.color import label2rgb
import cPickle as pickle

def detect(img_id):
    imrgb = manip.load_image(img_id)
    imrgb,regions = fe.detect(imrgb,bg)
    fe.show_prop (imrgb,regions)
    return imrgb,regions

def extract_feature():
    bg = fe.getbg()
    features = list()
    for i in range(10,159):
        im = manip.load_image(i)
        imrgb,labels,regions = fe.detect(im,bg)
        features.append(labels)
     
    # Saving the objects:
    with open('features.pkl', 'w') as f:
        pickle.dump(features, f)


if __name__ == "__main__":
    #bg = fe.getbg()
    #imrgb,regions = detect(34)
    extract_feature()

