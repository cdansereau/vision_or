__author__ = "Christian Dansereau"
__copyright__ = "Copyright 2015, Christian Dansereau"

import manip
from manip import show as show
from manip import showbw as showbw
import fextraction as fe
import numpy as np
from skimage.color import label2rgb

def getbg():
    images = manip.load_images(30)
    bg = np.median(images,2)
    return bg

def detect(img_id):
    imrgb = manip.load_image(img_id)
    im = imrgb.mean(2)
    mask_raw = fe.bg_mask(im-bg)
    mask = fe.erode(mask_raw)
    clean_img = fe.clean_mask(mask)
    label_image = fe.label(mask)
    borders = np.logical_xor(mask, clean_img)
    label_image[borders] = -1

    # Overlay the segmentation on the original image
    image_label_overlay = label2rgb(label_image, image=imrgb)

    regions = fe.get_metrics(label_image)
    fe.show_prop (imrgb,regions)
    return im, mask, label_image, image_label_overlay

if __name__ == "__main__":
    bg = getbg()
    detect(34)



