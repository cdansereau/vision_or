__author__ = "Christian Dansereau"
__copyright__ = "Copyright 2015, Christian Dansereau"

from skimage import io
from skimage.filter import threshold_otsu, threshold_adaptive
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import numpy as np
from skimage.morphology import label, closing, square, binary_erosion, binary_dilation, binary_closing, remove_small_objects
from skimage.measure import label
from skimage.segmentation import clear_border
from skimage.measure import regionprops
import math
from skimage.draw import ellipse
from skimage.transform import rotate
from skimage.color import label2rgb
import manip

def getbg():
    images = manip.load_images(30)
    bg = np.median(images,2)
    return bg

def bg_mask(image):
    # apply threshold
    #thresh = threshold_otsu(image)
    thresh = threshold_adaptive((image),200,offset=40)
    #thresh = np.abs(image)>(5*np.abs(image).mean())
    #bw = closing(image > thresh, square(3))
    bw = thresh == False;
    bw = binary_closing(bw, square(2))
    bw = binary_dilation(bw,square(4))
    return bw

def smooth_vol(image):
    image = binary_dilation(image,square(70))
    #put the border at zero
    image[:,(0,-1)] = np.zeros(image[:,(0,-1)].shape)
    image[(0,-1),:] = np.zeros(image[(0,-1),:].shape) 
    image = binary_erosion(image,square(70))
    
    image = label(image) 
    image = remove_small_objects(image,min_size=100)
    image = image > 0
    return image

def clean_mask(image):
    # remove artifacts connected to image border
    cleared = image.copy()
    clear_border(cleared)
    return cleared

def get_metrics(label_image):
    return regionprops(label_image)


def show_prop(image,regions):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap=plt.cm.gray)

    for props in regions:
        y0, x0 = props.centroid
        orientation = props.orientation
        x1 = x0 + math.cos(orientation) * 0.5 * props.major_axis_length
        y1 = y0 - math.sin(orientation) * 0.5 * props.major_axis_length
        x2 = x0 - math.sin(orientation) * 0.5 * props.minor_axis_length
        y2 = y0 - math.cos(orientation) * 0.5 * props.minor_axis_length

        ax.plot((x0, x1), (y0, y1), '-r', linewidth=1.5)
        ax.plot((x0, x2), (y0, y2), '-r', linewidth=1.5)
        ax.plot(x0, y0, '.g', markersize=10)

        minr, minc, maxr, maxc = props.bbox
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)
        ax.plot(bx, by, '-b', linewidth=1.5)

    ax.axis((0, image.shape[1], image.shape[0], 0))
    plt.show()

def show_prop_2class(image,regions,r_class):
    fig, ax = plt.subplots()
    ax.imshow(image)

    for idx in range(0,len(regions)):
        props = regions[idx]
        y0, x0 = props.centroid
        orientation = props.orientation
        x1 = x0 + math.cos(orientation) * 0.5 * props.major_axis_length
        y1 = y0 - math.sin(orientation) * 0.5 * props.major_axis_length
        x2 = x0 - math.sin(orientation) * 0.5 * props.minor_axis_length
        y2 = y0 - math.cos(orientation) * 0.5 * props.minor_axis_length

        #ax.plot((x0, x1), (y0, y1), '-r', linewidth=1.5)
        #ax.plot((x0, x2), (y0, y2), '-r', linewidth=1.5)
        #ax.plot(x0, y0, '.g', markersize=10)

        minr, minc, maxr, maxc = props.bbox
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)
        if r_class[idx] == 1:
            ax.plot(bx, by, '-r', linewidth=1)
        else:
            ax.plot(bx, by, '-y', linewidth=1)
    
        ax.axis((0, image.shape[1], image.shape[0], 0))
    plt.show()

def show_prop_2class_track(image,regions,r_class,trackingpoints):
    fig, ax = plt.subplots()
    ax.imshow(image)

    list_dots = np.nonzero(trackingpoints>0)
    for dotidx in range(0,len(list_dots[0])):
        x = list_dots[0][dotidx]
        y = list_dots[1][dotidx]
        if trackingpoints[x,y] == 1:
            ax.plot(y, x, '.m', markersize=10)
        else:
            ax.plot(y, x, '.c', markersize=10)

        ax.axis((0, image.shape[1], image.shape[0], 0))

    for idx in range(0,len(regions)):
        props = regions[idx]
        y0, x0 = props.centroid
        orientation = props.orientation
        x1 = x0 + math.cos(orientation) * 0.5 * props.major_axis_length
        y1 = y0 - math.sin(orientation) * 0.5 * props.major_axis_length
        x2 = x0 - math.sin(orientation) * 0.5 * props.minor_axis_length
        y2 = y0 - math.cos(orientation) * 0.5 * props.minor_axis_length

        #ax.plot((x0, x1), (y0, y1), '-r', linewidth=1.5)
        #ax.plot((x0, x2), (y0, y2), '-r', linewidth=1.5)
        #ax.plot(x0, y0, '.g', markersize=10)

        minr, minc, maxr, maxc = props.bbox
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)
        if r_class[idx] == 1:
            ax.plot(bx, by, '-r', linewidth=1)
        else:
            ax.plot(bx, by, '-y', linewidth=1)

        ax.axis((0, image.shape[1], image.shape[0], 0))
    plt.show()


def labels(image):
    # apply labels
    classif = label(image)
    return classif

def detect(imrgb, bg):
    im = imrgb.mean(2)
    mask = bg_mask((im-bg))
    #mask = erode(mask_raw)
    clean_img = clean_mask(mask)
    label_image = label(mask)
    label_image = remove_small_objects(label_image,min_size=100)
    # last dilation
    mask = label_image > 0
    mask = smooth_vol(mask)
    mask = smooth_vol(mask)
    mask = smooth_vol(mask)
    mask = smooth_vol(mask)
    label_image = label(mask)
    label_image = remove_small_objects(label_image,min_size=100)
    #borders = np.logical_xor(mask, mask_raw)
    background_idx = label_image == 0
    label_image[background_idx] = -1

    # Overlay the segmentation on the original image
    image_label_overlay = label2rgb(label_image, image=imrgb)
    regions = get_metrics(label_image)
    return image_label_overlay,label_image,regions 

if __name__ == "__main__":
    
    im = load_image(5).mean(2)
    bw = apply_bw(im)



