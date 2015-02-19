from skimage import io
from skimage.filter import threshold_otsu
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import numpy as np
from skimage.morphology import label, closing, square, binary_erosion
from skimage.measure import label
from skimage.segmentation import clear_border
from skimage.measure import regionprops
import math
from skimage.draw import ellipse
from skimage.transform import rotate

def bg_mask(image):
    # apply threshold
    thresh = threshold_otsu(image)
    bw = closing(image > thresh, square(3))
    bw = binary_erosion(bw,square(7))
    return bw

def clean_border(image):
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


def labels(image):
    # apply labels
    classif = label(image)
    return classif

if __name__ == "__main__":
    
    im = load_image(5).mean(2)
    bw = apply_bw(im)



