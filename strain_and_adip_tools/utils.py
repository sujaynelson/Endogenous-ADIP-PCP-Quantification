""" Top-level utility functions for pycellfit module."""

__author__ = "Nilai Vemula"

import os

import PIL.Image
import matplotlib.pyplot as plt
import numpy as np


def read_segmented_image(file_name, visualize=False):
    """Displays the segmented image using matplotlib

    :param file_name: file name of a segmented image in .tif format
    :type file_name: str
    :param visualize: if true, then image will be plotted using matplotlib.pyplot
    :type visualize: bool
    :raises TypeError: only accepts tif/tiff files as input
    :return: array of pixel values
    :rtype: numpy.ndarray
    """

    extension = os.path.splitext(file_name)[1]
    if not (extension == '.tif' or extension == '.tiff'):
        raise TypeError("Invalid File Type. File must be a .tif or .tiff")

    im = PIL.Image.open(file_name)
    img_array = np.array(im)

    if visualize:
        plt.imshow(img_array, cmap='gray', interpolation="nearest", vmax=255)

    return img_array
