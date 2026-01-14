import numpy as np


def fill_region(array_of_pixels, position, new_value):
    """ fills a region of a 2D numpy array with the same value

    :type array_of_pixels: np.ndarray
    :param array_of_pixels: 2D numpy array of all pixel values
    :param position: tuple with (row, col) location of pixel in the region to modify
    :type position: tuple
    :type new_value: float
    :param new_value: new value for pixel at `position` and all pixels in same region
    :return: None
    """

    xsize, ysize = array_of_pixels.shape
    old_value = array_of_pixels[position[0], position[1]]

    stack = {(position[0], position[1])}
    if new_value == old_value:
        raise ValueError("Region is already filled")

    while stack:
        x, y = stack.pop()

        if array_of_pixels[x, y] == old_value:
            array_of_pixels[x, y] = new_value
            if x > 0:
                stack.add((x - 1, y))
            if x < (xsize - 1):
                stack.add((x + 1, y))
            if y > 0:
                stack.add((x, y - 1))
            if y < (ysize - 1):
                stack.add((x, y + 1))


# constant value that the image array is padded with
PAD_VALUE = -1


def pad_with(vector, pad_width, iaxis, kwargs):
    """helper function that is called by np.pad to surround a nparray with a constant value
    Example: [[0,0],[0,0]] becomes [[-1,-1,-1, -1],[-1, 0, 0, -1],[-1, 0, 0, -1],[-1,-1,-1, -1]]
    """
    pad_value = kwargs.get('padder', PAD_VALUE)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
