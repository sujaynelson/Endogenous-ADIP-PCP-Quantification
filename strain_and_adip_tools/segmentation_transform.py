""" functions to convert between watershed and skeleton segmented images"""

import numpy as np
from scipy import stats

from strain_and_adip_tools.segmentation_transform_utils import pad_with, PAD_VALUE, fill_region


def skeleton_to_watershed(skeleton_image_array, region_value=0, boundary_value=255, keep_boundaries=False):
    """ converts a segmented skeleton image (all regions are same value with region boundaries being a second value)
    to a watershed segmented image (each region has a unique value and there are no boundary pixels, background
    region has value of zero)

    :param skeleton_image_array: 2D numpy array with pixel values of a skeleton segmented image
    :type skeleton_image_array: np.ndarray
    :param region_value: value of pixels in regions in skeleton_segmented images (default is 0)
    :type region_value: float
    :param boundary_value: value of boundary pixels of regions in skeleton_segmented images (default is 255)
    :type boundary_value: float
    :param keep_boundaries: if True, watershed image will keep boundaries in returned result
    :type keep_boundaries: bool
    :return: watershed_image_array
    :rtype watershed_image_array: np.ndarray
    """

    # throw error if skeleton_image_array is not np.ndarray
    if not isinstance(skeleton_image_array, np.ndarray):
        raise TypeError("skeleton_image_array must be of type numpy.ndarray")

    # throw error if skeleton_image_array is not two dimensions
    if np.ndim(skeleton_image_array) != 2:
        raise ValueError("skeleton_image_array must be two-dimensional")

    # throw error if array contains more than two values
    if not (np.unique(skeleton_image_array).shape == (2,)):
        raise ValueError("skeleton_image_array does not have only two values")

    # throw error if array contains values other than region_value and boundary_value
    if not (np.array_equal(np.unique(skeleton_image_array), [region_value, boundary_value]) or np.array_equal(np.unique(
            skeleton_image_array), [region_value, boundary_value])):
        raise ValueError("skeleton_image_array contains values other than region_value and boundary_value")

    # copy input array (skeleton image)
    watershed_image_array = np.copy(skeleton_image_array)

    # fill each region with unique value
    new_value = 1
    with np.nditer(watershed_image_array, flags=['multi_index'], op_flags=['readwrite']) as iterator:
        for old_value in iterator:
            if new_value == boundary_value:
                # make sure we don't label any regions with same value as boundary value
                new_value += 1
            # if old_value == boundary_value:
            # don't touch boundary pixels
            elif old_value == region_value:
                # if pixel has not been touched, fill region
                position = iterator.multi_index
                fill_region(watershed_image_array, position, new_value)
                new_value += 1

    # determine pixel value of background
    # potential background values are the values in the four corners of the array
    potential_background_values = watershed_image_array[[0, 0, -1, -1], [0, -1, 0, -1]]
    # we determine the background value to be the mode of the potential values
    background_value = stats.mode(potential_background_values)[0][0]

    # search for position of background value
    search_results = np.where(watershed_image_array == background_value)
    background_position = list(zip(search_results[0], search_results[1]))[0]

    # fill background region with zeros
    fill_region(watershed_image_array, background_position, 0)
    new_watershed_image_array = np.copy(watershed_image_array)

    if not keep_boundaries:
        # remove boundaries between regions
        while boundary_value in new_watershed_image_array:
            with np.nditer(new_watershed_image_array, flags=['multi_index'], op_flags=['readwrite']) as iterator:
                for pixel in iterator:
                    if pixel == 255:
                        position = iterator.multi_index
                        north = tuple(map(lambda i, j: i + j, position, (-1, 0)))
                        east = tuple(map(lambda i, j: i + j, position, (0, 1)))
                        northeast = tuple(map(lambda i, j: i + j, position, (-1, 1)))

                        if watershed_image_array[north] != boundary_value:
                            pixel[...] = watershed_image_array[north]
                        elif watershed_image_array[northeast] != boundary_value:
                            pixel[...] = watershed_image_array[northeast]
                        elif watershed_image_array[east] != boundary_value:
                            pixel[...] = watershed_image_array[east]
            watershed_image_array = np.copy(new_watershed_image_array)

    return new_watershed_image_array


def watershed_to_skeleton(watershed_image_array, region_value=0, boundary_value=255):
    """ converts a watershed segmented image (no boundaries between regions and each region has a different pixel
    value, background region has value of zero) to a skeleton segmented image (each region has the same pixel value
    and are separated by boundaries of a second value)

    :param watershed_image_array: 2D numpy array with pixel values of a watershed segmented image
    :type watershed_image_array: numpy.ndarray
    :param region_value: desired value of all regions in the output (skeleton segmented) array
    :type region_value: float
    :param boundary_value: desired value of boundary pixels in the output (skeleton segmented) array
    :type boundary_value: float
    :return: skeleton_image_array
    :rtype skeleton_image_array: np.ndarray
    """

    # create a one pixel wide padding around the edge of the image for easy iteration
    padded_array = np.pad(watershed_image_array, 1, pad_with)
    # print(padded_array.shape)

    # initialize a new nparray to hold skeleton segmented image (mask)
    skeleton_image_array = np.full(watershed_image_array.shape, region_value)

    # if a pixel neighbors a pixel of a different value, consider it a boundary point and add it to the skeleton image
    for row in range(1, padded_array.shape[0] - 1):
        for col in range(1, padded_array.shape[1] - 1):

            neighboring_values = [padded_array[row, col],
                                  padded_array[row, col + 1],
                                  padded_array[row + 1, col],
                                  padded_array[row + 1, col + 1]]

            # remove duplicates
            neighboring_values = list(set(neighboring_values))

            # remove padded value if in list of neighboring values
            try:
                neighboring_values.remove(PAD_VALUE)
            except ValueError:
                pass

            if len(neighboring_values) >= 2:
                skeleton_image_array[row - 1, col - 1] = boundary_value

    # FIXME: right most column and bottom row have boundary_value when they should not be
    # temporary fix: set values in last row and column to region_value
    skeleton_image_array[-1, :] = region_value
    skeleton_image_array[:, -1] = region_value

    return skeleton_image_array
