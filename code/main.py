__author__ = 'wmorrill'
# Thought Process:
# take training set and isolate lung with a smooth surface (we want to catch nodules that might be on the edge)
# probably add some buffer to this volume
# chop the lung volume into cubes (or possibly just search the whole image?)
# check each cube for a sphere like object
# use training set to determine which sphere like objects are not cancer and which might be
# (we only know which patients have chance not where it is or what it looks like)
# determine the difference in features between definite negatives and potential positives
# possible features; size, uniformity, density,


# Location of info to get tensorflow working
# http://www.heatonresearch.com/2017/01/01/tensorflow-windows-gpu.html

# retrain inception v3 on 2d images?

import tensorflow as tf
import numpy as np

IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
VALIDATION_SIZE = 5000  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 64
NUM_EPOCHS = 10
EVAL_BATCH_SIZE = 64
EVAL_FREQUENCY = 100  # Number of steps between evaluations.


def load_data(filename):
    """
    Args:
        filename: location of .npz data file

    Returns: 4 dimensional array
        0 dimension = patient
        1-3 dimension = density data x, y, z
    """
    raw_data = []
    raw_npz = np.load(filename)

    for i in range(len(raw_npz._files)):
        raw_data.append(raw_npz[raw_npz._files[i]])

    return raw_data


def mask_and_partition(image_array, mask_array, min_value, max_value):
    """
    takes an image and applies the mask to it to narrow down area of interest,
    then finds the points that fall within the bounds
    :param image_array:
    :param mask_array:
    :param min_value:
    :param max_value:
    :return:
    """
    # masked_image_array = tf.batch_matmul(image_array, mask_array)


def slice_n_dice(image_array, cube_size):
    """
    Takes a preprocessed image and sliced into a bunch of cubes of various sizes and locations for easier processing
    :param image_array: numpy array of the diacom image
    :param cube_size: how big of a cube do you want
    :return:
    """


def inspect_cube(cube_array):
    """
    takes a cube array subset of a 3D image and looks for something tumor-y
    :param cube_array:
    :return:
    """
    # is it tube shaped or not:
    # find the center of mass
    # find the mean distance for equally (with some buffer) dense pixels
    # is the mean equal in all/most directions?
    # are there bits touching the edges of the cube?
    # How big are the cross sections that intersect the cube wall


def make_2d(cube_array, x, y, z):
    """
    Takes a 3D array and x, y center point then outputs a few 2D slices through that centerpoint
    :param cube_array: 3D numpy array
    :param x: center point in x axis
    :param y: center point in y axis
    :param z: center point in z axis
    :return:
    """