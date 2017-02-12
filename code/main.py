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
import pandas as pd
import matplotlib.pyplot as plt

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
    pulls in npz patient files
    :param filename: location of .npz data file
    :return: dictionary with 3 dimensional array
        key = patient name
        value = 3d np array of density data x, y, z
    """
    raw_data = {}
    raw_npz = np.load(filename)

    for patient in raw_npz:
        raw_data[patient] = raw_npz[patient].astype(dtype='int32')

    return raw_data

def load_source_of_truth(filename):
    """
    imports the csv file that tells us which patients have cancer
    :param filename: location of *.csv file
    :return: dictionary {patient_id: has_cancer}
    """
    source_of_truth = pd.read_csv(filename)
    return dict(zip(list(source_of_truth.id), list(source_of_truth.cancer)))


def random_patient_with_cancer(source_of_truth):
    """
    surprise, surprise... this method returns a random patient who has cancer according to the source of truth
    :param source_of_truth: dictionary with format {patient_id: has_cancer}
    :return: patient_id (from above)
    """
    cancer_list = [patient for patient in source_of_truth if source_of_truth[patient]==1]
    return np.random.choice(cancer_list)


def mask_and_partition(image_array, mask_array, min_value = None, max_value = None):
    """
    takes an image and applies the mask to it to narrow down area of interest,
    then finds the points that fall within the bounds
    :param image_array:
    :param mask_array:
    :param min_value:
    :param max_value:
    :return:
    """
    masked_image_array = tf.Session().run(tf.pow(image_array, mask_array))
    return masked_image_array


def slice_n_dice(image_array, cube_size):
    """
    Takes a preprocessed image and sliced into a bunch of cubes of various sizes and locations for easier processing
    :param image_array: numpy array of the diacom image
    :param cube_size: how big of a cube do you want
    :return:
    """
    return


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
    return

def make_2d_funky(cube_array, xyz1, xyz2, xyz3):
    """
    make a 2d array given a 3d array and 3 point
    :param cube_array: 3d array of data
    :param xyz1: first point tuple (x,y,z)
    :param xyz2: second point tuple (x,y,z)
    :param xyz3: third point tuple (x,y,z)
    :return: 2d matrix
    """
    #TODO: We probably want to be able to get a slice through the center of mass, farthest point and another point
    # this would allow our image trainer to see some valuable images of the POI
    return


def make_2d(cube_array, x, y, z):
    """
    Takes a 3D array and x, y, z center point then outputs a few 2D slices through that centerpoint
    :param cube_array: 3D numpy array
    :param x: center point in x axis
    :param y: center point in y axis
    :param z: center point in z axis
    :return:
    """
    xy_slice = [[0 for col in cube_array[0][0]] for row in cube_array[0]]
    xz_slice = [[0 for col in cube_array[0][0]] for row in cube_array]
    yz_slice = [[0 for col in cube_array[0]] for row in cube_array]
    # take an xy slice
    xy_slice = cube_array[z]
    # take and xz slice
    for i in range(len(cube_array[0][0])):
        for j in range(len(cube_array)):
            xz_slice[j][i] = cube_array[j][y][i]
    # take a yz slice
    for i in range(len(cube_array[0])):
        for j in range(len(cube_array)):
            yz_slice[j][i] = cube_array[j][i][x]
    return xy_slice, xz_slice, yz_slice


def plot_3_by_3(raw, mask, masked, x=80, y=150, z=80):
    """
    makes a 3x3 subplot showing the raw, mask and masked image for each xy, xz and yz plane
    :param raw: 3d array of raw data
    :param mask: 3d array of mask
    :param masked: 3d array of masked data
    :param x: the x point you want to slice at (defaults to 80)
    :param y: the y point you want to slice at (defaults to 150)
    :param z: the z point you want to slice at (defaults to 80)
    :return: no return
    """
    raw_xy, raw_xz, raw_yz = make_2d(raw, x, y, z)
    mask_xy, mask_xz, mask_yz = make_2d(mask, x, y, z)
    masked_xy, masked_xz, masked_yz = make_2d(masked, x, y, z)
    ax = plt.subplot(3, 3, 1)
    ax.set_title("xy raw")
    plt.imshow(raw_xy, cmap=plt.cm.gray)
    ax = plt.subplot(3, 3, 2)
    ax.set_title("xy mask")
    plt.imshow(mask_xy, cmap=plt.cm.gray)
    ax = plt.subplot(3, 3, 3)
    ax.set_title("xy masked")
    plt.imshow(masked_xy, cmap=plt.cm.gray)
    ax = plt.subplot(3, 3, 4)
    ax.set_title("xz raw")
    plt.imshow(raw_xz, cmap=plt.cm.gray)
    ax = plt.subplot(3, 3, 5)
    ax.set_title("xz mask")
    plt.imshow(mask_xz, cmap=plt.cm.gray)
    ax = plt.subplot(3, 3, 6)
    ax.set_title("xz masked")
    plt.imshow(masked_xz, cmap=plt.cm.gray)
    ax = plt.subplot(3, 3, 7)
    ax.set_title("yz raw")
    plt.imshow(raw_yz, cmap=plt.cm.gray)
    ax = plt.subplot(3, 3, 8)
    ax.set_title("yz mask")
    plt.imshow(mask_yz, cmap=plt.cm.gray)
    ax = plt.subplot(3, 3, 9)
    ax.set_title("yz masked")
    plt.imshow(masked_yz, cmap=plt.cm.gray)
    plt.show()


if __name__ == "__main__":
    raw_patients_file = 'C:\\GIT\\kaggle_data_science_bowl_2017\\Data\\sample_patients_resampled.npz'
    patient_masks_file = 'C:\\GIT\\kaggle_data_science_bowl_2017\\Data\\sample_patients_segmented.npz'
    truth_file = 'C:\\GIT\\kaggle_data_science_bowl_2017\\stage1_labels.csv'
    source_of_truth = load_source_of_truth(truth_file)
    # print("Random Patient with Cancer: %s" % random_patient_with_cancer(source_of_truth))
    dict_of_patients = load_data(raw_patients_file)
    dict_of_masks = load_data(patient_masks_file)
    print("stuff loaded")
    lucky_winner = None
    while lucky_winner not in dict_of_patients:
        lucky_winner = random_patient_with_cancer(source_of_truth)
    print("Winner chosen at random")
    lucky_winner_raw_data = dict_of_patients[lucky_winner]
    lucky_winner_mask = dict_of_masks[lucky_winner]
    lucky_winner_masked_data = mask_and_partition(dict_of_patients[lucky_winner], dict_of_masks[lucky_winner])
    plot_3_by_3(lucky_winner_raw_data,lucky_winner_mask,lucky_winner_masked_data)