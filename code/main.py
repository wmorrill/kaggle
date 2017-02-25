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
from scipy import ndimage
import time

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

session = tf.Session()

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
    # masked_image_array = tf.Session().run(tf.pow(image_array, mask_array))
    # zeroes = np.zeros_like(image_array)
    # air = np.full_like(image_array, -1000)

    masked_image_array = session.run(tf.pow(image_array, mask_array))
    print("masked")
    zero = tf.constant(0, tf.int32)
    air = tf.constant(-1000, tf.int32)
    print("constants made")
    offset = session.run(tf.multiply(air,tf.cast(tf.equal(mask_array,zero), tf.int32)))
    print("offset")
    sum = session.run(tf.add(masked_image_array, offset))
    print("summed")
    del masked_image_array
    del offset
    return sum


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
    :return: 2d matrix  with the most non-zero values
    """
    return_matrix = []
    x1, y1, z1 = xyz1
    x2, y2, z2 = xyz2
    x3, y3, z3 = xyz3
    p1 = np.array([x1, y1, z1])
    p2 = np.array([x2, y2, z2])
    p3 = np.array([x3, y3, z3])
    # These two vectors are in the plane
    v1 = p3 - p1
    v2 = p2 - p1
    # the cross product is normal to the plane
    cp = np.cross(v1, v2)
    a, b, c = cp
    #a * x + b * y + c * z = d
    d = np.dot(cp, p1)
    # print('The equation is {0}x + {1}y + {2}z = {3}'.format(a, b, c, d))
    return_arr = []
    for z in range(len(cube_array)):
        row = []
        for y in range(len(cube_array[0])):
            x = int((c*z + b*y - d)/-a)
            if 0 <= x < len(cube_array[z][y]):
                value = cube_array[z][y][x]
                row.append(value)
            else:
                row.append(-1024)
        return_arr.append(row)

    return_arr2 = []
    for y in range(len(cube_array[0])):
        row = []
        for x in range(len(cube_array[0][0])):
            z = int((a*x + b*y - d)/-c)
            if 0 <= z < len(cube_array):
                value = cube_array[z][y][x]
                row.append(value)
            else:
                row.append(-1024)
        return_arr2.append(row)

    return_arr3 = []
    for z in range(len(cube_array)):
        row = []
        for x in range(len(cube_array[0][0])):
            y = int((a*x + c*z - d)/-b)
            if 0 <= y < len(cube_array[z]):
                value = cube_array[z][y][x]
                row.append(value)
            else:
                row.append(-1024)
        return_arr3.append(row)

    nonzero1 = np.count_nonzero(np.add(return_arr, 1024))
    nonzero2 = np.count_nonzero(np.add(return_arr2, 1024))
    nonzero3 = np.count_nonzero(np.add(return_arr3, 1024))
    # demo comparison of the 3 options
    # print(nonzero1, nonzero2, nonzero3)
    # plt.subplot(311)
    # plt.imshow(return_arr, cmap=plt.cm.gray)
    # plt.subplot(312)
    # plt.imshow(return_arr2, cmap=plt.cm.gray)
    # plt.subplot(313)
    # plt.imshow(return_arr3, cmap=plt.cm.gray)
    # plt.show()
    if nonzero1 >= nonzero2 and nonzero1 >= nonzero3:
        return return_arr
    elif nonzero2 > nonzero1 and nonzero2 >= nonzero3:
        return return_arr2
    else:
        return return_arr3


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


# def plot_3_by_3(raw, mask, masked, x=100, y=200, z=150):
#     """
#     makes a 3x3 subplot showing the raw, mask and masked image for each xy, xz and yz plane
#     :param raw: 3d array of raw data
#     :param mask: 3d array of mask
#     :param masked: 3d array of masked data
#     :param x: the x point you want to slice at (defaults to 100)
#     :param y: the y point you want to slice at (defaults to 200)
#     :param z: the z point you want to slice at (defaults to 150)
#     :return: no return
#     """
#     raw_xy, raw_xz, raw_yz = make_2d(raw, x, y, z)
#     mask_xy, mask_xz, mask_yz = make_2d(mask, x, y, z)
#     masked_xy, masked_xz, masked_yz = make_2d(masked, x, y, z)
#     ax = plt.subplot(3, 3, 1)
#     ax.set_title("xy raw")
#     plt.imshow(raw_xy, cmap=plt.cm.gray)
#     plt.plot(x, y, '+', color='red', ms=20)
#     ax = plt.subplot(3, 3, 2)
#     ax.set_title("xy mask")
#     plt.imshow(mask_xy, cmap=plt.cm.gray)
#     plt.plot(x, y, '+', color='red', ms=20)
#     ax = plt.subplot(3, 3, 3)
#     ax.set_title("xy masked")
#     plt.imshow(masked_xy, cmap=plt.cm.gray)
#     plt.plot(x, y, '+', color='red', ms=20)
#     ax = plt.subplot(3, 3, 4)
#     ax.set_title("xz raw")
#     plt.imshow(raw_xz, cmap=plt.cm.gray)
#     plt.plot(x, z, '+', color='red', ms=20)
#     ax = plt.subplot(3, 3, 5)
#     ax.set_title("xz mask")
#     plt.imshow(mask_xz, cmap=plt.cm.gray)
#     plt.plot(x, z, '+', color='red', ms=20)
#     ax = plt.subplot(3, 3, 6)
#     ax.set_title("xz masked")
#     plt.imshow(masked_xz, cmap=plt.cm.gray)
#     plt.plot(x, z, '+', color='red', ms=20)
#     ax = plt.subplot(3, 3, 7)
#     ax.set_title("yz raw")
#     plt.imshow(raw_yz, cmap=plt.cm.gray)
#     plt.plot(y, z, '+', color='red', ms=20)
#     ax = plt.subplot(3, 3, 8)
#     ax.set_title("yz mask")
#     plt.imshow(mask_yz, cmap=plt.cm.gray)
#     plt.plot(y, z, '+', color='red', ms=20)
#     ax = plt.subplot(3, 3, 9)
#     ax.set_title("yz masked")
#     plt.imshow(masked_yz, cmap=plt.cm.gray)
#     plt.plot(y, z, '+', color='red', ms=20)
#     plt.show()


def plot_3_by_3(img_tuple, points=[(100, 200, 150)]):
    """
    makes a 3x3 subplot showing the raw, mask and masked image for each xy, xz and yz plane
    :param raw: 3d array of raw data
    :param mask: 3d array of mask
    :param masked: 3d array of masked data
    :param x: the x point you want to slice at (defaults to 100)
    :param y: the y point you want to slice at (defaults to 200)
    :param z: the z point you want to slice at (defaults to 150)
    :return: no return
    """
    # raw, mask, masked = img_tuple
    for i, img in enumerate(img_tuple):
        if len(points) == 1:
            x = points[0][0]
            y = points[0][1]
            z = points[0][2]
        else:
            x = points[i][0]
            y = points[i][1]
            z = points[i][2]
        raw_xy, raw_xz, raw_yz = make_2d(img, x, y, z)
        ax = plt.subplot(3, len(img_tuple), i+1)
        ax.set_title("xy %d"%(i+1))
        plt.imshow(raw_xy, cmap=plt.cm.gray)
        plt.plot(x, y, '+', color='red', ms=20)

        ax = plt.subplot(3, len(img_tuple), len(img_tuple)+i+1)
        ax.set_title("xz %d"%(i+1))
        plt.imshow(raw_xz, cmap=plt.cm.gray)
        plt.plot(x, z, '+', color='red', ms=20)

        ax = plt.subplot(3, len(img_tuple), 2*len(img_tuple)+i+1)
        ax.set_title("yz %d"%(i+1))
        plt.imshow(raw_yz, cmap=plt.cm.gray)
        plt.plot(y, z, '+', color='red', ms=20)

    plt.show()


def mask_dilation(mask, iteration = 1):
    print("starting dilation")
    dilated = ndimage.binary_dilation(mask, iterations=iteration).astype(mask.dtype)
    erosion = ndimage.binary_erosion(dilated, iterations=iteration).astype(mask.dtype)
    del dilated
    return erosion


def mask_erosion(mask, iteration = 1):
    print("starting erosion")
    erosion = ndimage.binary_erosion(mask, iterations=iteration).astype(mask.dtype)
    dilated = ndimage.binary_dilation(erosion, iterations=iteration+1).astype(mask.dtype)
    del erosion
    return dilated


def density_mask(raw, value, threshold):
    limit = np.full_like(raw, threshold)
    adjustment = np.full_like(raw, value)
    # raw_adjusted = tf.Session().run(tf.subtract(raw, adjustment))
    # raw_adjusted_abs = tf.Session().run(tf.abs(raw_adjusted))
    mask = session.run(tf.less(tf.abs(tf.subtract(raw, adjustment)), limit))
    del limit
    del adjustment
    return mask


def find_cm(raw, iterations=1, window_size=100, x=0, y=0, z=0):
    for i in range(iterations):
        if x or y or z:
            # find x min and max
            xmin = x - int(window_size/2)
            if xmin < 0:
                xmin = 0
            xmax = xmin + window_size
            if xmax > len(raw[0][0])-1:
                xmax = len(raw[0][0])-1
                xmin = xmax - window_size
            # find y min and max
            ymin = y - int(window_size/2)
            if ymin < 0:
                ymin = 0
            ymax = ymin + window_size
            if ymax > len(raw[0])-1:
                ymax = len(raw[0])-1
                ymin = ymax - window_size
            # find z min and max
            zmin = z - int(window_size/2)
            if zmin < 0:
                zmin = 0
            zmax = zmin + window_size
            if zmax > len(raw)-1:
                zmax = len(raw)-1
                zmin = zmax - window_size
            windowed = raw[zmin:zmax,ymin:ymax,xmin:xmax]
            # plt.imshow(windowed[150], cmap=plt.cm.gray)
            # plt.show()
            value = ndimage.measurements.center_of_mass(windowed)
            if np.isnan(value).any():
                return x, y, z
            z, y, x = value
            print(x,y,z)
            x += xmin
            y += ymin
            z += zmin
            print(x,y,z)
        else:
            z, y, x = ndimage.measurements.center_of_mass(raw)
            print(x,y,z)
        # debug
        # plot on each iteration
        x = int(x)
        y = int(y)
        z = int(z)
        # plot on each iteration for debug
        # plot_3_by_3((raw,), [(x, y, z)])
    return x, y, z

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
    print("Winner chosen at random: ", end='')
    print(lucky_winner)
    lucky_winner_raw_data = dict_of_patients[lucky_winner]
    lucky_winner_mask = dict_of_masks[lucky_winner]
    # to save memory let's get rid of the rest of the data
    del dict_of_patients
    del dict_of_masks
    print("Apply Mask Dilation")
    lucky_winner_mask_dilation = mask_dilation(lucky_winner_mask, 6)

    # # apply mask and plot 3 by 3
    # lucky_winner_masked_data = mask_and_partition(dict_of_patients[lucky_winner], dict_of_masks[lucky_winner])
    # plot_3_by_3(lucky_winner_raw_data,lucky_winner_mask,lucky_winner_masked_data)
    # do it again with the modified mask
    print("done... getting masked data")
    lucky_winner_masked_data = mask_and_partition(lucky_winner_raw_data, lucky_winner_mask_dilation)
    print("done... getting density mask")
    lucky_winner_density_mask = density_mask(lucky_winner_masked_data, -50, 100)
    print("done... dilating density mask")
    lucky_winner_density_mask_dilation = mask_dilation(lucky_winner_density_mask, 3)
    print("done... erode density mask")
    #try to get rid of veins and keep only blobs
    lucky_winner_density_mask_erosion = mask_erosion(lucky_winner_density_mask, 1)
    print("done... plotting...")
    plot_3_by_3((lucky_winner_masked_data,lucky_winner_density_mask_dilation, lucky_winner_density_mask_erosion))
    #calc center of mass
    main_hotspot = find_cm(lucky_winner_density_mask_erosion, 5)
    zmax = len(lucky_winner_density_mask_erosion)-1
    ymax = len(lucky_winner_density_mask_erosion[0])-1
    xmax = len(lucky_winner_density_mask_erosion[0][0])-1
    quadrant_1_hotspot = find_cm(lucky_winner_density_mask_erosion, 3, int(zmax/2), x=1, y=1, z=1)
    quadrant_2_hotspot = find_cm(lucky_winner_density_mask_erosion, 3, int(zmax/2), x=xmax, y=1, z=1)
    quadrant_3_hotspot = find_cm(lucky_winner_density_mask_erosion, 3, int(zmax/2), x=xmax, y=ymax, z=1)
    quadrant_4_hotspot = find_cm(lucky_winner_density_mask_erosion, 3, int(zmax/2), x=xmax, y=ymax, z=zmax)
    quadrant_5_hotspot = find_cm(lucky_winner_density_mask_erosion, 3, int(zmax/2), x=1, y=ymax, z=1)
    quadrant_6_hotspot = find_cm(lucky_winner_density_mask_erosion, 3, int(zmax/2), x=1, y=ymax, z=zmax)
    quadrant_7_hotspot = find_cm(lucky_winner_density_mask_erosion, 3, int(zmax/2), x=1, y=1, z=zmax)
    quadrant_8_hotspot = find_cm(lucky_winner_density_mask_erosion, 3, int(zmax/2), x=xmax, y=1, z=zmax)
    raw_tuple = (lucky_winner_raw_data, lucky_winner_raw_data, lucky_winner_raw_data,
                 lucky_winner_raw_data, lucky_winner_raw_data, lucky_winner_raw_data,
                 lucky_winner_raw_data, lucky_winner_raw_data, lucky_winner_raw_data)
    hotspots = [main_hotspot,
                quadrant_1_hotspot,
                quadrant_2_hotspot,
                quadrant_3_hotspot,
                quadrant_4_hotspot,
                quadrant_5_hotspot,
                quadrant_6_hotspot,
                quadrant_7_hotspot,
                quadrant_8_hotspot]
    plot_3_by_3(raw_tuple, hotspots)

    # slice_2d = make_2d_funky(lucky_winner_raw_data, main_hotspot, quadrant_1_hotspot, quadrant_2_hotspot)
    # plt.imshow(slice_2d)
    # plt.show()


    # pick a random 3 points to make a plane and display that plane
    # p1 = (1, 1, 150)
    # p2 = (300, 100, 50)
    # p3 = (100, 300, 10)
    # slice_2d = make_2d_funky(lucky_winner_raw_data, p1, p2, p3)
    # plt.imshow(slice_2d)
    # plt.show()