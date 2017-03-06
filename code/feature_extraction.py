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
from scipy import spatial
from skimage.measure import label
from preprocess import plot_3d

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

# session = tf.Session()


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
    _source_of_truth = pd.read_csv(filename)
    return dict(zip(list(_source_of_truth.id), list(_source_of_truth.cancer)))


def random_patient_with_cancer(_source_of_truth):
    """
    surprise, surprise... this method returns a random patient who has cancer according to the source of truth
    
    :param _source_of_truth: dictionary with format {patient_id: has_cancer}
    :return: patient_id (from above)
    """
    cancer_list = [patient for patient in _source_of_truth if _source_of_truth[patient] == 1]
    return np.random.choice(cancer_list)


def apply_mask(image_array, mask_array):
    """
    takes an image and applies the mask to it

    :param image_array: (numpy 3D array) original image
    :param mask_array: (numpy 3D array) binary mask
    :return: (numpy 3D array) masked image
    """
    session = tf.Session()
    masked_image_array = session.run(tf.pow(image_array, mask_array))
    # print("masked")
    zero = tf.constant(0, tf.int32)
    air = tf.constant(-1000, tf.int32)
    # print("constants made")
    offset = session.run(tf.multiply(air, tf.cast(tf.equal(mask_array, zero), tf.int32)))
    # print("offset")
    sum_mask = session.run(tf.add(masked_image_array, offset))
    # print("summed")
    # do this to clear memory
    session.close()
    del masked_image_array
    del offset
    del image_array
    del mask_array
    return sum_mask


def make_2d_funky(cube_array, xyz1, xyz2, xyz3):
    """
    make a 2d array given a 3d array and 3 point
    
    :param cube_array: 3d array of data
    :param xyz1: first point tuple (x,y,z)
    :param xyz2: second point tuple (x,y,z)
    :param xyz3: third point tuple (x,y,z)
    :return: 2d matrix  with the most non-zero values
    """
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
    # a * x + b * y + c * z = d
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
    
    :param cube_array: (numpy 3D array) array to be sliced
    :param x: (int) center point in x axis
    :param y: (int) center point in y axis
    :param z: (int) center point in z axis
    :return: (tuple) returns a tuple of 2D arrays sliced on each major plane
    """
    # initialize some empty matrices
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


def plot_3_by_n(img_tuple, points=[(100, 200, 150)]):
    """
    makes a 3xn subplot showing the tuple of images for each xy, xz and yz plane around the point(s)
    
    :param img_tuple: (tuple) 2D images to be plotted
    :param points: (list) tuples of x,y,z points to plot around
    :return: nothing, shows the plot
    """
    # first we iterate the image tuple 
    for i, img in enumerate(img_tuple):
        # if there is only 1 point tuple, we use it for everything
        if len(points) == 1:
            x = points[0][0]
            y = points[0][1]
            z = points[0][2]
        # otherwise we use the corresponding point tuple
        else:
            # incase we are not passed equal length lists
            try:
                x = points[i][0]
                y = points[i][1]
                z = points[i][2]
            except IndexError:
                print("Warning: Inputs to Plot not equal length using previous x,y,z point")
        # get the 2D images around the x,y,z point
        raw_xy, raw_xz, raw_yz = make_2d(img, x, y, z)
        # make the first plot
        ax = plt.subplot(3, len(img_tuple), i + 1)
        ax.set_title("xy %d" % (i + 1))
        plt.imshow(raw_xy, cmap=plt.cm.gray)
        plt.plot(x, y, '+', color='red', ms=20)
        # make the second plot
        ax = plt.subplot(3, len(img_tuple), len(img_tuple) + i + 1)
        ax.set_title("xz %d" % (i + 1))
        plt.imshow(raw_xz, cmap=plt.cm.gray)
        plt.plot(x, z, '+', color='red', ms=20)
        # make the third plot
        ax = plt.subplot(3, len(img_tuple), 2 * len(img_tuple) + i + 1)
        ax.set_title("yz %d" % (i + 1))
        plt.imshow(raw_yz, cmap=plt.cm.gray)
        plt.plot(y, z, '+', color='red', ms=20)

    plt.show()


def mask_dilation(mask, iteration=1):
    """
    the operation first dilates the mask and then erodes it back. This should smooth edges
    operation runs both erosion and dilation equal amount of iterations
    
    :param mask: (numpy 3D array) original binary mask 
    :param iteration: (int) number of times to run the operations
    :return: (numpy 3D array) new binary mask
    """
    # print("starting dilation")
    dilated = ndimage.binary_dilation(mask, iterations=iteration).astype(mask.dtype)
    erosion = ndimage.binary_erosion(dilated, iterations=iteration).astype(mask.dtype)
    del dilated
    return erosion


def mask_erosion(mask, iteration=1):
    """
    the operation first erodes the mask and then dilates it back. This should get rid of small objects
    operation runs both erosion and dilation equal amount of iterations
    
    :param mask: (numpy 3D array) original binary mask 
    :param iteration: (int) number of times to run the operations
    :return: (numpy 3D array) new binary mask
    """
    # print("starting erosion")
    erosion = ndimage.binary_erosion(mask, iterations=iteration).astype(mask.dtype)
    dilated = ndimage.binary_dilation(erosion, iterations=iteration+1).astype(mask.dtype)
    del erosion
    return dilated


def density_mask(raw, value, threshold):
    """
    makes a mask from the raw image centered around a value within a threshold
    
    :param raw: (numpy 3D array) original matrix to look at (needs to be numbers)
    :param value: (int) the density to center the mask on
    :param threshold: (int) the bounds around the center value
    :return: (numpy 3D array) binary array, true where criteria are met
    """
    limit = np.full_like(raw, threshold)
    adjustment = np.full_like(raw, value)
    # raw_adjusted = tf.Session().run(tf.subtract(raw, adjustment))
    # raw_adjusted_abs = tf.Session().run(tf.abs(raw_adjusted))
    session = tf.Session()
    mask = session.run(tf.less(tf.abs(tf.subtract(raw, adjustment)), limit))
    session.close()
    del limit
    del adjustment
    return mask


def find_cm(raw, iterations=1, window_size=100, x=0, y=0, z=0):
    """
    returns the x, y, z position of the center of mass found after interations through the process. 
    Each iteration is windowed around the previous x,y,z center of mass
    Can short circuit to the window loop by passing an x,y,z value
    
    :param raw: (numpy 3D array) original matrix to look at (needs to be numbers)
    :param iterations: (int) number of times to iterate 
    :param window_size: (int) size of the window for introspection
    :param x: (int) coordinate
    :param y: (int) coordinate
    :param z: (int) coordinate
    :return: (tuple) x, y, z coordinate for center of mass
    """
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
            # make a windowed version of the original image
            windowed = raw[zmin:zmax, ymin:ymax, xmin:xmax]
            # find the CM of window
            value = ndimage.measurements.center_of_mass(windowed)
            # check to see if we got a NAN value (if there was no '1's in the window
            if np.isnan(value).any():
                return x, y, z
            z, y, x = value
            # print(x,y,z)
            # recalibrate x, y, z values back to the full image
            x += xmin
            y += ymin
            z += zmin
            # print(x,y,z)
        else:
            z, y, x = ndimage.measurements.center_of_mass(raw)
            # print(x,y,z)
        # cast as int because it returns a float
        x = int(x)
        y = int(y)
        z = int(z)
        # plot on each iteration for debug
        # plot_3_by_n((raw,), [(x, y, z)])
    return x, y, z


def find_unique_objects(mask, volume_min=800, volume_max=15000, ratio_min=1.5, ratio_max=4):
    """
    takes a mask and finds all the unique objects floating in that area

    :param mask: (np array) 3d binary image
    :param volume_min: (int) smallest acceptable volume
    :param volume_max: (int) largest acceptable volume
    :param ratio_min: (int) smallest acceptable volume/area
    :param ratio_max: (int) largest acceptable volume/area
    :return: (dict) unique_blob_dict, is a dictionary of arrays holding the coordinates,
    volume, area and original mask data for the given area
    """
    # get all the individual objects
    labeled_data, num_of_unique_blobs = label(mask, background=0, return_num=True)
    print("found %d unique objects" % num_of_unique_blobs)
    unique_blob_dict = {'coordinates':[], 'volume':[], 'area':[], 'raw':[]}
    # go through the objects and check if they are interesting
    for i in range(1,num_of_unique_blobs):
        # clears memory
        tf.reset_default_graph()
        # label gives each unique object a numeric value, so we sort based on that
        search_value = tf.constant(i, tf.int64)
        session = tf.Session()
        # generate a mask that only has the one object in it
        sub_mask = session.run(tf.equal(labeled_data, search_value))
        session.close()
        # get the bounding coordinates for this object
        coords = np.argwhere(sub_mask)
        # expand the coords to some standard size (30, 30, 30) or (50, 50, 50)
        # TODO: make coordinates and raw return a consistent size (easier for cnn later)
        top_left = coords.min(axis=0)
        bottom_right = coords.max(axis=0)
        # calculate the convex hull for this 3d object
        hull = spatial.ConvexHull(coords,3)
        # get the surface area and volume of the object
        area = float(hull.area)
        volume = float(hull.volume)
        # TODO: equate these arbitrary numbers to actual mm
        # if this object is within our interests, let's add it to the list
        if volume_min < volume < volume_max and ratio_min < (volume/area) < ratio_max:
            unique_blob_dict['coordinates'].append(find_cm(sub_mask, 1))
            unique_blob_dict['raw'].append(sub_mask[top_left[0]:bottom_right[0]+1,
                                                     top_left[1]:bottom_right[1]+1,
                                                     top_left[2]:bottom_right[2]+1])
            unique_blob_dict['volume'].append(volume)
            unique_blob_dict['area'].append(area)


            # for testing purposes
            print('Volume = %f' % unique_blob_dict['volume'][-1])
            print('Area = %f' % unique_blob_dict['area'][-1])
            print('Volume/Area = %f' % (float(unique_blob_dict['volume'][-1])/float(unique_blob_dict['area'][-1])))
            #plot_3_by_n((mask, sub_mask),[unique_blob_dict['coordinates'][-1]])

            # from preprocess import plot_3d
            # plot_3d(lucky_winner_density_mask_erosion, 0)
        del sub_mask
        del hull
        del coords

        # probably tumor
        # Volume = 1262.166667
        # Area = 618.409806
        # Volume/Area = 2.040987

        # probably tumor 0c0de3749d4fe175b7a5098b060982a1
        # Volume = 8161.666667
        # Area = 2178.180552
        # Volume/Area = 3.747011


    return unique_blob_dict


def get_mass_details(patient_raw_data, patient_mask_data):
    debug = True
    # dilate (expand) the mask to get all the little nodules and whatnot
    patient_mask_dilation = mask_dilation(patient_mask_data, 6)
    print("done... getting masked data")
    # apply mask to raw data
    patient_masked_data = apply_mask(patient_raw_data, patient_mask_dilation)
    print("done... getting density mask")
    # get new mask from density details in raw data
    patient_density_mask = density_mask(patient_masked_data, -50, 100)
    print("done... dilating density mask")
    # dilate (expand) the mask to smooth
    patient_density_mask_dilation = mask_dilation(patient_density_mask, 3)
    print("done... erode density mask")
    # # try to get rid of veins and keep only blobs
    patient_density_mask_erosion = mask_erosion(patient_density_mask_dilation, 1)
    print("done... plotting...")
    if debug:
        plot_3_by_n((patient_masked_data, patient_density_mask_dilation, patient_density_mask_erosion))

    del mask_dilation
    del patient_masked_data
    del patient_density_mask
    del patient_density_mask_dilation

    # find all the objects and get their details
    points_of_interest = find_unique_objects(patient_density_mask_erosion)
    if debug:
        for i in range(len(points_of_interest['area'])):
            print(points_of_interest['coordinates'][i])
            plot_3d(points_of_interest['raw'][i], 0)

    return points_of_interest


def get_lymph_node_details(patient_raw_data):
    # TODO: find ou if the patient lymph nodes are swollen. This is apparently indicative of a problem
    lymph_details = 1
    return lymph_details


if __name__ == "__main__":
    raw_patients_file = 'C:\\GIT\\kaggle_data_science_bowl_2017\\Data\\sample_patients_resampled.npz'
    patient_masks_file = 'C:\\GIT\\kaggle_data_science_bowl_2017\\Data\\sample_patients_segmented.npz'
    truth_file = 'C:\\GIT\\kaggle_data_science_bowl_2017\\stage1_labels.csv'
    source_of_truth = load_source_of_truth(truth_file)
    dict_of_patients = load_data(raw_patients_file)
    dict_of_masks = load_data(patient_masks_file)
    print("all patients loaded")
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
    print("done... Apply Mask Dilation")
    lucky_winner_masses = get_mass_details(lucky_winner_raw_data, lucky_winner_mask)
    lucky_winner_lymph_nodes = get_lymph_node_details(lucky_winner_raw_data)


