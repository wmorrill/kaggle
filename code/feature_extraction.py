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
from operator import itemgetter

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
    masked_image_array = tf.pow(tf.cast(image_array, tf.int32), tf.cast(mask_array, tf.int32))
    # print("masked")
    zero = tf.constant(0, tf.int32)
    air = tf.constant(-1000, tf.int32)
    # print("constants made")
    offset = tf.multiply(air, tf.cast(tf.equal(tf.cast(mask_array, tf.int32), zero), tf.int32))
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


def plot_3_by_n(img_tuple, points=[(100, 200, 150)], save=False, save_name=''):
    """
    makes a 3xn subplot showing the tuple of images for each xy, xz and yz plane around the point(s)
    
    :param img_tuple: (tuple) 2D images to be plotted
    :param points: (list) tuples of x,y,z points to plot around
    :param save: (boolean) to save fig locally or not
    :param save_name: (string) name for save file
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
    if save:
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
    else:
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


def find_unique_objects(raw_data, mask, volume_min=300, volume_max=15000, scale=1, save_name=''):
    """
    takes a mask and finds all the unique objects floating in that area

    :param raw_data: (np array) 3d binary image
    :param mask: (np array) 3d binary image
    :param volume_min: (int) smallest acceptable volume
    :param volume_max: (int) largest acceptable volume
    :param scale: (float) mm per index
    :param save_name: (string) name of file to save for plot
    :return: (dict) unique_blob_dict, is a dictionary of arrays holding the coordinates,
    volume, area and original mask data for the given area
    """
    # get all the individual objects
    labeled_data, num_of_unique_blobs = label(mask, background=0, return_num=True)
    print("found %d unique objects" % num_of_unique_blobs)
    # how many of each value exist
    bin_count = np.bincount(labeled_data.flat)
    # list of blobs bigger than some size
    big_blobs = [x for x in range(1,num_of_unique_blobs) if volume_max > bin_count[x] > volume_min]
    print("found %d unique BIG objects" % len(big_blobs))
    unique_blobs = []
    #{'coordinates':[], 'volume':[], 'area':[], 'raw':[], 'mask':[], 'spiculated':[]}
    # go through the objects and check if they are interesting
    for i in big_blobs:
        # generate a mask that only has the one object in it
        # session = tf.Session()
        # # # clears memory
        # # tf.reset_default_graph()
        # sub_mask = session.run(tf.equal(labeled_data, tf.constant(i, tf.int64)))
        # session.close()
        sub_mask = np.array(labeled_data == i)
        # get the bounding coordinates for this object
        coords = np.argwhere(sub_mask)
        # TODO: make coordinates and raw return a consistent size (easier for cnn later)
        # scale = mm per index
        original_size = raw_data.shape
        top_left = coords.min(axis=0)
        bottom_right = coords.max(axis=0)
        # expand the coords to some standard size (50, 50, 50)
        desired_box = (50, 50, 50)  # 2.5cm cube
        bounding_box = (bottom_right[0]-top_left[0], bottom_right[1]-top_left[1], bottom_right[2]-top_left[2])
        delta_box = np.subtract(desired_box, bounding_box)
        half_delta = [int(x/2) for x in delta_box]
        top_left = np.subtract(top_left, half_delta)
        bottom_right = np.add(top_left, desired_box)
        if top_left[0] < 0:
            bottom_right[0] -= top_left[0]
            top_left[0] = 0
        if top_left[1] < 0:
            bottom_right[1] -= top_left[1]
            top_left[1] = 0
        if top_left[2] < 0:
            bottom_right[2] -= top_left[2]
            top_left[2] = 0
        if bottom_right[0] > original_size[0]:
            top_left[0] -= bottom_right[0] - original_size[0]
            bottom_right[0] = original_size[0]
        if bottom_right[1] > original_size[1]:
            top_left[1] -= bottom_right[1] - original_size[1]
            bottom_right[1] = original_size[0]
        if bottom_right[2] > original_size[2]:
            top_left[2] -= bottom_right[2] - original_size[2]
            bottom_right[2] = original_size[2]
        # calculate the convex hull for this 3d object
        hull = spatial.ConvexHull(coords, 3)
        # get the surface area and volume of the object
        area = float(hull.area)*scale**3
        volume = float(hull.volume)*scale
        effective_r = np.sqrt(area/(4*np.pi))
        expected_volume = effective_r * area / 3
        thickness = find_hull_max(hull)*scale
        #volume/expected volume > 0.72 is a pretty good threshold for sphere-like things
        temp_dict = {}
        temp_dict['coordinates'].append(find_cm(sub_mask, 1))
        temp_dict['mask'].append(sub_mask[top_left[0]:bottom_right[0],
                                                 top_left[1]:bottom_right[1],
                                                 top_left[2]:bottom_right[2]])
        temp_dict['raw'].append(raw_data[top_left[0]:bottom_right[0],
                                                 top_left[1]:bottom_right[1],
                                                 top_left[2]:bottom_right[2]])
        temp_dict['volume'].append(volume)
        temp_dict['area'].append(area)
        temp_dict['spiculated'].append(is_mass_spiculated(unique_blob_dict['raw'][-1], unique_blob_dict['mask'][-1]))
        temp_dict['thickness'].append(thickness)
        unique_blobs.append(temp_dict)

        # for testing purposes
        print('Volume = %f' % temp_dict['volume'])
        print('Area = %f' % temp_dict['area'])
        print('Expected Volume= %f' % expected_volume)
        print('Volume Ratio (real/exp)= %f' % (temp_dict['volume']/expected_volume))
        print('Spiculated: {}'.format(temp_dict['spiculated']))
        print('Max Thickness: {}'.format(temp_dict['thickness']))
        if save_name:
            plot_3_by_n((raw_data, mask, sub_mask),
                        [temp_dict['coordinates']],
                        save=True,
                        save_name=save_name+"_"+ str(i))

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

    return unique_blobs


def find_hull_max(shape):
    vertex_coordinates = []
    for v in shape.vertices:
        vertex_coordinates.append(shape.points[v])
    for coordinates in vertex_coordinates:
        x_plane = []
        y_plane = []
        z_plane = []
        # check distances for items in the same plane
        for compare_coordinate in [x for x in vertex_coordinates if x[0]==coordinates[0]]:
            x_plane.append((abs(coordinates[1]-compare_coordinate[1]),
                            abs(coordinates[2]-compare_coordinate[2]),
                            np.sqrt((coordinates[1]-compare_coordinate[1])**2+(coordinates[2]-compare_coordinate[2])**2)))
        for compare_coordinate in [x for x in vertex_coordinates if x[1]==coordinates[1]]:
            y_plane.append((abs(coordinates[0]-compare_coordinate[0]),
                            abs(coordinates[2]-compare_coordinate[2]),
                            np.sqrt((coordinates[0]-compare_coordinate[0])**2+(coordinates[2]-compare_coordinate[2])**2)))
        for compare_coordinate in [x for x in vertex_coordinates if x[2]==coordinates[2]]:
            z_plane.append((abs(coordinates[1]-compare_coordinate[1]),
                            abs(coordinates[0]-compare_coordinate[0]),
                            np.sqrt((coordinates[1]-compare_coordinate[1])**2+(coordinates[0]-compare_coordinate[0])**2)))
    # find the max thickness
    # start by ordering largest to smallest on distance
    x_plane.sort(key=itemgetter(2), reverse=True)
    y_plane.sort(key=itemgetter(2), reverse=True)
    z_plane.sort(key=itemgetter(2), reverse=True)
    max_thickness = max(x_plane[0][2], y_plane[0][2], z_plane[0][2])
    # # find the min thickness
    # # sort biggest to smallest on plane[0]. opposite on plane[1]
    # x_plane.sort(key=itemgetter(1))
    # x_plane.sort(key=itemgetter(0), reverse=True)
    # # sort biggest to smallest on plane[0]. opposite on plane[1]
    # y_plane.sort(key=itemgetter(1))
    # y_plane.sort(key=itemgetter(0), reverse=True)
    # # sort biggest to smallest on plane[0]. opposite on plane[1]
    # z_plane.sort(key=itemgetter(1))
    # z_plane.sort(key=itemgetter(0), reverse=True)
    return max_thickness




def get_mass_details(patient_raw_data, patient_mask_data, scale=1, save_name=''):
    debug = False
    # # dilate (expand) the mask to get all the little nodules and whatnot
    # patient_mask_dilation = mask_dilation(patient_mask_data, 6)
    # print("done... getting masked data")
    # # apply mask to raw data
    # patient_masked_data = apply_mask(patient_raw_data, patient_mask_dilation)
    # print("done... getting density mask")
    # # get new mask from density details in raw data
    # patient_density_mask = density_mask(patient_masked_data, -50, 100)
    # print("done... dilating density mask")
    # # dilate (expand) the mask to smooth
    # patient_density_mask_dilation = mask_dilation(patient_density_mask, 3)
    # print("done... erode density mask")
    # # # try to get rid of veins and keep only blobs
    # patient_density_mask_erosion = mask_erosion(patient_density_mask_dilation, 1)
    # print("done... plotting...")

    # mask dilation
    dilated = ndimage.binary_dilation(patient_mask_data, iterations=6).astype(patient_mask_data.dtype)
    erosion = ndimage.binary_erosion(dilated, iterations=6).astype(patient_mask_data.dtype)
    # apply mask
    session = tf.Session()
    masked_image_array = tf.pow(tf.cast(patient_raw_data, tf.int32), tf.cast(erosion, tf.int32))
    zero = tf.constant(0, tf.int32)
    air = tf.constant(-1000, tf.int32)
    offset = tf.multiply(air, tf.cast(tf.equal(tf.cast(erosion, tf.int32), zero), tf.int32))
    sum_mask = tf.add(masked_image_array, offset)
    # get new mask from density
    limit = np.full_like(sum_mask, 100)
    adjustment = np.full_like(sum_mask, -50)
    mask = session.run(tf.less(tf.abs(tf.subtract(sum_mask, adjustment)), limit))
    session.close()
    # dilate density mask
    patient_density_mask_dilation = ndimage.binary_dilation(mask, iterations=3).astype(mask.dtype)
    patient_density_mask_erosion = ndimage.binary_erosion(patient_density_mask_dilation, iterations=3).astype(mask.dtype)
    # plot_3_by_n((patient_raw_data, patient_density_mask_erosion))

    if debug:
        plot_3_by_n((patient_raw_data, patient_density_mask_erosion))

    # del patient_mask_dilation
    # del patient_masked_data
    # del patient_density_mask
    # del patient_density_mask_dilation

    # find all the objects and get their details
    points_of_interest = find_unique_objects(patient_raw_data,
                                             patient_density_mask_erosion,
                                             scale=scale,
                                             save_name=save_name)
    print("Found {} masses".format(len(points_of_interest['area'])))
    if debug:
        for i in range(len(points_of_interest['area'])):
            print(points_of_interest['coordinates'][i])
            plot_3_by_n((points_of_interest['raw'][i], points_of_interest['mask'][i]), [(25,25,25)])
            plot_3d(points_of_interest['mask'][i], 0)

    return points_of_interest


def get_lymph_node_details(patient_raw_data):
    """
    decide if the patient's lymph nodes are swollen

    :param patient_raw_data: (np array) 3D array of density data
    :return: (float) probability that they are swollen, 0 (def not), 1 (def are)
    """
    # TODO: find out if the patient lymph nodes are swollen. This is apparently indicative of a problem
    lymph_details = 1
    return lymph_details


def is_mass_spiculated(raw, mask):
    """
    determine if the mass we found is spiculated, this is indicative of malignant tumor
    from wiki:
    In oncology, a spiculated mass is a lump of tissue with spikes or points on the surface.
    It is suggestive but not diagnostic of malignancy, i.e. cancer.
    It's a common mammography finding in carcinoma breast.

    :param raw: (np array) 3d array of just the area around the mass
    :return: (float) percentage chance the mass is spiculated 0 (def not), 1 (def yes)
    """
    new_threshold = np.array(raw > -402, dtype=np.int32)
    session = tf.Session()
    # eroded_mask = ndimage.binary_erosion(mask, iterations=1).astype(mask.dtype)
    eroded_mask = mask_erosion(mask, 3)
    # subtracted = session.run(tf.subtract(mask, eroded_mask))
    # session.close()
    subtracted = np.subtract(mask, eroded_mask)
    new_array, num_objects = label(subtracted, background=0, return_num=True)
    print(num_objects)
    if 1 or num_objects > 10:
        plot_3d(raw, -402)
        plot_3d(mask, 0)
        plot_3d(subtracted, 0)
        return 1
    return -1


if __name__ == "__main__":
    raw_patients_file = 'C:\\GIT\\kaggle_data_science_bowl_2017\\Data\\sample_patients_resampled.npz'
    patient_masks_file = 'C:\\GIT\\kaggle_data_science_bowl_2017\\Data\\sample_patients_segmented.npz'
    truth_file = 'C:\\GIT\\kaggle_data_science_bowl_2017\\stage1_labels.csv'
    source_of_truth = load_source_of_truth(truth_file)
    dict_of_patients = load_data(raw_patients_file)
    dict_of_masks = load_data(patient_masks_file)
    print("all patients loaded")
    lucky_winner = None
    # lucky_winner = '0c37613214faddf8701ca41e6d43f56e'
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
    lucky_winner_masses['spiculated'] = []
    for i, mass in enumerate(lucky_winner_masses['raw']):
        lucky_winner_masses['spiculated'].append(is_mass_spiculated(mass, lucky_winner_masses['mask'][i]))


