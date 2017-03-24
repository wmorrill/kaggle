import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dicom
import os
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import datetime
from skimage import measure, morphology, segmentation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import tensorflow as tf
import multiprocessing
import time

# Some constants
INPUT_FOLDER = 'C:\\GIT\\kaggle_data_science_bowl_2017\\Data\\Sample_data\\'
patients = os.listdir(INPUT_FOLDER)
patients.sort()

# HU Table upper limit
AIR = -1000
LUNG = -500
FAT = -50  # to -50 to -100
WATER = 0
CSF = 15
KIDNEY = 30
BLOOD = 45  # 30 to 45
MUSCLE = 40  #10 to 40
GREY_MATTER = 45  # 37 to 45
WHITE_MATTER = 30  # 20 to 30
LIVER = 60  # 40 to 60
SOFT_TISSUE = 300  # 100 to 300
BONE = 700
BONE2 = 3000

# Load the scans in given folder path
def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices

def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)

    return np.array(image, dtype=np.int16)

def resample(image, scan, new_spacing=[1,1,1]):
    print("resampling image size, starting at {}".format(image.shape))
    # Determine current pixel spacing
    spacing = map(float, ([scan[0].SliceThickness] + scan[0].PixelSpacing))
    spacing = np.array(list(spacing))
    #don't want to loose data, so let's go with the smallest current spacing
    new_spacing = [min(spacing),min(spacing),min(spacing)]
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = ndimage.interpolation.zoom(image, real_resize_factor)
    print("Done resampling, new size {}".format(image.shape))
    return image, new_spacing

def resample_tf(image, scan):
    print("resampling image size, starting at {}".format(image.shape))
    # Determine current pixel spacing
    spacing = map(float, ([scan[0].SliceThickness] + scan[0].PixelSpacing))
    spacing = np.array(list(spacing))
    #don't want to loose data, so let's go with the smallest current spacing
    new_spacing = [min(spacing),min(spacing),min(spacing)]
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    print(new_shape)
    session = tf.Session()
    size = (tf.cast(new_shape[0], tf.int32), tf.cast(new_shape[1], tf.int32))
    # images = [image[:,:,i] for i in range(len(image[0][0]))]
    image_t = tf.transpose(image, perm=[1,0,2], name='First_Tranpose')
    image_t_4d = tf.expand_dims(image_t, 3, name='expand')
    # resized_image_t_4d = tf.image.resize_bicubic(image_t_4d, size, name='Resample')
    resized_image_t_4d = tf.image.resize_bilinear(image_t_4d, size, name='Resample')
    resized_image_t = tf.squeeze(resized_image_t_4d, name='Squeeze')
    resized_image = session.run(tf.transpose(resized_image_t, perm=[1,0,2], name='Second_Tranpose'))
    session.close()
    print("Done resampling, new size {}".format(resized_image.shape))
    return resized_image

def plot_3d(image, threshold=-300):

    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
    p = p[:,:,::-1]

    verts, faces = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()

def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        biggest = vals[np.argmax(counts)]
    else:
        biggest = None
    return biggest

def generate_markers(image):
    """
    Used by the separate_lungs() function below
    :param image:
    :return:
    """
    #Creation of the internal Marker
    marker_internal = image < -400
    marker_internal = segmentation.clear_border(marker_internal)
    marker_internal_labels = measure.label(marker_internal)
    areas = [r.area for r in measure.regionprops(marker_internal_labels)]
    areas.sort()
    if len(areas) > 2:
        for region in measure.regionprops(marker_internal_labels):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                       marker_internal_labels[coordinates[0], coordinates[1]] = 0
    marker_internal = marker_internal_labels > 0
    #Creation of the external Marker
    external_a = ndimage.binary_dilation(marker_internal, iterations=10)
    external_b = ndimage.binary_dilation(marker_internal, iterations=55)
    marker_external = external_b ^ external_a
    #Creation of the Watershed Marker matrix
    marker_watershed = np.zeros(marker_internal.shape, dtype=np.int)
    marker_watershed += marker_internal * 255
    marker_watershed += marker_external * 128

    return marker_internal, marker_external, marker_watershed

def separate_lungs(image, return_list=None, iteration=-1):
    """
    This only takes in a 2D slice to make he lung segmentation and takes really long to run.
    But supposedly will get all corner cases. Not sure if mask from this is very good.
    Looks like the mask might be too dilated.

    :param image:
    :param return_list:
    :param iteration:
    :return:
    """
    #Creation of the markers as shown above:
    marker_internal, marker_external, marker_watershed = generate_markers(image)

    #Creation of the Sobel-Gradient
    sobel_filtered_dx = ndimage.sobel(image, 1)
    sobel_filtered_dy = ndimage.sobel(image, 0)
    sobel_gradient = np.hypot(sobel_filtered_dx, sobel_filtered_dy)
    sobel_gradient *= 255.0 / np.max(sobel_gradient)

    #Watershed algorithm
    watershed = morphology.watershed(sobel_gradient, marker_watershed)

    #Reducing the image created by the Watershed algorithm to its outline
    outline = ndimage.morphological_gradient(watershed, size=(3,3))
    outline = outline.astype(bool)

    #Performing Black-Tophat Morphology for reinclusion
    #Creation of the disk-kernel and increasing its size a bit
    blackhat_struct = [[0, 0, 1, 1, 1, 0, 0],
                       [0, 1, 1, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [0, 1, 1, 1, 1, 1, 0],
                       [0, 0, 1, 1, 1, 0, 0]]
    blackhat_struct = ndimage.iterate_structure(blackhat_struct, 8)
    #Perform the Black-Hat
    outline += ndimage.black_tophat(outline, structure=blackhat_struct)

    #Use the internal marker and the Outline that was just created to generate the lungfilter
    lungfilter = np.bitwise_or(marker_internal, outline)
    #Close holes in the lungfilter
    #fill_holes is not used here, since in some slices the heart would be reincluded by accident
    lungfilter = ndimage.morphology.binary_closing(lungfilter, structure=np.ones((5,5)), iterations=3)

    # #Apply the lungfilter (note the filtered areas being assigned -2000 HU)
    # segmented = np.where(lungfilter == 1, image, -2000*np.ones((512, 512)))
    if iteration >=0 and return_list:
        return_list[iteration] = lungfilter
    else:
        return lungfilter

def segment_lung_mask(image, fill_lung_structures=True):
    """
    This function works on 90% of images but fails on some corner cases

    :param image:
    :param fill_lung_structures:
    :return:
    """
    print("Creating Lung Mask")
    # TODO: Make this work on corner case: '0acbebb8d463b4b9ca88cf38431aac69'
    # # shrink image to get rid of threacheotomy? doesnt work.
    # # z, y, x = image.shape
    # shrink = 20
    # image_shrink = image[shrink:-shrink,shrink:-shrink,shrink:-shrink]
    # plt.subplot(211)
    # plt.imshow(image[80])
    # plt.subplot(212)
    # plt.imshow(image_shrink[80-shrink])
    # plt.show()
    # not actually binary, but 1 and 2.
    # 0 is treated as background, which we do not want
    binary_image = np.array(image < -320, dtype=np.int8)
    binary_image2 = np.array(image < -320, dtype=np.int8)
    # dilated = scipy.ndimage.binary_dilation(binary_image, iterations=20).astype(binary_image.dtype)
    # binary_image = scipy.ndimage.binary_erosion(dilated, iterations=20).astype(binary_image.dtype)
    labels = measure.label(binary_image)

    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air
    #   around the person in half
    # background_label = []
    # background_label.append(labels[0,0,0])
    # # background_label.append(labels[-1,0,0])
    # # background_label.append(labels[-1,-1,0])
    # # background_label.append(labels[-1,-1,-1])
    # # background_label.append(labels[0,-1,0])
    # # background_label.append(labels[0,-1,-1])
    # # background_label.append(labels[0,0,-1])
    # # background_label.append(labels[-1,0,-1])
    # background_set = set(background_label)

    # #Fill the air around the person
    # for value in background_set:
    #     binary_image[value == labels] = 0
        # print(value)
        # plt.imshow(binary_image[80])
        # plt.show()

    # binary_image = 1-binary_image # Invert it, lungs are now 0
    labels = measure.label(binary_image)
    unique, counts = np.unique(labels, return_counts=True)
    label_dict = dict(zip(unique, counts))
    little_chunks = [label_value for label_value in label_dict.keys() if label_dict[label_value] < 90000]
    # binary_image = 1-binary_image # Invert it, lungs are now 1
    for each_value in little_chunks:
        binary_image[labels == each_value] = 1
    # plt.imshow(binary_image[264], cmap=plt.cm.gray)
    # plt.show()

    # slice into 3's vertically and check to see if any of the labels touch more than 1 edge
    width = len(labels[0][0])

    left = binary_image[:, :, :int(width/3)]
    center = binary_image[:, :, int(width/3):int(width*2/3)]
    right = binary_image[:, :, int(width*2/3):]
    # plt.imshow(left[264], cmap=plt.cm.gray)
    # plt.show()
    # plt.imshow(center[264], cmap=plt.cm.gray)
    # plt.show()
    # plt.imshow(right[264], cmap=plt.cm.gray)
    # plt.show()
    left_labels = measure.label(left)
    top_slice = left_labels[0]
    bottom_slice = left_labels[-1]
    top_values = np.unique(top_slice)
    bottom_values = np.unique(bottom_slice)
    for each_value in (set(top_values) & set(bottom_values)):
        left[left_labels == each_value] = 0
    # plt.imshow(left[264], cmap=plt.cm.gray)
    # plt.show()
    center_labels = measure.label(center)
    top_slice = center_labels[0]
    bottom_slice = center_labels[-1]
    top_values = np.unique(top_slice)
    bottom_values = np.unique(bottom_slice)
    for each_value in (set(top_values) & set(bottom_values)):
        center[center_labels == each_value] = 0
    # plt.imshow(center[264], cmap=plt.cm.gray)
    # plt.show()
    right_labels = measure.label(right)
    top_slice = right_labels[0]
    bottom_slice = right_labels[-1]
    top_values = np.unique(top_slice)
    bottom_values = np.unique(bottom_slice)
    for each_value in (set(top_values) & set(bottom_values)):
        right[right_labels == each_value] = 0
    # plt.imshow(right[264], cmap=plt.cm.gray)
    # plt.show()
    total = np.dstack([left,center,right])
    # plt.imshow(total[264], cmap=plt.cm.gray)
    # plt.show()

    # Method of filling the lung structures (that is superior to something like
    # morphological closing)
    # if fill_lung_structures:
    #     # For every slice we determine the largest solid structure
    #     for i, axial_slice in enumerate(binary_image):
    #         axial_slice = axial_slice - 1
    #         labeling = measure.label(axial_slice)
    #         l_max = largest_label_volume(labeling, bg=0)
    #
    #         if l_max is not None: #This slice contains some lung
    #             binary_image[i][labeling != l_max] = 1
    #
    #
    # binary_image -= 1 #Make the image actual binary
    # binary_image = 1-binary_image # Invert it, lungs are now 1
    #
    # # Remove other air pockets inside body
    # labels = measure.label(binary_image, background=0)
    # l_max = largest_label_volume(labels, bg=0)
    # if l_max is not None: # There are air pockets
    #     binary_image[labels != l_max] = 0
    print("Done generating mask")
    return total


def store_patients(file, img):
    try:
        np.savez_compressed(file, **img)
    except:
        np.savez_compressed(file, img)


def load_patients(file):
    print("Loading Patient {}".format(file))
    patients = np.load(file)
    print("Done Loading")
    return patients


def get_meta_data(scan):
    # TODO: get meta-data from Dicom that might be useful for feature extraction
    meta_dict = {}
    # get slice thickness
    meta_dict['slice_thickness'] = scan[0].SliceThickness
    # get pixel spacing in mm
    meta_dict['pixel_spacing'] = scan[0].PixelSpacing
    # anything else?

    return meta_dict


if __name__ == '__main__':
    save_all = False
    save_raw = False
    save_resampled = False
    save_segmented = False
    print("%s - Preprocess Start" % datetime.datetime.now())
    #
    # patients_data = load_patients("C:\\GIT\\kaggle_data_science_bowl_2017\\Data\\one_patient_segmented.npz")
    # for patient_id in patients_data.files:
    #     plot_3d(patients_data[patient_id], 0)
    # exit()
    patients_data = {}
    # patients = patients[0:1]  # load only one patient
    patients = ['0acbebb8d463b4b9ca88cf38431aac69'] # this one is hard for some reason
    # # Load all the data
    # for patient in patients:
    #     raw_data = load_scan(INPUT_FOLDER + patient)
    #     raw_pixels = get_pixels_hu(raw_data)
    #     patients_data[patient] = raw_pixels
    # print("%s - Patients loaded" % datetime.datetime.now())
    # # save it raw
    # if save_all or save_raw:
    #     store_patients("C:\\GIT\\kaggle_data_science_bowl_2017\\Data\\sample_patients_raw", patients_data)
    #     print("Raw Saved")
    # # resample all the data
    # patients_data_resampled = {}
    # for patient in patients:
    #     patients_data_resampled[patient], spacing = resample(patients_data[patient], load_scan(INPUT_FOLDER + patient))
    # print("%s - Patients Resampled" % datetime.datetime.now())
    # # save it resampled
    # if save_all or save_resampled:
    #     store_patients("C:\\GIT\\kaggle_data_science_bowl_2017\\Data\\sample_patients_resampled", patients_data_resampled)
    #     print("Resampled Saved")
    # # segment
    # patients_data_mask = {}
    # for patient in patients:
    #     patients_data_mask[patient] = segment_lung_mask(patients_data_resampled[patient], True)
    # print("%s - Patients Segmented" % datetime.datetime.now())
    # # save it segmented
    # if save_all or save_segmented:
    #     store_patients("C:\\GIT\\kaggle_data_science_bowl_2017\\Data\\sample_patients_segmented", patients_data_mask)
    #     print("Segmentation Saved")

    first_patient = load_scan(INPUT_FOLDER + patients[0])
    first_patient_pixels = get_pixels_hu(first_patient)
    # plt.hist(first_patient_pixels.flatten(), bins=80, color='c')
    # plt.xlabel("Hounsfield Units (HU)")
    # plt.ylabel("Frequency")
    # plt.show()

    # # Show some slice in the middle
    # plt.imshow(first_patient_pixels[80], cmap=plt.cm.gray)
    # plt.show()
    t0 = time.clock()
    pix_resampled = resample_tf(first_patient_pixels, first_patient)
    print("took {} seconds to resample using GPU".format(time.clock()-t0))
    t0 = time.clock()
    pix_resampled, spacing = resample(first_patient_pixels, first_patient)
    print("took {} seconds to resample using CPU".format(time.clock()-t0))

    #
    # # plot_3d(pix_resampled, 400)  # This takes forever
    #
    # segmented_lungs = segment_lung_mask(pix_resampled, False)
    segmented_lungs_fill = segment_lung_mask(pix_resampled, True)

    # # multiprocessing took 40 mins
    # manager = multiprocessing.Manager()
    # segmented_lungs_fill = manager.list([None] * len(pix_resampled))
    # for i in range(len(pix_resampled)):
    #     p = multiprocessing.Process(target=seperate_lungs, args=(pix_resampled[i], segmented_lungs_fill, i))
    #     p.start()
    # print("%s - Preprocess End" % datetime.datetime.now())
    # plt.imshow(segmented_lungs_fill[80], cmap=plt.cm.gray)
    # plt.show()
    # # series processing also took 40 mins...
    # segmented_lungs_fill = list([None] * len(pix_resampled))
    # for i in range(len(pix_resampled)):
    #     segmented_lungs_fill[i] = separate_lungs(pix_resampled[i])
    # print("%s - Preprocess End" % datetime.datetime.now())
    plt.imshow(segmented_lungs_fill[264], cmap=plt.cm.gray)
    plt.show()
    # TODO: figure out how to speed this up? Maybe re-write seperate_lungs() function using tf

    # store_patients("C:\\GIT\\kaggle_data_science_bowl_2017\\Data\\one_patient_segmented", segmented_lungs_fill)
    # # plot_3d(segmented_lungs, 0)
    # # TODO: figure out why this fails (TypeError?)
    # plot_3d(segmented_lungs_fill, 0)
    # plot_3d(segmented_lungs_fill - segmented_lungs, 0)

