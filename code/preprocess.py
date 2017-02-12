import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt
import datetime
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import tensorflow as tf

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
    # Determine current pixel spacing
    spacing = map(float, ([scan[0].SliceThickness] + scan[0].PixelSpacing))
    spacing = np.array(list(spacing))

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)

    return image, new_spacing

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

def segment_lung_mask(image, fill_lung_structures=True):

    # not actually binary, but 1 and 2.
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8)+1
    labels = measure.label(binary_image)

    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air
    #   around the person in half
    background_label = labels[0,0,0]

    #Fill the air around the person
    binary_image[background_label == labels] = 2


    # Method of filling the lung structures (that is superior to something like
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)

            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1


    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1

    # Remove other air pockets inside body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0

    return binary_image

# MIN_BOUND = -1000.0
# MAX_BOUND = 400.0
#
# def normalize(image):
#     image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
#     image[image>1] = 1.
#     image[image<0] = 0.
#     return image
#
# PIXEL_MEAN = 0.25
#
# def zero_center(image):
#     image = image - PIXEL_MEAN
#     return image

MIN_BOUND = -1000.0
MAX_BOUND = 400.0
PIXEL_MEAN = 0.25
PIXEL_CORR = int((MAX_BOUND - MIN_BOUND) * PIXEL_MEAN) # in this case, 350

def zero_center(image):
    image = image - PIXEL_CORR
    return image

def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>(1-PIXEL_MEAN)] = 1.
    image[image<(0-PIXEL_MEAN)] = 0.
    return image

def store_patients(file, img):
    try:
        np.savez_compressed(file, **img)
    except:
        np.savez_compressed(file, img)

def load_patients(file):
    patients = np.load(file)
    return patients

def find_poi(array_3d, minimum=0, maximum=3000):
    foo = 1

def xy_slices(array_3d):
    # create a single large 2d array from the 3d array in xy plane
    pass

def yz_slices(array_3d):
    # create a single large 2d array from the 3d array in yz plane
    pass

def xz_slices(array_3d):
    # create a single large 2d array from the 3d array in xz plane
    pass

def tf_poi(matrix, mask):
    tf_mask = tf.constant(mask)
    tf_matrix = tf.constant(matrix)
    something = tf.mul(tf_mask, tf_matrix)
    with tf.Session() as sess:
        result = sess.run([something])
        return result


if __name__ == '__main__':
    save_all = False
    save_raw = False
    save_resampled = False
    save_segmented = True
    print("%s - Preprocess Start" % datetime.datetime.now())
    #
    # patients_data = load_patients("C:\\GIT\\kaggle_data_science_bowl_2017\\Data\\one_patient_segmented.npz")
    # for patient_id in patients_data.files:
    #     plot_3d(patients_data[patient_id], 0)
    # exit()
    patients_data = {}
    # patients = patients[0:1]  # load only one patient
    # Load all the data
    for patient in patients:
        raw_data = load_scan(INPUT_FOLDER + patient)
        raw_pixels = get_pixels_hu(raw_data)
        patients_data[patient] = raw_pixels
    print("%s - Patients loaded" % datetime.datetime.now())
    # save it raw
    if save_all or save_raw:
        store_patients("C:\\GIT\\kaggle_data_science_bowl_2017\\Data\\sample_patients_raw", patients_data)
        print("Raw Saved")
    # resample all the data
    patients_data_resampled = {}
    for patient in patients:
        patients_data_resampled[patient], spacing = resample(patients_data[patient], load_scan(INPUT_FOLDER + patient))
    print("%s - Patients Resampled" % datetime.datetime.now())
    # save it resampled
    if save_all or save_resampled:
        store_patients("C:\\GIT\\kaggle_data_science_bowl_2017\\Data\\sample_patients_resampled", patients_data_resampled)
        print("Resampled Saved")
    # segment
    patients_data_mask = {}
    for patient in patients:
        patients_data_mask[patient] = segment_lung_mask(patients_data_resampled[patient], True)
    print("%s - Patients Segmented" % datetime.datetime.now())
    # save it segmented
    if save_all or save_segmented:
        store_patients("C:\\GIT\\kaggle_data_science_bowl_2017\\Data\\sample_patients_segmented", patients_data_mask)
        print("Segmentation Saved")

    # first_patient = load_scan(INPUT_FOLDER + patients[0])
    # first_patient_pixels = get_pixels_hu(first_patient)
    # # plt.hist(first_patient_pixels.flatten(), bins=80, color='c')
    # # plt.xlabel("Hounsfield Units (HU)")
    # # plt.ylabel("Frequency")
    # # plt.show()
    #
    # # Show some slice in the middle
    # # plt.imshow(first_patient_pixels[80], cmap=plt.cm.gray)
    # # plt.show()
    #
    # pix_resampled, spacing = resample(first_patient_pixels, first_patient, [1,1,1])
    # print("Shape before resampling\t", first_patient_pixels.shape)
    # print("Shape after resampling\t", pix_resampled.shape)
    #
    # # plot_3d(pix_resampled, 400)  # This takes forever
    #
    # segmented_lungs = segment_lung_mask(pix_resampled, False)
    # segmented_lungs_fill = segment_lung_mask(pix_resampled, True)
    #
    #
    # store_patients("C:\\GIT\\kaggle_data_science_bowl_2017\\Data\\one_patient_segmented", segmented_lungs_fill)
    # # plot_3d(segmented_lungs, 0)
    # # plot_3d(segmented_lungs_fill, 0)
    # # plot_3d(segmented_lungs_fill - segmented_lungs, 0)

