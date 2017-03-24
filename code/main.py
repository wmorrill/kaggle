__author__ = 'wmorrill'

from preprocess import *
from feature_extraction import *
import os

INPUT_FOLDER = 'C:\\GIT\\kaggle_data_science_bowl_2017\\Data\\Sample_data\\'
patients = os.listdir(INPUT_FOLDER)
patients = ['0acbebb8d463b4b9ca88cf38431aac69']
for patient in patients:
    patient_dicom = load_scan(INPUT_FOLDER + patient)
    patient_data = get_pixels_hu(patient_dicom)
    patient_resampled = resample_tf(patient_data, patient_dicom)
    del patient_dicom
    del patient_data
    patient_lungs_mask = segment_lung_mask(patient_resampled, True)
    masses = get_mass_details(patient_resampled, patient_lungs_mask)
    # patient_meta = get_meta_data(patient_dicom)