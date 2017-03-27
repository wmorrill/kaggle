__author__ = 'wmorrill'

from preprocess import *
from feature_extraction import *
import os
import csv

INPUT_FOLDER = 'C:\\GIT\\kaggle_data_science_bowl_2017\\Data\\Sample_data\\'
save_path = '\\'.join(INPUT_FOLDER.split('\\')[:-2])+'\\output\\'
print(save_path)
patients = os.listdir(INPUT_FOLDER)
patients = ['0acbebb8d463b4b9ca88cf38431aac69']
for patient in patients:
    print(patient)
    patient_dicom = load_scan(INPUT_FOLDER + patient)
    patient_data = get_pixels_hu(patient_dicom)
    patient_resampled = resample_tf(patient_data, patient_dicom)
    patient_meta = get_meta_data(patient_dicom)
    del patient_dicom
    del patient_data
    patient_lungs_mask = segment_lung_mask(patient_resampled, True)
    scaling_factor = min(min(patient_meta['pixel_spacing']), patient_meta['slice_thickness'])
    masses = get_mass_details(patient_resampled,
                              patient_lungs_mask,
                              scale=scaling_factor,
                              save_name=save_path+patient)
    with open(save_path+patient) as csvfile:
        fieldnames = list(masses[0].keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(masses)
