__author__ = 'wmorrill'

import os
import csv
import time
from preprocess import segment_lung_mask, resample_tf, get_pixels_hu, load_scan, get_meta_data
from feature_extraction import get_mass_details
from datetime import datetime

# INPUT_FOLDER = 'C:\\GIT\\kaggle_data_science_bowl_2017\\Data\\Sample_data\\'
INPUT_FOLDER = 'D:\\kaggle_data\\stage1\\'
save_path = '\\'.join(INPUT_FOLDER.split('\\')[:-2])+'\\output\\'
print(save_path)
patients = os.listdir(INPUT_FOLDER)
patients = patients[2:]
# patients = ['0acbebb8d463b4b9ca88cf38431aac69']
# patients = ['003f41c78e6acfa92430a057ac0b306e']  # this one takes forever to segment
for patient in patients:
    # print(datetime.now())
    t0 = time.clock()
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
    dialect = csv.excel()
    dialect.lineterminator = '\n'
    f = open(save_path+patient+".csv", 'a')
    writer = csv.DictWriter(f, list(masses[0].keys()), 'NaN', extrasaction='ignore', dialect=dialect)
    # writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(masses)
    f.close()
    print("time to process: {}".format(time.clock()-t0))
