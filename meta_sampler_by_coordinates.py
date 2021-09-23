#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import os, sys
import copy
from nibabel.nifti1 import Nifti1Image
from nilearn.image import resample_img
import numpy as np
import pandas as pd
import re
import six

from collections import OrderedDict
import time

import nibabel as nib
import nilearn as nil
from typing import Dict

import textdistance

class Timer(object):

    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name,)
        print('Elapsed: %s' % (time.time() - self.tstart))


def simple_autocorrect(input_word, vocabulary):
    _input_word = input_word.lower()

    if _input_word in vocabulary:
        return input_word
    else:
        similarities = [1 - (textdistance.Jaccard(qval=2).distance(v, input_word)) for v in vocabulary]
        ordered = sorted(zip(vocabulary,similarities),key= lambda x: x[1], reverse=True)
        return ordered[0][0]


def voxel_to_world(affine: np.ndarray, voxel_index: (tuple,np.ndarray,list)) -> np.ndarray:
    """This function transforms the voxel index to world coordinate\n"""
    return nib.affines.apply_affine(affine,voxel_index)


def world_to_voxel(affine: np.ndarray, world_coord: (tuple,np.ndarray,list)) -> np.ndarray:
    """This function transforms the world coordinate to voxel index\n"""
    inv_affine = np.linalg.inv(affine)
    _voxel_index = np.array(nib.affines.apply_affine(inv_affine, world_coord)).astype(int)
    return _voxel_index


def get_patient_dict(data_dir):
    """
    :param data_dir: root directory for patient images/segmentations. Patient directories should be start with patient number.\n
    :return: dictionary of patients (patient_id : path)
    """
    _dirs = [f for f in os.listdir(data_dir) if re.match('[0-9]+',f)]
    dirs = [os.path.join(data_dir,f) for f in _dirs]
    dirs = list(filter(lambda x: os.path.isdir(x),dirs))
    patient_ids = [f"patient{re.match('[0-9]+', os.path.basename(d))[0].zfill(2)}" for d in dirs]
    res = dict(zip(patient_ids,dirs))

    return res


def get_image_dict(image_dir, vocabulary = None):
    """Initializes images for faster processing.\n
    Warning! This function reads EVERY nii, or nii.gz images in the given directory.\n"""
    img_names = os.listdir(image_dir)
    img_dict = {}
    for img_name in img_names:
        if not (str(img_name).endswith(".nii") or str(img_name).endswith(".nii.gz")):
            continue
        _name = str(img_name).split(".")[0].strip().replace(" ","")

        if isinstance(vocabulary,list):
            #correct name with vocabulary
            _name = simple_autocorrect(_name,vocabulary)

        img_path = os.path.join(image_dir, img_name)
        try:
            nib_img = nib.load(img_path)
            img_dict[_name] = nib_img
        except Exception as E:
            raise E
    return img_dict


def get_segmentation(patient_id, segmentation_dir, segmentation_mask = "*Segmentation*nii*",
                     LUT_mask = "*ColorTable*", vocabulary = None) -> (nib.Nifti1Image, Dict[str, int]):
    """
    This function reads the segmentaion image and extracts the label dictionary form Slicer segmentation files.
    Warning! More than one segemntation output in the segementation_dir directory which matches the given masks
    can lead to malfunctions.\n
    :param patient_id:  patient id of the patient...
    :param segmentation_dir: directory contains the segmentation for the given patient
    :param segmentation_mask: mask for the segmentation image file
    :param LUT_mask: mask for the LUT file
    :return: segmentation image and label dictionary
    """
    seg_image_path = glob.glob(os.path.join(segmentation_dir,segmentation_mask))
    if len(seg_image_path) > 0:
        seg_image_path = seg_image_path[0]
    else:
        raise IOError(f"no segmentation found for subject {patient_id}")

    seg_img = nib.load(seg_image_path)

    seg_LUT_path = glob.glob(os.path.join(segmentation_dir,LUT_mask))
    if len(seg_LUT_path) > 0:
        seg_LUT_path = seg_LUT_path[0]
    else:
        raise IOError(f"no LUT found for subject {patient_id}")

    seg_label_df = pd.read_csv(seg_LUT_path, skiprows=2, delimiter=" ")
    seg_label_df.columns = ["value", "label_name", "R", "G", "B", "A"]

    seg_vals = np.unique(seg_img.get_fdata())[1:]
    seg_label_df = seg_label_df[seg_label_df["value"].isin(seg_vals)]
    seg_label_df["label_name"] = seg_label_df["label_name"].apply(lambda x: str(x).replace("_", "").replace("mask", ""))

    if isinstance(vocabulary, list):
        # correct name with vocabulary
        seg_label_df["label_name"] = seg_label_df["label_name"].apply(lambda x: simple_autocorrect(x,vocabulary))

    seg_label_dict = seg_label_df.set_index("label_name")["value"].to_dict()

    return seg_img, seg_label_dict


def get_intersecting_volume(image_dict, image_name_filter = "apt"):
    image_volumes = pd.DataFrame()

    assert isinstance(image_dict,dict)

    for image_name, nifti_image in six.iteritems(image_dict):
        if not isinstance(nifti_image,Nifti1Image):
            continue
        if image_name_filter not in image_name:
          continue
        image_start = voxel_to_world(nifti_image.affine,(0,0,0))
        image_end = voxel_to_world(nifti_image.affine,nifti_image.shape)
        image_start[0],image_end[0] = sorted([image_start[0],image_end[0]])
        image_start[1],image_end[1] = sorted([image_start[1],image_end[1]])
        image_start[2],image_end[2] = sorted([image_start[2],image_end[2]])

        image_info = {"name":image_name}
        image_info["x0"],image_info["y0"], image_info["z0"] = image_start[0], image_start[1], image_start[2]
        image_info["x1"],image_info["y1"], image_info["z1"] = image_end[0], image_end[1], image_end[2]
        image_volumes = image_volumes.append(image_info,ignore_index=True)

    image_volumes = image_volumes[["name","x0","x1","y0","y1","z0","z1"]]
    image_volumes = image_volumes.sort_values(by=["x0","y0","z0","x1","y1","z1","name"],ascending=[0,0,0,1,1,1,1]).reset_index(drop=True)    
    
    smallest_image_name = image_volumes.head(1).name.tolist()[0]
    smallest_image = image_dict[smallest_image_name]

    return smallest_image.affine, smallest_image.shape


def process_patient(patient_id: str, image_dict: dict, segmentation_image: nib.Nifti1Image, label_dict: dict,
                    processed_dir: str = None, voxel_limit: int = 5000,
                    unlimited_labels:list = None, limiting_labels: list = None) -> pd.DataFrame:
    """
    This function samples every image of image_dict with the disjoint masks of the segmentation_image for every labels
    in the label_dict.\n
    Notice: Since this operation is calculated with multiple (voxel -> world -> int(voxel')) coordinate transformations,
    images with lower resolution than the segmentation_image will produce redundant data.\n
    :param patient_id: patient id of the patient...
    :param image_dict: dictionary of Nifti1Image s
    :param segmentation_image: segmentation image
    :param label_dict: corresponding label dictionary extracted from the Slicer generated LUT of the segmentation
    :param processed_dir: output directory
    :param voxel_limit: default voxel limit
    :param unlimited_labels: labels not affected by the voxel limit
    :param limiting_labels: labels that can re-define the voxel limit
    :return: resulting pandas dataframe
    """

    if not isinstance(image_dict,dict):
        raise TypeError("image_dict should be a dictionary")

    if not isinstance(segmentation_image, nib.Nifti1Image):
        raise TypeError("Segmentation image format is not supported")

    if not isinstance(label_dict,dict):
        raise TypeError("label_dict should be a dictionary")

    if isinstance(limiting_labels,type(None)): #these labels will not be limited by the voxel limit and these can set a new voxel_limit
        limiting_labels = ["tumors", "tumormelanoma"]

    if isinstance(unlimited_labels,type(None)): #these labels will not be limited by the voxel_limit
        unlimited_labels = ["unspecificwmlesion"]

    result_cols = ["patientNumber","voxelID","worldCoordinate","class_segmentation"] + sorted(list(image_dict.keys()))
    result_df = pd.DataFrame(columns= result_cols)

    limiting_labels = [l for l in label_dict.keys() if any([str(l).startswith(_l) for _l in limiting_labels])]
    unlimited_labels = [l for l in label_dict.keys() if any([str(l).startswith(ul) for ul in unlimited_labels])]

    label_names = limiting_labels + unlimited_labels + list(set(label_dict.keys()).difference(set(limiting_labels+unlimited_labels)))

    voxel_index_cols = [f"{it}_vi" for it in image_dict.keys()]

    for label_name in label_names:
        print(f"processing label '{label_name}'")
        label_value = label_dict[label_name]
        label_voxel_indices = np.argwhere(segmentation_image.get_fdata()==label_value)

        if label_name in limiting_labels:
            if label_voxel_indices.shape[0] > 0:
                voxel_limit = label_voxel_indices.shape[0]*2
        elif label_name not in unlimited_labels:
            _voxel_limit = min(voxel_limit,label_voxel_indices.shape[0])
            _indices = np.random.choice(np.arange(label_voxel_indices.shape[0]),_voxel_limit,replace=False)
            label_voxel_indices = label_voxel_indices[_indices,:]

        print(f"number of voxels = {label_voxel_indices.shape[0]}")

        with Timer(label_name):
            for _i in range(label_voxel_indices.shape[0]):
                voxel_index = label_voxel_indices[_i]
                world_coordinate = voxel_to_world(segmentation_image.affine,voxel_index)

                result_dict_row = OrderedDict()
                result_dict_row["patientNumber"] = patient_id
                result_dict_row["voxelID"] = _i
                result_dict_row["worldCoordinate"] = world_coordinate
                result_dict_row["class_segmentation"] = label_name

                for img_type in image_dict.keys():
                    try:
                        _voxel_index = world_to_voxel(image_dict[img_type].affine,world_coordinate)
                        intensity = image_dict[img_type].get_fdata()[_voxel_index[0],_voxel_index[1],_voxel_index[2]]

                    except IndexError:
                        intensity = np.nan
                        _voxel_index = np.array([np.nan]*3)

                    result_dict_row[img_type] = intensity
                    result_dict_row[f"{img_type}_vi"] = _voxel_index.tobytes()

                result_df = result_df.append(result_dict_row, ignore_index=True)

    #marking duplicates

    nan_val = np.array([np.nan]*3).tobytes()
    result_df["is_duplicate"] = False
    for vic in voxel_index_cols:
        result_df["_is_duplicate"] = result_df.duplicated(subset=[vic],keep="first")
        result_df["is_duplicate"] = result_df.apply(lambda x: x["is_duplicate"] or (x[vic]!=nan_val and x["_is_duplicate"]),axis=1)
    print(f"total voxel count = {len(result_df.index.to_list())}\nnumber of duplicates = {result_df['is_duplicate'].sum}")

    result_df = result_df[result_cols + ["is_duplicate"]]

    if os.path.isdir(processed_dir):
        result_df.to_csv(os.path.join(processed_dir,f"{patient_id}_results.csv"), index=False)

    return result_df


if __name__ == '__main__':
    study_path = "/local_data/switzerland"
    data_dir = os.path.join(study_path, "data_nifti2")
    seg_dir = os.path.join(study_path, "segmentation")
    processed_dir = os.path.join(study_path, "processed")

    os.makedirs(processed_dir,exist_ok=True)

    patient_image_dict = get_patient_dict(data_dir)
    patient_segment_dict = get_patient_dict(seg_dir)

    patient_list = list(set(patient_image_dict.keys()).intersection(set(patient_segment_dict.keys())))
    patient_list.sort()

    results_df = pd.DataFrame()

    series_vocabulary = "ADC,APT_SSFSE,FLAIR,T1,T1_KM,T2,apt10_error_index,apt1_freewatert2t1,apt2_t2t1,apt3_bindingwatermt,apt4_mt_m0cm0a,apt5_mtr_asym,apt6_t1a,apt7_t2a,apt8_noe,apt9_b0".split(",")
    series_vocabulary = [str(s).lower() for s in series_vocabulary]

    label_vocabulary = "postresections,tumormelanoma,whitematters,greymatters,postresection,tumors,nonspecificwmlesions,tumorglioma".split(",")
    label_vocabulary = [str(s).lower() for s in label_vocabulary]

    for patient in patient_list:
        print(f"processing {patient}")
        with Timer(patient):
            try:

                _image_dict = get_image_dict(patient_image_dict[patient], series_vocabulary)
                resampled_img_dir = os.path.join(patient_image_dict[patient],"resampled")
                segmentation_image, label_dict = get_segmentation(patient,patient_segment_dict[patient], vocabulary= label_vocabulary)

                 #_results = process_patient(patient,_image_dict, segmentation_image, label_dict, processed_dir= processed_dir)
            

                _results = process_patient_with_resampled_volumes(patient,_image_dict, segmentation_image, 
                                                                  label_dict, processed_dir= processed_dir, resampled_dir = resampled_img_dir)

                if isinstance(_results,pd.DataFrame):
                    results_df = results_df.append(_results,ignore_index=True)

            except Exception as e:
                print(e)
                # raise e
                continue
                
           

    #results_df.to_csv(os.path.join(processed_dir,"results_2.csv"),index=False)
    results_df.to_csv(os.path.join(processed_dir,"results_resampling.csv"),index=False, na_rep='NaN')
