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

from skimage import morphology as skim
from scipy.ndimage import morphology as morph
import scipy.ndimage as ndimage

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
    _vocabulary = [re.sub('[^0-9a-zA-Z]+','',str(v).lower()) for v in vocabulary]
    _input_word = re.sub('[^0-9a-zA-Z]+','',str(input_word).lower())

    if input_word in vocabulary:
        return input_word, 1
    else:
        similarities = [1 - (textdistance.Jaccard(qval=2).distance(re.sub('[^0-9a-zA-Z]+','',str(v).lower()), _input_word)) for v in vocabulary]
        ordered = sorted(zip(vocabulary,similarities),key= lambda x: x[1], reverse=True)
        return ordered[0]


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
        score = 1
        if isinstance(vocabulary,list):
            #correct name with vocabulary
            _name, score = simple_autocorrect(_name,vocabulary)
            # print(f"{_name} : {score}")
            if score<0.5:
              continue

        img_path = os.path.join(image_dir, img_name)
        try:
            nib_img = nib.load(img_path)
            img_dict[_name] = nib_img
            print(f"{_name} = {img_path}")
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
        seg_label_df["label_name"] = seg_label_df["label_name"].apply(lambda x: simple_autocorrect(x,vocabulary)[0])

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


def process_patient_with_resampled_volumes_nn_segment(patient_id: str, image_dict: dict, segmentation_image: nib.Nifti1Image, label_dict: dict,
                  processed_dir: str = None, resampled_dir: str = None,
                  voxel_limit: int = 5000,
                  unlimited_labels:list = None, limiting_labels: list = None):
  
    if not isinstance(image_dict,dict):
      raise TypeError("image_dict should be a dictionary")

    if not isinstance(segmentation_image, nib.Nifti1Image):
      raise TypeError("Segmentation image format is not supported")

    if not isinstance(label_dict,dict):
      raise TypeError("label_dict should be a dictionary")

    if isinstance(limiting_labels,type(None)): #these labels will not be limited by the voxel limit and these can set a new voxel_limit
      limiting_labels = ["tumors", "tumormelanoma","tumor","meta"]

    if isinstance(unlimited_labels,type(None)): #these labels will not be limited by the voxel_limit
      # unlimited_labels = ["unspecificwmlesion","nonspecific"]
      unlimited_labels = ["unspec","nonspec"]

    resampling_affine, resampling_shape = get_intersecting_volume(image_dict = _image_dict)

    limiting_labels = [l for l in label_dict.keys() if any([str(l).startswith(_l) for _l in limiting_labels])]
    unlimited_labels = [l for l in label_dict.keys() if any([str(l).startswith(ul) for ul in unlimited_labels])]

    label_names = limiting_labels + unlimited_labels + list(set(label_dict.keys()).difference(set(limiting_labels+unlimited_labels)))

    # TODO implement 'process_patient_with_resamling', which resamples image dict and the segmentation image with the
    # previously calculated 'resampling_affine' and 'resampling_shape' using the nilearn.image.resample_image function, 
    # then extract the intensity and label information from all images and all remaining voxels...

    if not isinstance(resampled_dir,type(None)):
      os.makedirs(resampled_dir,exist_ok=True)
      save_resampled = True
    else:
      save_resampled = False

    resampled_image_dict = {}
    resampled_voxel_validity_dict = {}
    for image_type in image_dict.keys():
      try:
        _resampled_image = resample_img(image_dict[image_type],target_affine=resampling_affine, target_shape=resampling_shape, interpolation= "linear", fill_value= np.nan)
        if save_resampled:
          try:
            out_path = os.path.join(resampled_dir,f"{image_type}.nii.gz")
            print(f"saving {out_path}")
            nib.save(_resampled_image,out_path)
          except Exception as _e:
            print(_e)
        
        validity_image_data = np.ones_like(image_dict[image_type].get_fdata(),dtype=int)
        validity_image = nib.Nifti1Image(validity_image_data,image_dict[image_type].affine,image_dict[image_type].header)
        resampled_validity_image = resample_img(validity_image,target_affine=resampling_affine, target_shape=resampling_shape,
        interpolation= "nearest", fill_value= 0)
        validity_data = resampled_validity_image.get_fdata().astype(bool)
        resampled_voxel_validity_dict[image_type] = validity_data
        resampled_image_dict[image_type] = _resampled_image

      except Exception as __e:
        print(__e)
        continue

    result_cols = ["patientNumber","voxelID","class_segmentation"] + sorted([str(k).upper() for k in list(image_dict.keys())])
    result_df = pd.DataFrame(columns= result_cols)

    resampled_segment = resample_img(segmentation_image,target_affine=resampling_affine, target_shape=resampling_shape,
          interpolation = "nearest")

    if save_resampled:
      try:
        out_path = os.path.join(resampled_dir,f"segmentation.nii.gz")
        print(f"saving {out_path}")
        resampled_segment.set_data_dtype(np.int16)
        nib.save(resampled_segment,out_path)
      except Exception as _e:
        print(_e)

    resampled_segment_data = resampled_segment.get_fdata()

    for label_name in label_names:
      print(f"label type:{label_name}")
      label_value = label_dict[label_name] 

      label_voxel_indices = np.argwhere(resampled_segment_data==label_value)

      if label_name in limiting_labels:
        if label_voxel_indices.shape[0] > 0:
          voxel_limit = label_voxel_indices.shape[0]*2
      elif label_name not in unlimited_labels:
        _voxel_limit = min(voxel_limit,label_voxel_indices.shape[0])
        _indices = np.random.choice(np.arange(label_voxel_indices.shape[0]),_voxel_limit,replace=False)
        label_voxel_indices = label_voxel_indices[_indices,:]

      print(f"number of voxels = {label_voxel_indices.shape[0]}")

      for _i in range(label_voxel_indices.shape[0]):
        voxel_index = label_voxel_indices[_i]
        result_dict_row = OrderedDict()
        result_dict_row["patientNumber"] = patient_id
        result_dict_row["voxelID"] = _i
        result_dict_row["class_segmentation"] = label_name
    
        for image_type in resampled_image_dict.keys():
          if (not resampled_voxel_validity_dict[image_type][voxel_index[0],voxel_index[1],voxel_index[2]]):
            intensity = np.nan
          else:
            intensity = resampled_image_dict[image_type].get_fdata()[voxel_index[0],voxel_index[1],voxel_index[2]]
          result_dict_row[str(image_type).upper()] = intensity

        result_df = result_df.append(result_dict_row, ignore_index=True)

    if os.path.isdir(processed_dir):
      result_df.to_csv(os.path.join(processed_dir,f"{patient_id}_results.csv"), index=False, na_rep='NaN')
    return result_df

def process_patient_with_resampled_volumes_trilinear_segment(patient_id: str, image_dict: dict, segmentation_image: nib.Nifti1Image, label_dict: dict,
                  processed_dir: str = None, resampled_dir: str = None,
                  voxel_limit: int = 5000,
                  unlimited_labels:list = None, limiting_labels: list = None):
  
    if not isinstance(image_dict,dict):
      raise TypeError("image_dict should be a dictionary")

    if not isinstance(segmentation_image, nib.Nifti1Image):
      raise TypeError("Segmentation image format is not supported")

    if not isinstance(label_dict,dict):
      raise TypeError("label_dict should be a dictionary")

    if isinstance(limiting_labels,type(None)): #these labels will not be limited by the voxel limit and these can set a new voxel_limit
      # limiting_labels = ["tumors", "tumormelanoma","tumor","meta"]
      limiting_labels = []

    if isinstance(unlimited_labels,type(None)): #these labels will not be limited by the voxel_limit
      # unlimited_labels = ["unspecificwmlesion","nonspecific"]
      # unlimited_labels = ["unspec","nonspec"]
      unlimited_labels = []

    resampling_affine, resampling_shape = get_intersecting_volume(image_dict = _image_dict)

    limiting_labels = [l for l in label_dict.keys() if any([str(l).startswith(_l) for _l in limiting_labels])]
    unlimited_labels = [l for l in label_dict.keys() if any([str(l).startswith(ul) for ul in unlimited_labels])]

    label_names = limiting_labels + unlimited_labels + list(set(label_dict.keys()).difference(set(limiting_labels+unlimited_labels)))


    if not isinstance(resampled_dir,type(None)):
      os.makedirs(resampled_dir,exist_ok=True)
      save_resampled = True
    else:
      save_resampled = False

    resampled_image_dict = {}
    resampled_voxel_validity_dict = {}
    for image_type in image_dict.keys():
      try:
        _resampled_image = resample_img(image_dict[image_type],target_affine=resampling_affine, target_shape=resampling_shape, interpolation= "linear", fill_value= np.nan)
        if save_resampled:
          try:
            out_path = os.path.join(resampled_dir,f"{image_type}.nii.gz")
            print(f"saving {out_path}")
            nib.save(_resampled_image,out_path)
          except Exception as _e:
            print(_e)
        
        validity_image_data = np.ones_like(image_dict[image_type].get_fdata(),dtype=int)
        validity_image = nib.Nifti1Image(validity_image_data,image_dict[image_type].affine,image_dict[image_type].header)
        resampled_validity_image = resample_img(validity_image,target_affine=resampling_affine, target_shape=resampling_shape,
        interpolation= "nearest", fill_value= 0)
        validity_data = resampled_validity_image.get_fdata().astype(bool)
        resampled_voxel_validity_dict[image_type] = validity_data
        resampled_image_dict[image_type] = _resampled_image

      except Exception as __e:
        print(__e)
        continue

    result_cols = ["patientNumber","voxelID","class_segmentation"] + sorted([str(k).upper() for k in list(image_dict.keys())])
    result_df = pd.DataFrame(columns= result_cols)

    # segment-wise trilinear interpolation
    segmentation_values = np.unique(segmentation_image.get_fdata())[1:]
    limit = 0.01
    dilation_radius = 0
    close_radius = 1
    __resampled_segmentation_data = np.zeros([segmentation_values.shape[0]]+list(resampling_shape))
    for sv_index in range(segmentation_values.shape[0]):
      sv = segmentation_values[sv_index]
      _segment_data = np.zeros_like(segmentation_image.get_fdata()).astype(int)
      _segment_data[segmentation_image.get_fdata()==sv] = 1
      if close_radius > 0:
        _segment_data = morph.binary_closing(_segment_data,skim.ball(close_radius))
      if dilation_radius > 0:
        _segment_data = morph.binary_dilation(_segment_data,skim.ball(dilation_radius))
      _segment_data = _segment_data.astype(float)
      _segment_data[_segment_data!=0] = 1.0001
      _segmentation = nib.Nifti1Image(_segment_data,segmentation_image.affine)
      _segmentation.set_data_dtype(float)
      _resampled_segmentation = resample_img(_segmentation,target_affine=resampling_affine, target_shape=resampling_shape,
          interpolation = "linear")
      __resampled_segmentation_data[sv_index,:,:,:] = _resampled_segmentation.get_fdata()      

    _resampled_segmentation_data = np.zeros(resampling_shape)
    # find the highest probability index
    _resampled_segmentation_data = np.argmax(__resampled_segmentation_data,axis = 0)
    #replace with the orig value
    resampled_segmentation_data = np.zeros(resampling_shape)
    for sv_index in range(segmentation_values.shape[0]):
      resampled_segmentation_data[_resampled_segmentation_data==sv_index] = segmentation_values[sv_index]
    resampled_segmentation_data[np.max(__resampled_segmentation_data,axis = 0)<limit] = 0
    resampled_segmentation_data = resampled_segmentation_data.astype(int)

    if save_resampled:
      try:
        out_path = os.path.join(resampled_dir,f"trilinear_segmentation.nii.gz")
        print(f"saving {out_path}")
        resammpled_seg_image = nib.Nifti1Image(resampled_segmentation_data.astype(int),resampling_affine)
        resammpled_seg_image.set_data_dtype(np.int16)
        nib.save(resammpled_seg_image,out_path)
      except Exception as _e:
        print(_e)


    for label_name in label_names:
      print(f"label type:{label_name}")
      label_value = label_dict[label_name] 

      label_voxel_indices = np.argwhere(resampled_segmentation_data==label_value)

      if label_name in limiting_labels:
        if label_voxel_indices.shape[0] > 0:
          voxel_limit = label_voxel_indices.shape[0]*2
      elif label_name not in unlimited_labels:
        _voxel_limit = min(voxel_limit,label_voxel_indices.shape[0])
        _indices = np.random.choice(np.arange(label_voxel_indices.shape[0]),_voxel_limit,replace=False)
        label_voxel_indices = label_voxel_indices[_indices,:]

      print(f"number of voxels = {label_voxel_indices.shape[0]}")

      for _i in range(label_voxel_indices.shape[0]):
        voxel_index = label_voxel_indices[_i]
        result_dict_row = OrderedDict()
        result_dict_row["patientNumber"] = patient_id
        result_dict_row["voxelID"] = _i
        result_dict_row["class_segmentation"] = label_name
    
        for image_type in resampled_image_dict.keys():
          if (not resampled_voxel_validity_dict[image_type][voxel_index[0],voxel_index[1],voxel_index[2]]):
            intensity = np.nan
          else:
            intensity = resampled_image_dict[image_type].get_fdata()[voxel_index[0],voxel_index[1],voxel_index[2]]
          result_dict_row[str(image_type).upper()] = intensity

        result_df = result_df.append(result_dict_row, ignore_index=True)

    if os.path.isdir(processed_dir):
      result_df.to_csv(os.path.join(processed_dir,f"{patient_id}_results.csv"), index=False, na_rep='NaN')
    return result_df

if __name__ == '__main__':
  study_path = "/local_data/switzerland/new_20210907"
  processed_dir = os.path.join(study_path, "processed")

  os.makedirs(processed_dir,exist_ok=True)

  patient_image_dict = get_patient_dict(study_path)

  patient_list = list(patient_image_dict.keys())
  patient_list.sort()

  results_df = pd.DataFrame()

  series_vocabulary = "ADC,APT_SSFSE,FLAIR,T1,T1_KM,T2,apt10_error_index,apt1_freewatert2t1,apt2_t2t1,apt3_bindingwatermt,apt4_mt_m0cm0a,apt5_mtr_asym,apt6_t1a,apt7_t2a,apt8_noe,apt9_b0".split(",")
  series_vocabulary = [str(s).lower() for s in series_vocabulary]

  label_vocabulary = ("postopresection,tumors_lung,metastase_melanom_all,nonspecificlesions,tumors_lung_all,nonspecificwmlesions"+
  ","+"white_matter,grey_matter,non_specific_lesions,tumors_melanom,metastases_lung").split(",")
  label_vocabulary = [str(s).lower() for s in label_vocabulary]

  for patient in patient_list:
    print(f"processing {patient}")
    with Timer(patient):
      try:
        # if patient != "patient21":
        #   continue

        _image_dict = get_image_dict(patient_image_dict[patient], series_vocabulary)
        
        resampled_img_dir = os.path.join(patient_image_dict[patient],"resampled")
        segmentation_dir = os.path.join(patient_image_dict[patient],"Scene")
        if not os.path.isdir(segmentation_dir):
          continue
        segmentation_image, label_dict = get_segmentation(patient,segmentation_dir, vocabulary= label_vocabulary)
        # continue
         #_results = process_patient(patient,_image_dict, segmentation_image, label_dict, processed_dir= processed_dir)
      

        # print(label_dict)
        # continue
        _results = process_patient_with_resampled_volumes_trilinear_segment(patient,_image_dict, segmentation_image, 
                                  label_dict, processed_dir= processed_dir, resampled_dir = resampled_img_dir)

        #_results = process_patient_with_resampled_volumes_nn_segment(patient,_image_dict, segmentation_image, 
        #                          label_dict, processed_dir= processed_dir, resampled_dir = resampled_img_dir)

        if isinstance(_results,pd.DataFrame):
          results_df = results_df.append(_results,ignore_index=True)

      except Exception as e:
        print(e)
        # raise e
        continue
        
       
  results_df.to_csv(os.path.join(processed_dir,"results_resampling.csv"),index=False, na_rep='NaN')
