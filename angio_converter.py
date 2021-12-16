from re import I, T
import pydicom as dicom
import nibabel as nib
import nibabel.affines
import glob
import os,sys
import numpy as np
import skimage
import pandas as pd
from collections import OrderedDict
import datetime

import json

def convert_dcm_angio_images(root_dir, out_dir, default_time_step = 1, default_spacing = 1):
  # find images
  dcm_angio_batches = []
  for root, dirs, files in os.walk(root_dir):
    batch_data = []
    batch_path = []
    for f in files:
      try:
        _f = os.path.join(root,f)
        D = dicom.read_file(_f)

        if not hasattr(D,"pixel_array"):
          continue

        batch_data.append(D)
        batch_path.append(_f)
      
      except Exception as e:
        print(e)
        continue
    
    if len(batch_data)>0:
      dcm_angio_batches.append({"data":batch_data,"path":batch_path})
          
  # convert batches to nii.gz images by stacking them on time with uniform time step
  if not os.path.isdir(out_dir):
    os.makedirs(out_dir,exist_ok = True)

  for batch_index in range(len(dcm_angio_batches)):

    print(f"processing batch {batch_index+1}")
    batch = dcm_angio_batches[batch_index]
    data = batch["data"]
    path = batch["path"]

    # sort files by acqusition time
    _d = zip(path,data)
    _d = sorted(_d,key= lambda x: x[1].ContentTime)
    __d = zip(*_d)
    path,data = [list(___d) for ___d in __d]
    path = [os.path.relpath(p,root_dir) for p in path]

    sample_data = data[0]
    spacing = None
    if hasattr(sample_data,"PixelSpacing"):
      spacing = np.array(sample_data.PixelSpacing)
    elif hasattr(sample_data,"ImagerPixelSpacing"):
      spacing = np.array(sample_data.ImagerPixelSpacing) / sample_data.EstimatedRadiographicMagnificationFactor
    else:
      if np.array(default_spacing).ndim == 0:
        spacing = np.array([default_spacing]*2)
      else:
        spacing = np.array(default_spacing)
      print("using default pixel spacing")
    
    print(f"spacing : {spacing} mm")
    im_count = len(data)
    print(f"image count : {im_count}")
    im_shape = sample_data.pixel_array.shape
    print(f"image shape : {im_shape}")


    content_times = [str(d.ContentTime) for d in data]
    print(f"Content times: {content_times}")
    time_zero = datetime.datetime.strptime(content_times[0],"%H%M%S.%f")
    _times = [ datetime.datetime.strptime(str(t),"%H%M%S.%f") for t in content_times]
    time_scale = [(t-time_zero).seconds + (t-time_zero).microseconds/np.power(10,6) for t in _times]

    if len(time_scale) == 1:
      time_step = default_time_step
      print("using default pixel spacing")
    else:
      time_step = np.mean(np.diff(time_scale))
    
    print(f"time step : {time_step} s")

    affine = np.eye(4)
    affine = affine * np.array([spacing[1],spacing[0],1,time_step])


    image_matrix = np.zeros(tuple(list(sample_data.pixel_array.shape)+[1,len(data)]))
    
    for j in range(len(data)):
      image_matrix[:,:,0,j] = data[j].pixel_array.T
      
    img = nib.Nifti1Image(image_matrix,affine=affine)
    img_path = os.path.join(out_dir,f"img_{str(batch_index+1).zfill(4)}.nii.gz")
    nib.save(img,img_path)

    json_path = os.path.join(out_dir,f"img_{str(batch_index+1).zfill(4)}.json")
    json_dict = OrderedDict()
    json_dict["source_root"] = os.path.relpath(root_dir,out_dir)
    json_dict["batch_count"] = len(dcm_angio_batches)
    json_dict["batch_index"] = batch_index+1
    json_dict["nii_file"] = os.path.relpath(img_path,out_dir)
    json_dict["patient_id"] = sample_data.PatientID
    json_dict["batch_len"] = im_count
    json_dict["batch_files"] = path
    json_dict["content_times"] = content_times
    json_dict["time_scale"] = time_scale
    json_dict["time_step"] = time_step

    json_dict["image_shape"] = im_shape
    json_dict["pixel_spacing"] = spacing.tolist()
    json_dict["time_step"] = time_step

    with open(json_path,"w") as json_stream:
      json.dump(json_dict,json_stream)
    

if __name__ == '__main__':
  convert_dcm_angio_images("/local_data/switzerland/angio/data_example2","/local_data/switzerland/angio/data_example2_converted")