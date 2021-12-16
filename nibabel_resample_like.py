import os, sys
import nibabel

from nibabel.nifti1 import Nifti1Image
from nilearn.image import resample_img
import numpy as np

def nibabel_resample_from_to(input_img_path, reference_img_path, out_img_path, resampling_method = "linear", fill_value = 0,
  return_image = False, propagate_error = False):
  """
  Simple method for image resampling with nibabel. This method resamples the image stored at the 'input_image_path' 
  into the image matrix of the image stored at the 'reference_img_path' using the 'resampling_method' resampling method.
  This method only works on 3D images!
  """
  try:
    assert(os.path.isfile(input_img_path))
    assert(os.path.isfile(reference_img_path))
    assert(os.path.isdir(os.path.dirname(out_img_path)))
    assert(resampling_method in ['continuous', 'linear','nearest'])
    
    input_img = nibabel.load(input_img_path)
    reference_img = nibabel.load(reference_img_path)

    assert(input_img.get_fdata().ndim == 3)
    assert(reference_img.get_fdata().ndim == 3)

    print(f"'{resampling_method}' resampling '{input_img_path}' -> '{reference_img_path}' = '{out_img_path}'")

    resampled_img = resample_img(input_img,target_affine=reference_img.affine, target_shape=reference_img.shape,
        interpolation= resampling_method, fill_value= fill_value)
    
    resampled_img.set_data_dtype(input_img.get_data_dtype())
    
    nibabel.save(resampled_img,out_img_path)
    if return_image:
      return resampled_img
    return True
  except Exception as e:
    print(e)
    if propagate_error:
      raise e
    return False

if __name__ == "__main__":
  ref = "/local_data/switzerland/fsl_coreg/my_processing/fbrainmask-in-rawavg.nii.gz"
  input = "/local_data/switzerland/fsl_coreg/my_processing/MTT.nii.gz"
  output = "/local_data/switzerland/fsl_coreg/my_processing/nibl-MTT.nii.gz"
  nibabel_resample_from_to(input,ref,output)