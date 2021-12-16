import os, sys
import nibabel

from nibabel.nifti1 import Nifti1Image
from nilearn.image import resample_img
import numpy as np

def nibabel_check_align(input_img_path, reference_img_path, propagate_error = False, 
alignment_rel_tol = 0.01, origin_rel_tol = 0.01, affine_rel_tol = 0.01, steps_rel_tol = 0.01, shape_tol = 0):
  """
  Simple method to check if two image has same (or equal) spatial alignment - stored in their affine matrices - and shapes.
  This function assumes time as last dimension.
  """
  check_dict = {"alignment":None,"origin":None,"affine":None, "steps":None, "ndim":None, "shape":None, "frame_shape": None}

  try:
    assert(os.path.isfile(input_img_path))
    assert(os.path.isfile(reference_img_path))

    
    input_img = nibabel.load(input_img_path)
    reference_img = nibabel.load(reference_img_path)

    assert input_img.affine.shape == reference_img.affine.shape

    
    check_dict["alignment"] = np.allclose(input_img.affine[:3,:3],reference_img.affine[:3,:3],rtol = alignment_rel_tol)
    check_dict["origin"] = np.allclose(input_img.affine[:-1,-1],reference_img.affine[:-1,-1],rtol = origin_rel_tol)
    check_dict["affine"] = np.allclose(input_img.affine,reference_img.affine,affine_rel_tol)
    if len(input_img.header.get_zooms()) != reference_img.header.get_zooms():
      check_dict["steps"] = False
    else:
      check_dict["steps"] = np.allclose(input_img.header.get_zooms(),reference_img.header.get_zooms(),steps_rel_tol)
    
    dim1 = input_img.get_fdata().ndim
    dim2 = reference_img.get_fdata().ndim

    check_dict["ndim"] = dim1 == dim2
    if check_dict["ndim"]:
      check_dict["shape"] = np.allclose(input_img.get_fdata().shape,reference_img.get_fdata().shape, atol=shape_tol)
      check_dict["frame_shape"] = check_dict["shape"]
    else:
      check_dict["shape"] = False
      if dim1>dim2:
        check_dict["frame_shape"] = np.allclose(np.array(input_img.get_fdata().shape)[:-1],np.array(reference_img.get_fdata().shape), atol=shape_tol)
      else:
        check_dict["frame_shape"] = np.allclose(np.array(input_img.get_fdata().shape),np.array(reference_img.get_fdata().shape)[:-1], atol=shape_tol)

    return check_dict
  except Exception as e:
    print(e)
    if propagate_error:
      raise e
    return check_dict

if __name__ == "__main__":
  ref = "/local_data/switzerland/fsl_coreg/my_processing/fbrainmask-in-rawavg.nii.gz"
  input = "/local_data/switzerland/fsl_coreg/my_processing/TMAX.nii.gz"
  ref = "/local_data/switzerland/fsl_coreg/my_processing/fbrainmask-in-rawavg.nii.gz"
  input = "/local_data/switzerland/fsl_coreg/my_processing/MTT.nii.gz"
  output = "/local_data/switzerland/fsl_coreg/my_processing/nibl-MTT.nii.gz"
  print(nibabel_check_align(ref,output))