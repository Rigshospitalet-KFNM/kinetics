"""These function exists just for me to create an easy environment for debugging"""
from pathlib import Path
from pydicom import dcmread
import numpy
from nibabel import load as load_nifti

from rt_utils import RTStruct
from rt_utils.ds_helper import create_rtstruct_dataset

def load_data():
  ct_ds = [
    dcmread(p) for p in Path('data/ct_rest/dicom').glob('*.dcm')
  ]

  rt_ds = create_rtstruct_dataset(ct_ds)

  aorta = load_nifti('data/ct_rest/nifti_seg/aorta.nii.gz')
  aorta_data = aorta.get_fdata().astype(numpy.bool_)

  rt =  RTStruct(ct_ds, rt_ds)

  rt.add_roi(
    aorta_data,
    [255,255,255],
    'aorta',
    'Total segmentator aorta'
  )

  return rt
