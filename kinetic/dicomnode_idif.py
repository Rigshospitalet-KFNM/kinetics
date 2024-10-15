

from rt_utils import RTStruct

from dicomnode.dicom.rt_structs import get_mask
from dicomnode.math import bounding_box

def get_idif(rt_struct: RTStruct, ):
  mask = get_mask(rt_struct, 'aorta')

  (x_min, x_max), (y_min, y_max), (z_min, z_max) = bounding_box(mask)
