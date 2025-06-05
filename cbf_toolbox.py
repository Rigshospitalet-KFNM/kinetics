"""Alright this is a toolbox for CBF pipeline

So some of this code is kinda weird because my code and Thomas's code doesn't
really agree about how you should order the data. It's the x,y,z vs z,y,x
thing I have been talking about...

Still I hope that it's mostly straightforward to take the parts out of the file
that you need.


"""


from os import environ
from pathlib import Path
from typing import Any, List, Optional,Tuple, Union

from matplotlib.axes import Axes
import nibabel as nib
import numpy as np
from pydicom import Dataset, read_file
from pydicom.uid import RTImageStorage, CTImageStorage
from rt_utils import RTStruct
from skimage.measure import label, regionprops
from scipy.ndimage import binary_dilation

# Dicomnode imports
from dicomnode.lib.exceptions import CouldNotCompleteDIMSEMessage
from dicomnode.lib.validators import OptionsValidator
from dicomnode.dicom.rt_structs import get_mask, get_mask_wrong_way
from dicomnode.dicom.dimse import Address, send_images
from dicomnode.dicom.series import FramedDicomSeries
from dicomnode.math import transpose_nifti_coords
from dicomnode.math.interpolation import resample
from dicomnode.server.grinders import RTStructGrinder, LargeDynamicPetSeriesGrinder
from dicomnode.server.input import AbstractInput

from rhkinetics.lib import roi


AORTA_DESC_LOWER = 1
AORTA_ARCH = 2
AORTA_DESC_UPPER = 3
AORTA_ASH = 4


CONTOUR_COLORS = ['lime', 'red', 'purple', 'blue']
CONTOUR_LEVELS = [0.5, 1.5, 2.5, 3.5, 4.5]

#region SMALL FUNCTIONS

def find_true_sub_ranges(bools: np.ndarray)\
    -> np.ndarray:
  """Generates a list of indexes for sub-ranges where there's consecutive true
  values

  Args:
      bools (numpy.ndarray[Tuple[int], numpy.bool_]): _description_

  Returns:
      numpy.ndarray[Tuple[int, Literal[2]], int]: A list of indexes

  Example:
  >>>find_true_sub_ranges(
    [False, True, True, False, True, True, True, False, False, True]
  )
  array([[ 1,  3],
         [ 4,  7],
         [ 9, 10]])
  """

  return np.where(
      np.diff(
        np.hstack(
          ([False],bools,[False])
        )
      )
    )[0].reshape(-1,2)

def sort_greatest_first(idx_pairs: np.ndarray):
  """Sorts the array such that the tuple with the greatest difference is the
  first element, and the second element is second greatest difference and so
  forth

  Args:
    idx_pairs (List[Tuple[int,int]]): A list of indexes indicating ranges

  Returns:
      List[ndarray]: A sorted list

  Example:
  >>> sort_greatest_first([(1,3),(5,9),(11,12),(20,56)])
  [(20,56), (5,9), (1,3), (11,12)]
  """
  return sorted(idx_pairs, reverse=True, key=np.diff)

def get_longest_sub_range(ranges: np.ndarray)\
    -> np.ndarray:
  """Finds the longest sub range in an array of pairs.

  Args:
    ranges: ndarray[Tuple[int, Literal[2]], int]

  Returns:
      numpy.ndarray[Tuple[Literal[2]], int]: the indexes that are the longest

  Example:
  >>>get_longest_sub_range([[1,3], [4,5], [8,12]])
  [8,12]
  """
  return ranges[np.diff(ranges, axis=1).argmax()]

def ordered_range(x, y):
  """Creates a range from X to Y or from Y to X depending on which of them is
  larger

  Args:
      x (int): starting or ending point of the range
      y (int): starting or ending point of the range

  Returns:
      Range: A range from X to Y or a range from Y to X
  """
  if x < y:
    return range(x, y)
  else:
    return range(y, x)

def suv_frame_selection(pet_series: FramedDicomSeries) -> np.ndarray[Tuple[int], Any]:
  """Selects the frames used for that provides the activity for the SUV
  calculation. For this implementation it's the first 40 seconds of frames.

  But this is just a human choice...

  """
  frame_start_time = pet_series.frame_acquisition_time - pet_series.frame_acquisition_time[0]
  return frame_start_time < np.timedelta64(40 * 1000, 'ms')

def calculate_suv_row_major(image: np.ndarray, activity_Bq, patient_weight_kg):
  if len(image.shape) == 4:
    return image.mean(axis=0) * patient_weight_kg * 1000  / activity_Bq
  return image * patient_weight_kg * 1000 / activity_Bq

def calculate_suv_column_major(image: np.ndarray, activity_Bq, patient_weight_kg):
  if len(image.shape) == 4:
    return image.mean(axis=-1) * patient_weight_kg * 1000 / activity_Bq
  return image * patient_weight_kg * 1000 / activity_Bq

def input_extract_mask(segmentation_input, ct_series=None) -> np.ndarray:
    # Input extraction
  if isinstance(segmentation_input, nib.nifti1.Nifti1Image)\
    or isinstance(segmentation_input, nib.nifti2.Nifti2Image):

    # mask = segmentation_nifti.get_fdata() == AORTA_FLAG
    mask = segmentation_input.get_data()
  elif isinstance(segmentation_input, Dataset):
    if ct_series is None:
      raise TypeError("Cannot construct the Segmentation without the underlying CT series")
    mask = get_mask_wrong_way(ct_series, segmentation_input, 'aorta')
  elif isinstance(segmentation_input, str) or isinstance(segmentation_input, Path):
    if isinstance(segmentation_input, str):
      segmentation_path = Path(segmentation_input)
    else:
      segmentation_path = segmentation_input

    if segmentation_path.name.endswith('.dcm') and segmentation_path.exists():
      segmentation_dataset = read_file(segmentation_path)

      if ct_series is None:
        raise TypeError("Cannot construct the Segmentation without the underlying CT series")
      mask = get_mask_wrong_way(ct_series, segmentation_dataset, 'aorta')
    elif segmentation_path.name.endswith('.nii.gz') and segmentation_path.exists():
      segmentation_nifti = nib.loadsave.load(segmentation_path)

      mask = segmentation_nifti.get_fdata() # type: ignore
      # So here the segmentation might include multiple segmentation in that case:
      # mask = segmentation_nifti.get_fdata() == AORTA_FLAG
    else:
      raise ValueError("Unable to convert files to a mask")
  elif isinstance(segmentation_input, np.ndarray):
      mask = segmentation_input
  else:
    raise TypeError("Aorta_segmentation: Unable to convert Segmentation Input to numpy mask")

  return mask

# Region Aorta Segmentation
def aorta_preproccesing(aortamask: np.ndarray, SUV, SUV_median):
  # Threshold aortamask with median(SUV)/1.5
  out = aortamask*np.int8(SUV>SUV_median/1.5)

  xdim, ydim, nslices = out.shape

  # Count number of clusters
  nclusters = roi.count_clusters(out)
  #print(f'Number of Clusters found in Aorta Segmentation: {nclusters}')

  if nclusters > 1:
    # Handle the mystery
    #print(' Handling multiple clusters')

    # Keep only cluster above threshold
    volthreshold=20
    aortamask_tmp, nclusters = roi.threshold_clusters(out, volthreshold=volthreshold)

    # Still have multiple clusters - now try to extrapolate
    if nclusters > 1:
      # Loop over axial slices and count number of clusters
      nrois = np.zeros((nslices,),dtype=int)
      for slc in range(nslices-1,-1,-1):
        nrois[slc] = roi.count_clusters(out[:,:,slc])

        if nrois[slc] == 0 or np.count_nonzero(out[:,:,slc])<3:
          # Get bounding box for the two slices below
          xmin_tmp, xsize_tmp, ymin_tmp, ysize_tmp, _, _ = roi.bbox(out[:,:,slc+1:slc+3])

          xmid_tmp = xmin_tmp+xsize_tmp//2
          ymid_tmp = ymin_tmp+ysize_tmp//2

          # Keep only largest cluster if multiple
          maskimg = roi.keep_largest_cluster(SUV[xmid_tmp-5:xmid_tmp+5,ymid_tmp-5:ymid_tmp+5,slc]>SUV_median/1.5)

          # Run region props on binary mask add SUV image for weighted centroid estimation
          regions = regionprops(label(maskimg), intensity_image=SUV[xmid_tmp-5:xmid_tmp+5,ymid_tmp-5:ymid_tmp+5,slc])
          for props in regions:
            out[xmid_tmp-5+int(props.centroid_weighted[0])-3:xmid_tmp-5+int(props.centroid_weighted[0])+4,
                ymid_tmp-5+int(props.centroid_weighted[1])-3:ymid_tmp-5+int(props.centroid_weighted[1])+4,slc] = 1

        # Dilate and threshold to account for aortavoxels outside segmentation
        aortamask_dilated = binary_dilation(out[:,:,slc])
        out[:,:,slc] = aortamask_dilated*np.int8(SUV[:,:,slc]>SUV_median/1.5)
  return out


def aorta_segmentation(
    segmentation_input,
    ct_series: Optional[List[Dataset]]= None
  ) -> np.ndarray:
  """Segments the aorta with an similiar to the IDIF.py from kinetic library

  Note that IDIF does some preprocessing with regarding to the SUV, which this
  function doesn't do. If you want to do those you have

  Args:
      segmentation_input: An object that can be converted to numpy array in
                          Column major order (x,y,z)
      ct_series (Optional[List[Dataset]], optional): Incase you pass a RT
      struct, you need pass the Series that was used to create the RT struct,
      because the spacial information isn't saved in the RT struct, only the
        Defaults to None.

  Raises:
      TypeError: If input_extract_mask is unable to pull a mask from the args,
                then it throws

  Returns:
      np.ndarray[Tuple[int,int,int], np.bool_]: A mask segmented in a similar
        way to IDIF.py
  """
  mask = input_extract_mask(segmentation_input, ct_series)

  # End of input extraction

  def do_the_segmentation(aortamask: np.ndarray):
    # Get image dimensions
    xdim,ydim,nslices = aortamask.shape

    # Allocate aortamask_segmented
    aortamask_segmented = aortamask.astype(int)

    # Loop over all axial slices and count number of clusters
    nclusters = np.zeros((nslices,),dtype=int)

    for slc in range(nslices):
      label_img, nclusters[slc] = label(aortamask[:,:,slc], return_num=True)


    # Correct
    nclusters[nclusters>2] = 2

    # Find islands of connected ones in nclusters
    # Get start, stop index pairs for islands/seq. of 1s
    idx_pairs = np.where(np.diff(np.hstack(([False],nclusters==1,[False]))))[0].reshape(-1,2)

    # Get the island lengths, whose argmax would give us the ID of longest island.
    # Start index of that island would be the desired output
    start_longest_seq_1 = idx_pairs[np.diff(idx_pairs,axis=1).argmax(),0]

    # Aorta descendens is the biggest island of connected slices
    slices_desc = range(start_longest_seq_1, nslices)

    # Segment aorta descendens lower
    aortamask_segmented[:,:,slices_desc] = aortamask[:,:,slices_desc] * 4

    # Aorta arch is the biggest island of connected twos

    # Get start, stop index pairs for islands/seq. of 1s
    idx_pairs = np.where(np.diff(np.hstack(([False],nclusters==2,[False]))))[0].reshape(-1,2)
    # Get the island lengths, whose argmax would give us the ID of longest island.
    # Start index of that island would be the desired output
    start_longest_seq_2 = idx_pairs[np.diff(idx_pairs,axis=1).argmax(),0]
    slices_arch = range(0,start_longest_seq_2)
    slices_two = range(start_longest_seq_2,start_longest_seq_1)


    # Find upper descending part of aorta by finding connection with lower part
    label_img_2, nclusters = label(aortamask[:,:,slices_two[0]:slices_desc[0]+1], return_num=True)

    for cluster in range(1,nclusters+1):
        if np.sum((label_img_2==cluster)*(aortamask_segmented[:,:,slices_two[0]:slices_desc[0]+1]==4)):
            aortamask_segmented[:,:,slices_two[0]:slices_desc[0]] =  aortamask_segmented[:,:,slices_two[0]:slices_desc[0]] + (label_img_2[:,:,0:-1]==cluster)*2

    # Find middle slice in slices_two
    aortamask_segmented[:,:,slices_arch] = aortamask[:,:,slices_arch] * 2
    ##########
    return aortamask_segmented

  try:
    return do_the_segmentation(mask)
  except IndexError:
    # If aorta is the wrong way, then flip it around and try again
    return do_the_segmentation(mask[:,:,::-1])

def rog_driven_VOI_selection(SUV_segment: np.ndarray, N):
  _,_, nslices = SUV_segment.shape
  VOI = np.zeros_like(SUV_segment, dtype=bool)

  # Create slice profile in z-direction
  # Get median value of each slice
  slicemedian = np.zeros(nslices)
  for slc in range(nslices):
    SUV_slice = SUV_segment[:,:,slc]
    if SUV_slice.any():
      slicemedian[slc] = np.median(SUV_slice[SUV_slice>0])

  # Sliding window average of slice profile
  middleslc = np.argmax(np.convolve(slicemedian, np.ones(N)/N, mode='valid'))+N//2

  # Position 3x3xN VOI within segment with detected center slice
  for slc in range(middleslc-N//2,middleslc+N//2):
    SUV_slice = SUV_segment[:,:,slc]
    if SUV_slice.any():
      x0,y0 = roi.cog(SUV_slice)
      VOI[y0-1:y0+2,x0-1:x0+2,slc] = 1

  return VOI

def data_driven_VOI_selection(SUV_segment, voxdim):
  VOI = np.zeros_like(SUV_segment, dtype=bool)

  thr = 1000 // voxdim

  prc = 99.99
  volume = 0

  while volume <= thr:
    percentile = np.percentile(SUV_segment,prc)

    VOI = SUV_segment >= percentile

    label_img, nclusters = label(VOI, return_num=True)

    clustersize = np.zeros((nclusters,))

    for cluster in range(nclusters):
      clustersize[cluster] = np.sum(label_img==cluster+1)

    # Find largest cluster
    maxclusteridx = np.argmax(clustersize)+1

    VOI = label_img==maxclusteridx

    volume = clustersize[maxclusteridx-1]
    prc -= 0.5

  return VOI


# region IDIF CLASS
def calculate_idif(deck, suv, segmentation, voxdim):
  xdim,ydim, nslices, nframes = deck.shape

  ### Create VOI inside aorta arch of approx 1 mL ###
  # Allocate
  VOI = np.zeros((xdim,ydim,nslices,4), dtype=bool)
  idif = np.zeros((nframes,4))

  N = int(np.round((1000/(voxdim)*3*3))/2)*2
  #print(f"N: {N}")

  for seg in range(4):
    SUV_segment = suv * (segmentation == seg+1)

    if seg in [0,2,3]:
      VOI[:,:,:, seg] = rog_driven_VOI_selection(SUV_segment, N)
    else:
      VOI[:,:,:, seg] = data_driven_VOI_selection(SUV_segment, voxdim)

    ### Extract IDIF as the mean inside the VOI ###
    idif[:,seg] = np.mean(deck[np.squeeze(VOI[:,:,:,seg])], axis=0)
  return idif, VOI

class IDIF:
  def __init__(self, rt_struct: RTStruct, dynamic_pets: FramedDicomSeries):
    # Step 0 Get SUV

    x_vox_dim, y_vox_dim, z_vox_dim = dynamic_pets.pixel_volume
    voxdim = x_vox_dim * y_vox_dim * z_vox_dim

    pivot_dataset = dynamic_pets.datasets[0][0]
    total_dose = pivot_dataset.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose,
    patient_weight = pivot_dataset.PatientWeight

    suv_frames = suv_frame_selection(dynamic_pets)
    suv = calculate_suv_row_major(
      dynamic_pets.raw[suv_frames, :,:,:],
      total_dose,
      patient_weight
    )

    # Here I transform the coordinates back to
    SUV = transpose_nifti_coords(suv)
    deck = transpose_nifti_coords(dynamic_pets.image.raw)

    # Step 1 resample the segmentation to the pet space

    segmentation = get_mask(rt_struct, 'aorta')
    segmentation._raw = segmentation._raw.astype(np.uint8)

    resampled_segmentation = resample(segmentation, dynamic_pets.image.space)
    raw_mask = resampled_segmentation.raw

    median_suv = np.median(suv[raw_mask.astype(np.bool_)])

    row_major_mask = transpose_nifti_coords(raw_mask)

    aorta_preprocced = aorta_preproccesing(row_major_mask, SUV, median_suv)
    aortamask_segmented = aorta_segmentation(aorta_preprocced)


    # Allocate
    self.idif, self.VOI = calculate_idif(deck, suv, aortamask_segmented, voxdim)
  # End of IDIF __init__

  def __str__(self):
    return ""


def aorta_segment_row_major(aorta_mask: np.ndarray):
  """Segments the aorta such in four regions:
    * AORTA_DESC_LOWER = 1
    * AORTA_ARCH = 2
    * AORTA_DESC_UPPER = 3
    * AORTA_ASCENDENS = 4

    There's a pretty comment in the code to showcase where the different regions
    are located on the aorta. (which isn't here because formatting fuck it up)

  Args:
      aorta_mask (numpy.ndarray[Tuple[int,int,int], numpy.bool_]): _description_
  """

  # Also yeah I know that it's different to the docs to the aorta_segment, but
  # that's the result of their code...

  # Pretty comment:
  # The aorta seen from the side will look something like this.
  #
  #            ---
  #           /  2  \  Arch
  #          /-------\
  #         / 3  /\ 1 \   Desc
  #        /----/  \---\
  #        |   |
  #        |   |
  #        | 4 |  Asc
  #        |   |
  #        |   |
  #        |---|
  #
  # I take advantage of the fact that "All" aortas looks like this
  # Namely that


  # Get image dimensions
  number_of_slices, _ , _ = aorta_mask.shape

  # Allocate aortamask_segmented
  aorta_mask_segmented = aorta_mask.astype(int)

  # Loop over all axial slices and count number of clusters
  n_clusters = np.zeros((number_of_slices,),dtype=int)

  for slc in range(number_of_slices):
    _, n_clusters[slc] = label(aorta_mask[slc,:,:], return_num=True)

  # Correct
  n_clusters[n_clusters>2] = 2

  idx_pairs = find_true_sub_ranges(n_clusters==1)
  sorted_idx_pairs = sort_greatest_first(idx_pairs)

  asc_start, asc_end = sorted_idx_pairs[0]
  arch_start, arch_end = sorted_idx_pairs[1]

  desc_start = asc_end
  desc_end = arch_start

  range_asc = range(asc_start, asc_end)
  range_arch = range(arch_start, arch_end)

  # Aorta descendens is the biggest island of connected slices
  range_desc = ordered_range(desc_start, desc_end)

  # Find upper descending part of aorta by finding connection with lower part
  label_img_2, n_clusters = label(aorta_mask[range_desc], return_num=True)

  sizes = []

  for i in range(n_clusters):
    cluster = i + 1
    sizes.append(np.count_nonzero(label_img_2==cluster))

  cluster_1_size, cluster_2_size = sizes

  if cluster_1_size < cluster_2_size:
    label_img_2[np.where(label_img_2 == 1)] = AORTA_DESC_UPPER
    label_img_2[np.where(label_img_2 == 2)] = AORTA_DESC_LOWER
  else:
    label_img_2[np.where(label_img_2 == 1)] = AORTA_DESC_LOWER
    label_img_2[np.where(label_img_2 == 2)] = AORTA_DESC_UPPER

  aorta_mask_segmented[range_desc,:,:] = label_img_2
  aorta_mask_segmented[range_asc, :, :] = aorta_mask[range_asc,:,:] * 4
  aorta_mask_segmented[range_arch,:,:] = aorta_mask[range_arch,:,:] * 2
  ##########
  return aorta_mask_segmented


def print_contour(axes: Axes, axis, segmentation):
  axes.contourf(
    np.max(segmentation, axis=axis),
    alpha = 0.75,
    colors=CONTOUR_COLORS,
    levels=CONTOUR_LEVELS,
    antialiased=True
  )

# UPDATE THESE

ENV_VAR_ADDRESS_NAME = "PIPELINE_RT_ADDRESS"
ENV_VAR_PORT_NAME = "PIPELINE_RT_PORT"
ENV_VAR_AE_TITLE_NAME = "PIPELINE_RT_AE_TITLE"
ENV_VAR_SUV_FRAME_SELECTION = "PIPELINE_SUV_SELECTION"

SERVICE_ADDRESS = environ.get(ENV_VAR_ADDRESS_NAME)
SERVICE_PORT = environ.get(ENV_VAR_PORT_NAME)
SERVICE_AE_TITLE = environ.get(ENV_VAR_AE_TITLE_NAME)
#region Inputs
class RemoteRTConsumer(AbstractInput):
  image_grinder = RTStructGrinder()

  required_values = {
    0x0008_0016 : OptionsValidator([
      RTImageStorage, CTImageStorage
    ])
  }

  def __init__(self, options: AbstractInput.Options = AbstractInput.Options()):
    super().__init__(options)
    self.has_rt_struct = False
    self.has_send_datasets = False

  def add_image(self, dicom: Dataset) -> int:
    return_value = super().add_image(dicom)
    if return_value != 0 and dicom.SOPClassUID == RTImageStorage:
      self.has_rt_struct = True
    return return_value

  def validate(self) -> bool:
    ready = 1 < self.images and self.has_rt_struct
    if not ready and 0 < self.images and not self.has_send_datasets:
      self.has_send_datasets = True

      if SERVICE_ADDRESS is not None and\
         SERVICE_PORT is not None and\
         SERVICE_AE_TITLE is not None and\
         self.options.ae_title is not None:

        try:
          send_images(self.options.ae_title,
                      Address(SERVICE_ADDRESS, int(SERVICE_PORT),SERVICE_AE_TITLE),
                      self)
        except CouldNotCompleteDIMSEMessage:
          error_message = "Could send images with parameters:\n"\
                         f"IP:   {SERVICE_ADDRESS}\n"\
                         f"port: {SERVICE_PORT}\n"\
                         f"scp: {SERVICE_AE_TITLE}\n"\
                         f"scu: {self.options.ae_title}\n"
          self.logger.error(error_message)
      else:
        error_message = f"This Input is incorrectly configured the following "\
          "values should not be None:\n"\
          f"Environment Variable: {ENV_VAR_ADDRESS_NAME} - {SERVICE_ADDRESS}\n"\
          f"Environment Variable: {ENV_VAR_PORT_NAME} - {SERVICE_PORT}\n"\
          f"Environment Variable: {ENV_VAR_AE_TITLE_NAME} - {SERVICE_AE_TITLE}\n"\
          f"AE SCU: {self.options.ae_title}"
        self.logger.error(error_message)

    return ready

# If you do not remotely process I highly suggest you use Total Segmentator in
# the process function rather than the input processing to ensure GPU access
# As the process function can acquire the GPU as a resource while the Input
# Handler cannot!

class PETDynamicSeriesInput(AbstractInput):
  image_grinder = LargeDynamicPetSeriesGrinder()

  def validate(self):
    for dataset in self:
      total_images = dataset.NumberOfSlices * dataset.NumberOfTimeSlices
      return total_images == len(self)
    return False
#region Pipeline

RT_STRUCT_ARG_KW = "RT_STRUCT"
DYNAMIC_ARG_KW = "DYNAMIC"

all = [
  'aorta_segmentation',
  'IDIF'

]

if __name__ == '__main__':
  """python3 cbf_toolbox.py /raid/source/1906480944/1.2.752.24.5.554629087.20240828032323.23605341/1.3.12.2.1107.5.1.4.10006.30000024091206111368900400939 /franklyn/derivatives/1906480944/1.2.752.24.5.554629087.20240828032323.23605341/1.3.12.2.1107.5.1.4.10006.30000024091208301151200000355/totalsegmentator-v2/DICOM /franklyn/derivatives/1906480944/1.2.752.24.5.554629087.20240828032323.23605341/1.3.12.2.1107.5.1.4.10006.30000024091208301151200000355/totalsegmentator-v2/rtstruct/segmentations.dcm"""
  print("PLEASE DON'T USE THIS AS PART OF A SCRIPT, JUST IMPORT AND BE HAPPY")
  import argparse
  from dicomnode.lib.io import DicomLazyIterator
  from dicomnode.dicom.series import DicomSeries

  parser = argparse.ArgumentParser(
    "CBF_TOOLKIT_DEBUGING",
    description="This program is for testing if the cbf toolkit is working"\
      "correctly. Note that this program uses quite a bit of ram and processing")

  parser.add_argument("pet", type=Path, help="Path to the directory containing all the pet dicoms")
  parser.add_argument("ct", type=Path, help="Path to the directory containing all the ct dicoms, that the segmentation was constructed from")
  parser.add_argument("segmentation", type=Path, help="Path to the RT structure")

  namespace = parser.parse_args()

  pet_datasets = FramedDicomSeries(DicomLazyIterator(namespace.pet))
  ct_series = DicomSeries([ds for ds in DicomLazyIterator(namespace.ct)])

  rt_dataset = read_file(namespace.segmentation)

  rt_struct = RTStruct(ct_series.datasets, rt_dataset)

  idif = IDIF(
    rt_struct, pet_datasets
  )

  print(idif)
