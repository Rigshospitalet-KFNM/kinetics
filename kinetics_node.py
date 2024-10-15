# Python Standard Library
from logging import DEBUG
from pathlib import Path
from os import environ
import argparse
from typing import Literal, Tuple

# Third party imports
import numpy
from pydicom import read_file, Dataset
from pydicom.uid import RTImageStorage, CTImageStorage
from rt_utils import RTStruct
from skimage.measure import label


# Dicomnode imports
from dicomnode.lib.exceptions import CouldNotCompleteDIMSEMessage
from dicomnode.lib.validators import OptionsValidator
from dicomnode.dicom.rt_structs import get_mask
from dicomnode.dicom.dimse import Address, send_images
from dicomnode.dicom.series import LargeDynamicPetSeries
from dicomnode.math import bounding_box, center_of_gravity
from dicomnode.server.pipeline_tree import InputContainer
from dicomnode.server.output import PipelineOutput
from dicomnode.server.grinders import RTStructGrinder, LargeDynamicPetSeriesGrinder
from dicomnode.server.nodes import AbstractQueuedPipeline
from dicomnode.server.input import AbstractInput
from dicomnode.server.output import NoOutput

#region Environment
ENV_VAR_ADDRESS_NAME = "PIPELINE_RT_ADDRESS"
ENV_VAR_PORT_NAME = "PIPELINE_RT_PORT"
ENV_VAR_AE_TITLE_NAME = "PIPELINE_RT_AE_TITLE"
ENV_VAR_SUV_FRAME_SELECTION = "PIPELINE_SUV_SELECTION"

SERVICE_ADDRESS = environ.get(ENV_VAR_ADDRESS_NAME)
SERVICE_PORT = environ.get(ENV_VAR_PORT_NAME)
SERVICE_AE_TITLE = environ.get(ENV_VAR_AE_TITLE_NAME)

def find_true_sub_ranges(bools: numpy.ndarray[Tuple[int], numpy.bool_])\
    -> numpy.ndarray[Tuple[int, Literal[2]], int]:
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

  return numpy.where(
      numpy.diff(
        numpy.hstack(
          ([False],bools,[False])
        )
      )
    )[0].reshape(-1,2)

def get_longest_sub_range(ranges: numpy.ndarray[Tuple[int, Literal[2]], int])\
    -> numpy.ndarray[Tuple[Literal[2]], int]:
  """Finds the longest sub

  Returns:
      _type_: _description_
  """
  return ranges[numpy.diff(ranges, axis=1).argmax()]

def suv_frame_selection(pet_series: LargeDynamicPetSeries):
  frame_start_time = pet_series.frame_acquisition_time - pet_series.frame_acquisition_time[0]
  return frame_start_time < numpy.datetime64(40, 's')

def calculate_suv(image: numpy.ndarray, activity_Bq, patient_weight_kg):
  if len(image.shape) == 4:
    return image.mean(axis=0) * patient_weight_kg * 1000  / activity_Bq
  return image * patient_weight_kg * 1000 / activity_Bq

def handle_multiple_clusters(mask, labeled_mask, number_of_clusters):
  return mask

def aorta_segment(aorta_mask: numpy.ndarray[Tuple[int,int,int], numpy.bool_]):
     # Get image dimensions
    number_of_slices,_ , _ = aorta_mask.shape

    # Allocate aortamask_segmented
    aorta_mask_segmented = aorta_mask.astype(int)

    # Loop over all axial slices and count number of clusters
    n_clusters = numpy.zeros((number_of_slices,),dtype=int)

    for slc in range(number_of_slices):
      _, n_clusters[slc] = label(aorta_mask[slc,:,:], return_num=True)

    # Correct
    n_clusters[n_clusters>2] = 2

    idx_pairs = find_true_sub_ranges(n_clusters==1)

    # Get the island lengths, whose argmax would give us the ID of longest island.
    # Start index of that island would be the desired output
    start_longest_seq_1, _ = get_longest_sub_range(idx_pairs)

    # Aorta descendens is the biggest island of connected slices
    slices_desc = range(start_longest_seq_1, number_of_slices)

    # Segment aorta descendens lower
    aorta_mask_segmented[slices_desc,:,:] = aorta_mask[slices_desc,:,:] * 4

    # Aorta arch is the biggest island of connected twos
    n_clusters_twos = n_clusters
    n_clusters_twos[n_clusters_twos==1] = 0

    # Get start, stop index pairs for islands/seq. of 1s
    idx_pairs = find_true_sub_ranges(n_clusters==2)

    # Get the island lengths, whose argmax would give us the ID of longest island.
    # Start index of that island would be the desired output
    start_longest_seq_2 = find_true_sub_ranges(idx_pairs)
    slices_arch = range(0,start_longest_seq_2)
    slices_two = range(start_longest_seq_2,start_longest_seq_1)

    # Find upper descending part of aorta by finding connection with lower part
    label_img_2, n_clusters = label(
      aorta_mask[slices_two[0]:slices_desc[0]+1,:,:],
      return_num=True
    )

    for cluster in range(1,n_clusters+1):
      if numpy.sum(
        (label_img_2==cluster) * (aorta_mask_segmented[slices_two[0]:slices_desc[0]+1, :,:]==4)
      ):
        aorta_mask_segmented[slices_two[0]:slices_desc[0], :,:] = \
          aorta_mask_segmented[slices_two[0]:slices_desc[0], :,:] + \
            (label_img_2[0:-1,:,:]==cluster) * 2


    aorta_mask_segmented[slices_arch,:,:] = aorta_mask[slices_arch,:,:] * 2
    ##########
    return aorta_mask_segmented


def processing(rt_struct: RTStruct, dynamic_pet_series: LargeDynamicPetSeries):
  aorta_mask = get_mask(rt_struct, "aorta")

  suv_frames = suv_frame_selection(dynamic_pet_series)
  suv = calculate_suv(
    dynamic_pet_series.raw[suv_frames, :,:,:],
    dynamic_pet_series.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose,
    dynamic_pet_series.PatientWeight
  )

  median_suv = numpy.median(suv[aorta_mask])
  #
  aorta_mask = aorta_mask * numpy.int8(suv > median_suv / 1.5)
  labeled_mask, number_of_clusters = label(aorta_mask, return_num=True)

  if 1 < number_of_clusters:
    aorta_mask = handle_multiple_clusters(aorta_mask, labeled_mask, number_of_clusters)

  segments_names = ['Aorta asc', 'Aortic arch', 'Aorta desc (upper)', 'Aorta desc (lower)']
  segments = len(segments_names)
  aorta_mask_segmented = aorta_segment(aorta_mask)

  (z_min, z_max),(y_min, y_max),(x_min, x_max) = bounding_box(aorta_mask_segmented)
  threshold = 1000 // numpy.prod(dynamic_pet_series.pixel_volume)

  number_of_slices = dynamic_pet_series.NumberOfSlices
  volume_of_interest = numpy.zeros((segments,
                                    number_of_slices,
                                    dynamic_pet_series.Rows,
                                    dynamic_pet_series.Cols), dtype=numpy.bool_)

  input_derived_image_function = numpy.zeros((number_of_slices, segments))

  # Wat?
  N = int(numpy.round(
    (1000 / (numpy.prod(dynamic_pet_series.pixel_volume) * 3 * 3)) / 2
  ) * 2)

  for segment_index in range(segments):
    segment_id = segment_index + 1
    if segment_index != 1:
      slice_median = numpy.zeros(number_of_slices)
      for slice_ in range(number_of_slices):
        if numpy.sum(aorta_mask_segmented[slice_,:,:] == segment_id):
           SUV_segment = suv[slice_,:,:]*(aorta_mask_segmented[slice_:,:]== segment_id)
          slice_median[slice_] = numpy.median(SUV_segment[SUV_segment>0])

      # Sliding window average of slice profile
      middle_slice = numpy.argmax(
        numpy.convolve(slice_median, numpy.ones(N)/N, mode='valid')
      ) + N // 2

      # Position 3x3xN VOI within segment with detected center slice
      for slice_ in range(middle_slice-N//2,middle_slice+N//2):
          if numpy.sum(aorta_mask_segmented[slice_,:,:] == segment_id):
              x0,y0 = center_of_gravity(suv[slice_,:,:]*(aorta_mask_segmented[slice_,:,:] == segment_id))
              volume_of_interest[segment_index,slice_,y0-1:y0+2,x0-1:x0+2] = 1


#region Inputs
class RTConsumer(AbstractInput):
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

class PETDynamicSeriesInput(AbstractInput):
  required_tags = LargeDynamicPetSeries.REQUIRED_TAGS

  image_grinder = LargeDynamicPetSeriesGrinder()

  def validate(self):
    for dataset in self:
      total_images = dataset.NumberOfSlices * dataset.NumberOfTimeSlices
      return total_images == len(self)
    return False
#region Pipeline

RT_STRUCT_ARG_KW = "RT_STRUCT"
DYNAMIC_ARG_KW = "DYNAMIC"


class KineticPipeline(AbstractQueuedPipeline):
  ae_title = "KINETICDIAMOX"
  port = 11112

  input = {
    RT_STRUCT_ARG_KW : RTConsumer,
    DYNAMIC_ARG_KW : PETDynamicSeriesInput,
  }

  def process(self, input_container: InputContainer):
    rt_struct: RTStruct = input_container[RT_STRUCT_ARG_KW]
    dynamic_series: LargeDynamicPetSeries = input_container[DYNAMIC_ARG_KW]

    processing(rt_struct, dynamic_series)

    return NoOutput()


#region main
if __name__ == '__main__':
  print("running")