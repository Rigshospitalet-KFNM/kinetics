# Python Standard Library
from logging import DEBUG
from pathlib import Path
import threading
import os
import signal
import tqdm
import inspect

# Third party imports
from pydicom import read_file
from rt_utils import RTStruct
from pydicom.uid import CTImageStorage
from totalsegmentator.python_api import totalsegmentator

# Dicomnode imports
from dicomnode.dicom.dimse import Address
from dicomnode.server.pipeline_tree import InputContainer
from dicomnode.server.output import PipelineOutput
from dicomnode.server.grinders import IdentityGrinder
from dicomnode.server.nodes import AbstractQueuedPipeline
from dicomnode.server.input import AbstractInput
from dicomnode.server.output import DicomOutput, NoOutput

# God fucking dammit
tqdm.tqdm.monitor_interval = 0

# Constants Setup
INPUT_KEY = "CT_IMAGES"

# Environment setup
TOTAL_SEG_ENV_PORT = "TOTAL_SEG_ENV_PORT"
TOTAL_SEG_ENV_AE_TITLE = "TOTAL_SEG_ENV_AE_TITLE"
TOTAL_SEG_ENV_INPUT = "TOTAL_SEG_ENV_INPUT"
TOTAL_SEG_ENV_OUTPUT = "TOTAL_SEG_ENV_OUTPUT"

input_node_path = Path(os.environ.get(TOTAL_SEG_ENV_INPUT, os.getcwd() + '/inputs'))
#output_node_path = Path(os.environ.get(TOTAL_SEG_ENV_OUTPUT, os.getcwd() + '/output'))

node_port = int(os.environ.get(TOTAL_SEG_ENV_PORT, 42069))
node_ae_title = os.environ.get(TOTAL_SEG_ENV_AE_TITLE, "TOTALSEG")

class TotalSegmentatorInput(AbstractInput):
  image_grinder = IdentityGrinder()

  required_tags = [
    0x0020_0013
  ]

  required_values = {
    0x0008_0016 : CTImageStorage,
    0x0008_0060 : 'CT',
  }

  def validate(self) -> bool:
    if self.images == 0:
      return False

    max_instance_number = -1

    for image in self:
      max_instance_number = max(max_instance_number, image.InstanceNumber)

    return max_instance_number - 1 == self.images

destination_address = Address('172.16.82.175', 11112, "KINETICDIAMOX")


class TotalSegmentatorPipeline(AbstractQueuedPipeline):
  input = {
    INPUT_KEY : TotalSegmentatorInput
  }

  ae_title = node_ae_title
  data_directory = input_node_path
  port = node_port

  def process(self, input_data: InputContainer) -> PipelineOutput:
    if input_data.paths is None:
      raise Exception("This can never happen!")

    destination = None

    for source_dataset in input_data.datasets.values():
      if 0x1337_0101 in source_dataset and 0x1337_0102 in source_dataset and 0x1337_0103:
        destination = Address(
          source_dataset[0x1337_0101].value,
          source_dataset[0x1337_0102].value,
          source_dataset[0x1337_0103].value,
        )
      break
    if destination is None:
        destination = destination_address

    input_path = input_data.paths[INPUT_KEY]
    output_path = input_path.parent / "output"

    totalsegmentator(input_path,
                     output_path,
                     output_type="dicom",
                     device="gpu",
                     quiet=True,
                     body_seg=True,
                     ml=True,
    )

    output_dataset = read_file(output_path / "segmentations.dcm")

    self.logger.info(f"Sending segmentation to {destination.ae_title} ({destination.ip}:{destination.port})")

    return DicomOutput([(destination, [output_dataset])],self.ae_title)


if __name__ == "__main__":
  pipeline = TotalSegmentatorPipeline()
  pipeline.open()