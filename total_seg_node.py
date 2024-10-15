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
ENV_NAME_NODE_PATH = "TOTAL_SEGMENTATOR_PATH"
node_path = Path(os.environ.get(ENV_NAME_NODE_PATH, os.getcwd() + "/total_segmentator_node"))
output_path = Path(os.getcwd()) / "outputs"

def SIGUSR1_handler(signum, frane):
  for thread in threading.enumerate():
    print(thread._target)
    if thread._target is not None:
      print(inspect.getsourcelines(thread._target))

signal.signal(signal.SIGUSR1, SIGUSR1_handler)

class TotalSegmentatorInput(AbstractInput):
  image_grinder = IdentityGrinder()

  required_tags = [
    0x00200013
  ]

  required_values = {
    0x00080016 : CTImageStorage,
    0x00080060 : 'CT',
  }

  def validate(self) -> bool:
    if self.images == 0:
      return False

    for image in self:
      if self.images < image.InstanceNumber:
        return False

    return True

destination_address = Address('10.146.12.194', 11112, "KINETICDIAMOX")


class TotalSegmentatorPipeline(AbstractQueuedPipeline):
  input = {
    INPUT_KEY : TotalSegmentatorInput
  }

  ae_title = "TOTALSEG"
  data_directory = node_path
  port = 11113

  def process(self, input_data: InputContainer) -> PipelineOutput:
    if input_data.paths is None:
      raise Exception("This can never happen!")

    input_path = input_data.paths[INPUT_KEY]
    #output_path = input_path.parent / "output"

    totalsegmentator(input_path,
                     output_path,
                     output_type="dicom",
                     device="gpu",
                     quiet=True,
                     body_seg=True,
                     ml=True,
    )

    output_dataset = read_file(output_path / "segmentations.dcm")
    return NoOutput()
    #return DicomOutput([(destination_address, output_dataset)],self.ae_title)


if __name__ == "__main__":
  pipeline = TotalSegmentatorPipeline()
  pipeline.open()