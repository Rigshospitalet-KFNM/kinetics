# Python Standard Library
from logging import DEBUG
from pathlib import Path
import os
import argparse

# Third party imports
from pydicom import read_file
from pydicom.uid import CTImageStorage
from totalsegmentator.python_api import totalsegmentator

# Dicomnode imports 
from dicomnode.server.pipeline_tree import InputContainer
from dicomnode.server.output import PipelineOutput
from dicomnode.server.grinders import IdentityGrinder
from dicomnode.server.nodes import AbstractQueuedPipeline
from dicomnode.server.input import AbstractInput
from dicomnode.server.output import NoOutput

# Constants Setup
INPUT_KEY = "CT_IMAGES"


# Environment setup
ENV_NAME_NODE_PATH = "TOTAL_SEGMENTATOR_PATH"
node_path = Path(os.environ.get(ENV_NAME_NODE_PATH, os.getcwd() + "/total_segmentator_node"))


parser = argparse.ArgumentParser("Total segmentator pipeline, sets up a dicomnode for CT segmentation.",
                                 epilog="""Total segmentator is written by Jakob Wasserthal and is not intended for clinical usage
Please cite total segmentator at: https://pubs.rsna.org/doi/10.1148/ryai.230024""")

# Positional Arguments
parser.add_argument('--dry-run', action='store_true', default=False, help="Executes this script without opening a server")
#

args = parser.parse_args()


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


class TotalSegmentatorPipeline(AbstractQueuedPipeline):
  input = {
    INPUT_KEY : TotalSegmentatorInput
  }

  ae_title = "TOTAL_SEG"
  data_directory = node_path
  port = 11112
  log_level = DEBUG

  def process(self, input_data: InputContainer) -> PipelineOutput:
    if input_data.paths is None:
      raise Exception("This can never happen!")

    input_path = input_data.paths[INPUT_KEY]
    output_path = input_path.parent / "output"

    totalsegmentator(input_path, output_path, output_type="dicom", device="gpu", quiet=True)

    output_file = output_path / "segmentations.dcm"

    print(read_file(output_file))

    return NoOutput()


if __name__ == "__main__":
  if not args.dry_run:
    pipeline = TotalSegmentatorPipeline()
    pipeline.open()
  else:
    print("Not running")
