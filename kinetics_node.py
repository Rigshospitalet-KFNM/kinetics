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


