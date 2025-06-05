#! /usr/bin/env python3

if __name__ != '__main__':
  print("This is a script dont import it :(")
  exit(1)

import os
from pathlib import Path
import argparse

parser = argparse.ArgumentParser("select_the_n_first_frames")
parser.add_argument('frames', type=int)
parser.add_argument("output", type=Path)
parser.add_argument("--frames-from", type=int, default=0)

namespace = parser.parse_args()

frames = namespace.frames

if frames < 1:
  print("You can't select less than 1 frame")
  exit(1)

output_path: Path = namespace.output

virtual_path = Path(os.environ['VIRTUAL_ENV'])
data_folder = virtual_path.parent / "data" / "2804873069" / "1.3.51.0.1.1.10.143.20.159.19475642.12758627" / "1.3.12.2.1107.5.1.4.10006.30000022102710151772000044648"

if not output_path.exists():
  output_path.mkdir()

dicom_path = output_path / 'dicom'

if not dicom_path.exists():
  dicom_path.mkdir()

from dicomnode.lib.io import save_dicom, DicomLazyIterator

for ds in DicomLazyIterator(data_folder):
  if 'NumberOfSlices' not in ds or 'ImageIndex' not in ds and 'NumberOfTimeSlices' not in ds:
    raise Exception("The dicom files are not a PET series?")
  if ds.NumberOfTimeSlices < frames:
    raise Exception(f"You can't require more than {ds.NumberOfTimeSlices} frames for this series!")

  series_number = (ds.ImageIndex - 1)  // ds.NumberOfSlices

  if namespace.frames_from <= series_number and series_number < frames + namespace.frames_from:
    new_index = ds.ImageIndex - (ds.NumberOfSlices * namespace.frames_from)

    ds.NumberOfTimeSlices = frames
    ds.InstanceNumber = new_index
    ds.ImageIndex = new_index

    save_dicom(
      dicom_path / f"{ds.ImageIndex}.dcm",
      ds
    )
