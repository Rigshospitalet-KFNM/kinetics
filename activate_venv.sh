#!/bin/bash

virtuel_env=".venv"

if [ ! -d "$virtuel_env" ]; then
  python3 -m venv .venv

  pip install git+https://github.com/Rigshospitalet-KFNM/DicomNode.git
fi

source .venv/bin/activate
