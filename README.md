# Kinetic

This library is process for running the kinetic pipeline. Which previously
have been a series of scrips being execute. Now the goal is that process should
be fully automated.

This is achieved with the library dicomnode, which is a end2end package for
creating dicom endpoints.

The main file and server is `kinetic_node.py` and the directory `kinetic`
contains all of the used