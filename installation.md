<p align="center">
  <img src="logo.png" width="35%"></img>
</p>

-----
# ISTHMUS: The marching windows between voxels and surface mesh

## Software Description

## System Requirements

## Installation


- conda env for marching cubes as Vijay's steps
    contains: trimesh, lazy_loader, numpy, Cython, scipy

- conda env create -n <new env name> -f environment.yml

- compile using 
    python -W ignore setup.py build_ext --inplace

- add src folder to PYTHONPATH

- for gpu:
    - Test using check_numba_cuda.sh
    - From scratch:
        - install cuda-toolkit depending on your system from nividia website
        - conda install numba cudatoolkit pyculib
    - use a complete conda env:
        - might run out of disk quota. Change pkgs fldr to another location:
        "conda config --add pkgs_dirs /your/custom/path"
        clean using "conda clean -a" 
    - HPC (LCC):
        - module load ccs/cuda/12.2.0_535.54.03
        - /opt/ohpc/pub/libs/conda/env/numba-0.54.0
