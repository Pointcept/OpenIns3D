
# Installation


Here we provide the installation guide for the entire OpenIns3D pipeline, including Mask, Snap, and Lookup modules.

## Requirements

- CUDA: 11.6
- PyTorch: 11.3
- Hardware: one 24G memory GPU is sufficent to produce all results

## Setup 

Install dependencies by running:

### Snap Module
```bash
conda create -n openins3d python=3.9
conda activate openins3d

conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
pip install mkl==2024.0.0
conda install pytorch3d -c pytorch3d
pip install gdown plyfile pandas

pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.13.1%2Bcu116.html 
# if you have problem for torch-scatter visit https://pytorch-geometric.com/whl/ and choose a version that fit your current torch and cuda version.
# Up here is sufficent for snap module.

```
### Lookup Module

```bash

# we then need to install off-shell module for 2D detectors, here we choose two methods, ODISE and Yoloworld.

# installation of YOLOWORLD
pip install -U openmim 
mim install mmcv==2.0.0
mim install mmdet==3.0.0
mim install mmengine==0.10.3
mim install mmyolo==0.6.0 
pip install supervision==0.18.0
pip install transformers==4.38.1

# install odise
export CPATH=/usr/local/cuda/include:$CPATH
conda install lightning -c conda-forge
conda install -c "nvidia/label/cuda-11.6.1" libcusolver-dev
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
cd third_party
git clone https://github.com/NVlabs/ODISE.git
cd ODISE
pip install -e .
cd ../..
# Up here is sufficent for Snap-Lookup, all results in the paper can be reproduced with this installation
```

### Mask Module

```bash

export CPATH=/usr/local/cuda/include:$CPATH
# Install Mask module to produce mask
conda install nltk
cd third_party/pointnet2
python setup.py install
cd ../

# install MinkowskiEngine for MPM
git clone --recursive "https://github.com/NVIDIA/MinkowskiEngine" # clone the repo to third_party
cd MinkowskiEngine
git checkout 02fc608bea4c0549b0a7b00ca1bf15dee4a0b228
python setup.py install --force_cuda --blas=openblas
pip install hydra-core loguru albumentations open3d
cd ../../

# Up here you have the full pipeline of OpenIns3D. you can now do the zero-shot inference on your own data
```