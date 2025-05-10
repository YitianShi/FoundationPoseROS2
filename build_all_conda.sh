#!/bin/bash
source /opt/ros/humble/setup.bash
export PATH=/usr/local/cuda-12.4/bin/${PATH:+:${PATH}}~

PROJ_ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Install dependencies
pip install torchvision==0.19.1 torchaudio torch==2.4.1
MAX_JOBS=20 pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
python -m pip install -r requirements.txt

# Clone source repository of FoundationPose
git clone git@gitlab.kit.edu:kit/ifl/gruppen/air/algorithm/FoundationPose.git

# Create the weights directory and download the pretrained weights from FoundationPose
gdown --folder https://drive.google.com/drive/folders/1BEQLZH69UO5EOfah-K9bfI3JyP9Hf7wC -O FoundationPose/weights/2023-10-28-18-33-37 
gdown --folder https://drive.google.com/drive/folders/12Te_3TELLes5cim1d7F7EBTwUSe7iRBj -O FoundationPose/weights/2024-01-11-20-02-45

# conda install cmake

# Install pybind11
cd ${PROJ_ROOT}/FoundationPose && git clone https://github.com/pybind/pybind11 && \
    cd pybind11 && git checkout v2.10.0 && \
    mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DPYBIND11_INSTALL=ON -DPYBIND11_TEST=OFF && \
    sudo make -j6 && sudo make install

# Install Eigen
cd ${PROJ_ROOT}/FoundationPose && wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz && \
    tar xvzf ./eigen-3.4.0.tar.gz && rm ./eigen-3.4.0.tar.gz && \
    cd eigen-3.4.0 && \
    mkdir build && \
    cd build && \
    cmake .. && \
    sudo make install

# Clone and install nvdiffrast
cd ${PROJ_ROOT}/FoundationPose && git clone https://github.com/NVlabs/nvdiffrast && \
    cd /nvdiffrast && pip install .

# Install mycpp
cd ${PROJ_ROOT}/FoundationPose/mycpp/ && \
rm -rf build && mkdir -p build && cd build && \
cmake .. && \
sudo make -j$(nproc)

# Install mycuda
# !! DONT FORGET TO CHANGE THE C++ DEPENDENCY !! {PROJ_ROOT}/FoundationPose/bundlesdf/mycuda/setup.py c++14 -> c++17
cd ${PROJ_ROOT}/FoundationPose/bundlesdf/mycuda && \
rm -rf build *egg* *.so && \
python3 -m pip install -e .

cd ${PROJ_ROOT}

git clone --recursive https://github.com/geopavlakos/hamer.git
pip install -e hamer[all]
pip install -v -e hamer/third-party/ViTPose
bash hamer/fetch_demo_data.sh
rm -f hamer_demo_data.tar.gz

git clone https://github.com/lixiny/manotorch.git
pip install -e manotorch

# Define color codes
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Now you need to download the MANO_RIGHT.pkl file from the following link and put it in the _DATA/data/mano directory: https://mano.is.tue.mpg.de/${NC}"
echo -e "${YELLOW}Replace vitpose_model.py line 15 with: current_dir = os.path.dirname(__file__); VIT_DIR = os.path.join(current_dir, "third-party/ViTPose")"
