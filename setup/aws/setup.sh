#!/bin/bash

set -ex

sudo apt-get update && \
    DEBIAN_FRONTEND=noninteractive sudo apt-get install -y python3.8 python3.8-venv python3.8-dev cmake git git-lfs clang

echo "doing gpu setup"
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt-get update
sudo apt-get install -y cuda-libraries-11-4 cuda-command-line-tools-11-4 nvidia-headless-470 nvidia-utils-470 nvidia-modprobe

printf "Please copy the libcudnn deb file, then press ENTER...\n(https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.2.4/11.4_20210831/Ubuntu18_04-x64/libcudnn8_8.2.4.15-1+cuda11.4_amd64.deb)"; read
sudo dpkg -i $HOME/libcudnn8_8.2.4.15-1+cuda11.4_amd64.deb

mkdir -p $HOME/Projects

echo "Please enter the clone command for the repository (will cd into ~/Projects before)"
read clone_command
(cd $HOME/Projects/ && eval $clone_command)

cd $HOME/Projects/forennsic
python3.8 -m venv venv --prompt forennsic
source venv/bin/activate

# the AMI comes with python3.8 and pip 9.x, first upgrade pip so we can actually install tensorflow
pip install --upgrade pip

# manually install wheel first
pip install wheel
git submodule update --init --recursive
pip install -e innfrastructure
pip install -r requirements.txt

# pull git lfs objects
git lfs pull -X experiments/results

# ensure everything works
(\
cd $HOME/Projects/forennsic/experiments; \
source ../venv/bin/activate; \
innfcli predict cifar10_medium data/datasets/cifar10/samples.npy
)
# cd experiments
# innfcli
