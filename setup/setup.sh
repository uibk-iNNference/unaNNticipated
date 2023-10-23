#!/bin/bash

set -ex

GPU_SETUP=0

while getopts ":g" opt; do
    case $opt in
        g) GPU_SETUP=1;
    esac
done

sudo apt-get update && \
    DEBIAN_FRONTEND=noninteractive sudo apt-get install -y python3 python3-venv python3-dev cmake git clang valgrind

if [ $GPU_SETUP -gt 0 ]; then
    echo "doing gpu setup"
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
    sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
    sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
    sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC # install cudatools@nvidia.com key
    sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
    sudo apt-get update
    sudo apt-get install -y cuda-libraries-11-4 cuda-command-line-tools-11-4 nvidia-headless-470 nvidia-utils-470 nvidia-modprobe
    sudo dpkg -i /mnt/repo/libcudnn8_8.2.4.15-1+cuda11.4_amd64.deb
fi

mkdir -p $HOME/Projects
cp -r /mnt/repo/forennsic $HOME/Projects/forennsic

cd $HOME/Projects/forennsic
python3 -m venv venv --prompt forennsic
source venv/bin/activate

# manually install wheel first
pip install wheel
pip install -e innfrastructure
pip install -r requirements.txt
