#!/usr/bin/env bash

PYTHON_VER="3.8"

read -p "Create a conda environment called 'sumo-rl'(y/n)? " answer
case ${answer:0:1} in
    y|Y )
        echo Creating conda environment 'sumo-rl'
        conda create -n sumo-rl -y python=$PYTHON_VER

    ;;
    * )
        echo Skipping environment creation ...
    ;;
esac

pip install stable-baselines3
pip install pettingzoo supersuit psutil

conda install -y -c conda-forge gym numpy pandas
conda install -y -c pytorch -c conda-forge pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=11.3

echo "Done."
