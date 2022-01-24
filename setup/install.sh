#!/usr/bin/env bash
set -e
CWD="$(pwd)"
ENV_NAME=""   # name of your conda environment
PYTHON_VER="" # python version being used i.e. 3.6, 3.7, 3.8, etc.
CONDA_SCRIPT="$HOME/miniconda3/etc/profile.d/conda.sh"
source $CONDA_SCRIPT # to enable activating conda env

# use this only if you need to create a new conda environment
# conda create -n "$ENV_NAME" python="$PYTHON_VER" -y
conda activate "$ENV_NAME" &&
    conda install -y pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia

# conda install -c conda-forge -y tensorflow keras
conda install -c conda-forge -y jupyterlab ipykernel nb_black numpy matplotlib

sudo apt-get install -y sumo sumo-tools sumo-doc
python3 -m pip install -r setup/requirements.txt

cd CWD
# install sumo-rl
if [ -d "sumo-rl" ]; then
    echo "sumo-rl already exists!"
else
    git clone git@github.com:ChristoAdis/sumo-rl.git
fi
cd sumo-rl && pip install -e .

cd CWD
# install ray
pip install -U https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-2.0.0.dev0-cp38-cp38-manylinux2014_x86_64.whl &&
    git clone https://github.com/ray-project/ray.git &&
    cd ray &&
    python python/ray/setup-dev.py
