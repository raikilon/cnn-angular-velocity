#!/bin/bash -l
#SBATCH --job-name=RoboticsProject
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=log_%j.out
#SBATCH --error=log_%j.err
#SBATCH --partition=gpu
#SBATCH --mem=40000
#SBATCH --exclusive
# your commands
module load cudatoolkit/9.0.176
module load cudnn/5.0.5
export LC_ALL=en_US.utf-8
export LANG=en_US.utf-8

$HOME/libs/bin/python3.5 -m pip install numpy --user
$HOME/libs/bin/python3.5 -m pip install torch==1.1.0 -f https://download.pytorch.org/whl/cu90/torch_stable.html --user
$HOME/libs/bin/python3.5 -m pip install torchvision==0.3.0 -f https://download.pytorch.org/whl/cu90/torch_stable.html --user
$HOME/libs/bin/python3.5 -m pip install scikit-image --user
$HOME/libs/bin/python3.5 -m pip install tensorboardX --user
$HOME/libs/bin/python3.5 -m pip install wandb --user
$HOME/libs/bin/python3.5 -m pip install sklearn --user
$HOME/libs/bin/python3.5 -m pip install pandas --user
$HOME/libs/bin/python3.5 -m pip install mlxtend==0.16.0 --user


wandb login e0416b77a0b46deaa6ba3e2ac62fba762e87a599

$HOME/libs/bin/python3.5 train_model.py data

