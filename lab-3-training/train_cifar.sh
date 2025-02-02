#!/usr/bin/env bash
#SBATCH --partition gpu
#SBATCH --time 0-00:30
#SBATCH --account comsm0045
#SBATCH --reservation comsm0045-lab3
#SBATCH --mem 64GB
#SBATCH --gres gpu:1

# get rid of any modules already loaded
module purge
# load in the module dependencies for this script
module load "languages/anaconda3/2019.07-3.6.5-tflow-1.14"

python train_cifar.py --sgd-momentum 0
python train_cifar.py --sgd-momentum 0.9
python train_cifar.py --batch-norm True --sgd-momentum 0
python train_cifar.py --batch-norm True --sgd-momentum 0.9
python train_cifar.py --learning-rate 1e-2 --sgd-momentum 0
python train_cifar.py --learning-rate 1e-2 --sgd-momentum 0.9
python train_cifar.py --learning-rate 1e-2 --batch-norm True --sgd-momentum 0
python train_cifar.py --learning-rate 1e-2 --batch-norm True --sgd-momentum 0.9
