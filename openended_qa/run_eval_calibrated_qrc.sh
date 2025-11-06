#!/bin/bash
#SBATCH --partition=gpu-shared
#SBATCH --account=ddp433
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --job-name=calib_qrc
#SBATCH --output=logs/calib_qrc_%j.log
#SBATCH --error=logs/calib_qrc_%j.err

echo "Job starting: $(date)"
source ~/.bashrc
conda activate qrc

python eval_calibrated_qrc.py

echo "Job completed: $(date)"
