#!/bin/bash
#SBATCH --partition=gpu-shared
#SBATCH --account=ddp433
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --job-name=calib_qrc_resume
#SBATCH --output=logs/calib_qrc_resume_%j.log
#SBATCH --error=logs/calib_qrc_resume_%j.err

module load cuda/12.1
source ~/.bashrc
conda activate qrc  

echo "Job starting: $(date)"
python eval_calibrated_qrc_resume.py
echo "Job completed: $(date)"
