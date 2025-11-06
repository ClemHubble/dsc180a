#!/bin/bash
#SBATCH --partition=gpu-shared
#SBATCH --account=ddp433
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --job-name=qrc_openqa_fast
#SBATCH --output=logs/qrc_openqa_fast_%j.out
#SBATCH --error=logs/qrc_openqa_fast_%j.err

source ~/.bashrc
conda activate qrc
cd ~/llama_qrc_project

echo "Job starting: $(date)"
python qrc_openqa_eval_fast.py
echo "Job completed: $(date)"
