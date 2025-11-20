#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --job-name=qrc_grid
#SBATCH --output=logs/qrc_%j.out
#SBATCH --error=logs/qrc_%j.err

mkdir -p logs

echo "========== Job Info =========="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "=============================="

# OPTIONAL: Activate a conda env (user must fill in)
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate myenv

echo "Python version:"
python --version

echo "Starting experiment..."
START=$(date +%s)

python qrc_with_heatmap.py
STATUS=$?

END=$(date +%s)
RUNTIME=$((END - START))

echo "Finished with code $STATUS"
echo "Runtime: ${RUNTIME}s"

# Print result summary
if [ -d "results" ]; then
    echo "Generated results files:"
    ls -lh results
fi

exit $STATUS
