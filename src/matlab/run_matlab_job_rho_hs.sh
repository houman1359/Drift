#!/bin/bash
#SBATCH --job-name=drift_rho
#SBATCH --output=logs/matlab_output_%j.log
#SBATCH --error=logs/matlab_error_%j.log
#SBATCH --time=4:00:00
#SBATCH --partition=shared
#SBATCH --mem=20G
#SBATCH --cpus-per-task=8

module load matlab/R2024b-fasrc01

cd /n/holylabs/kempner_dev/Users/hsafaai/Code/Drift/FromFarhad

matlab -nodisplay -nosplash -nodesktop -r "noisecorr_run; exit"

