#!/bin/bash
#SBATCH --job-name=drif
#SBATCH --output=logs/matlab_output_%j.log
#SBATCH --error=logs/matlab_error_%j.log
#SBATCH --time=2:00:00
#SBATCH --partition=shared
#SBATCH --mem=20G
#SBATCH --cpus-per-task=8

# Load MATLAB module
module load matlab/R2024b-fasrc01

# Change to the script directory
cd /n/holylabs/kempner_dev/Users/hsafaai/Code/Drift/FromFarhad

# Run MATLAB script
# matlab -nodisplay -nosplash -nodesktop -r "run_vs_dim; exit"
# matlab -nodisplay -nosplash -nodesktop -r "run_vs_dim; exit"
matlab -nodisplay -nosplash -nodesktop -r "inputdim_run; exit"