#!/bin/bash
#SBATCH -A p_scads            
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8         
#SBATCH --ntasks=1         
#SBATCH --time=16:00:00
#SBATCH -J "d-t-mrtb-nd"
#SBATCH --output=/scratch/ws/trng859b-dme/log-mrtb-no_dist-%j.out
#SBATCH --mem=16192

OUTFILE=""

srun python3 main.py with dme.evenly_distributed=False 'dme.extras=["mrtb"]' dme.test_all=False dme.use_validation=True

exit 0
