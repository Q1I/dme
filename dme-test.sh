#!/bin/bash
#SBATCH -A p_scads            
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8         
#SBATCH --ntasks=1         
#SBATCH --time=16:00:00
#SBATCH -J "dme-test"
#SBATCH --output=/scratch/ws/trng859b-dme/log-%j.out
#SBATCH --mem=16192

OUTFILE=""

srun python3 main.py with dme.evenly_distributed=True 'dme.extras=["no-extras"]' dme.test_all=False dme.use_validation=True

exit 0
