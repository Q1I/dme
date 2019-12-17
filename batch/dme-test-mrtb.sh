#!/bin/bash
#SBATCH -A p_scads            
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8         
#SBATCH --ntasks=1         
#SBATCH --time=16:00:00
#SBATCH -J "dme-t-mrtb"
#SBATCH --output=/scratch/ws/trng859b-dme/log-%j-mrtb.out
#SBATCH --mem=16192

OUTFILE=""

srun python3 main.py with dme.evenly_distributed=True 'dme.extras=["mrtb"]' dme.test_all=False dme.use_validation=True

exit 0
