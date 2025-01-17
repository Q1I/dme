#!/bin/bash
#SBATCH -A p_scads            
#SBATCH --nodes=1            
#SBATCH --mincpus=8         
#SBATCH --ntasks=1         
#SBATCH --cpus-per-task=1
#SBATCH --time=16:00:00
#SBATCH -J "dme-t-extras"
#SBATCH --output=/scratch/ws/trng859b-dme/log-extras-%j.out
#SBATCH --mem=16192


OUTFILE=""

srun python3 main.py with dme.evenly_distributed=True dme.test_all=False dme.use_validation=True

exit 0
