#!/bin/bash
#SBATCH -A p_scads            
#SBATCH --nodes=1            
#SBATCH --mincpus=1         
#SBATCH --ntasks=1         
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00
#SBATCH -J "dme"
#SBATCH --output=/scratch/ws/trng859b-dme/experiment-master-%j-extras-even.out
#SBATCH --mem=16192


OUTFILE=""

srun python3 main.py with dme.evenly_distributed=True

exit 0
