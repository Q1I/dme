#!/bin/bash
#SBATCH -A p_scads            
#SBATCH --nodes=1            
#SBATCH --mincpus=1         
#SBATCH --ntasks=1         
#SBATCH --cpus-per-task=8
#SBATCH --time=19:00:00
#SBATCH -J "dme-mrtb"
#SBATCH --output=/scratch/ws/trng859b-dme/experiment-master-%j-mrtb-even.out
#SBATCH --mem=16192

OUTFILE=""

srun python3 main.py with dme.evenly_distributed=True "dme.extras=['mrtb']"

exit 0
