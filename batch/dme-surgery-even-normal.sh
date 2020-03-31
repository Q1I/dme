#!/bin/bash
#SBATCH -A p_scads            
#SBATCH --nodes=1            
#SBATCH --mincpus=1         
#SBATCH --ntasks=1         
#SBATCH --cpus-per-task=8
#SBATCH --time=26:00:00
#SBATCH -J "dme-surgery"
#SBATCH --output=/scratch/ws/1/trng859b-dme/experiment-master-%j-surgery-even.out
#SBATCH --mem=19192


OUTFILE=""

srun python3 main.py with dme.evenly_distributed=True "dme.extras=['surgery_yes','surgery_no']" dme.num_extra=2

exit 0
