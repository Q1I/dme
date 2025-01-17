#!/bin/bash
#SBATCH -A p_scads            
#SBATCH --nodes=1            
#SBATCH --mincpus=1         
#SBATCH --ntasks=1         
#SBATCH --cpus-per-task=8
#SBATCH --time=26:00:00
#SBATCH -J "dme-hba1c"
#SBATCH --output=/scratch/ws/trng859b-dme/experiment-master-%j-hba1c-even.out
#SBATCH --mem=19192


OUTFILE=""

srun python3 main.py with dme.evenly_distributed=True "dme.extras=['hba1c']" dme.num_extra=1

exit 0
