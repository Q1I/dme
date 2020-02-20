#!/bin/bash
#SBATCH -A p_scads            
#SBATCH --nodes=1            
#SBATCH --mincpus=1         
#SBATCH --ntasks=1         
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH -J "mv-bcva_delta"
#SBATCH --output=/scratch/ws/trng859b-dme/experiment-mv-%j-bcva_delta.out
#SBATCH --mem=34192


OUTFILE=""

#srun python3 missing_values_run.py with missing_values.evenly_distributed=False "missing_values.extras=['hba1c']" missing_values.num_extra=1
srun python3 missing_values_run.py
exit 0
