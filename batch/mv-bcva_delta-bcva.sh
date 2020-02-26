#!/bin/bash
#SBATCH -A p_scads            
#SBATCH --nodes=1            
#SBATCH --mincpus=18        
#SBATCH --ntasks=1         
#SBATCH --cpus-per-task=1
#SBATCH --time=6:30:00
#SBATCH -J "mv-bcva_delta-bcva"
#SBATCH --output=/scratch/ws/trng859b-dme/experiment-mv-%j-bcva_delta-bcva.out
#SBATCH --mem=34192

OUTFILE=""

srun python3 missing_values_run.py with "missing_values.extras=['bcva_delta_m0_m12','bcva']" missing_values.num_extra=2
exit 0
