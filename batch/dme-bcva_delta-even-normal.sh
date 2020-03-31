#!/bin/bash
#SBATCH -A p_scads            
#SBATCH --nodes=1            
#SBATCH --mincpus=1         
#SBATCH --ntasks=1         
#SBATCH --cpus-per-task=8
#SBATCH --time=26:00:00
#SBATCH -J "dme-b-delta"
#SBATCH --output=/scratch/ws/1/trng859b-dme/experiment-master-%j-bcva_delta-even.out
#SBATCH --mem=19192


OUTFILE=""

srun python3 main.py with dme.evenly_distributed=True "dme.extras=['bcva_delta_m0_m12']" dme.num_extra=1

exit 0
