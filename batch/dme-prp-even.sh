#!/bin/bash
#SBATCH -A p_scads            
#SBATCH --nodes=1            
#SBATCH --mincpus=1         
#SBATCH --ntasks=1         
#SBATCH --cpus-per-task=1
#SBATCH --time=5:00:00
#SBATCH -J "dme-prp"
#SBATCH --output=/scratch/ws/1/trng859b-dme/experiment-master-%j-prp-even.out
#SBATCH --gres=gpu:1           # use 1 GPU per node (i.e. use one GPUper task)
#SBATCH -p gpu2
#SBATCH --mem=14192


OUTFILE=""

srun python3 main.py with dme.evenly_distributed=True "dme.extras=['prp_yes','prp_no']" dme.num_extra=2

exit 0
