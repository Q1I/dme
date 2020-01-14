#!/bin/bash
#SBATCH -A p_scads            
#SBATCH --nodes=1            
#SBATCH --mincpus=1         
#SBATCH --ntasks=1         
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH -J "dme-duration"
#SBATCH --output=/scratch/ws/trng859b-dme/experiment-master-%j-duration-even.out
#SBATCH --gres=gpu:1           # use 1 GPU per node (i.e. use one GPUper task)
#SBATCH -p gpu2
#SBATCH --mem=14192


OUTFILE=""

srun python3 main.py with dme.evenly_distributed=True "dme.extras=['duration']" dme.num_extra=1

exit 0
