#!/bin/bash
#SBATCH -A p_scads            
#SBATCH --nodes=1            
#SBATCH --mincpus=1         
#SBATCH --ntasks=1         
#SBATCH --cpus-per-task=1
#SBATCH --time=3:00:00
#SBATCH -J "dme-cstb"
#SBATCH --output=/scratch/ws/trng859b-dme/experiment-master-%j-cstb.out
#SBATCH --gres=gpu:1           # use 1 GPU per node (i.e. use one GPUper task)
#SBATCH -p gpu2
#SBATCH --mem=13192


OUTFILE=""

srun python3 main.py with dme.evenly_distributed=False "dme.extras=['cstb']"

exit 0
