#!/bin/bash
#SBATCH -A p_scads            
#SBATCH --nodes=1            
#SBATCH --mincpus=2         
#SBATCH --ntasks=1         
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH -J "dme-test"
#SBATCH --output=/scratch/ws/trng859b-dme/log-%j.out
#SBATCH --mem=13192
#SBATCH --gres=gpu:1           # use 1 GPU per node (i.e. use one GPUper task)
#SBATCH -p gpu2,gpu1


OUTFILE=""

srun python3 main.py with dme.evenly_distributed=True dme.test_all=False dme.use_validation=True 'dme.extras=["no-extras"]'

exit 0
