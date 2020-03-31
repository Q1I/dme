#!/bin/bash
#SBATCH --nodes=1
#SBATCH -p clara-job
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH -J "dme-extras"
#SBATCH --output=/home/sc.uni-leipzig.de/mx492cyci/dme/out/experiment-master-%j-extras-even.out
#SBATCH --mem=14192


OUTFILE=""

srun python3 main.py with dme.evenly_distributed=True

exit 0
