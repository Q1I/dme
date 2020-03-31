#!/bin/bash
#SBATCH --nodes=1
#SBATCH -p clara-job
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH -J "dme-surgery"
#SBATCH --output=/home/sc.uni-leipzig.de/mx492cyci/dme/out/experiment-master-%j-surgery-even.out
#SBATCH --mem=14192


OUTFILE=""

srun python3 main.py with dme.evenly_distributed=True "dme.extras=['surgery_yes','surgery_no']" dme.num_extra=2

exit 0
