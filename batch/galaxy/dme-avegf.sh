#!/bin/bash
#SBATCH --nodes=1
#SBATCH -p clara-job
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH -J "dme-avegf"
#SBATCH --output=/home/sc.uni-leipzig.de/mx492cyci/dme/out/experiment-master-%j-avegf-even.out
#SBATCH --mem=14192


OUTFILE=""

srun python3 main.py with dme.evenly_distributed=True "dme.extras=['avegf_ranibizumab','avegf_aflibercept','avegf_bevacizumab']" dme.num_extra=3

exit 0
