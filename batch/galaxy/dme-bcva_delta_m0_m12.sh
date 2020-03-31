#!/bin/bash
#SBATCH --nodes=1
#SBATCH -p clara-job
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH -J "dme-bd0_12"
#SBATCH --output=/home/sc.uni-leipzig.de/mx492cyci/dme/out/experiment-master-%j-bcva_delta_m0_m12-even.out
#SBATCH --mem=14192


OUTFILE=""

srun python3 main.py with dme.evenly_distributed=True "dme.extras=['bcva_delta_m0_m12']" dme.num_extra=1

exit 0
