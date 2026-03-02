#!/bin/bash
#SBATCH --partition=hitz-exclusive
#SBATCH --account=hitz-exclusive
#SBATCH --job-name=WSDBaseline_LLM
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32GB
#SBATCH --gres=gpu:a100:1
#SBATCH --output=./SLURM/LOG/irteeraOWiC.log
#SBATCH --error=./SLURM/ERRORE/erroreOWiC.err

source ~/inguruneak/GRAL/bin/activate

echo "Sentence Bert"
srun python WSD/baseline.py

