#!/bin/bash
#SBATCH -t 165:00:00
#SBATCH -A LU2023-2-11
#SBATCH -p gpua40
#SBATCH --gres=gpu:1   # Requesting 1 GPU
#SBATCH -N 1
#SBATCH --tasks-per-node=12
#SBATCH --mem-per-cpu=40000  # Increase memory allocation per CPU
#SBATCH -J run_Down_Pangea
#SBATCH -o run_autoen.out
#SBATCH -e run_autoen.err
#SBATCH --mail-user=sara.behnamian@biol.lu.se
#SBATCH --mail-type=END

module purge
module load Anaconda3/2022.05
conda run -n newenv python unet_evaluation.py

