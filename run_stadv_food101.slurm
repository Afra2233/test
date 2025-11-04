#!/bin/bash
#SBATCH -p gpu-shortxxx
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1  
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --cpus-per-task=1
module load anaconda3/2022.05 
source activate stadv

cd /scratch/hpc/07/zhang303/test
srun python stadv_clip_food101.py --device cuda --batch-size 16 --steps 20
