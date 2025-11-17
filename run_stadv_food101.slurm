#!/bin/bash
#SBATCH -p gpu-short
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1  
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --cpus-per-task=4
module load anaconda3/2022.05 
source activate stadv

cd /scratch/hpc/07/zhang303/test
#srun python stadv_clip_food101.py --device cuda --batch-size 16 --steps 20
# srun python patchclip.py 
srun python train_and_eval_patch.py \
    --workdir ./work \
    --device cuda \
    --load-resnet \
    --epochs 30 \
    --batch-size 128 \
    --pretrained \
    --max_iter 300

    # ./work/resnet50_cifar100.pth 