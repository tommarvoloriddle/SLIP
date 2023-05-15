#!/bin/bash

#SBATCH --job-name=resnet18_pokemon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --gres=gpu
#SBATCH --mem=32GB
#SBATCH --output="vgg_pokemon_ouptut" 

module purge

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK;


singularity exec --nv \
	    --overlay /scratch/sg7729/hpml/my_pytorch2.ext3:ro \
	    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
	    /bin/bash -c "source /ext3/env.sh; python /scratch/sg7729/DL_project/Classifier/VGG.py"