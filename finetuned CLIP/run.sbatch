#!/bin/bash

#SBATCH --job-name=C2Lab2CPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu
#SBATCH --job-name=torch_c2cpu
#SBATCH --output="trained clip pokemon gs" 


module purge

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK;

      
singularity exec --nv \
	    --overlay /scratch/$USER/path/to/singularity.ext3:ro \
	    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
	    /bin/bash -c "source /ext3/env.sh; python /path/to/clip_grid_search.py"
