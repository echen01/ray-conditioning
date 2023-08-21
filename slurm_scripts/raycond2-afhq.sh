#!/bin/bash

#SBATCH -J AFHQ2                           # Job name
#SBATCH -o AFHQ2.o                              # Name of stdout output file (%j expands to jobId)
#SBATCH -e AFHQ2.o                              # Name of stderr output file (%j expands to jobId)
#SBATCH -N 1                                            # Total number of nodes (physical machine) requested 
#SBATCH -n 6                                            # Number of cores requested
#SBATCH --mem=30000                                     # Memory pool (MB)
#SBATCH -t 96:00:00                                     # Run time (hh:mm:ss) 
#SBATCH --partition=cuvl,davis,snavely
#SBATCH --gres=gpu:a6000:2                         # gpu:gpu_type:how_many_gpu

# below is the script that specifies all the commands to run
# or you could just add a single command
# e.g. echo "hello world!"
# better to use absolute path

python /share/phoenix/nfs04/S7/emc348/ray-conditioning/train.py --outdir=/share/phoenix/nfs04/S7/emc348/stylegan3/training-runs --data=/share/phoenix/nfs04/S7/emc348/nerf/eg3d/processed_afhq.zip --cfg=raycond2 --gpus=2 --batch=32 --gamma=2.5 --snap=40 --cond=1 --aug=noaug --resume=/share/phoenix/nfs04/S7/emc348/nerf/eg3d/afhqcat.pkl
