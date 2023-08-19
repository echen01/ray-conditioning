#!/bin/bash

#SBATCH -J Cars                           # Job name
#SBATCH -o Cars3.o                              # Name of stdout output file (%j expands to jobId)
#SBATCH -e Cars3.o                              # Name of stderr output file (%j expands to jobId)
#SBATCH -N 1                                            # Total number of nodes (physical machine) requested 
#SBATCH -n 8                                            # Number of cores requested
#SBATCH --mem=60000                                     # Memory pool (MB)
#SBATCH -t 96:00:00                                     # Run time (hh:mm:ss) 
#SBATCH --partition=cuvl,gpu
#SBATCH --gres=gpu:1080ti:4                         # gpu:gpu_type:how_many_gpu

# below is the script that specifies all the commands to run
# or you could just add a single command
# e.g. echo "hello world!"
# better to use absolute path

#python /share/phoenix/nfs04/S7/emc348/stylegan3/train.py --outdir=/share/phoenix/nfs04/S7/emc348/stylegan3/training-runs --data=/share/phoenix/nfs04/S7/emc348/nerf/eg3d/cars_train.zip --cfg=cam-field --gpus=2 --batch=32 --gamma=0.3 --snap=20 --metrics=none --cond=1 --aug=noaug --resume=/share/cuvl/emc348/pretrained_models/stylegan2-ffhq-512x512.pkl
python /share/phoenix/nfs04/S7/emc348/ray-conditioning/train.py --outdir=/share/phoenix/nfs04/S7/emc348/stylegan3/training-runs --data=/share/phoenix/nfs04/S7/emc348/nerf/eg3d/cars_train.zip --cfg=raycond3-t --gpus=4 --batch=32 --gamma=0.5 --snap=20 --metrics=fid50k_full --cond=1 --aug=noaug 

