#!/bin/bash

#SBATCH -J ShapeNetCarsNoAug                           # Job name
#SBATCH -o ShapeNetCarsNoAug.o                              # Name of stdout output file (%j expands to jobId)
#SBATCH -e ShapeNetCarsNoAug.o                              # Name of stderr output file (%j expands to jobId)
#SBATCH -N 1                                            # Total number of nodes (physical machine) requested 
#SBATCH -n 6                                            # Number of cores requested
#SBATCH --mem=30000                                     # Memory pool (MB)
#SBATCH -t 72:00:00                                     # Run time (hh:mm:ss) a
#SBATCH --partition=cuvl
#SBATCH --gres=gpu:2080ti:2                         # gpu:gpu_type:how_many_gpu

# below is the script that specifies all the commands to run
# or you could just add a single command
# e.g. echo "hello world!"
# better to use absolute path

python /share/phoenix/nfs04/S7/emc348/stylegan3/train.py --outdir=/share/phoenix/nfs04/S7/emc348/stylegan3/training-runs --data=/share/phoenix/nfs04/S7/emc348/nerf/eg3d/cars_train.zip --cfg=stylegan3-t --gpus=2 --batch=32 --gamma=0.5 --snap=20 --metrics=none --aug=noaug
# python /share/phoenix/nfs04/S7/emc348/SynthesizingAcrossTime/stylegan2-ada-pytorch/train.py --outdir=/share/phoenix/nfs04/S7/emc348/SynthesizingAcrossTime/stylegan2-ada-pytorch/training-runs/ --data=/scratch/datasets/emc348/rebuttal_filtered.zip --gpus=4 --resume=ffhq256 --snap=20 --mirror=1 --gamma=2.4 --metrics=none --cfg=paper256
# sbatch --requeue /share/phoenix/nfs04/S7/emc348/SynthesizingAcrossTime/stylegan2-ada-pytorch/train.sh
