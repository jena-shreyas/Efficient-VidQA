#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node 1
#SBATCH --time=10:0:0
#SBATCH --account=def-egranger
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shreyas.jena.1@etsmtl.net

cd $SCRATCH
wget -i clevrer_files.txt
