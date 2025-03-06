#!/bin/bash
# SBATCH --job-name=gpu_notebook
# SBATCH --gpus-per-node=1
# SBATCH --cpus-per-task=5
# SBATCH --time=06:00:00

# Clear the module environment
module purge
# Load the Python version that has been used to construct the virtual environment
# we are using below
module load Python/3.11.5-GCCcore-13.2.0

# Activate the virtual environment
source ~/pytorch_env/bin/activate

# Start the jupyter server, using the hostname of the node as the way to connect to it
jupyter notebook --no-browser --ip=$( hostname )

# After this, tunnel the notebook using ssh:
# ssh studentnumber@login1.hb.hpc.rug.nl -L 1996:habrok_node:jupyter_port

# Then, in the browser go to:
# http://127.0.0.1:1996/?token=THETOKENINTHEFILE

# sbatch: Memory requested 2000 partition will be regular