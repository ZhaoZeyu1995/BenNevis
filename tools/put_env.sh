#!/bin/bash
# This script is used to set the environment variables for the application
# More specifically, it writes a `env.sh` file to the current directory
# given the conda envrionment path.
# Here is an example of env.sh file that is written:
# ```
# env_path=$LOCAL_HOME/conda/k2
# if [ $CONDA_PREFIX != "$env_path" ]; then
#     echo "Activating conda environment $env_path"
#     . $CONDA_PREFIX/etc/profile.d/conda.sh && conda deactivate && conda activate $env_path
# fi
# ```
# Usage:
#  ./put_env.sh <env_path>
#  where <env_path> is the path to the conda environment
# e.g. ./put_env.sh /disk/scratch/zzhao/conda/k2
# Authors:
#   * Zeyu Zhao (The University of Edinburgh) 2024

env_path=$1
echo "env_path=$env_path" > env.sh
echo 'if [ $CONDA_PREFIX != "$env_path" ]; then' >> env.sh
echo '    echo "Activating conda environment $env_path"' >> env.sh
echo '    . $CONDA_PREFIX/etc/profile.d/conda.sh && conda deactivate && conda activate $env_path' >> env.sh
echo 'fi' >> env.sh

echo "$0: env.sh is written to the current directory with the env_path=$env_path."
