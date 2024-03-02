env_path=$LOCAL_HOME/conda/k2
if [ $CONDA_PREFIX != "$env_path" ]; then
    echo "Activating conda environment $env_path"
    . $CONDA_PREFIX/etc/profile.d/conda.sh && conda deactivate && conda activate $env_path
fi
