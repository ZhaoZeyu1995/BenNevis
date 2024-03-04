env_path=/disk/scratch3/zzhao/conda/k2
if [ $CONDA_PREFIX != "$env_path" ]; then
    echo "Activating conda environment $env_path"
    . $CONDA_PREFIX/etc/profile.d/conda.sh && conda deactivate && conda activate $env_path
fi
