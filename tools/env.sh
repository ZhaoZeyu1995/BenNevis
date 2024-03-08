venv_path=../../../tools/venv
export PYTHONPATH=../../..:$PYTHONPATH
if [ -z $VIRTUAL_ENV ]; then
    echo "Activating virtual environment $venv_path"
    source $venv_path/bin/activate
else
    if [ $VIRTUAL_ENV != $(realpath $venv_path) ]; then
        echo "Deactivating current virtual environment $VIRTUAL_ENV"
        deactivate
        echo "Activating virtual environment $venv_path"
        source $venv_path/bin/activate
    else
        echo "Virtual environment $venv_path is already activated"
    fi
fi
