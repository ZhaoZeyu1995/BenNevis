#!/bin/bash
# Create a virtual environment for BenNevis
# After running this script, there will be a virtual environment named "venv" in the current directory
# Usage: ./create_env.sh

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install k2==1.24.4.dev20231220+cuda11.8.torch2.1.0 -f https://k2-fsa.github.io/k2/cuda.html
# for the progress bar support
pip install tqdm  
# for configuration management
pip install hydra-core 
# a very powerful logger
pip install wandb 
# for model summary display
pip install torchinfo 
# for kaldi-format data IO
pip install kaldiio 
# for whisper finetuning
pip install openai-whisper
