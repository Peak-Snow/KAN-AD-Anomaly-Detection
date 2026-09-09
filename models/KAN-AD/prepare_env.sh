#!/bin/sh
# This script sets up the environment for the project

# install uv if not already installed
if ! command -v uv &> /dev/null
then
    echo "uv could not be found, installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
else
    echo "uv is already installed"
fi

# clone datasets
if [ ! -d "datasets" ]; then
    echo "Cloning datasets repository..."
    git clone https://github.com/CSTCloudOps/datasets.git
fi

# use uv to sync env
uv sync

# echo some text
echo "Environment setup complete. You can now run the experiements with uv run run_exp.py"