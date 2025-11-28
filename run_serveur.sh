#!/bin/bash

# Exit immediately if a command fails
set -e

# Name of virtual environment
VENV_DIR="venv"

# Check if venv exists, if not, create it
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv $VENV_DIR
fi

# Activate virtual environment
source $VENV_DIR/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies from requirements.txt
pip install -r requirements.txt

# Run the FastAPI server
echo "Starting FastAPI server..."
uvicorn ml_server:app --host 0.0.0.0 --port 5000
