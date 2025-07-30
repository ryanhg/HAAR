#!/bin/bash

# Install system-level libraries if missing
echo "Installing system libraries..."
apt-get update && apt-get install -y libgl1-mesa-glx

# Download model files only if not present
chmod +x ./scripts/download.sh
./scripts/download.sh

# Create results directory
mkdir -p ./api_results

# Start the FastAPI server in the background
echo "Starting HAAR FastAPI server..."
python api.py &

# Keep the container alive for terminal access
sleep infinity
