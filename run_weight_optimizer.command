#!/bin/bash
#
# This script runs the weight_optimizer.py script within the virtual environment.

# Get the directory of the script itself
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Change to the script's directory to ensure relative paths work correctly
cd "$SCRIPT_DIR"

# Activate the virtual environment using its absolute path
source "venv/bin/activate"

# Run the weight_optimizer.py script using the python from the activated venv
python "weight_optimizer.py"

# Deactivate the virtual environment (optional, but good practice)
deactivate

echo "Weight optimization process finished."
