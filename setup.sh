#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Set Acados directories using absolute paths
export ACADOS_SOURCE_DIR=$SCRIPT_DIR/acados
export ACADOS_INSTALL_DIR=$SCRIPT_DIR/acados/build

# Add library paths for Acados
export DYLD_LIBRARY_PATH=$ACADOS_SOURCE_DIR/lib:$DYLD_LIBRARY_PATH
export LD_LIBRARY_PATH=$ACADOS_SOURCE_DIR/lib:$LD_LIBRARY_PATH

# Add Python path for Acados templates
export PYTHONPATH=$ACADOS_SOURCE_DIR/interfaces/acados_template:$PYTHONPATH

# Activate virtual environment
source $SCRIPT_DIR/venv/bin/activate

# Print setup information
echo "Environment setup completed:"
echo "ACADOS_SOURCE_DIR=$ACADOS_SOURCE_DIR"
echo "ACADOS_INSTALL_DIR=$ACADOS_INSTALL_DIR"
echo "Python virtual environment activated"
echo ""
echo "To deactivate the virtual environment when done, run: deactivate" 