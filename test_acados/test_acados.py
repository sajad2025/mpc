#!/usr/bin/env python3

import os
import sys
import numpy as np
from acados_template import AcadosOcp, AcadosModel

def main():
    """
    Simple test to verify Acados installation.
    """
    print("Testing Acados installation...")
    
    # Print environment variables
    print(f"ACADOS_SOURCE_DIR: {os.environ.get('ACADOS_SOURCE_DIR', 'Not set')}")
    print(f"ACADOS_INSTALL_DIR: {os.environ.get('ACADOS_INSTALL_DIR', 'Not set')}")
    
    # Create a simple model
    model = AcadosModel()
    model.name = "test_model"
    
    # Create a simple OCP
    ocp = AcadosOcp()
    ocp.model = model
    
    # Print Acados version info
    print("\nAcados modules successfully imported!")
    print("Python version:", sys.version)
    print("NumPy version:", np.__version__)
    
    # Check if CasADi is available
    try:
        import casadi
        print("CasADi version:", casadi.__version__)
    except ImportError:
        print("CasADi not found")
    
    print("\nAcados basic test completed successfully!")

if __name__ == "__main__":
    main() 