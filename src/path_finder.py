#!/usr/bin/env python3

import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import subprocess
import numpy as np
import matplotlib.pyplot as plt
from contextlib import contextmanager
from core_solver import generate_controls, EgoConfig, SimConfig
from plots import plot_results
from utils.io_utils import SuppressOutput, clean_acados_files  # Import SuppressOutput and clean_acados_files from utils

def find_path(ego, sim_cfg):
    """
    Find a feasible path using the specified duration.
    
    Args:
        ego: Object containing vehicle parameters and constraints
        sim_cfg: duration and dt
        verbose: Whether to print progress messages and compilation output
        
    Returns:
        Results dictionary containing the path and control inputs
    """

    # Generate controls
    try:
        # Suppress output if not verbose
        if not ego.verbose:
            with SuppressOutput():
                results = generate_controls(ego, sim_cfg)
        else:
            results = generate_controls(ego, sim_cfg)
            
        status = results['status']
        
        if status == 0:
            if ego.verbose:
                print(f"✓ Path planning successful")
            return results
        else:
            if ego.verbose:
                print(f"✗ Path planning failed with status {status}")
            return None
        
    except Exception as e:
        if ego.verbose:
            print(f"✗ Error during path planning: {str(e)}")
        return None

if __name__ == "__main__":
    # Clean up any existing Acados files
    clean_acados_files()

    # Create simulation config
    sim_cfg = SimConfig()
    sim_cfg.duration = 40
    
    # Create ego vehicle configuration
    ego = EgoConfig()
    ego.verbose = True

    ego.state_start = [-20, 20, -np.pi/4, 0.0, 0]
    ego.state_final = [20, -20, -np.pi/4, 0.0, 0]
    ego.corridor_width = 10.0
    ego.velocity_min = 0.0
    
    # Set obstacles [x, y, radius, safety_margin]
    multiple_obstacles = [
        [-7,  12,  4.0, 1.0], 
        [7,  -12,  4.0, 1.0]
    ]
    ego.obstacles = multiple_obstacles
    
    results = find_path(ego, sim_cfg)
    
    if results is not None:
        plot_results(results, ego, save_path='docs/path_finder.png', show_xy_plot=True)
    else:
        print("No feasible path found.") 