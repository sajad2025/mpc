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
from core_solver_geodesic import generate_controls_geodesic, EgoConfig, SimConfig
from plots import plot_results_geodesic
from utils.io_utils import clean_acados_files, SuppressOutput

def find_path_geodesic(ego, sim_cfg):
    """
    Find a feasible path using the specified duration.
    
    Args:
        ego: Object containing vehicle parameters and constraints
        sim_cfg: duration and dt
        
    Returns:
        Results dictionary containing the path and control inputs
    """
    # Generate controls
    try:
        # Suppress output if not verbose
        if not ego.verbose:
            with SuppressOutput():
                results = generate_controls_geodesic(ego, sim_cfg)
        else:
            results = generate_controls_geodesic(ego, sim_cfg)
            
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
    clean_acados_files()

    sim_cfg = SimConfig()

    ego = EgoConfig()
    ego.verbose = False

    r = 5.5
    print(f"min turning radius {ego.L / np.tan(ego.steering_max)}")

    ego.state_start     = [-r,  0,  np.pi/2, -0.5] # [x, y, theta, steering]
    ego.state_final     = [0,   r,  0,       -0.5]
    ego.corridor_width  = 1
    sim_cfg.duration    = (2 *np.pi * r) * .251
    
    results = find_path_geodesic(ego, sim_cfg)
    
    if results is not None:
        plot_results_geodesic(results, ego, save_path='docs/geodesic.png')
    else:
        print("No feasible path.") 