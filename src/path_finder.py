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

class SuppressOutput:
    """
    A context manager that redirects stdout and stderr to devnull.
    This works for both Python's output and C-level output.
    """
    def __init__(self):
        self._stdout = None
        self._stderr = None
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for _ in range(2)]
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Save current file descriptors and redirect stdout/stderr to devnull
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)
        sys.stdout.flush()
        sys.stderr.flush()
        
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore file descriptors
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        
        # Close all file descriptors
        for fd in self.null_fds + self.save_fds:
            try:
                os.close(fd)
            except:
                pass

def find_path(ego, duration, dt=0.1, verbose=True, obstacles=None):
    """
    Find a feasible path using the specified duration.
    
    Args:
        ego: Object containing vehicle parameters and constraints
        duration: Fixed duration for path planning (seconds)
        dt: Time step for discretization
        verbose: Whether to print progress messages and compilation output
        obstacles: List of obstacles, each defined as [x, y, radius, safety_margin]
        
    Returns:
        Results dictionary containing the path and control inputs
    """
    if verbose:
        start_pos = ego.state_start[:2]
        end_pos = ego.state_final[:2]
        distance = np.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
        print(f"Distance from start to goal: {distance:.2f} meters")
        print(f"Using duration: {duration:.2f} seconds")
    
    # Set obstacles if provided
    if obstacles is not None:
        ego.obstacles = obstacles
        if verbose:
            print(f"Added {len(obstacles)} obstacles to the path planning problem")
    
    # Create simulation config
    sim_cfg = SimConfig()
    sim_cfg.duration = duration
    sim_cfg.dt = dt
    
    # Generate controls
    try:
        # Suppress output if not verbose
        if not verbose:
            with SuppressOutput():
                results = generate_controls(ego, sim_cfg)
        else:
            results = generate_controls(ego, sim_cfg)
            
        status = results['status']
        
        if status == 0:
            if verbose:
                print(f"✓ Path planning successful with duration {duration:.1f}s")
            return results
        else:
            if verbose:
                print(f"✗ Path planning failed with status {status}")
            return None
    except Exception as e:
        if verbose:
            print(f"✗ Error during path planning: {str(e)}")
        return None

def clean_acados_files():
    """
    Clean up Acados generated files.
    """
    print("Cleaning up Acados generated files...")
    try:
        with SuppressOutput():
            # Remove generated files
            os.system("rm -f acados_solver_kinematic_car.o")
            os.system("rm -f libacados_ocp_solver_kinematic_car.dylib")
            os.system("rm -f acados_ocp_kinematic_car.json")
    except:
        print("Error cleaning up files.")

if __name__ == "__main__":
    # Clean up any existing Acados files
    clean_acados_files()
    
    # Create ego vehicle configuration
    ego = EgoConfig()

    # Set start and end states
    ego.state_start = [-20, 20, -np.pi/4, 0.0, 0]
    ego.state_final = [20, -20, -np.pi/4, 0.0, 0]
    ego.corridor_width = 10.0
    ego.velocity_min = 0.0
    
    # Set obstacles [x, y, radius, safety_margin]
    multiple_obstacles = [
        [-7,  12,  4.0, 1.0], 
        [7,  -12,  4.0, 1.0]
    ]
    
    results = find_path(ego, duration=40, dt=0.1, verbose=True, obstacles=multiple_obstacles)
    if results is not None:
        plot_results(results, ego, save_path='docs/path_finder.png', show_xy_plot=True)
    else:
        print("No feasible path found for Example 3.") 