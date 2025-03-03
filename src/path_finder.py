#!/usr/bin/env python3

import os
import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from contextlib import contextmanager
from core_solver import generate_controls, EgoConfig, SimConfig, calculate_path_duration
import time

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

def find_path(ego, duration, dt=0.1, verbose=True):
    """
    Find a feasible path using the specified duration.
    
    Args:
        ego: Object containing vehicle parameters and constraints
        duration: Fixed duration for path planning (seconds)
        dt: Time step for discretization
        verbose: Whether to print progress messages and compilation output
        
    Returns:
        Results dictionary containing the path and control inputs
    """
    if verbose:
        start_pos = ego.state_start[:2]
        end_pos = ego.state_final[:2]
        distance = np.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
        print(f"Distance from start to goal: {distance:.2f} meters")
        print(f"Using duration: {duration:.2f} seconds")
    
    # Create simulation config
    sim_cfg = SimConfig()
    sim_cfg.duration = duration
    sim_cfg.dt = dt
    
    # Generate controls
    try:
        # Suppress output if not verbose
        if not verbose:
            with SuppressOutput():
                results = generate_controls(ego, sim_cfg, use_nn_init=False)
        else:
            results = generate_controls(ego, sim_cfg, use_nn_init=False)
            
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
    # Example usage
    ego = EgoConfig()
    # Now we can directly set angles without explicit normalization
    # The EgoConfig class will handle normalization internally
    
    scenarios = [
        # {
        #     'name': 'U-Turn',
        #     'start': [0, 0, 0, 0, 0],
        #     'goal': [-5, 0, np.pi, 0, 0],
        #     'duration': 40.0
        # },
        {
            'name': 'd245',
            'start': [10, 20, np.pi, 0, 0],
            'goal': [0, 0, 0, 0, 0],
            'duration': 80.0
        }
    ]
    
    # Clean up any existing Acados files
    clean_acados_files()
    
    # Calculate duration based on distance and velocity
    # start_pos = ego.state_start[:2]
    # end_pos = ego.state_final[:2]
    # duration = calculate_path_duration(start_pos, end_pos, ego.velocity_max)
    
    # print("Running path planning silently...")
    # results = find_path(ego, duration=3*duration, dt=0.1, verbose=True)
    # print("Path planning completed.")

    for i, scenario in enumerate(scenarios):
        print(f"\nScenario {i+1}: {scenario['name']}")
        
        # Create ego config
        ego = EgoConfig()
        ego.state_start = scenario['start']
        ego.state_final = scenario['goal']
        
        # Calculate distance
        # start_pos = ego.state_start[:2]
        # end_pos = ego.state_final[:2]
        # distance = np.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
        
        # # Calculate duration based on distance
        # duration = distance / ego.velocity_max + 5.0
        
        # Create simulation config
        sim_cfg = SimConfig()
        sim_cfg.duration = scenario['duration']
        sim_cfg.dt = 0.1
        
        # Run with default initialization
        # print("Running with default initialization...")
        # start_time = time.time()
        # default_results = generate_controls(ego, sim_cfg, use_nn_init=False)
        # default_results = find_path(ego, duration=duration, dt=0.1, verbose=True)
        default_results = generate_controls(ego, sim_cfg, use_nn_init=False)
        # default_time = time.time() - start_time
        
        # print(f"Default initialization: Time = {default_time:.3f}s, Status = {default_results['status']}")
    
        # Import plot_results only if needed
        if default_results is not None:
            from plots import plot_results
            plot_results(default_results, ego, save_path='docs/path_finder_results.png', show_plot=True)
        else:
            print("No feasible path found.") 