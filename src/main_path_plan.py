#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
from core_solver import EgoConfig, SimConfig, generate_controls

def plot_results(results, ego, save_path=None):
    """
    Plot the path planning results.
    
    Args:
        results: Dictionary containing t, x, and u from generate_controls
        ego: Object containing vehicle parameters
        save_path: Optional path to save the plot
    """
    t = results['t']
    x = results['x']
    u = results['u']
    
    # Create subplots (3x2 grid now)
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(10.5, 7.35))
    
    # Left column: States
    # Plot trajectory
    ax1.plot(x[:, 0], x[:, 1], 'b-', label='path (m)')
    ax1.plot(ego.state_start[0], ego.state_start[1], 'go', label='start')
    ax1.plot(ego.state_final[0], ego.state_final[1], 'ro', label='goal')
    
    # Add circles for car position at each sample
    for i in range(0, len(t), 5):  # Plot every 5th circle to avoid overcrowding
        circle = plt.Circle((x[i, 0], x[i, 1]), ego.L, fill=False, linestyle='--', 
                          color='blue', alpha=0.2, linewidth=0.5)
        ax1.add_patch(circle)
    
    ax1.grid(True)
    # Keep the xy plot legend at the upper right
    ax1.legend(bbox_to_anchor=(1.15, 1), loc='upper left')
    # Make sure the aspect ratio is equal so circles look circular
    ax1.set_aspect('equal')
    
    # Plot velocity
    ax3.plot(t, x[:, 3], 'b-', label='velocity (m/s)')
    ax3.axhline(y=ego.velocity_max, color='k', linestyle='--', alpha=0.3, label='bounds')
    ax3.axhline(y=ego.velocity_min, color='k', linestyle='--', alpha=0.3)
    ax3.grid(True)
    # Move legend to southwest corner
    ax3.legend(loc='lower left')
    
    # Plot steering angle (moved to bottom left)
    ax5.plot(t, np.rad2deg(x[:, 4]), 'r-', label='steering angle (deg)')
    ax5.axhline(y=np.rad2deg(ego.steering_max), color='k', linestyle='--', alpha=0.3, label='bounds')
    ax5.axhline(y=np.rad2deg(ego.steering_min), color='k', linestyle='--', alpha=0.3)
    ax5.set_xlabel('time (s)')
    ax5.grid(True)
    # Move legend to southwest corner
    ax5.legend(loc='lower left')
    
    # Right column: Controls and heading
    # Plot heading (moved to top right)
    ax2.plot(t, np.rad2deg(x[:, 2]), 'b-', label='current heading (deg)')
    ax2.axhline(y=np.rad2deg(ego.state_final[2]), color='r', linestyle='--', label='target heading')
    ax2.grid(True)
    # Move legend to southwest corner
    ax2.legend(loc='lower left')
    
    # Plot acceleration
    ax4.plot(t[:-1], u[:, 0], 'g-', label='acceleration (m/s²)')
    ax4.axhline(y=ego.acceleration_max, color='k', linestyle='--', alpha=0.3, label='bounds')
    ax4.axhline(y=ego.acceleration_min, color='k', linestyle='--', alpha=0.3)
    ax4.grid(True)
    # Move legend to southwest corner
    ax4.legend(loc='lower left')
    
    # Plot steering rate
    ax6.plot(t[:-1], np.rad2deg(u[:, 1]), 'm-', label='steering rate (deg/s)')
    ax6.axhline(y=np.rad2deg(ego.steering_rate_max), color='k', linestyle='--', alpha=0.3, label='bounds')
    ax6.axhline(y=np.rad2deg(ego.steering_rate_min), color='k', linestyle='--', alpha=0.3)
    ax6.set_xlabel('time (s)')
    ax6.grid(True)
    # Move legend to southwest corner
    ax6.legend(loc='lower left')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    # Show the plot and keep it open
    plt.show(block=True)

def find_minimum_duration(ego, initial_duration=None, min_duration=None, max_duration=None, 
                       duration_range_margin=5.0, step=0.5, dt=0.1):
    """
    Find the minimum duration that results in a feasible path.
    
    Args:
        ego: Object containing vehicle parameters and constraints
        initial_duration: Starting duration to test (if None, will be calculated)
        min_duration: Minimum duration to test (if None, will be calculated)
        max_duration: Maximum duration to test (if None, will be calculated)
        duration_range_margin: Margin to add/subtract from middle duration to set search range (default: 5.0 seconds)
        step: Step size for decreasing duration
        dt: Time step for discretization
        
    Returns:
        Minimum feasible duration and the corresponding results
    """
    # Calculate a reasonable duration range based on distance and max velocity
    start_x, start_y = ego.state_start[0], ego.state_start[1]
    end_x, end_y = ego.state_final[0], ego.state_final[1]
    
    # Calculate Euclidean distance
    distance = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
    
    # Calculate middle duration based on distance and max velocity
    duration_middle = distance / ego.velocity_max
    
    # Set search range if not provided
    if initial_duration is None:
        initial_duration = duration_middle
    if min_duration is None:
        min_duration = max(duration_middle - duration_range_margin, 1.0)  # Ensure min_duration is at least 1 second
    if max_duration is None:
        max_duration = duration_middle + duration_range_margin
    
    print(f"Distance from start to goal: {distance:.2f} meters")
    print(f"Estimated middle duration: {duration_middle:.2f} seconds")
    print(f"Search range: {min_duration:.1f} to {max_duration:.1f} seconds")
    print("Finding minimum feasible duration...")
    
    # Start with the initial duration
    current_duration = initial_duration
    last_feasible_duration = None
    last_feasible_results = None
    
    # First try the initial duration
    print(f"Testing initial duration: {current_duration:.1f} seconds")
    sim_cfg = SimConfig()
    sim_cfg.duration = current_duration
    sim_cfg.dt = dt
    
    try:
        results = generate_controls(ego, sim_cfg)
        status = results['status']
        
        if status == 0:
            print(f"✓ Duration {current_duration:.1f}s: Feasible solution found")
            last_feasible_duration = current_duration
            last_feasible_results = results
        else:
            print(f"✗ Duration {current_duration:.1f}s: Solver failed with status {status}")
            # Try a longer duration
            current_duration = min(current_duration + step, max_duration)
    except Exception as e:
        print(f"✗ Duration {current_duration:.1f}s: Error - {str(e)}")
        # Try a longer duration
        current_duration = min(current_duration + step, max_duration)
    
    # If initial duration was feasible, try decreasing
    if last_feasible_duration is not None:
        current_duration = last_feasible_duration - step
        
        # Try decreasing durations until we find the minimum
        while current_duration >= min_duration:
            print(f"Testing duration: {current_duration:.1f} seconds")
            
            sim_cfg = SimConfig()
            sim_cfg.duration = current_duration
            sim_cfg.dt = dt
            
            try:
                results = generate_controls(ego, sim_cfg)
                status = results['status']
                
                if status == 0:
                    print(f"✓ Duration {current_duration:.1f}s: Feasible solution found")
                    last_feasible_duration = current_duration
                    last_feasible_results = results
                    # Decrease duration and try again
                    current_duration -= step
                else:
                    print(f"✗ Duration {current_duration:.1f}s: Solver failed with status {status}")
                    # We've found the minimum feasible duration
                    break
            except Exception as e:
                print(f"✗ Duration {current_duration:.1f}s: Error - {str(e)}")
                # We've found the minimum feasible duration
                break
    # If initial duration was not feasible, try increasing
    else:
        # Try increasing durations until we find a feasible solution
        while current_duration <= max_duration:
            print(f"Testing duration: {current_duration:.1f} seconds")
            
            sim_cfg = SimConfig()
            sim_cfg.duration = current_duration
            sim_cfg.dt = dt
            
            try:
                results = generate_controls(ego, sim_cfg)
                status = results['status']
                
                if status == 0:
                    print(f"✓ Duration {current_duration:.1f}s: Feasible solution found")
                    last_feasible_duration = current_duration
                    last_feasible_results = results
                    break  # Found a feasible solution, no need to increase further
                else:
                    print(f"✗ Duration {current_duration:.1f}s: Solver failed with status {status}")
                    # Try a longer duration
                    current_duration += step
            except Exception as e:
                print(f"✗ Duration {current_duration:.1f}s: Error - {str(e)}")
                # Try a longer duration
                current_duration += step
    
    if last_feasible_duration is None:
        print("No feasible solution found within the tested range.")
        return None, None
    
    print(f"\nMinimum feasible duration: {last_feasible_duration:.1f} seconds")
    return last_feasible_duration, last_feasible_results

if __name__ == "__main__":
    # Example usage
    # Create ego configuration
    ego = EgoConfig()
    
    # Example of customizing weights
    # Uncomment and modify these lines to change the behavior
    # ego.weight_acceleration = 0.5      # Lower to allow more aggressive acceleration
    # ego.weight_steering_rate = 2.0     # Higher to encourage smoother steering changes
    # ego.weight_steering_angle = 10.0   # Higher to encourage straighter paths
    
    # Terminal weights
    # ego.weight_terminal_position_x = 200.0  # Higher to ensure precise final x position
    # ego.weight_terminal_position_y = 200.0  # Higher to ensure precise final y position
    # ego.weight_terminal_heading = 200.0     # Higher to ensure precise final heading
    # ego.weight_terminal_velocity = 50.0     # Higher to ensure precise final velocity
    # ego.weight_terminal_steering = 50.0     # Higher to ensure precise final steering angle
    
    # Find minimum feasible duration using the distance-based search range
    min_duration, min_results = find_minimum_duration(
        ego,
        duration_range_margin=5.0,  # +/- 5 seconds around the middle duration
        step=1.0,
        dt=0.1
    )
    
    # Plot results if a feasible solution was found
    if min_results is not None:
        plot_results(min_results, ego, save_path='docs/path_planning_results.png') 