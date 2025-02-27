#!/usr/bin/env python3

import numpy as np
from core_solver import SimConfig, EgoConfig, generate_controls, calc_time_range
from plots import plot_results

def find_geodesic(ego, min_duration=1, max_duration=50, time_steps=0.5, dt=0.1):
    """
    Find the minimum duration that results in a feasible path.
    
    Args:
        ego: Object containing vehicle parameters and constraints
        min_duration: Minimum duration to test (default: 1 second)
        max_duration: Maximum duration to test (default: 50 seconds)
        time_steps: Step size for duration search (default: 0.5 seconds)
        dt: Time step for discretization (default: 0.1 seconds)
        
    Returns:
        Minimum feasible duration and the corresponding results
    """
    # Calculate distance for logging purposes
    start_x, start_y = ego.state_start[0], ego.state_start[1]
    end_x, end_y = ego.state_final[0], ego.state_final[1]
    distance = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
    
    # Use the provided time_steps parameter
    step = time_steps
    
    print(f"Distance from start to goal: {distance:.2f} meters")
    print(f"Search range: {min_duration:.1f} to {max_duration:.1f} seconds")
    print(f"Step size: {step:.1f} seconds")
    print("Finding minimum feasible duration...")
    
    # Start with the minimum duration
    current_duration = min_duration
    last_feasible_duration = None
    last_feasible_results = None
    
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
    ego.state_start = [20, 0, 0, 0, 0]
    ego.state_final = [0,  0, 0, 0, 0]

    ego.velocity_max = 0.0  # Set velocity max to 0 for backward motion
    ego.velocity_min = -3.0 # Use negative velocity for backward motion

    # Find minimum feasible duration
    min_feasible_duration, min_results = find_geodesic(
        ego,
        min_duration=1,  # Use the calculated minimum duration
        max_duration=10,  # Use the calculated maximum duration
        time_steps=2,               # Search with 2 second steps
        dt=0.1
    )
    
    # Plot results if a feasible solution was found
    if min_results is not None:
        plot_results(min_results, ego, save_path='docs/geodesic_results.png') 
    
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
    
    # Calculate time range using calc_time_range (optional)
    # _, min_duration, max_duration, _ = calc_time_range(
    #     ego,
    #     duration_range_margin=5.0
    # )
    # print(f"Calculated time range: min={min_duration:.1f}s, max={max_duration:.1f}s")
    
    