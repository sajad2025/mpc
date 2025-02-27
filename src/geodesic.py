#!/usr/bin/env python3

import numpy as np
from core_solver import SimConfig, EgoConfig, generate_controls
from plots import plot_results

def find_geodesic(ego, initial_duration=None, min_duration=None, max_duration=None, 
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
    min_duration, min_results = find_geodesic(
        ego,
        duration_range_margin=5.0,  # +/- 5 seconds around the middle duration
        step=1.0,
        dt=0.1
    )
    
    # Plot results if a feasible solution was found
    if min_results is not None:
        plot_results(min_results, ego, save_path='docs/geodesic_results.png') 