#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
from core_solver import EgoConfig, SimConfig, calculate_path_duration
from path_finder import find_path

def grid_path_planning(goal_x, goal_y, goal_theta, grid_size=20, grid_step=1, num_angles=8):
    """
    Perform path planning from different initial positions in a grid around the goal.
    
    Args:
        goal_x: Goal x-coordinate
        goal_y: Goal y-coordinate
        goal_theta: Goal heading angle (radians)
        grid_size: Size of the grid in meters (centered on goal)
        grid_step: Step size for grid points in meters
        num_angles: Number of different heading angles to try at each position
        
    Returns:
        List of successful path planning results
    """
    # Calculate grid boundaries
    half_size = grid_size / 2
    x_min = goal_x - half_size
    x_max = goal_x + half_size
    y_min = goal_y - half_size
    y_max = goal_y + half_size
    
    # Generate grid points
    x_points = np.arange(x_min, x_max + grid_step, grid_step)
    y_points = np.arange(y_min, y_max + grid_step, grid_step)
    
    # Generate angles (evenly spaced around the circle)
    angles = np.linspace(0, 2*np.pi, num_angles, endpoint=False)
    
    # Store successful results
    successful_results = []
    
    # Total number of combinations to try
    total_combinations = len(x_points) * len(y_points) * len(angles)
    current_combination = 0
    
    print(f"Grid search with {len(x_points)}x{len(y_points)} positions and {num_angles} angles")
    print(f"Total combinations to try: {total_combinations}")
    
    # Create base ego configuration
    base_ego = EgoConfig()
    
    # Set goal state
    base_ego.state_final = [goal_x, goal_y, goal_theta, 0, 0]
    
    # Loop through all grid points and angles
    for x in x_points:
        for y in y_points:
            for theta in angles:
                current_combination += 1
                print(f"\nCombination {current_combination}/{total_combinations}")
                print(f"Testing start position: x={x:.1f}, y={y:.1f}, theta={np.rad2deg(theta):.1f}°")
                
                # Skip if too close to goal (within 1 meter)
                distance_to_goal = np.sqrt((x - goal_x)**2 + (y - goal_y)**2)
                if distance_to_goal < 1.0:
                    print(f"Skipping - too close to goal ({distance_to_goal:.2f}m)")
                    continue
                
                # Create a new ego config for this combination
                ego = EgoConfig()
                ego.state_final = base_ego.state_final.copy()
                ego.state_start = [x, y, theta, 0, 0]
                
                # Calculate initial duration based on distance and velocity
                start_pos = [x, y]
                end_pos = [goal_x, goal_y]
                duration = calculate_path_duration(start_pos, end_pos, ego.velocity_max)
                
                # First attempt with calculated duration
                try:
                    # Use verbose=False to suppress compilation output
                    results = find_path(ego, duration=duration, dt=0.1, verbose=False)
                    
                    # If first attempt successful, store results
                    if results is not None:
                        print(f"✓ Path planning successful")
                        successful_results.append({
                            'start': (x, y, theta),
                            'results': results,
                            'ego': ego
                        })
                        continue  # Move to next combination
                    
                    # If first attempt failed, try again with tripled duration
                    print(f"✗ First attempt failed, trying again with tripled duration...")
                    tripled_duration = duration * 3
                    results = find_path(ego, duration=tripled_duration, dt=0.1, verbose=False)
                    
                    if results is not None:
                        print(f"✓ Path planning successful with tripled duration")
                        successful_results.append({
                            'start': (x, y, theta),
                            'results': results,
                            'ego': ego
                        })
                    else:
                        print(f"✗ Path planning failed for second time")
                    
                except Exception as e:
                    print(f"✗ Error during path planning: {str(e)}")
    
    print(f"\nCompleted grid search. Successful paths: {len(successful_results)}/{total_combinations}")
    return successful_results

def plot_all_paths(successful_results, goal_x, goal_y, save_path=None):
    """
    Plot all successful paths on a single 2D plot.
    
    Args:
        successful_results: List of successful path planning results
        goal_x: Goal x-coordinate
        goal_y: Goal y-coordinate
        save_path: Optional path to save the plot
    """
    # Create figure with proper layout for colorbar
    fig = plt.figure(figsize=(12, 10))
    gs = plt.GridSpec(1, 20)  # 1 row, 20 columns
    ax = fig.add_subplot(gs[0, :19])  # Main plot takes 19/20 of the width
    cax = fig.add_subplot(gs[0, 19])  # Colorbar takes 1/20 of the width
    
    # Plot goal position
    ax.plot(goal_x, goal_y, 'ro', markersize=10, label='Goal')
    
    # Plot each path with a different color
    cmap = plt.cm.viridis
    colors = [cmap(i) for i in np.linspace(0, 1, len(successful_results))]
    
    for i, result in enumerate(successful_results):
        # Extract path data
        path = result['results']['x']
        start_x, start_y, start_theta = result['start']
        
        # Plot path
        ax.plot(path[:, 0], path[:, 1], '-', color=colors[i], linewidth=1, alpha=0.7)
        
        # Plot start position with an arrow to show heading
        arrow_length = 1.0
        dx = arrow_length * np.cos(start_theta)
        dy = arrow_length * np.sin(start_theta)
        ax.arrow(start_x, start_y, dx, dy, head_width=0.3, head_length=0.5, 
                fc=colors[i], ec=colors[i], alpha=0.7)
    
    # Add grid
    ax.grid(True)
    
    # Set equal aspect ratio
    ax.set_aspect('equal')
    
    # Add labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'Path Planning from Multiple Initial Positions\n{len(successful_results)} successful paths')
    
    # Add colorbar to show path index
    norm = plt.Normalize(0, len(successful_results)-1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, cax=cax, label='Path Index')
    
    # Tight layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    # Show plot
    plt.show()

if __name__ == "__main__":
    # Define goal position
    goal_x = 0
    goal_y = 0
    goal_theta = 0  # radians
    
    # Perform grid search
    successful_results = grid_path_planning(
        goal_x=goal_x,
        goal_y=goal_y,
        goal_theta=goal_theta,
        grid_size=20,  # 20x20 meter grid
        grid_step=10,   # 2 meter steps (reduced from 1m for faster execution)
        num_angles=2   # 8 different heading angles
    )
    
    # Plot all successful paths
    if successful_results:
        plot_all_paths(successful_results, goal_x, goal_y, save_path='docs/grid_path_planning_results.png')
    else:
        print("No successful paths found.") 