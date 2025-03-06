#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle
import argparse
from core_solver import EgoConfig, SimConfig
from path_finder import find_path
from plots import plot_all_paths

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

    # Create simulation config
    sim_cfg = SimConfig()
    sim_cfg.duration = 80
    
    # fixed state_final configuration
    state_final = [goal_x, goal_y, goal_theta, 0, 0]
    
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
                ego.state_final = state_final.copy()
                ego.state_start = [x, y, theta, 0, 0]
                ego.verbose     = False
                
                try:
                    results = find_path(ego, sim_cfg)
                    if results is not None:
                        print(f"✓ Path planning successful")
                        successful_results.append({
                            'start': (x, y, theta),
                            'results': results,
                            'ego': ego
                        })
                    
                except Exception as e:
                    print(f"✗ Error during path planning: {str(e)}")
    
    print(f"\nCompleted grid search. Successful paths: {len(successful_results)}/{total_combinations}")
    return successful_results

def create_vehicle_patches(x, y, theta, color='blue', alpha=0.8):
    """Create patches representing a more realistic vehicle."""
    # More realistic vehicle dimensions
    length = 4.5  # meters (typical car length)
    width = 2.0   # meters (typical car width)
    
    # Calculate vehicle corners
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    # Calculate the corner position for a centered rectangle
    corner_x = x - (length/2 * cos_theta - width/2 * sin_theta)
    corner_y = y - (length/2 * sin_theta + width/2 * cos_theta)
    
    # Create vehicle body (main rectangle)
    vehicle_body = Rectangle(
        (corner_x, corner_y),
        length, width,
        angle=np.degrees(theta),
        facecolor=color,
        alpha=alpha,
        edgecolor='black',
        linewidth=1.5
    )
    
    # Create headlights (two small circles at the front)
    headlight_radius = 0.3
    headlight_positions = [
        (length/2 - headlight_radius, width/3),  # front left
        (length/2 - headlight_radius, -width/3)  # front right
    ]
    
    headlights = []
    for hx, hy in headlight_positions:
        # Apply rotation and translation
        hx_rot = hx * cos_theta - hy * sin_theta + x
        hy_rot = hx * sin_theta + hy * cos_theta + y
        
        headlight = Circle(
            (hx_rot, hy_rot),
            headlight_radius,
            facecolor='yellow',
            alpha=alpha,
            edgecolor='black'
        )
        headlights.append(headlight)
    
    # Combine all vehicle parts
    vehicle_parts = [vehicle_body] + headlights
    
    return vehicle_parts

def create_grid_paths_animation(successful_results, max_paths=None):
    """Create animation of vehicles following multiple paths from grid search."""
    # Limit the number of paths if specified
    if max_paths is not None and max_paths > 0:
        successful_results = successful_results[:max_paths]
    
    num_paths = len(successful_results)
    print(f"Creating animation for {num_paths} paths")
    
    # Find the maximum trajectory length to determine animation frames
    max_length = 0
    for result in successful_results:
        trajectory = result['results']['x']
        max_length = max(max_length, len(trajectory))
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Extract goal position from the first result
    goal_x, goal_y, goal_theta = successful_results[0]['ego'].state_final[:3]
    
    # Plot goal position
    ax.plot(goal_x, goal_y, 'ro', markersize=10, label='Goal')
    
    # Create a list to store all vehicle patches for each path
    all_vehicle_parts = []
    
    # Create a colormap for different paths
    cmap = plt.cm.viridis
    colors = [cmap(i / num_paths) for i in range(num_paths)]
    
    # Plot the complete paths with faded lines
    for i, result in enumerate(successful_results):
        trajectory = result['results']['x']
        ax.plot(trajectory[:, 0], trajectory[:, 1], '--', color=colors[i], alpha=0.3)
        
        # Plot start position
        start_x, start_y, start_theta = result['start']
        ax.plot(start_x, start_y, 'o', color=colors[i], markersize=6)
        
        # Initialize empty list for this path's vehicle parts
        all_vehicle_parts.append([])
    
    # Set plot properties
    ax.set_aspect('equal')
    ax.set_xlabel('X position (m)')
    ax.set_ylabel('Y position (m)')
    ax.set_title(f'Path Planning from Multiple Initial Positions\n{num_paths} successful paths')
    
    # Add grid
    ax.grid(True)
    
    # Calculate plot limits with some padding
    all_x = []
    all_y = []
    for result in successful_results:
        trajectory = result['results']['x']
        all_x.extend(trajectory[:, 0])
        all_y.extend(trajectory[:, 1])
    
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    
    # Add padding (10% of range)
    x_padding = (x_max - x_min) * 0.1
    y_padding = (y_max - y_min) * 0.1
    
    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)
    
    def init():
        """Initialize animation."""
        return []
    
    def animate(frame):
        """Update animation frame."""
        # Remove all previous vehicle patches
        for parts in all_vehicle_parts:
            for part in parts:
                part.remove()
            parts.clear()
        
        # List to collect all patches for blit
        all_patches = []
        
        # Create new vehicle patches for each path
        for i, result in enumerate(successful_results):
            trajectory = result['results']['x']
            
            # Skip if this trajectory is shorter than current frame
            if frame < len(trajectory):
                x, y, theta = trajectory[frame, 0:3]
                vehicle_parts = create_vehicle_patches(x, y, theta, color=colors[i])
                
                # Add patches to plot and store them
                for part in vehicle_parts:
                    ax.add_patch(part)
                    all_patches.append(part)
                
                # Store for later removal
                all_vehicle_parts[i] = vehicle_parts
        
        # Update title with progress
        progress = frame / max_length * 100
        ax.set_title(f'Path Planning from Multiple Initial Positions\n{num_paths} successful paths (Progress: {progress:.1f}%)')
        
        return all_patches
    
    # Create animation
    frames = max_length
    interval = 16  # milliseconds between frames (60 fps)
    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=frames, interval=interval, blit=True
    )
    
    # Save animation
    docs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'docs')
    os.makedirs(docs_dir, exist_ok=True)
    output_file = os.path.join(docs_dir, 'grid_paths_animation.mp4')
    
    # Save animation with high quality
    anim.save(output_file, writer='ffmpeg', fps=60,
              dpi=200, bitrate=2000,
              metadata={'title': 'Grid Paths Animation'})
    
    plt.close()
    print(f"Animation saved to: {output_file}")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Grid path planning with animation')
    parser.add_argument('--grid-size', type=float, default=40,
                        help='Size of the grid in meters (centered on goal)')
    parser.add_argument('--grid-step', type=float, default=20,
                        help='Step size for grid points in meters')
    parser.add_argument('--num-angles', type=int, default=4,
                        help='Number of different heading angles to try at each position')
    parser.add_argument('--goal-x', type=float, default=0,
                        help='Goal x-coordinate')
    parser.add_argument('--goal-y', type=float, default=0,
                        help='Goal-y-coordinate')
    parser.add_argument('--goal-theta', type=float, default=0,
                        help='Goal heading angle (degrees)')
    parser.add_argument('--max-paths', type=int, default=None,
                        help='Maximum number of paths to animate (default: all)')
    parser.add_argument('--animate', action='store_true',
                        help='Create animation of the paths')
    
    args = parser.parse_args()
    
    # Convert goal theta from degrees to radians
    goal_theta_rad = np.deg2rad(args.goal_theta)
    
    # Perform grid search
    successful_results = grid_path_planning(
        goal_x=args.goal_x,
        goal_y=args.goal_y,
        goal_theta=goal_theta_rad,
        grid_size=args.grid_size,
        grid_step=args.grid_step,
        num_angles=args.num_angles
    )
    
    if successful_results:
        docs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'docs')
        os.makedirs(docs_dir, exist_ok=True)
        plot_all_paths(successful_results, args.goal_x, args.goal_y, 
                      save_path=os.path.join(docs_dir, 'grid_path_planning_results.png'))
        
        # Create animation if requested
        if args.animate:
            create_grid_paths_animation(successful_results, max_paths=args.max_paths)
    else:
        print("No successful paths found.") 