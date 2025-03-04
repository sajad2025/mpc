#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from utils.corridor import corridor_cal
import os

def plot_results(results, ego, save_path=None, show_xy_plot=False):
    """
    Plot the path planning results.
    
    Args:
        results: Dictionary containing t, x, and u from generate_controls
        ego: Object containing vehicle parameters
        save_path: Optional path to save the plot. If provided:
            - Main plot will be saved as {save_path}
            - XY plot (if show_xy_plot=True) will be saved as {save_path_without_ext}_xy{ext}
    """
    t = results['t']
    x = results['x']
    u = results['u']
    
    # Create subplots (3x2 grid now)
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(10.5, 7.35))
    
    def plot_xy_trajectory(ax, title=None, show_arrows=False):
        """Helper function to plot xy trajectory on any axes"""
        # Plot trajectory
        ax.plot(x[:, 0], x[:, 1], 'b-', label='path (m)')
        ax.plot(ego.state_start[0], ego.state_start[1], 'go', label='start')
        ax.plot(ego.state_final[0], ego.state_final[1], 'ro', label='goal')
        
        # Add direction arrows along the path (only if requested)
        if show_arrows:
            sample_interval = 10
            sampled_points = x[::sample_interval]
            directions_x = np.cos(sampled_points[:, 2])
            directions_y = np.sin(sampled_points[:, 2])
            ax.quiver(sampled_points[:, 0], sampled_points[:, 1], 
                     directions_x, directions_y,
                     color='blue', alpha=0.6, scale=5, scale_units='inches')
        
        # Calculate and plot corridor
        point_a = (ego.state_start[0], ego.state_start[1])
        point_b = (ego.state_final[0], ego.state_final[1])
        c1, a1, b1, c2, a2, b2 = corridor_cal(point_a, point_b, ego.corridor_width)
        
        # Calculate plot bounds with margin
        margin = ego.corridor_width * 2
        x_min = min(point_a[0], point_b[0]) - margin
        x_max = max(point_a[0], point_b[0]) + margin
        
        # Use parametric approach to draw boundary lines
        t = np.linspace(-1, 2, 100)  # Parametric range covering the segment and beyond
        
        # Calculate direction vector of the line
        dx = point_b[0] - point_a[0]
        dy = point_b[1] - point_a[1]
        
        # Normalize
        length = np.sqrt(dx**2 + dy**2)
        dx, dy = dx/length, dy/length
        
        # Normal vector (perpendicular to direction)
        nx, ny = -dy, dx
        
        # Points along the original line (extended)
        line_x = point_a[0] + t * dx * length
        line_y = point_a[1] + t * dy * length
        
        # Upper boundary
        upper_x = line_x + nx * ego.corridor_width
        upper_y = line_y + ny * ego.corridor_width
        
        # Lower boundary
        lower_x = line_x - nx * ego.corridor_width
        lower_y = line_y - ny * ego.corridor_width
        
        # Plot corridor boundaries
        ax.plot(upper_x, upper_y, 'r--', alpha=0.3, label='corridor')
        ax.plot(lower_x, lower_y, 'r--', alpha=0.3)
        
        ax.grid(True)
        ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left')
        ax.set_aspect('equal')
        
        if title:
            ax.set_title(title)
    
    # Plot xy trajectory in main figure (without arrows)
    plot_xy_trajectory(ax1, show_arrows=False)
    
    # Create separate window for xy plot if requested (with arrows)
    xy_fig = None
    if show_xy_plot:
        xy_fig = plt.figure(figsize=(12, 10))
        xy_ax = xy_fig.add_subplot(111)
        plot_xy_trajectory(xy_ax, title='Vehicle Trajectory', show_arrows=True)
        xy_fig.tight_layout()
    
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
    
    # Save plots if path is provided
    if save_path:
        # Save main plot
        fig.savefig(save_path)
        
        # Save xy plot if it exists
        if xy_fig and show_xy_plot:
            # Split the save_path into name and extension
            save_path_base, ext = os.path.splitext(save_path)
            xy_save_path = f"{save_path_base}_xy{ext}"
            xy_fig.savefig(xy_save_path)
    
    # Show the plot and keep it open
    plt.show(block=True)

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
    
    # Create colormap for velocity
    cmap = plt.cm.coolwarm  # Red for forward, blue for backward
    
    # Find min and max velocities across all paths for normalization
    all_velocities = []
    for result in successful_results:
        path = result['results']['x']
        velocities = path[:, 3]  # Velocity is the 4th state variable (index 3)
        all_velocities.extend(velocities)
    
    # Set velocity range for colormap
    vel_min = min(all_velocities)
    vel_max = max(all_velocities)
    vel_abs_max = max(abs(vel_min), abs(vel_max))
    norm = plt.Normalize(-vel_abs_max, vel_abs_max)  # Symmetric around zero
    
    # Create a custom colormap with black at zero and smooth transitions
    # Define the colors for our custom colormap with the requested scheme
    colors = [
        (0, 0.4, 1),      # Bright blue (most negative velocities)
        (0, 0, 0.4),      # Dark blue (slightly negative velocities)
        (0, 0, 0),        # Black (zero velocity)
        (0.4, 0, 0),      # Dark red (slightly positive velocities)
        (1, 0.4, 0)       # Bright red (most positive velocities)
    ]
    
    # Define the positions for these colors on a scale from 0 to 1
    # This ensures black is exactly at the center (0.5)
    positions = [0, 0.4, 0.5, 0.6, 1]
    
    # Create the colormap
    cdict = {'red': [], 'green': [], 'blue': []}
    for pos, color in zip(positions, colors):
        cdict['red'].append((pos, color[0], color[0]))
        cdict['green'].append((pos, color[1], color[1]))
        cdict['blue'].append((pos, color[2], color[2]))
    
    custom_cmap = mcolors.LinearSegmentedColormap('custom_velocity', cdict)
    
    # Plot each path
    for i, result in enumerate(successful_results):
        # Extract path data
        path = result['results']['x']
        start_x, start_y, start_theta = result['start']
        
        # Extract velocities for this path
        velocities = path[:, 3]
        
        # Plot path segments with colors based on velocity
        for j in range(len(path) - 1):
            # Get segment points
            x_seg = path[j:j+2, 0]
            y_seg = path[j:j+2, 1]
            
            # Get velocity for this segment (use average of endpoints)
            vel = (velocities[j] + velocities[j+1]) / 2
            
            # Plot segment with color based on velocity
            ax.plot(x_seg, y_seg, '-', color=custom_cmap(norm(vel)), linewidth=1.5, alpha=0.7)
        
        # Plot start position with an arrow to show heading
        arrow_length = 1.0
        dx = arrow_length * np.cos(start_theta)
        dy = arrow_length * np.sin(start_theta)
        ax.arrow(start_x, start_y, dx, dy, head_width=0.3, head_length=0.5, 
                fc='black', ec='black', alpha=0.7)
    
    # Add grid
    ax.grid(True)
    
    # Set equal aspect ratio
    ax.set_aspect('equal')
    
    # Add labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'Path Planning from Multiple Initial Positions\n{len(successful_results)} successful paths')
    
    # Add colorbar to show velocity
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label('Velocity (m/s)')
    
    # Tight layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    # Show plot
    plt.show() 