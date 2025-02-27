#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

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