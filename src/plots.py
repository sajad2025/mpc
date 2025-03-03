#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def plot_results(results, ego, save_path=None, show_plot=False):
    """
    Plot the path planning results.
    
    Args:
        results: Dictionary containing t, x, and u from generate_controls
        ego: Object containing vehicle parameters
        save_path: Optional path to save the plot
        show_plot: Whether to keep the plot window open (True) or close it (False, default)
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
    
    # Add heading arrows along the trajectory
    arrow_step = 5  # Plot an arrow every 5 steps
    for i in range(0, len(t), arrow_step):
        dx = np.cos(x[i, 2])
        dy = np.sin(x[i, 2])
        ax1.quiver(x[i, 0], x[i, 1], dx, dy,
                  color='b', scale=10, alpha=0.3)
    
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
    
    # Show the plot if show_plot is True, otherwise close it
    if show_plot:
        plt.show()
    else:
        plt.close()

def plot_all_paths(successful_results, goal_x, goal_y, save_path=None):
    """
    Plot all successful paths on a single 2D plot.
    
    Args:
        successful_results: List of successful path planning results
        goal_x: Goal x-coordinate
        goal_y: Goal y-coordinate
        save_path: Optional path to save the plot
    """
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    
    # Plot goal position
    ax.plot(goal_x, goal_y, 'ro', markersize=10, label='Goal')
    
    # Define colors for different heading directions
    # Blue for north/south (around 0 and π radians)
    # Green for east/west (around π/2 and 3π/2 radians)
    
    # Create a legend handles list
    legend_handles = [
        plt.Line2D([0], [0], color='blue', lw=2, label='North/South Initial Heading'),
        plt.Line2D([0], [0], color='green', lw=2, label='East/West Initial Heading')
    ]
    
    for result in successful_results:
        # Extract path data
        path = result['results']['x']
        start_x, start_y, start_theta = result['start']
        
        # Normalize angle to [0, 2π)
        normalized_theta = start_theta % (2 * np.pi)
        
        # Determine color based on heading direction
        # North/South: around 0, π (with some tolerance)
        # East/West: around π/2, 3π/2 (with some tolerance)
        north_south_angles = [0, np.pi]
        east_west_angles = [np.pi/2, 3*np.pi/2]
        
        # Check if angle is close to north/south or east/west
        # Using a tolerance of π/4 radians (45 degrees)
        tolerance = np.pi/4
        
        is_north_south = any(abs((normalized_theta - angle + np.pi) % (2*np.pi) - np.pi) < tolerance 
                             for angle in north_south_angles)
        
        # If it's not north/south, it must be east/west
        if is_north_south:
            color = 'blue'
        else:
            color = 'green'
        
        # Plot path
        ax.plot(path[:, 0], path[:, 1], '-', color=color, linewidth=1, alpha=0.7)
        
        # Plot start position with an arrow to show heading
        arrow_length = 1.0
        dx = arrow_length * np.cos(start_theta)
        dy = arrow_length * np.sin(start_theta)
        ax.arrow(start_x, start_y, dx, dy, head_width=0.3, head_length=0.5, 
                fc=color, ec=color, alpha=0.7)
    
    # Add grid
    ax.grid(True)
    
    # Set equal aspect ratio
    ax.set_aspect('equal')
    
    # Add labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'Path Planning from Multiple Initial Positions\n{len(successful_results)} successful paths')
    
    # Add legend
    ax.legend(handles=legend_handles, loc='best')
    
    # Tight layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    # Show plot
    plt.show()

def plot_comparison(default_results, nn_results, ego, computation_times=None, durations=None, save_path=None):
    """
    Plot comparison between default and neural network initialization results.
    
    Args:
        default_results: Dictionary containing results from default initialization
        nn_results: Dictionary containing results from neural network initialization
        ego: Object containing vehicle parameters
        computation_times: Optional tuple of (default_time, nn_time) for displaying computation times
        durations: Optional tuple of (default_duration, nn_duration) for displaying durations
        save_path: Optional path to save the plot
    """
    # Create subplots (3x2 grid)
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(12, 10))
    
    # Time arrays
    t_default = default_results['t']
    t_nn = nn_results['t']
    
    # Plot trajectories
    ax1.plot(default_results['x'][:, 0], default_results['x'][:, 1], 'b-', label='Default Init')
    ax1.plot(nn_results['x'][:, 0], nn_results['x'][:, 1], 'r-', label='NN Init')
    ax1.plot(ego.state_start[0], ego.state_start[1], 'go', markersize=10, label='Start')
    ax1.plot(ego.state_final[0], ego.state_final[1], 'ko', markersize=10, label='Goal')
    
    # Add car orientation arrows
    arrow_step = 5  # Plot an arrow every 5 steps
    max_arrows = min(len(t_default), len(t_nn))  # Use the shorter trajectory length
    for i in range(0, max_arrows, arrow_step):
        # Default initialization arrows
        dx = np.cos(default_results['x'][i, 2])
        dy = np.sin(default_results['x'][i, 2])
        ax1.quiver(default_results['x'][i, 0], default_results['x'][i, 1], dx, dy,
                  color='b', scale=10, alpha=0.3)
        
        # Neural network initialization arrows
        dx = np.cos(nn_results['x'][i, 2])
        dy = np.sin(nn_results['x'][i, 2])
        ax1.quiver(nn_results['x'][i, 0], nn_results['x'][i, 1], dx, dy,
                  color='r', scale=10, alpha=0.3)
    
    ax1.grid(True)
    ax1.set_aspect('equal')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    
    # Add computation times and durations to title if provided
    title = 'Path Comparison\n'
    if computation_times:
        default_time, nn_time = computation_times
        improvement = (default_time - nn_time) / default_time * 100
        title += f'Solve time: Default={default_time:.3f}s, NN={nn_time:.3f}s ({improvement:.1f}% faster)\n'
    if durations:
        default_duration, nn_duration = durations
        title += f'Duration: Default={default_duration:.1f}s, NN={nn_duration:.1f}s'
    ax1.set_title(title)
    ax1.legend()
    
    # Plot heading
    ax2.plot(t_default, np.rad2deg(default_results['x'][:, 2]), 'b-', label='Default Init')
    ax2.plot(t_nn, np.rad2deg(nn_results['x'][:, 2]), 'r-', label='NN Init')
    ax2.axhline(y=np.rad2deg(ego.state_final[2]), color='k', linestyle='--', label='Target')
    ax2.grid(True)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Heading (deg)')
    ax2.set_title('Heading Angle')
    ax2.legend()
    
    # Plot velocity
    ax3.plot(t_default, default_results['x'][:, 3], 'b-', label='Default Init')
    ax3.plot(t_nn, nn_results['x'][:, 3], 'r-', label='NN Init')
    ax3.axhline(y=ego.velocity_max, color='k', linestyle='--', alpha=0.3)
    ax3.axhline(y=ego.velocity_min, color='k', linestyle='--', alpha=0.3)
    ax3.grid(True)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Velocity (m/s)')
    ax3.set_title('Velocity Profile')
    ax3.legend()
    
    # Plot steering angle
    ax4.plot(t_default, np.rad2deg(default_results['x'][:, 4]), 'b-', label='Default Init')
    ax4.plot(t_nn, np.rad2deg(nn_results['x'][:, 4]), 'r-', label='NN Init')
    ax4.axhline(y=np.rad2deg(ego.steering_max), color='k', linestyle='--', alpha=0.3)
    ax4.axhline(y=np.rad2deg(ego.steering_min), color='k', linestyle='--', alpha=0.3)
    ax4.grid(True)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Steering Angle (deg)')
    ax4.set_title('Steering Profile')
    ax4.legend()
    
    # Plot acceleration
    ax5.plot(t_default[:-1], default_results['u'][:, 0], 'b-', label='Default Init')
    ax5.plot(t_nn[:-1], nn_results['u'][:, 0], 'r-', label='NN Init')
    ax5.axhline(y=ego.acceleration_max, color='k', linestyle='--', alpha=0.3)
    ax5.axhline(y=ego.acceleration_min, color='k', linestyle='--', alpha=0.3)
    ax5.grid(True)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Acceleration (m/s²)')
    ax5.set_title('Acceleration Profile')
    ax5.legend()
    
    # Plot steering rate
    ax6.plot(t_default[:-1], np.rad2deg(default_results['u'][:, 1]), 'b-', label='Default Init')
    ax6.plot(t_nn[:-1], np.rad2deg(nn_results['u'][:, 1]), 'r-', label='NN Init')
    ax6.axhline(y=np.rad2deg(ego.steering_rate_max), color='k', linestyle='--', alpha=0.3)
    ax6.axhline(y=np.rad2deg(ego.steering_rate_min), color='k', linestyle='--', alpha=0.3)
    ax6.grid(True)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Steering Rate (deg/s)')
    ax6.set_title('Steering Rate Profile')
    ax6.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close() 