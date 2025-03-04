#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arrow
import matplotlib.animation as animation

def load_sequential_results():
    """Load the sequential path planning results."""
    docs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'docs')
    results_file = os.path.join(docs_dir, "sequential_path_results.npy")
    return np.load(results_file, allow_pickle=True).item()

def create_vehicle_patches(x, y, theta, color='blue', alpha=0.7):
    """Create patches representing the vehicle."""
    # Vehicle dimensions
    length = 2.7  # meters
    width = 1.8   # meters
    
    # Calculate vehicle corners
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    # Calculate the corner position for a centered rectangle
    corner_x = x - (length/2 * cos_theta - width/2 * sin_theta)
    corner_y = y - (length/2 * sin_theta + width/2 * cos_theta)
    
    # Create vehicle rectangle
    vehicle = Rectangle(
        (corner_x, corner_y),
        length, width,
        angle=np.degrees(theta),
        facecolor=color,
        alpha=alpha,
        edgecolor='black'
    )
    
    # Create direction arrow at the center
    arrow_length = length * 0.8
    arrow = Arrow(
        x, y,
        arrow_length * cos_theta,
        arrow_length * sin_theta,
        width=width * 0.5,
        color='white'
    )
    
    return vehicle, arrow

def create_animation():
    """Create animation of the vehicle following the path."""
    # Load results
    results = load_sequential_results()
    trajectory = results['x']
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot the complete path
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'b--', alpha=0.3, label='Path')
    
    # Plot waypoints
    waypoints = {
        'A': (15, 35),
        'B': (0, 30),
        'C': (0, 20),
        'D': (15, 15),
        'E': (20, 10),
        'F': (20, 0)
    }
    
    for label, (x, y) in waypoints.items():
        if label == 'A':
            ax.plot(x, y, 'go', markersize=10, label='Start')
        elif label == 'F':
            ax.plot(x, y, 'ro', markersize=10, label='Goal')
        else:
            ax.plot(x, y, 'ko', markersize=8)
        ax.text(x + 1, y + 1, label, fontsize=12)
    
    # Set plot properties
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlabel('X position (m)')
    ax.set_ylabel('Y position (m)')
    ax.set_title('Vehicle Path Animation: A to F')
    ax.legend()
    
    # Set plot limits with extended left side view
    ax.set_xlim(-10, 25)  # Extended left limit from -5 to -10
    ax.set_ylim(-5, 40)  # Keeping y limits the same
    
    # Initialize vehicle patches
    vehicle_patch = None
    arrow_patch = None
    
    def init():
        """Initialize animation."""
        return []
    
    def animate(frame):
        """Update animation frame."""
        nonlocal vehicle_patch, arrow_patch
        
        # Remove previous vehicle patches
        if vehicle_patch is not None:
            vehicle_patch.remove()
        if arrow_patch is not None:
            arrow_patch.remove()
        
        # Create new vehicle patches
        x, y, theta = trajectory[frame, 0:3]
        vehicle_patch, arrow_patch = create_vehicle_patches(x, y, theta)
        
        # Add patches to plot
        ax.add_patch(vehicle_patch)
        ax.add_patch(arrow_patch)
        
        # Add progress text
        progress = frame / len(trajectory) * 100
        ax.set_title(f'Vehicle Path Animation: A to F (Progress: {progress:.1f}%)')
        
        return [vehicle_patch, arrow_patch]
    
    # Create animation
    frames = len(trajectory)
    interval = 50  # milliseconds between frames
    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=frames, interval=interval, blit=True
    )
    
    # Save animation
    docs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'docs')
    output_file = os.path.join(docs_dir, 'path_animation.mp4')
    
    # Save animation with high quality
    anim.save(output_file, writer='ffmpeg', fps=20,
              dpi=200, bitrate=2000,
              metadata={'title': 'Vehicle Path Animation'})
    
    plt.close()
    print(f"Animation saved to: {output_file}")

if __name__ == "__main__":
    create_animation() 