#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arrow, Polygon, Circle
import matplotlib.animation as animation
from matplotlib.transforms import Affine2D

def load_sequential_results():
    """Load the sequential path planning results."""
    docs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'docs')
    results_file = os.path.join(docs_dir, "sequential_path_results.npy")
    return np.load(results_file, allow_pickle=True).item()

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

def create_animation():
    """Create animation of the vehicle following the path."""
    # Load results
    results = load_sequential_results()
    trajectory = results['x']
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Add road lines based on the diagram
    # Blue lines (main roads)
    blue_road_segments = [
        # Top row
        [(-40, 40), (-30, 40)],
        [(-30, 40), (-30, 50)],
        [(0, 40), (0, 50)],
        [(0, 40), (50, 40)],
        
        # Middle row
        [(0, 25), (50, 25)],
        
        # Bottom row
        [(-40, 10), (-30, 10)],
        [(-30, 10), (-30, -30)],
        [(0, 10), (0, -30)],
        [(0, 10), (30, 10)],
        [(30, 10), (30, -30)],
        [(45, 10), (45, -30)],
        [(45, 10), (50, 10)]
    ]
    
    # Orange lines (secondary roads)
    orange_road_segments = [
        # Top row
        [(-15, 40), (-15, 50)],
        
        # Middle row
        [(-40, 25), (-30, 25)],
        
        # Bottom row
        [(-15, 10), (-15, -30)]
    ]
    
    # Plot blue road segments
    for start, end in blue_road_segments:
        ax.plot([start[0], end[0]], [start[1], end[1]], 'b-', linewidth=2.5)
    
    # Plot orange road segments
    for start, end in orange_road_segments:
        ax.plot([start[0], end[0]], [start[1], end[1]], color='orange', linewidth=2.5)
    
    # Plot the complete path
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'b--', alpha=0.3, label='Path')
    
    # Plot waypoints
    waypoints = {
        'A': (30, 35),
        'B': (0, 30),
        'C': (0, 20),
        'D': (30, 15),
        'E': (37, 8),
        'F': (37, -22)
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
    ax.set_aspect('equal')
    ax.set_xlabel('X position (m)')
    ax.set_ylabel('Y position (m)')
    ax.set_title('Vehicle Path Animation: A to F')
    ax.legend()
    
    # Set plot limits to show the complete path with some padding
    ax.set_xlim(-45, 55)  # Extend to show full road network
    ax.set_ylim(-35, 55)  # Extend to show full road network
    
    # Initialize vehicle patches
    vehicle_parts = []
    
    def init():
        """Initialize animation."""
        return []
    
    def animate(frame):
        """Update animation frame."""
        nonlocal vehicle_parts
        
        # Remove previous vehicle patches
        for part in vehicle_parts:
            part.remove()
        vehicle_parts = []
        
        # Create new vehicle patches
        x, y, theta = trajectory[frame, 0:3]
        vehicle_parts = create_vehicle_patches(x, y, theta, color='royalblue')
        
        # Add patches to plot
        for part in vehicle_parts:
            ax.add_patch(part)
        
        # Add progress text
        progress = frame / len(trajectory) * 100
        ax.set_title(f'Vehicle Path Animation: A to F (Progress: {progress:.1f}%)')
        
        return vehicle_parts
    
    # Create animation
    frames = len(trajectory)
    interval = 16  # milliseconds between frames (3x faster than original 50ms)
    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=frames, interval=interval, blit=True
    )
    
    # Save animation
    docs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'docs')
    output_file = os.path.join(docs_dir, 'path_animation.mp4')
    
    # Save animation with high quality
    anim.save(output_file, writer='ffmpeg', fps=60,  # Increased fps from 20 to 60 for smoother animation
              dpi=200, bitrate=2000,
              metadata={'title': 'Vehicle Path Animation'})
    
    plt.close()
    print(f"Animation saved to: {output_file}")

if __name__ == "__main__":
    create_animation() 