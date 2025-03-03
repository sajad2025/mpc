#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def load_trajectory_dataset(dataset_path='models/trajectory_dataset.npz'):
    """Load the training dataset."""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    data = np.load(dataset_path)
    return data

def calculate_distances(inputs):
    """Calculate distances from start to end positions for each trajectory.
    
    Args:
        inputs: Array of input features where indices 0:2 are start x,y and 5:7 are end x,y
    """
    # Extract start and end positions
    start_positions = inputs[:, :2]  # x, y of start state
    end_positions = inputs[:, 5:7]   # x, y of end state
    
    # Calculate Euclidean distances
    distances = np.sqrt(np.sum((end_positions - start_positions)**2, axis=1))
    return distances

def main():
    # Create docs directory if it doesn't exist
    os.makedirs('docs', exist_ok=True)
    
    # Load the dataset
    data = load_trajectory_dataset()
    inputs = data['inputs']
    durations = data['durations']
    
    # Calculate distances
    distances = calculate_distances(inputs)
    
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(distances, durations)
    line = slope * distances + intercept
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create scatter plot with different colors for different final headings
    headings = inputs[:, 7]  # Index 7 is the final heading
    unique_headings = np.unique(headings)
    colors = ['blue', 'red', 'green', 'purple']
    
    # Create a list to store scatter plot objects
    scatter_plots = []
    
    for i, heading in enumerate(unique_headings):
        mask = (headings == heading)
        scatter = ax.scatter(distances[mask], durations[mask], alpha=0.5, 
                           label=f'Final heading = {np.rad2deg(heading):.0f}°',
                           color=colors[i], picker=True)
        scatter_plots.append(scatter)
    
    # Plot the linear fit
    x_fit = np.array([0, np.max(distances)])
    y_fit = slope * x_fit + intercept
    ax.plot(x_fit, y_fit, 'k--', label=f'Linear Fit', linewidth=2)
    
    # Create annotation object (initially empty)
    annot = ax.annotate("", xy=(0,0), xytext=(10,10), textcoords="offset points",
                       bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9),
                       arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)
    
    def update_annot(ind, scatter):
        """Update the annotation with the data for the hovered point"""
        pos = scatter.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        
        # Get the index in the original data
        mask = distances == pos[0]
        mask &= durations == pos[1]
        data_idx = np.where(mask)[0][0]
        
        # Get goal state information
        goal_x = inputs[data_idx, 5]  # Index 5 is goal x
        goal_y = inputs[data_idx, 6]  # Index 6 is goal y
        goal_theta = inputs[data_idx, 7]  # Index 7 is goal theta
        
        text = f'Goal state:\nx: {goal_x:.1f}m\ny: {goal_y:.1f}m\nθ: {np.rad2deg(goal_theta):.0f}°\nDuration: {durations[data_idx]:.1f}s'
        annot.set_text(text)
    
    def hover(event):
        """Handle mouse hover events"""
        if event.inaxes == ax:
            for scatter in scatter_plots:
                cont, ind = scatter.contains(event)
                if cont:
                    update_annot(ind, scatter)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                    return
            annot.set_visible(False)
            fig.canvas.draw_idle()
    
    # Connect the hover event
    fig.canvas.mpl_connect("motion_notify_event", hover)
    
    plt.xlabel('Distance (m)')
    plt.ylabel('Duration (s)')
    plt.title(f'Duration vs Distance (n={len(distances)} samples)\n'
              f'Linear Fit: duration = {slope:.3f} * distance + {intercept:.3f}\n'
              f'R² = {r_value**2:.3f}')
    
    # Set axis limits to show the full range
    plt.xlim(-5, 75)  # Add some padding around the expected range (0-70.7m)
    plt.ylim(0, np.max(durations) * 1.1)  # Add 10% padding above max duration
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save the plot first
    plt.savefig('docs/duration_vs_distance_raw.png', dpi=300, bbox_inches='tight')
    
    # Print statistics
    print(f"Plot saved as docs/duration_vs_distance_raw.png")
    print(f"Number of samples: {len(distances)}")
    print(f"Linear fit: duration = {slope:.3f} * distance + {intercept:.3f}")
    print(f"R² value: {r_value**2:.3f}")
    
    print(f"\nDistance statistics:")
    print(f"  Min distance: {np.min(distances):.1f}m")
    print(f"  Max distance: {np.max(distances):.1f}m")
    print(f"  Mean distance: {np.mean(distances):.1f}m")
    print(f"\nDuration statistics:")
    print(f"  Min duration: {np.min(durations):.1f}s")
    print(f"  Max duration: {np.max(durations):.1f}s")
    print(f"  Mean duration: {np.mean(durations):.1f}s")
    
    # Show the plot and keep window open
    plt.show()

if __name__ == "__main__":
    main() 