#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def corridor_cal(point_a: tuple, point_b: tuple, distance: float) -> tuple:
    """
    Calculate corridor parameters around a line segment defined by points A and B.
    
    Args:
        point_a (tuple): (x, y) coordinates of point A
        point_b (tuple): (x, y) coordinates of point B
        distance (float): perpendicular distance from the center line to corridor boundary
        
    Returns:
        tuple: (a1, b1, a2, b2) where:
            The corridor is defined by two inequalities:
            1. y + a1*x + b1 > 0  (points above this line are above the upper boundary)
            2. y + a2*x + b2 < 0  (points below this line are below the lower boundary)
            Points satisfying BOTH inequalities are inside the corridor.
            
            In other words:
            y = -a1*x - b1  defines the upper boundary line
            y = -a2*x - b2  defines the lower boundary line
    """
    # Extract coordinates
    x1, y1 = point_a
    x2, y2 = point_b
    
    # Calculate line direction vector
    dx = x2 - x1
    dy = y2 - y1
    
    # Normalize direction vector
    length = np.sqrt(dx**2 + dy**2)
    if length == 0:
        raise ValueError("Points A and B cannot be the same")
        
    dx, dy = dx/length, dy/length
    
    # Calculate normal vector (rotate direction vector by 90 degrees)
    nx = -dy
    ny = dx
    
    # For nearly vertical lines, use vertical line handling
    if abs(dx) < 1e-10:
        return 1.0, -point_a[0] + distance, 1.0, -point_a[0] - distance
    
    # Calculate line slope and intercept
    m = dy/dx
    
    # Calculate parallel lines offset by distance
    d_offset = distance * np.sqrt(1 + m**2)
    
    # For upper boundary (y + a1*x + b1 > 0)
    a1 = -m
    b1 = y1 - m*x1 + d_offset
    
    # For lower boundary (y + a2*x + b2 < 0)
    a2 = -m
    b2 = y1 - m*x1 - d_offset
    
    return a1, b1, a2, b2

def verify_point_in_corridor(point: tuple, a1: float, b1: float, a2: float, b2: float) -> bool:
    """
    Verify if a point lies within the corridor.
    
    Args:
        point (tuple): (x, y) coordinates of the point to check
        a1, b1, a2, b2 (float): Corridor parameters
        
    Returns:
        bool: True if point is inside corridor, False otherwise
    """
    x, y = point
    upper_bound = y + a1*x + b1
    lower_bound = y + a2*x + b2
    
    return upper_bound > 0 and lower_bound < 0

def plot_corridor(point_a: tuple, point_b: tuple, a1: float, b1: float, a2: float, b2: float, distance: float):
    """
    Visualize the corridor with test points.
    
    Args:
        point_a (tuple): Start point of the line segment
        point_b (tuple): End point of the line segment
        a1, b1 (float): Parameters for upper boundary
        a2, b2 (float): Parameters for lower boundary
        distance (float): Corridor width
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot original line segment
    ax.plot([point_a[0], point_b[0]], [point_a[1], point_b[1]], 'k-', label='Original Line')
    
    # Calculate plot bounds
    margin = distance * 2
    x_min = min(point_a[0], point_b[0]) - margin
    x_max = max(point_a[0], point_b[0]) + margin
    y_min = min(point_a[1], point_b[1]) - margin
    y_max = max(point_a[1], point_b[1]) + margin
    
    # Generate points for corridor boundaries
    x = np.linspace(x_min, x_max, 100)
    
    # Plot corridor boundaries
    y_upper = -a1 * x - b1
    y_lower = -a2 * x - b2
    
    # Clip the corridor to a reasonable region around the line segment
    mask = (x >= x_min) & (x <= x_max) & (y_upper >= y_min) & (y_upper <= y_max)
    ax.plot(x[mask], y_upper[mask], 'r--', label='Upper Boundary')
    ax.plot(x[mask], y_lower[mask], 'b--', label='Lower Boundary')
    
    # Generate and plot test points
    np.random.seed(42)  # For reproducibility
    n_points = 50
    x_range = np.random.uniform(x_min, x_max, n_points)
    y_range = np.random.uniform(y_min, y_max, n_points)
    
    for x_test, y_test in zip(x_range, y_range):
        point = (x_test, y_test)
        is_inside = verify_point_in_corridor(point, a1, b1, a2, b2)
        ax.scatter(x_test, y_test, 
                  color='green' if is_inside else 'red',
                  alpha=0.6,
                  marker='o' if is_inside else 'x')
    
    # Add legend for points
    ax.scatter([], [], color='green', alpha=0.6, label='Points Inside Corridor')
    ax.scatter([], [], color='red', alpha=0.6, marker='x', label='Points Outside Corridor')
    
    # Plot start and end points
    ax.scatter(point_a[0], point_a[1], color='black', s=100, label='Start Point')
    ax.scatter(point_b[0], point_b[1], color='black', s=100, label='End Point')
    
    # Customize plot
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Corridor Visualization')
    ax.legend()
    
    # Set axis limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Make axes equal for better visualization
    ax.set_aspect('equal')
    
    return fig, ax

if __name__ == "__main__":
    # Example usage
    point_a = (0, 0)
    point_b = (10, 10)
    distance = 2.0
    
    a1, b1, a2, b2 = corridor_cal(point_a, point_b, distance)
    print(f"Corridor parameters:")
    print(f"Upper boundary: y + {a1:.3f}x + {b1:.3f} > 0")
    print(f"Lower boundary: y + {a2:.3f}x + {b2:.3f} < 0")
    
    # Create visualization
    fig, ax = plot_corridor(point_a, point_b, a1, b1, a2, b2, distance)
    plt.show()
    
    # Test some specific points
    test_points = [(5, 5), (5, 7), (5, 3)]
    for point in test_points:
        is_inside = verify_point_in_corridor(point, a1, b1, a2, b2)
        print(f"Test point {point} is {'inside' if is_inside else 'outside'} the corridor") 