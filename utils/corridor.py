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
        tuple: (c1, a1, b1, c2, a2, b2) where:
            The corridor is defined by two inequalities:
            1. c1*y + a1*x + b1 > 0  (points above this line are above the upper boundary)
            2. c2*y + a2*x + b2 < 0  (points below this line are below the lower boundary)
            Points satisfying BOTH inequalities are inside the corridor.
            
            All coefficients are normalized so that sqrt(c^2 + a^2) = 1 for each boundary.
            
            In other words:
            c1*y = -a1*x - b1  defines the upper boundary line
            c2*y = -a2*x - b2  defines the lower boundary line
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
    # This is already normalized since (dx, dy) was normalized
    nx = -dy  # Normal vector x component
    ny = dx   # Normal vector y component
    
    # The normal vector (nx, ny) is already normalized and gives us our coefficients
    # For the line equation: nx*x + ny*y + d = 0
    c = ny    # coefficient of y
    a = nx    # coefficient of x
    
    # Calculate the offset for both boundaries using the normal vector
    # The base offset is -(nx*x1 + ny*y1) which gives us distance from origin
    base_offset = -(nx*x1 + ny*y1)
    
    # For upper boundary (c*y + a*x + b1 > 0)
    b1 = base_offset + distance
    
    # For lower boundary (c*y + a*x + b2 < 0)
    b2 = base_offset - distance
    
    return c, a, b1, c, a, b2

def verify_point_in_corridor(point: tuple, c1: float, a1: float, b1: float, c2: float, a2: float, b2: float) -> bool:
    """
    Verify if a point lies within the corridor.
    
    Args:
        point (tuple): (x, y) coordinates of the point to check
        c1, a1, b1, c2, a2, b2 (float): Normalized corridor parameters
        
    Returns:
        bool: True if point is inside corridor, False otherwise
    """
    x, y = point
    upper_bound = c1*y + a1*x + b1
    lower_bound = c2*y + a2*x + b2
    
    return upper_bound > 0 and lower_bound < 0

def plot_corridor(point_a: tuple, point_b: tuple, c1: float, a1: float, b1: float, c2: float, a2: float, b2: float, distance: float):
    """
    Visualize the corridor with test points.
    
    Args:
        point_a (tuple): Start point of the line segment
        point_b (tuple): End point of the line segment
        c1, a1, b1 (float): Parameters for upper boundary
        c2, a2, b2 (float): Parameters for lower boundary
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
    
    # Use a parametric approach to draw boundary lines
    # This works for all line orientations (vertical, horizontal, diagonal)
    t = np.linspace(-1, 2, 300)  # Parametric range covering the segment and beyond
    
    # Calculate direction vector of the line
    dx = point_b[0] - point_a[0]
    dy = point_b[1] - point_a[1]
    
    # Normalize
    length = np.sqrt(dx**2 + dy**2)
    dx, dy = dx/length, dy/length
    
    # Normal vector (perpendicular to direction)
    nx, ny = -dy, dx
    
    # Generate boundary lines using parametric equations
    # Line segment: point_a + t * (point_b - point_a), t ∈ [0,1]
    # Extended to t ∈ [-1,2] to show more of the corridor
    
    # Points along the original line (extended)
    line_x = point_a[0] + t * dx * length
    line_y = point_a[1] + t * dy * length
    
    # Upper boundary
    upper_x = line_x + nx * distance
    upper_y = line_y + ny * distance
    
    # Lower boundary
    lower_x = line_x - nx * distance
    lower_y = line_y - ny * distance
    
    # Clip points to the visible area
    mask_upper = (upper_x >= x_min) & (upper_x <= x_max) & (upper_y >= y_min) & (upper_y <= y_max)
    mask_lower = (lower_x >= x_min) & (lower_x <= x_max) & (lower_y >= y_min) & (lower_y <= y_max)
    
    # Plot the boundary lines
    ax.plot(upper_x[mask_upper], upper_y[mask_upper], 'r--', label='Upper Boundary')
    ax.plot(lower_x[mask_lower], lower_y[mask_lower], 'b--', label='Lower Boundary')
    
    # Generate and plot test points
    np.random.seed(42)  # For reproducibility
    n_points = 50
    x_range = np.random.uniform(x_min, x_max, n_points)
    y_range = np.random.uniform(y_min, y_max, n_points)
    
    for x_test, y_test in zip(x_range, y_range):
        point = (x_test, y_test)
        is_inside = verify_point_in_corridor(point, c1, a1, b1, c2, a2, b2)
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
    # Test Case 1: Diagonal line
    point_a_diagonal = (0, 0)
    point_b_diagonal = (10, 10)
    distance = 2.0
    
    c1, a1, b1, c2, a2, b2 = corridor_cal(point_a_diagonal, point_b_diagonal, distance)
    print(f"\nDiagonal corridor parameters:")
    print(f"Upper boundary: {c1:.3f}y + {a1:.3f}x + {b1:.3f} > 0")
    print(f"Lower boundary: {c2:.3f}y + {a2:.3f}x + {b2:.3f} < 0")
    
    # Create visualization for diagonal case
    fig1, ax1 = plot_corridor(point_a_diagonal, point_b_diagonal, c1, a1, b1, c2, a2, b2, distance)
    ax1.set_title('Corridor Visualization - Diagonal Line')
    # fig1.savefig('utils/corridor_diagonal.png', dpi=300, bbox_inches='tight')
    
    # Test Case 2: Horizontal line
    point_a_horizontal = (0, 5)
    point_b_horizontal = (10, 5)
    
    c1, a1, b1, c2, a2, b2 = corridor_cal(point_a_horizontal, point_b_horizontal, distance)
    print(f"\nHorizontal corridor parameters:")
    print(f"Upper boundary: {c1:.3f}y + {a1:.3f}x + {b1:.3f} > 0")
    print(f"Lower boundary: {c2:.3f}y + {a2:.3f}x + {b2:.3f} < 0")
    
    # Create visualization for horizontal case
    fig2, ax2 = plot_corridor(point_a_horizontal, point_b_horizontal, c1, a1, b1, c2, a2, b2, distance)
    ax2.set_title('Corridor Visualization - Horizontal Line')
    # fig2.savefig('utils/corridor_horizontal.png', dpi=300, bbox_inches='tight')
    
    # Test Case 3: Vertical line
    point_a_vertical = (5, 0)
    point_b_vertical = (5, 10)
    
    c1, a1, b1, c2, a2, b2 = corridor_cal(point_a_vertical, point_b_vertical, distance)
    print(f"\nVertical corridor parameters:")
    print(f"Upper boundary: {c1:.3f}y + {a1:.3f}x + {b1:.3f} > 0")
    print(f"Lower boundary: {c2:.3f}y + {a2:.3f}x + {b2:.3f} < 0")
    
    # Create visualization for vertical case
    fig3, ax3 = plot_corridor(point_a_vertical, point_b_vertical, c1, a1, b1, c2, a2, b2, distance)
    ax3.set_title('Corridor Visualization - Vertical Line')
    # fig3.savefig('utils/corridor_vertical.png', dpi=300, bbox_inches='tight')
    
    # Test some specific points for each case
    test_points_diagonal = [(5, 5), (5, 7), (5, 3)]
    print("\nTesting diagonal corridor:")
    for point in test_points_diagonal:
        is_inside = verify_point_in_corridor(point, c1, a1, b1, c2, a2, b2)
        print(f"Test point {point} is {'inside' if is_inside else 'outside'} the corridor")
    
    plt.show() 