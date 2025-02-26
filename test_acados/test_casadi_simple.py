#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import casadi as ca

def main():
    """
    Very simple optimization example using CasADi.
    This example finds the minimum of a simple function.
    """
    print("Testing CasADi with a very simple optimization example...")
    
    # Create optimization variables
    x = ca.SX.sym('x')
    y = ca.SX.sym('y')
    
    # Define objective function: f(x,y) = x^2 + y^2
    f = x**2 + y**2
    
    # Create an optimization problem
    nlp = {'x': ca.vertcat(x, y), 'f': f}
    
    # Create solver
    solver = ca.nlpsol('solver', 'ipopt', nlp, {'ipopt.print_level': 0, 'print_time': 0})
    
    # Solve the problem
    print("Solving optimization problem...")
    sol = solver(x0=[10, 10])
    
    # Get the solution
    x_opt = sol['x']
    
    print(f"Optimal solution: x = {float(x_opt[0]):.4f}, y = {float(x_opt[1]):.4f}")
    print(f"Minimum value: f(x,y) = {float(sol['f']):.4f}")
    
    # Create a simple plot
    x_vals = np.linspace(-5, 5, 100)
    y_vals = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = X**2 + Y**2
    
    plt.figure(figsize=(10, 8))
    
    # Contour plot
    plt.contourf(X, Y, Z, 20, cmap='viridis')
    plt.colorbar(label='f(x,y) = x² + y²')
    
    # Mark the optimal point
    plt.plot(float(x_opt[0]), float(x_opt[1]), 'ro', markersize=10)
    plt.text(float(x_opt[0])+0.1, float(x_opt[1])+0.1, 'Minimum', fontsize=12)
    
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Simple Optimization with CasADi')
    
    plt.tight_layout()
    # Save plot
    plt.savefig('test_acados/casadi_simple_results.png')
    plt.close()
    print("Results plot saved as 'test_acados/casadi_simple_results.png'")
    
    print("CasADi simple test completed successfully!")

if __name__ == "__main__":
    main() 