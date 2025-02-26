#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import casadi as ca

def main():
    """
    Simple optimal control example using CasADi.
    This example solves a linear quadratic regulator problem
    for a simple double integrator system.
    """
    print("Testing CasADi with a simple optimal control example...")
    
    # Problem dimensions
    nx = 2  # states: [position, velocity]
    nu = 1  # controls: [acceleration]
    N = 20  # number of control intervals
    
    # Time horizon
    T = 5.0  # seconds
    dt = T/N  # time step
    
    # Initial and target states
    x0 = [1.0, 0.0]  # initial state: position=1, velocity=0
    xf = [0.0, 0.0]  # target state: position=0, velocity=0
    
    # Cost weights
    Q = np.diag([1.0, 0.1])  # State cost
    R = np.array([[0.01]])   # Control cost
    
    # Control bounds
    u_min = -2.0
    u_max = 2.0
    
    # Create optimization variables
    opti = ca.Opti()
    
    # Decision variables
    X = opti.variable(nx, N+1)  # state trajectory
    U = opti.variable(nu, N)    # control trajectory
    
    # Cost function
    cost = 0
    
    # Add stage cost
    for k in range(N):
        x_err = X[:, k] - np.array(xf).reshape(nx, 1)
        cost += ca.mtimes([x_err.T, Q, x_err]) + ca.mtimes([U[:, k].T, R, U[:, k]])
    
    # Add terminal cost
    x_err = X[:, N] - np.array(xf).reshape(nx, 1)
    cost += ca.mtimes([x_err.T, Q, x_err])
    
    # Set objective
    opti.minimize(cost)
    
    # System dynamics (double integrator)
    A = np.array([[1.0, dt], [0.0, 1.0]])
    B = np.array([[0.5*dt*dt], [dt]])
    
    # Initial condition constraint
    opti.subject_to(X[:, 0] == x0)
    
    # Dynamic constraints
    for k in range(N):
        opti.subject_to(X[:, k+1] == ca.mtimes(A, X[:, k]) + ca.mtimes(B, U[:, k]))
    
    # Control constraints
    for k in range(N):
        opti.subject_to(opti.bounded(u_min, U[:, k], u_max))
    
    # Set solver options
    opts = {"ipopt.print_level": 0, "print_time": 0}
    opti.solver('ipopt', opts)
    
    # Solve the optimization problem
    print("Solving optimization problem...")
    sol = opti.solve()
    
    # Extract solution
    x_sol = np.array(sol.value(X))
    u_sol = np.array(sol.value(U)).flatten()  # Convert to 1D array
    
    # Print results
    print(f"Initial state: position = {x0[0]:.2f}, velocity = {x0[1]:.2f}")
    print(f"Final state: position = {x_sol[0, N]:.4f}, velocity = {x_sol[1, N]:.4f}")
    print("Control inputs (acceleration):")
    for i in range(min(5, N)):
        print(f"  u[{i}] = {u_sol[i]:.4f}")  # Access as 1D array
    if N > 5:
        print("  ...")
    
    # Plot results
    t = np.linspace(0, T, N+1)
    t_u = np.linspace(0, T, N)
    
    plt.figure(figsize=(10, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(t, x_sol[0, :])
    plt.grid(True)
    plt.ylabel('Position')
    plt.title('Double Integrator Optimal Control (CasADi)')
    
    plt.subplot(3, 1, 2)
    plt.plot(t, x_sol[1, :])
    plt.grid(True)
    plt.ylabel('Velocity')
    
    plt.subplot(3, 1, 3)
    plt.step(t_u, u_sol, where='post')  # Plot u_sol directly as 1D array
    plt.grid(True)
    plt.ylabel('Acceleration')
    plt.xlabel('Time [s]')
    
    plt.tight_layout()
    plt.savefig('test_acados/casadi_results.png')
    plt.close()
    print("Results plot saved as 'test_acados/casadi_results.png'")
    
    print("CasADi test completed successfully!")

if __name__ == "__main__":
    main() 