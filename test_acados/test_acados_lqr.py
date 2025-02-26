#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
from acados_template import AcadosOcp, AcadosOcpSolver
import json

def main():
    """
    Simple LQR example using Acados.
    This example solves a linear quadratic regulator problem
    for a simple double integrator system.
    """
    print("Testing Acados with a simple LQR example...")
    
    # Create an Acados OCP object
    ocp = AcadosOcp()
    
    # OCP dimensions
    nx = 2  # states: [position, velocity]
    nu = 1  # controls: [acceleration]
    N = 20  # number of shooting nodes
    
    # OCP time horizon
    T = 5.0  # seconds
    
    # Model dynamics (double integrator)
    x = ca.SX.sym('x', nx)
    u = ca.SX.sym('u', nu)
    
    # State variables
    position = x[0]
    velocity = x[1]
    
    # Control variables
    acceleration = u[0]
    
    # Linear dynamics: x_dot = Ax + Bu
    # [position_dot]   = [0 1] [position]   + [0] [acceleration]
    # [velocity_dot]     [0 0] [velocity]     [1]
    f_expl = ca.vertcat(
        velocity,
        acceleration
    )
    
    # Set model
    ocp.model.x = x
    ocp.model.u = u
    ocp.model.f_expl_expr = f_expl
    ocp.model.name = 'double_integrator'
    
    # Set dimensions
    ocp.dims.N = N
    
    # Set cost (standard LQR cost)
    ocp.cost.cost_type = 'LINEAR_LS'
    ocp.cost.cost_type_e = 'LINEAR_LS'
    
    # Cost matrices
    Q = np.diag([1.0, 0.1])  # State cost
    R = np.array([[0.01]])   # Control cost
    
    # Linear cost terms
    Vx = np.zeros((nx+nu, nx))
    Vx[0:nx, 0:nx] = np.eye(nx)
    
    Vu = np.zeros((nx+nu, nu))
    Vu[nx:nx+nu, 0:nu] = np.eye(nu)
    
    # Set cost matrices
    ocp.cost.W = np.block([
        [Q, np.zeros((nx, nu))],
        [np.zeros((nu, nx)), R]
    ])
    
    ocp.cost.W_e = Q
    
    # Set cost function matrices
    ocp.cost.Vx = Vx
    ocp.cost.Vu = Vu
    
    # Terminal cost
    ocp.cost.Vx_e = np.eye(nx)
    
    # Reference for tracking (target at origin)
    yref = np.zeros((nx+nu,))
    yref_e = np.zeros((nx,))
    
    # Set references
    ocp.cost.yref = yref
    ocp.cost.yref_e = yref_e
    
    # Initial condition (start at position=1, velocity=0)
    x0 = np.array([1.0, 0.0])
    ocp.constraints.x0 = x0
    
    # Set constraints on control
    ocp.constraints.lbu = np.array([-2.0])  # Lower bound on control
    ocp.constraints.ubu = np.array([2.0])   # Upper bound on control
    ocp.constraints.idxbu = np.array([0])   # Index of control to constrain
    
    # Set solver options
    ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'  # Use SQP_RTI for linear systems
    
    # Set prediction horizon
    ocp.solver_options.tf = T
    
    # Create solver
    solver = AcadosOcpSolver(ocp, json_file='acados_ocp_lqr.json')
    
    # Solve OCP
    status = solver.solve()
    
    # Get solution
    x_sol = np.zeros((N+1, nx))
    u_sol = np.zeros((N, nu))
    
    for i in range(N):
        x_sol[i, :] = solver.get(i, "x")
        u_sol[i, :] = solver.get(i, "u")
    x_sol[N, :] = solver.get(N, "x")
    
    # Print results
    print(f"Solver status: {status}")
    print(f"Initial state: position = {x0[0]:.2f}, velocity = {x0[1]:.2f}")
    print(f"Final state: position = {x_sol[N, 0]:.4f}, velocity = {x_sol[N, 1]:.4f}")
    print("Control inputs (acceleration):")
    for i in range(min(5, N)):
        print(f"  u[{i}] = {u_sol[i, 0]:.4f}")
    if N > 5:
        print("  ...")
    
    # Plot results
    t = np.linspace(0, T, N+1)
    t_u = np.linspace(0, T, N)
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(t, x_sol[:, 0])
    plt.grid(True)
    plt.ylabel('Position')
    plt.title('Double Integrator LQR Control')
    
    plt.subplot(3, 1, 2)
    plt.plot(t, x_sol[:, 1])
    plt.grid(True)
    plt.ylabel('Velocity')
    
    plt.subplot(3, 1, 3)
    plt.step(t_u, u_sol[:, 0], where='post')
    plt.grid(True)
    plt.ylabel('Acceleration')
    plt.xlabel('Time [s]')
    
    plt.tight_layout()
    plt.savefig('test_acados/lqr_results.png')
    print("Results plot saved as 'test_acados/lqr_results.png'")
    
    # Save generated JSON to file
    with open('test_acados/acados_ocp_lqr.json', 'w') as f:
        json.dump(solver.acados_ocp.json_file, f, indent=4)
    
    print("Acados LQR test completed successfully!")

if __name__ == "__main__":
    main() 