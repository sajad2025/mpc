#!/usr/bin/env python3

import os
import numpy as np
import scipy.linalg
from casadi import *
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import matplotlib.pyplot as plt

def generate_controls(ego, sim_cfg):
    """
    Generate optimal controls for path planning using Acados.
    
    Args:
        ego: Object containing vehicle parameters and constraints
        sim_cfg: Object containing simulation parameters
    
    Returns:
        Dictionary containing:
        - t: time points
        - x: state trajectory
        - u: control inputs
    """
    # Create Acados OCP object
    ocp = AcadosOcp()
    
    # Set model dimensions
    nx = 5  # [x, y, theta, velocity, steering]
    nu = 2  # [acceleration, steering_rate]
    N = int(sim_cfg.duration / sim_cfg.dt)  # number of shooting nodes
    
    # Set model name
    ocp.model_name = 'kinematic_car'
    
    # Create symbolic variables
    x = SX.sym('x', nx)
    u = SX.sym('u', nu)
    xdot = SX.sym('xdot', nx)
    
    # Named symbolic variables for better readability
    pos_x, pos_y, theta, v, steering = x[0], x[1], x[2], x[3], x[4]
    acc, steering_rate = u[0], u[1]
    
    # Model equations
    x_dot = v * cos(theta)
    y_dot = v * sin(theta)
    theta_dot = v * tan(steering) / ego.L
    v_dot = acc
    steering_dot = steering_rate
    
    # Explicit ODE right hand side
    f_expl = vertcat(x_dot, y_dot, theta_dot, v_dot, steering_dot)
    
    # Create model
    model = AcadosModel()
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = ocp.model_name
    
    # Set model
    ocp.model = model
    
    # Set dimensions
    ocp.dims.N = N
    
    # Set cost type
    ocp.cost.cost_type = 'LINEAR_LS'
    ocp.cost.cost_type_e = 'LINEAR_LS'
    
    # Only penalize control inputs (acceleration and steering rate)
    ny = nu  # number of outputs in cost function - only control inputs
    ny_e = 0  # no terminal cost
    
    ocp.dims.ny = ny
    ocp.dims.ny_e = ny_e
    
    # Cost matrix - only penalize controls
    R = np.diag([1.0, 1.0])  # Equal weights on acceleration and steering rate
    
    ocp.cost.W = R
    
    # No terminal cost since we're using constraints
    ocp.cost.W_e = np.zeros((0, 0))
    
    # Linear output functions - only for controls
    ocp.cost.Vx = np.zeros((ny, nx))
    ocp.cost.Vu = np.eye(nu)
    
    ocp.cost.Vx_e = np.zeros((0, nx))
    
    # Reference trajectory - zero control reference
    ocp.cost.yref = np.zeros(nu)  # Reference is zero control
    ocp.cost.yref_e = np.zeros(0)  # No terminal cost
    
    # Set prediction horizon
    ocp.solver_options.tf = sim_cfg.duration
    
    # Set options
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.nlp_solver_max_iter = 100
    ocp.solver_options.levenberg_marquardt = 1e-2
    ocp.solver_options.qp_solver_cond_N = 5
    
    # Additional numerical options for better convergence
    ocp.solver_options.qp_solver_iter_max = 50
    ocp.solver_options.tol = 1e-3
    ocp.solver_options.print_level = 1
    
    # Set constraints with slack
    eps = 1e-1  # Increased slack for better numerical stability
    
    # Control input constraints
    ocp.constraints.lbu = np.array([ego.acceleration_min, ego.steering_rate_min])
    ocp.constraints.ubu = np.array([ego.acceleration_max, ego.steering_rate_max])
    ocp.constraints.idxbu = np.array(range(nu))
    
    # State constraints - relaxed bounds for better convergence
    x_max = 100.0
    y_max = 100.0
    ocp.constraints.lbx = np.array([-x_max, -y_max, -2*np.pi, 0.0, -0.5])  # Modified velocity and steering bounds
    ocp.constraints.ubx = np.array([x_max, y_max, 2*np.pi, 3.0, 0.5])
    ocp.constraints.idxbx = np.array(range(nx))
    
    # Initial state constraint
    ocp.constraints.x0 = np.array(ego.state_start)
    
    # Terminal constraints with larger slack
    ocp.constraints.lbx_e = np.array([
        ego.state_final[0] - eps,
        ego.state_final[1] - eps,
        ego.state_final[2] - eps,
        -eps,
        -eps
    ])
    ocp.constraints.ubx_e = np.array([
        ego.state_final[0] + eps,
        ego.state_final[1] + eps,
        ego.state_final[2] + eps,
        eps,
        eps
    ])
    ocp.constraints.idxbx_e = np.array(range(nx))
    
    # Create solver
    acados_solver = AcadosOcpSolver(ocp, json_file='acados_ocp_' + ocp.model_name + '.json')
    
    # Initialize solution containers
    simX = np.ndarray((N+1, nx))
    simU = np.ndarray((N, nu))
    
    # Initialize with a simple straight line trajectory
    for i in range(N):
        t = float(i) / N
        x_init = ego.state_start[0] + t * (ego.state_final[0] - ego.state_start[0])
        y_init = ego.state_start[1] + t * (ego.state_final[1] - ego.state_start[1])
        theta_init = ego.state_start[2] + t * (ego.state_final[2] - ego.state_start[2])
        v_init = 1.0  # Initial guess for velocity
        steering_init = 0.0
        acados_solver.set(i, "x", np.array([x_init, y_init, theta_init, v_init, steering_init]))
    
    # Solve OCP
    status = acados_solver.solve()
    
    # Get solution
    for i in range(N):
        simX[i] = acados_solver.get(i, "x")
        simU[i] = acados_solver.get(i, "u")
    simX[N] = acados_solver.get(N, "x")
    
    # Create time array
    t = np.linspace(0, sim_cfg.duration, N+1)
    
    # Return results
    return {
        't': t,
        'x': simX,
        'u': simU
    }

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
    # Move legend outside of ax1 to the right
    ax1.legend(bbox_to_anchor=(1.15, 1), loc='upper left')
    # Make sure the aspect ratio is equal so circles look circular
    ax1.set_aspect('equal')
    
    # Plot velocity
    ax3.plot(t, x[:, 3], 'b-', label='velocity (m/s)')
    ax3.axhline(y=ego.velocity_max, color='k', linestyle='--', alpha=0.3, label='bounds')
    ax3.axhline(y=ego.velocity_min, color='k', linestyle='--', alpha=0.3)
    ax3.grid(True)
    ax3.legend()
    
    # Plot steering angle (moved to bottom left)
    ax5.plot(t, np.rad2deg(x[:, 4]), 'r-', label='steering angle (deg)')
    ax5.axhline(y=np.rad2deg(ego.steering_max), color='k', linestyle='--', alpha=0.3, label='bounds')
    ax5.axhline(y=np.rad2deg(ego.steering_min), color='k', linestyle='--', alpha=0.3)
    ax5.set_xlabel('time (s)')
    ax5.grid(True)
    ax5.legend()
    
    # Right column: Controls and heading
    # Plot heading (moved to top right)
    ax2.plot(t, np.rad2deg(x[:, 2]), 'b-', label='current heading (deg)')
    ax2.axhline(y=np.rad2deg(ego.state_final[2]), color='r', linestyle='--', label='target heading')
    ax2.grid(True)
    ax2.legend()
    
    # Plot acceleration
    ax4.plot(t[:-1], u[:, 0], 'g-', label='acceleration (m/s²)')
    ax4.axhline(y=ego.acceleration_max, color='k', linestyle='--', alpha=0.3, label='bounds')
    ax4.axhline(y=ego.acceleration_min, color='k', linestyle='--', alpha=0.3)
    ax4.grid(True)
    ax4.legend()
    
    # Plot steering rate
    ax6.plot(t[:-1], np.rad2deg(u[:, 1]), 'm-', label='steering rate (deg/s)')
    ax6.axhline(y=np.rad2deg(ego.steering_rate_max), color='k', linestyle='--', alpha=0.3, label='bounds')
    ax6.axhline(y=np.rad2deg(ego.steering_rate_min), color='k', linestyle='--', alpha=0.3)
    ax6.set_xlabel('time (s)')
    ax6.grid(True)
    ax6.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    # Show the plot and keep it open
    plt.show(block=True)

if __name__ == "__main__":
    # Example usage
    class EgoConfig:
        def __init__(self):
            self.velocity_max = 3.0
            self.velocity_min = 0.0
            self.acceleration_max = 2.0
            self.acceleration_min = -2.0
            self.steering_max = 0.5
            self.steering_min = -0.5
            self.steering_rate_max = 1.0
            self.steering_rate_min = -1.0
            self.state_start = [0, 0, 0, 0, 0]
            self.state_final = [20, 20, np.pi/2, 0, 0]
            self.L = 2.7

    class SimConfig:
        def __init__(self):
            self.duration = 30.0
            self.dt = 0.1

    # Create configurations
    ego = EgoConfig()
    sim_cfg = SimConfig()
    
    # Generate and plot results
    results = generate_controls(ego, sim_cfg)
    plot_results(results, ego, save_path='path_planning_results.png') 