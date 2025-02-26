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
    
    # State and control reference
    x_ref = np.array([ego.state_final[0], ego.state_final[1], ego.state_final[2], 0.0, 0.0])
    u_ref = np.array([0.0, 0.0])
    
    # Cost matrices
    Q = np.diag([1.0, 1.0, 0.1, 0.1, 0.1])  # state cost
    R = np.diag([0.1, 0.1])  # control cost
    
    # Set cost parameters
    ny = nx + nu  # number of outputs in cost function
    ny_e = nx    # number of outputs in terminal cost function
    
    ocp.dims.ny = ny
    ocp.dims.ny_e = ny_e
    
    # Cost matrices
    ocp.cost.W = np.zeros((ny, ny))
    ocp.cost.W[:nx, :nx] = Q
    ocp.cost.W[nx:, nx:] = R
    
    ocp.cost.W_e = Q
    
    # Linear output functions
    ocp.cost.Vx = np.zeros((ny, nx))
    ocp.cost.Vx[:nx, :nx] = np.eye(nx)
    
    ocp.cost.Vu = np.zeros((ny, nu))
    ocp.cost.Vu[nx:, :] = np.eye(nu)
    
    ocp.cost.Vx_e = np.eye(nx)
    
    # Reference trajectory
    ocp.cost.yref = np.concatenate((x_ref, u_ref))
    ocp.cost.yref_e = x_ref
    
    # Set prediction horizon
    ocp.solver_options.tf = sim_cfg.duration
    
    # Set constraints
    ocp.constraints.lbu = np.array([ego.acceleration_min, ego.steering_rate_min])
    ocp.constraints.ubu = np.array([ego.acceleration_max, ego.steering_rate_max])
    ocp.constraints.idxbu = np.array(range(nu))
    
    # State constraints
    x_max = 100.0  # Large enough for our problem
    y_max = 100.0
    ocp.constraints.lbx = np.array([-x_max, -y_max, -2*np.pi, ego.velocity_min, ego.steering_min])
    ocp.constraints.ubx = np.array([x_max, y_max, 2*np.pi, ego.velocity_max, ego.steering_max])
    ocp.constraints.idxbx = np.array(range(nx))
    
    # Initial state constraint
    ocp.constraints.x0 = np.array(ego.state_start)
    
    # Set options
    ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    
    # Create solver
    acados_solver = AcadosOcpSolver(ocp, json_file='acados_ocp_' + ocp.model_name + '.json')
    
    # Initialize solution containers
    simX = np.ndarray((N+1, nx))
    simU = np.ndarray((N, nu))
    
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
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot trajectory
    ax1.plot(x[:, 0], x[:, 1], 'b-', label='Vehicle path')
    ax1.plot(ego.state_start[0], ego.state_start[1], 'go', label='Start')
    ax1.plot(ego.state_final[0], ego.state_final[1], 'ro', label='Goal')
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')
    ax1.set_title('Vehicle Trajectory')
    ax1.grid(True)
    ax1.legend()
    
    # Plot velocity and steering
    ax2.plot(t, x[:, 3], 'b-', label='Velocity')
    ax2.plot(t, x[:, 4], 'r-', label='Steering angle')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Value')
    ax2.set_title('Velocity and Steering')
    ax2.grid(True)
    ax2.legend()
    
    # Plot controls
    ax3.plot(t[:-1], u[:, 0], 'g-', label='Acceleration')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Acceleration [m/s²]')
    ax3.set_title('Acceleration Control')
    ax3.grid(True)
    ax3.legend()
    
    ax4.plot(t[:-1], u[:, 1], 'm-', label='Steering rate')
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Steering rate [rad/s]')
    ax4.set_title('Steering Rate Control')
    ax4.grid(True)
    ax4.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

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