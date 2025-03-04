#!/usr/bin/env python3

import os
import numpy as np
import scipy.linalg
from casadi import *
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
from utils.corridor import corridor_cal

class EgoConfig:
    def __init__(self):
        # Vehicle physical parameters
        self.L = 2.7  # Wheelbase length (m)
        
        # State constraints
        self.velocity_max = 3.0
        self.velocity_min = -3.0
        self.acceleration_max = 2.0
        self.acceleration_min = -2.0
        self.steering_max = 0.5
        self.steering_min = -0.5
        self.steering_rate_max = 0.4
        self.steering_rate_min = -0.4
        
        # Corridor parameters
        self.corridor_width = 15.0  # Width of the corridor in meters
        
        # Start and goal states [x, y, theta, velocity, steering]
        self._state_start = [0, 0, 0, 0, 0]
        self._state_final = [20, 20, np.pi/2, 0, 0]
        
        # Cost function weights
        # Path cost weights
        self.weight_acceleration = 100.0
        self.weight_steering_rate = 100.0
        self.weight_steering_angle = 1.0
        self.weight_velocity = 0.0
        
        # Terminal cost weights for all states
        self.weight_terminal_position_x = 100.0
        self.weight_terminal_position_y = 100.0
        self.weight_terminal_heading = 100.0
        self.weight_terminal_velocity = 10.0
        self.weight_terminal_steering = 10.0
    
    @staticmethod
    def normalize_angle(angle):
        """
        Normalize angle to be within [-π, π]
        """
        return ((angle + np.pi) % (2 * np.pi)) - np.pi
    
    @property
    def state_start(self):
        return self._state_start
    
    @state_start.setter
    def state_start(self, state):
        """
        Set the start state with angle normalization.
        Ensures the heading angle (theta) is normalized to [-π, π].
        """
        if len(state) != 5:
            raise ValueError("State must have 5 elements: [x, y, theta, velocity, steering]")
        
        # Create a copy of the state to avoid modifying the input
        normalized_state = state.copy()
        
        # Normalize the heading angle (index 2)
        normalized_state[2] = self.normalize_angle(state[2])
        
        self._state_start = normalized_state
    
    @property
    def state_final(self):
        return self._state_final
    
    @state_final.setter
    def state_final(self, state):
        """
        Set the final state with angle normalization.
        Ensures the heading angle (theta) is normalized to [-π, π].
        """
        if len(state) != 5:
            raise ValueError("State must have 5 elements: [x, y, theta, velocity, steering]")
        
        # Create a copy of the state to avoid modifying the input
        normalized_state = state.copy()
        
        # Normalize the heading angle (index 2)
        normalized_state[2] = self.normalize_angle(state[2])
        
        self._state_final = normalized_state

class SimConfig:
    def __init__(self):
        self.duration = 30.0  # seconds
        self.dt = 0.1  # seconds

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
        - status: solver status (0 = success)
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
    
    # Penalize control inputs, steering angle, and velocity
    ny = nu + 2  # number of outputs in cost function: [acceleration, steering_rate, steering_angle, velocity]
    ny_e = nx  # Terminal cost for all states: [x, y, theta, velocity, steering]
    
    ocp.dims.ny = ny
    ocp.dims.ny_e = ny_e
    
    # Cost matrix - weights for [acceleration, steering_rate, steering_angle, velocity]
    # Use weights from ego config
    R = np.diag([
        ego.weight_acceleration,
        ego.weight_steering_rate,
        ego.weight_steering_angle,
        ego.weight_velocity
    ])
    
    ocp.cost.W = R
    
    # Terminal cost to ensure all states reach target values
    # Use weights from ego config
    W_e = np.diag([
        ego.weight_terminal_position_x,
        ego.weight_terminal_position_y,
        ego.weight_terminal_heading,
        ego.weight_terminal_velocity,
        ego.weight_terminal_steering
    ])
    ocp.cost.W_e = W_e
    
    # Linear output functions - for controls, steering angle, and velocity
    # Map state and control to the cost function outputs
    Vx = np.zeros((ny, nx))
    Vx[2, 4] = 1.0  # Extract steering angle (5th state)
    Vx[3, 3] = 1.0  # Extract velocity (4th state)
    
    Vu = np.zeros((ny, nu))
    Vu[0, 0] = 1.0  # Extract acceleration (1st control)
    Vu[1, 1] = 1.0  # Extract steering rate (2nd control)
    
    ocp.cost.Vx = Vx
    ocp.cost.Vu = Vu
    
    # Terminal cost function - extract all states
    Vx_e = np.eye(nx)  # Identity matrix to extract all states
    ocp.cost.Vx_e = Vx_e
    
    # Reference trajectory - zero reference for all outputs
    ocp.cost.yref = np.zeros(ny)  # Reference is zero for all outputs
    
    # Terminal reference - target values for all states
    yref_e = np.array(ego.state_final)  # Target for all states
    ocp.cost.yref_e = yref_e
    
    # Set prediction horizon
    ocp.solver_options.tf = sim_cfg.duration
    
    # Set options
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.nlp_solver_max_iter = 200
    ocp.solver_options.levenberg_marquardt = 1e-2
    ocp.solver_options.qp_solver_cond_N = 5
    
    # Additional numerical options for better convergence
    ocp.solver_options.qp_solver_iter_max = 50
    ocp.solver_options.tol = 1e-3
    ocp.solver_options.print_level = 1
    
    # Set constraints with slack
    # Use smaller slack for better precision at the terminal state
    eps_pos = 1e-1  # Slack for position
    eps_angle = 1e-2  # Tighter slack for angles
    eps_vel = 1e-2  # Tighter slack for velocity and steering
    
    # Control input constraints
    ocp.constraints.lbu = np.array([ego.acceleration_min, ego.steering_rate_min])
    ocp.constraints.ubu = np.array([ego.acceleration_max, ego.steering_rate_max])
    ocp.constraints.idxbu = np.array(range(nu))
    
    # Calculate corridor constraints
    point_a = (ego.state_start[0], ego.state_start[1])  # (x, y) of start
    point_b = (ego.state_final[0], ego.state_final[1])  # (x, y) of end
    c1, a1, b1, c2, a2, b2 = corridor_cal(point_a, point_b, ego.corridor_width)
    
    # Add corridor constraints as linear constraints
    # Original constraints are: 
    # c1*y + a1*x + b1 > 0  ->  -c1*y - a1*x <= b1
    # c2*y + a2*x + b2 < 0  ->   c2*y + a2*x <= -b2
    
    # Set up linear constraints: C*x + D*u <= ug
    ocp.dims.ng = 2  # Two linear constraints
    
    # Define constraint matrices
    C = np.zeros((2, nx))
    # First row: -c1*y - a1*x <= b1
    C[0, 0] = float(-a1)  # x coefficient
    C[0, 1] = float(-c1)  # y coefficient
    # Second row: c2*y + a2*x <= -b2
    C[1, 0] = float(a2)   # x coefficient
    C[1, 1] = float(c2)   # y coefficient
    
    # No control input in constraints
    D = np.zeros((2, nu))
    
    # Set the constraint matrices
    ocp.constraints.C = C
    ocp.constraints.D = D
    
    # Set bounds for the linear constraints
    ocp.constraints.lg = np.array([-1e9, -1e9], dtype=float)  # Lower bounds
    ocp.constraints.ug = np.array([float(b1), float(-b2)])    # Upper bounds
    
    # Apply same constraints to terminal state
    ocp.constraints.C_e = C.copy()
    ocp.constraints.lg_e = np.array([-1e9, -1e9], dtype=float)
    ocp.constraints.ug_e = np.array([float(b1), float(-b2)])
    
    # State constraints - relaxed bounds for better convergence
    x_max = 100.0
    y_max = 100.0
    ocp.constraints.lbx = np.array([-x_max, -y_max, -np.pi, ego.velocity_min, ego.steering_min])  # Use ego velocity limits
    ocp.constraints.ubx = np.array([x_max, y_max, np.pi, ego.velocity_max, ego.steering_max])  # Use ego velocity limits
    ocp.constraints.idxbx = np.array(range(nx))
    
    # Initial state constraint
    ocp.constraints.x0 = np.array(ego.state_start)
    
    # Terminal constraints with tighter slack for angles
    ocp.constraints.lbx_e = np.array([
        ego.state_final[0] - eps_pos,
        ego.state_final[1] - eps_pos,
        ego.state_final[2] - eps_angle,
        ego.state_final[3] - eps_vel,  # Use the target velocity with slack
        ego.state_final[4] - eps_vel
    ])
    ocp.constraints.ubx_e = np.array([
        ego.state_final[0] + eps_pos,
        ego.state_final[1] + eps_pos,
        ego.state_final[2] + eps_angle,
        ego.state_final[3] + eps_vel,  # Use the target velocity with slack
        ego.state_final[4] + eps_vel
    ])
    ocp.constraints.idxbx_e = np.array(range(nx))
    
    # Create solver
    acados_solver = AcadosOcpSolver(ocp, json_file='acados_ocp_' + ocp.model_name + '.json')
    
    # Initialize solution containers
    simX = np.ndarray((N+1, nx))
    simU = np.ndarray((N, nu))
    
    # Initialize with a trajectory that respects the corridor
    # Calculate center line of the corridor
    for i in range(N+1):
        t = float(i) / N
        # Linear interpolation for x and y
        x_init = ego.state_start[0] + t * (ego.state_final[0] - ego.state_start[0])
        y_init = ego.state_start[1] + t * (ego.state_final[1] - ego.state_start[1])
        
        # Ensure initial guess satisfies corridor constraints
        y_center = -(a1*x_init + b1/2 + a2*x_init + b2/2)/2
        y_init = min(max(y_init, y_center - ego.corridor_width/2), y_center + ego.corridor_width/2)
        
        # Linear interpolation for other states
        theta_init = ego.state_start[2] + t * (ego.state_final[2] - ego.state_start[2])
        v_init = 1.0 * (1 - t)  # Start with velocity and slow down
        steering_init = 0.0
        
        if i < N:
            acados_solver.set(i, "x", np.array([x_init, y_init, theta_init, v_init, steering_init]))
    
    # Set the final state
    acados_solver.set(N, "x", np.array(ego.state_final))
    
    # Solve OCP
    status = acados_solver.solve()
    
    # Return None if solver failed
    if status != 0:
        print(f"Solver failed with status {status}")
        return None
    
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
        'u': simU,
        'status': status
    } 