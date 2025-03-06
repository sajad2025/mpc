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
        self.steering_max = 0.5
        self.steering_min = -0.5
        self.steering_rate_max = 0.4
        self.steering_rate_min = -0.4
        
        # Corridor parameters
        self.corridor_width = 15.0  # Width of the corridor in meters
        
        # Start and goal states [x, y, theta, steering]
        self._state_start = [-40, 0, 0, 0]
        self._state_final = [+40, 0, 0, 0]
        
        # Obstacle list - each obstacle is [x, y, radius, safety_margin]
        self.obstacles = None
        
        # Cost function weights
        # Path cost weights
        self.weight_steering_rate   = 10.0
        self.weight_steering_angle  = 10.0
        self.weight_position_x      = 10.0
        self.weight_position_y      = 10.0
        self.weight_heading         = 1.0

        # Output verbosity
        self.verbose = True  # Whether to print detailed output from solvers

    
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
        if len(state) != 4:
            raise ValueError("State must have 4 elements: [x, y, theta, steering]")
        
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
        if len(state) != 4:
            raise ValueError("State must have 4 elements: [x, y, theta, steering]")
        
        # Create a copy of the state to avoid modifying the input
        normalized_state = state.copy()
        
        # Normalize the heading angle (index 2)
        normalized_state[2] = self.normalize_angle(state[2])
        
        self._state_final = normalized_state

class SimConfig:
    def __init__(self):
        self.duration = 80.0  # seconds
        self.dt = 0.1  # seconds

def generate_controls_geodesic(ego, sim_cfg):
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

    ## 1 Acados Model
    # Create Acados OCP object
    ocp = AcadosOcp()
    
    # Set model dimensions
    nx = 4  # [x, y, theta, steering]
    nu = 1  # [steering_rate]
    N = int(sim_cfg.duration / sim_cfg.dt)  # number of shooting nodes
    
    # Set model name
    ocp.model_name = 'kinematic_car'
    
    # Create symbolic variables
    x = SX.sym('x', nx)
    u = SX.sym('u', nu)
    xdot = SX.sym('xdot', nx)
    
    # Named symbolic variables for better readability
    pos_x, pos_y, theta, steering = x[0], x[1], x[2], x[3]
    steering_rate = u[0]
    
    # Model equations
    x_dot = 1 * cos(theta)
    y_dot = 1 * sin(theta)
    theta_dot = 1 * tan(steering) / ego.L
    steering_dot = steering_rate
    
    # Explicit ODE right hand side
    f_expl = vertcat(x_dot, y_dot, theta_dot, steering_dot)
    
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
    

    ## 2 cost function
    # Set cost type
    ocp.cost.cost_type = 'LINEAR_LS'
    
    # Penalize control inputs, and steering angle
    ny = nu + nx  # [steering_rate]+[x, y, theta, steering]
    ocp.dims.ny = ny
    
    # Cost matrix - weights for [steering_rate, x, y, theta, steering]
    R = np.diag([
        ego.weight_steering_rate, # control
        ego.weight_position_x,    # states
        ego.weight_position_y,
        ego.weight_heading,
        ego.weight_steering_angle
    ])
    ocp.cost.W = R
    
    # Linear output functions - for controls, steering angle, and velocity
    # Map state and control to the cost function outputs
    
    # Controls go in the first rows
    Vu = np.zeros((ny, nu))
    Vu[0, 0] = 1.0  # Extract steering rate (1st control)

    # states
    Vx = np.zeros((ny, nx))
    Vx[1, 0] = 1.0  # state x
    Vx[2, 1] = 1.0  # state y
    Vx[3, 2] = 1.0  # state theta
    Vx[4, 3] = 1.0  # state steering

    ocp.cost.Vx = Vx
    ocp.cost.Vu = Vu
    
    # We want the reference to be "u=0, x=state_final"
    yref = np.zeros(ny)
    yref[1] = ego.state_final[0]         # x
    yref[2] = ego.state_final[1]         # y
    yref[3] = ego.state_final[2]         # theta
    yref[4] = ego.state_final[3]         # steering
    ocp.cost.yref = yref

    ## 3 solver options
    # Set prediction horizon
    ocp.solver_options.tf = sim_cfg.duration
    
    # Set options, 
    # PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES,    
    ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    ocp.solver_options.nlp_solver_max_iter = 200
    ocp.solver_options.levenberg_marquardt = 1e-2
    ocp.solver_options.qp_solver_cond_N = 5
    
    # Additional numerical options for better convergence
    ocp.solver_options.qp_solver_iter_max = 50
    ocp.solver_options.tol = 1e-3
    ocp.solver_options.print_level = 0 if not ego.verbose else 1


    ## 4 constraints
    # Set constraints with slack
    # Use smaller slack for better precision at the terminal state
    eps_pos = .1  # Slack for position
    eps_angle = 1e-2  # slack for angles
    eps_steer = 1e-1  # slack for steering
    

    ## 4-1 general constraints
    # Control input constraints
    ocp.constraints.lbu = np.array([ego.steering_rate_min])
    ocp.constraints.ubu = np.array([ego.steering_rate_max])
    ocp.constraints.idxbu = np.array(range(nu))

    # State constraints - relaxed bounds for better convergence
    x_max = 100.0
    y_max = 100.0
    ocp.constraints.lbx = np.array([-x_max, -y_max, -np.pi, ego.steering_min])  # Use ego velocity limits
    ocp.constraints.ubx = np.array([x_max, y_max, np.pi, ego.steering_max])  # Use ego velocity limits
    ocp.constraints.idxbx = np.array(range(nx))
    
    # Initial state constraint
    ocp.constraints.x0 = np.array(ego.state_start)
    
    # Terminal constraints with tighter slack for angles
    ocp.constraints.lbx_e = np.array([
        ego.state_final[0] - eps_pos,
        ego.state_final[1] - eps_pos,
        ego.state_final[2] - eps_angle,
        ego.state_final[3] - eps_steer
    ])
    ocp.constraints.ubx_e = np.array([
        ego.state_final[0] + eps_pos,
        ego.state_final[1] + eps_pos,
        ego.state_final[2] + eps_angle,
        ego.state_final[3] + eps_steer
    ])
    ocp.constraints.idxbx_e = np.array(range(nx))
    
    ## 5 init
    # Create solver
    acados_solver = AcadosOcpSolver(ocp, json_file='acados_ocp_' + ocp.model_name + '.json')
    
    # Initialize solution containers
    simX = np.ndarray((N+1, nx))
    simU = np.ndarray((N, nu))
    
    for i in range(N+1):
        t = float(i) / N
        # Linear interpolation for x and y
        x_init = ego.state_start[0] + t * (ego.state_final[0] - ego.state_start[0])
        y_init = ego.state_start[1] + t * (ego.state_final[1] - ego.state_start[1])
        theta_init = ego.state_start[2] + t * (ego.state_final[2] - ego.state_start[2])
        steering_init = 0.0
        
        if i < N:
            acados_solver.set(i, "x", np.array([x_init, y_init, theta_init, steering_init]))
    

    ## 6 solve
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