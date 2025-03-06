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
        self._state_start = [-40, 0, 0, 0, 0]
        self._state_final = [+40, 0, 0, 0, 0]
        
        # Obstacle list - each obstacle is [x, y, radius, safety_margin]
        self.obstacles = None
        
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
    ocp.solver_options.print_level = 0 if not ego.verbose else 1
    
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
    # Second row: c2*y + a2*x + b2 <= 0
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

    # nonlinear constraint for obstacle avoidance
    if ego.obstacles is not None and len(ego.obstacles) > 0:
        # Create a list to store all distance expressions
        dist_exprs = []
        
        # Add a constraint for each obstacle
        for obs in ego.obstacles:
            obs_x, obs_y, obs_r, safety_margin = obs
            # Distance formula: (x-x0)^2 + (y-y0)^2 - (r+safety)^2
            dist_expr = (x[0] - obs_x)**2 + (x[1] - obs_y)**2 - (obs_r + safety_margin)**2
            dist_exprs.append(dist_expr)
        
        # Combine all distance expressions into a single constraint vector
        ocp.model.con_h_expr = vertcat(*dist_exprs)
        ocp.dims.nh = len(dist_exprs)
        
        # Set lower and upper bounds for all constraints
        # Lower bound of 0 means the vehicle must stay outside the obstacle
        ocp.constraints.lh = np.zeros(len(dist_exprs))
        # Upper bound of inf means there's no maximum distance
        ocp.constraints.uh = np.ones(len(dist_exprs)) * 1e9
    
    # Create solver
    acados_solver = AcadosOcpSolver(ocp, json_file='acados_ocp_' + ocp.model_name + '.json')
    
    # Initialize solution containers
    simX = np.ndarray((N+1, nx))
    simU = np.ndarray((N, nu))
    
    def adjust_point_for_obstacles(x_init, y_init):
        """
        Check if a point is inside any obstacle and move it to the boundary if it is.
        When an obstacle is on the path-line (line connecting start to goal),
        shift the entire path-line and project the point onto this shifted line.
        
        Args:
            x_init, y_init: Initial point coordinates
            
        Returns:
            x_adjusted, y_adjusted: Point coordinates after obstacle avoidance
        """
        if ego.obstacles is None:
            return x_init, y_init
            
        x_adjusted, y_adjusted = x_init, y_init
        
        # Define the path-line (line from start to goal)
        start_x, start_y = ego.state_start[0], ego.state_start[1]
        goal_x, goal_y = ego.state_final[0], ego.state_final[1]
        
        # Calculate path direction vector and normalize it
        path_dx = goal_x - start_x
        path_dy = goal_y - start_y
        path_length = np.sqrt(path_dx**2 + path_dy**2)
        
        if path_length < 1e-6:  # Avoid division by zero
            return x_init, y_init
            
        path_dx /= path_length
        path_dy /= path_length
        
        # Calculate normal vector to the path (90 degrees counterclockwise)
        normal_dx = -path_dy
        normal_dy = path_dx
        
        # Check if any obstacle is on the path-line
        path_shift = 0
        shift_direction = 1  # 1 for positive normal direction, -1 for negative
        
        for obs in ego.obstacles:
            obs_x, obs_y, obs_r, safety_margin = obs
            total_radius = obs_r + safety_margin + 0.05
            
            # Vector from start to obstacle
            obs_dx = obs_x - start_x
            obs_dy = obs_y - start_y
            
            # Project obstacle vector onto path vector
            projection = obs_dx * path_dx + obs_dy * path_dy
            
            # Find closest point on path to obstacle
            closest_x = start_x + projection * path_dx
            closest_y = start_y + projection * path_dy
            
            # Distance from obstacle center to path-line
            dist_to_path = np.sqrt((closest_x - obs_x)**2 + (closest_y - obs_y)**2)
            
            # Check if obstacle is on path (within obstacle radius and projection is within path length)
            if dist_to_path < total_radius and 0 <= projection <= path_length:
                # Determine which side of the path the point is currently on
                point_side = (x_init - start_x) * normal_dx + (y_init - start_y) * normal_dy
                if point_side < 0:
                    shift_direction = -1
                
                # Calculate how much we need to shift the path to avoid this obstacle
                needed_shift = total_radius - dist_to_path
                if needed_shift > path_shift:
                    path_shift = needed_shift
        
        # If we need to shift the path
        if path_shift > 0:
            # Create shifted path-line
            shift_amount = path_shift + 0.1  # Add a small buffer
            shifted_start_x = start_x + shift_direction * normal_dx * shift_amount
            shifted_start_y = start_y + shift_direction * normal_dy * shift_amount
            shifted_goal_x = goal_x + shift_direction * normal_dx * shift_amount
            shifted_goal_y = goal_y + shift_direction * normal_dy * shift_amount
            
            # Project the initial point onto this shifted line
            # Vector from shifted_start to point
            v_dx = x_init - shifted_start_x
            v_dy = y_init - shifted_start_y
            
            # Project this vector onto the path direction
            proj = v_dx * path_dx + v_dy * path_dy
            proj = max(0, min(proj, path_length))  # Clamp to path length
            
            # The adjusted point is on the shifted line at the projected distance
            x_adjusted = shifted_start_x + proj * path_dx
            y_adjusted = shifted_start_y + proj * path_dy
            
            return x_adjusted, y_adjusted
        
        # If no path shift was needed, check if the point itself is in an obstacle
        for obs in ego.obstacles:
            obs_x, obs_y, obs_r, safety_margin = obs
            total_radius = obs_r + safety_margin + 0.05
            
            # Calculate distance from point to obstacle center
            dx = x_adjusted - obs_x
            dy = y_adjusted - obs_y
            distance = np.sqrt(dx**2 + dy**2)
            
            # If point is inside obstacle (including safety margin)
            if distance < total_radius:
                # Calculate angle of point relative to obstacle center
                angle = np.arctan2(dy, dx)
                
                # Move point to boundary
                x_adjusted = obs_x + total_radius * np.cos(angle)
                y_adjusted = obs_y + total_radius * np.sin(angle)
        
        return x_adjusted, y_adjusted
    
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
        
        # Adjust point if it's inside any obstacle
        x_init, y_init = adjust_point_for_obstacles(x_init, y_init)
        
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