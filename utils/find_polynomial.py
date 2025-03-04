import numpy as np
import cvxpy as cp

def find_polynomial_coefficients(T, f0, fT, f_min, f_max, fp_min, fp_max, dt=0.01, degree=3):
    """
    Find coefficients of a polynomial f(t) = a0 + a1*t + a2*t^2 + ... + an*t^n
    that satisfies the following constraints:
    
    Boundary conditions:
    - f(0) = f0
    - f(T) = fT
    - f'(0) = 0
    - f'(T) = 0
    
    Inequality constraints (for all 0 < t < T):
    - f_min < f(t) < f_max
    - fp_min < f'(t) < fp_max
    
    Args:
        T (float): End time
        f0 (float): Function value at t=0
        fT (float): Function value at t=T
        f_min (float): Minimum allowed function value
        f_max (float): Maximum allowed function value
        fp_min (float): Minimum allowed derivative value
        fp_max (float): Maximum allowed derivative value
        dt (float): Time step for discretization
        degree (int): Degree of the polynomial
    
    Returns:
        numpy.ndarray: Coefficients [a0, a1, ..., an] of the polynomial
    """
    # Create time discretization
    t_values = np.arange(0, T + dt, dt)
    
    # Define the optimization variables (polynomial coefficients)
    coeffs = cp.Variable(degree + 1)
    
    # Define constraints
    constraints = []
    
    # Boundary conditions for function values
    # f(0) = f0 => a0 = f0
    constraints.append(coeffs[0] == f0)
    
    # f(T) = fT => a0 + a1*T + a2*T^2 + ... + an*T^n = fT
    T_powers = np.array([T**i for i in range(degree + 1)])
    constraints.append(coeffs @ T_powers == fT)
    
    # Boundary conditions for derivatives
    # f'(0) = 0 => a1 = 0
    constraints.append(coeffs[1] == 0)
    
    # f'(T) = 0 => a1 + 2*a2*T + 3*a3*T^2 + ... + n*an*T^(n-1) = 0
    derivative_T_powers = np.array([i * T**(i-1) if i > 0 else 0 for i in range(degree + 1)])
    constraints.append(coeffs @ derivative_T_powers == 0)
    
    # Inequality constraints at discretized points
    for t in t_values:
        if t == 0 or t == T:
            continue  # Skip boundary points as they're already constrained
            
        # Function value constraints: f_min < f(t) < f_max
        t_powers = np.array([t**i for i in range(degree + 1)])
        constraints.append(coeffs @ t_powers <= f_max)
        constraints.append(coeffs @ t_powers >= f_min)
        
        # Derivative constraints: fp_min < f'(t) < fp_max
        derivative_powers = np.array([i * t**(i-1) if i > 0 else 0 for i in range(degree + 1)])
        constraints.append(coeffs @ derivative_powers <= fp_max)
        constraints.append(coeffs @ derivative_powers >= fp_min)
    
    # Objective: minimize the sum of squared coefficients (regularization)
    # This helps find a "simpler" polynomial with smaller coefficients
    objective = cp.Minimize(cp.sum_squares(coeffs))
    
    # Solve the problem
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve()
        
        if problem.status == 'optimal' or problem.status == 'optimal_inaccurate':
            return coeffs.value
        else:
            print(f"Problem status: {problem.status}")
            return None
    except cp.error.SolverError:
        print("Solver error. Try adjusting the constraints or using a different solver.")
        return None

def evaluate_polynomial(coeffs, t):
    """
    Evaluate the polynomial at time t
    
    Args:
        coeffs (numpy.ndarray): Polynomial coefficients [a0, a1, ..., an]
        t (float): Time at which to evaluate
        
    Returns:
        float: Value of the polynomial at time t
    """
    return sum(a * t**i for i, a in enumerate(coeffs))

def evaluate_polynomial_derivative(coeffs, t):
    """
    Evaluate the derivative of the polynomial at time t
    
    Args:
        coeffs (numpy.ndarray): Polynomial coefficients [a0, a1, ..., an]
        t (float): Time at which to evaluate
        
    Returns:
        float: Value of the polynomial derivative at time t
    """
    return sum(i * a * t**(i-1) for i, a in enumerate(coeffs) if i > 0)

def verify_solution(coeffs, T, f0, fT, f_min, f_max, fp_min, fp_max, dt=0.01):
    """
    Verify that the solution satisfies all constraints
    
    Args:
        coeffs (numpy.ndarray): Polynomial coefficients [a0, a1, ..., an]
        T (float): End time
        f0 (float): Function value at t=0
        fT (float): Function value at t=T
        f_min (float): Minimum allowed function value
        f_max (float): Maximum allowed function value
        fp_min (float): Minimum allowed derivative value
        fp_max (float): Maximum allowed derivative value
        dt (float): Time step for verification
        
    Returns:
        bool: True if all constraints are satisfied, False otherwise
    """
    t_values = np.arange(0, T + dt, dt)
    
    # Check boundary conditions
    f_start = evaluate_polynomial(coeffs, 0)
    f_end = evaluate_polynomial(coeffs, T)
    fp_start = evaluate_polynomial_derivative(coeffs, 0)
    fp_end = evaluate_polynomial_derivative(coeffs, T)
    
    if not np.isclose(f_start, f0) or not np.isclose(f_end, fT):
        print(f"Boundary condition violation: f(0)={f_start}, f(T)={f_end}")
        return False
    
    if not np.isclose(fp_start, 0) or not np.isclose(fp_end, 0):
        print(f"Derivative boundary condition violation: f'(0)={fp_start}, f'(T)={fp_end}")
        return False
    
    # Check inequality constraints
    for t in t_values:
        f_t = evaluate_polynomial(coeffs, t)
        fp_t = evaluate_polynomial_derivative(coeffs, t)
        
        if f_t < f_min or f_t > f_max:
            print(f"Function value constraint violation at t={t}: f(t)={f_t}")
            return False
        
        if fp_t < fp_min or fp_t > fp_max:
            print(f"Derivative constraint violation at t={t}: f'(t)={fp_t}")
            return False
    
    return True

def plot_solution(coeffs, T, f0, fT, f_min, f_max, fp_min, fp_max, dt=0.01):
    """
    Plot the polynomial solution and its derivative
    
    Args:
        coeffs (numpy.ndarray): Polynomial coefficients [a0, a1, ..., an]
        T (float): End time
        f0 (float): Function value at t=0
        fT (float): Function value at t=T
        f_min (float): Minimum allowed function value
        f_max (float): Maximum allowed function value
        fp_min (float): Minimum allowed derivative value
        fp_max (float): Maximum allowed derivative value
        dt (float): Time step for plotting
    """
    try:
        import matplotlib.pyplot as plt
        
        t_values = np.arange(0, T + dt, dt)
        f_values = [evaluate_polynomial(coeffs, t) for t in t_values]
        fp_values = [evaluate_polynomial_derivative(coeffs, t) for t in t_values]
        
        plt.figure(figsize=(12, 8))
        
        # Plot function
        plt.subplot(2, 1, 1)
        plt.plot(t_values, f_values, 'b-', label='f(t)')
        plt.axhline(y=f_min, color='r', linestyle='--', label='f_min')
        plt.axhline(y=f_max, color='g', linestyle='--', label='f_max')
        plt.scatter([0, T], [f0, fT], color='k', s=50, label='Boundary points')
        plt.xlabel('Time (t)')
        plt.ylabel('Function value')
        plt.title('Polynomial Function')
        plt.legend()
        plt.grid(True)
        
        # Plot derivative
        plt.subplot(2, 1, 2)
        plt.plot(t_values, fp_values, 'b-', label="f'(t)")
        plt.axhline(y=fp_min, color='r', linestyle='--', label='fp_min')
        plt.axhline(y=fp_max, color='g', linestyle='--', label='fp_max')
        plt.scatter([0, T], [0, 0], color='k', s=50, label='Boundary derivatives')
        plt.xlabel('Time (t)')
        plt.ylabel('Derivative value')
        plt.title('Polynomial Derivative')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("Matplotlib is required for plotting. Install it with 'pip install matplotlib'")

if __name__ == "__main__":
    # Example usage
    T = 5.0
    f0 = 0.0
    fT = 2.0
    f_min = -3.0
    f_max = 3.0
    fp_min = -2.0
    fp_max = 2.0
    dt = 0.1

    degree = 3
    
    coeffs = find_polynomial_coefficients(T, f0, fT, f_min, f_max, fp_min, fp_max, dt, degree)
    
    if coeffs is not None:
        print("Polynomial coefficients:", coeffs)
        
        is_valid = verify_solution(coeffs, T, f0, fT, f_min, f_max, fp_min, fp_max, dt)
        print(f"Solution is {'valid' if is_valid else 'invalid'}")
        
        # Plot the solution
        plot_solution(coeffs, T, f0, fT, f_min, f_max, fp_min, fp_max, dt)
    else:
        print("Failed to find a solution. Try relaxing the constraints or increasing the polynomial degree.") 