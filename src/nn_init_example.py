#!/usr/bin/env python3

import os
import time
import numpy as np
from core_solver import EgoConfig, SimConfig, generate_controls
from plots import plot_comparison
from nn_init_main import train_and_save_duration_predictor, train_and_save_trajectory_predictor, find_best_duration

def run_solver(ego, sim, use_nn_init):
    """
    Run the solver and handle failures.
    
    Returns:
        tuple: (results, solve_time)
        - results: None if solver failed, otherwise the results dictionary
        - solve_time: float, np.inf if solver failed, otherwise the actual solve time
    """
    start_time = time.time()
    try:
        results = generate_controls(ego, sim, use_nn_init=use_nn_init)
        solve_time = time.time() - start_time
        
        # Check if solver was successful
        if results['status'] != 0:
            print(f"Solver failed with status: {results['status']}")
            return None, np.inf
            
        return results, solve_time
    except Exception as e:
        print(f"Solver failed with error: {str(e)}")
        return None, np.inf

def run_example():
    """
    Run example scenarios comparing default and neural network initialization.
    """
    # Create models and docs directories if they don't exist
    os.makedirs('models', exist_ok=True)
    docs_dir = 'docs'
    results_dir = os.path.join(docs_dir, 'nn_example_results')
    os.makedirs(docs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Check if models exist, if not train new ones
    duration_model_path = 'models/duration_predictor.pt'
    trajectory_model_path = 'models/trajectory_predictor.pt'
    
    if not os.path.exists(duration_model_path) or not os.path.exists(trajectory_model_path):
        print("Training new models (this may take a few minutes)...")
        # Train duration predictor first
        print("Training duration predictor...")
        train_and_save_duration_predictor(
            num_samples=500,  # Small dataset for example
            batch_size=32,
            num_epochs=10,
            learning_rate=1e-3
        )
        
        # Then train trajectory predictor
        print("Training trajectory predictor...")
        train_and_save_trajectory_predictor(
            num_samples=500,  # Small dataset for example
            batch_size=32,
            num_epochs=10,
            hidden_size=128,
            num_layers=2,
            dropout=0.1,
            learning_rate=1e-3
        )
        print("Model training complete!")
    
    # Define test scenarios (without pre-determined durations)
    scenarios = [
        {
            'name': 'Simple Forward',
            'start': [0, 0, 0, 0, 0],
            'goal': [10, 0, 0, 0, 0],
            'default_duration': 10.0  # Used only for default initialization
        },
        {
            'name': 'Diagonal with Rotation',
            'start': [0, 0, 0, 0, 0],
            'goal': [10, 10, np.pi/4, 0, 0],
            'default_duration': 15.0
        },
        {
            'name': 'U-Turn',
            'start': [0, 0, 0, 0, 0],
            'goal': [-5, 0, np.pi, 0, 0],
            'default_duration': 20.0
        },
        {
            'name': 'Complex Maneuver',
            'start': [0, 0, 0, 0, 0],
            'goal': [5, 10, -np.pi/2, 0, 0],
            'default_duration': 25.0
        }
    ]
    
    # Run scenarios
    successful_scenarios = []
    failed_scenarios = []
    
    for scenario in scenarios:
        print(f"\nRunning scenario: {scenario['name']}")
        
        # Create configuration objects
        ego = EgoConfig()
        sim = SimConfig()
        
        # Set start and goal states
        ego.state_start = scenario['start']
        ego.state_final = scenario['goal']
        sim.dt = 0.1
        
        # Calculate distance for information
        start_pos = scenario['start'][:2]
        goal_pos = scenario['goal'][:2]
        distance = np.sqrt(np.sum((np.array(goal_pos) - np.array(start_pos))**2))
        
        # Run with default initialization using pre-determined duration
        print("Running with default initialization...")
        sim.duration = scenario['default_duration']
        print(f"Path distance: {distance:.2f}m, Default duration: {sim.duration:.2f}s")
        default_results, default_time = run_solver(ego, sim, use_nn_init=False)
        
        if default_time == np.inf:
            print("Default initialization failed to find a solution")
            failed_scenarios.append(scenario['name'])
            continue
        
        print(f"Default initialization completed in {default_time:.3f}s")
        
        # Run with neural network initialization using predicted duration
        print("Running with neural network initialization...")
        # Get predicted duration from the neural network
        predicted_duration = find_best_duration(
            state_start=ego.state_start,
            state_end=ego.state_final,
            dt=sim.dt
        )
        sim.duration = predicted_duration
        print(f"Path distance: {distance:.2f}m, Predicted duration: {sim.duration:.2f}s")
        nn_results, nn_time = run_solver(ego, sim, use_nn_init=True)
        
        if nn_time == np.inf:
            print("Neural network initialization failed to find a solution")
            failed_scenarios.append(scenario['name'])
            continue
            
        print(f"Neural network initialization completed in {nn_time:.3f}s")
        
        # Calculate improvement
        if default_time > 0 and default_time != np.inf:
            improvement = (default_time - nn_time) / default_time * 100
            print(f"Time improvement: {improvement:.1f}%")
        
        # Create comparison plot
        save_path = os.path.join(results_dir, f"{scenario['name'].lower().replace(' ', '_')}_comparison.png")
        plot_comparison(
            default_results=default_results,
            nn_results=nn_results,
            ego=ego,
            computation_times=(default_time, nn_time),
            durations=(scenario['default_duration'], predicted_duration),  # Add durations to plot
            save_path=save_path
        )
        
        successful_scenarios.append(scenario['name'])
    
    # Print summary
    print("\nExample complete!")
    if successful_scenarios:
        print("\nSuccessful scenarios:")
        for name in successful_scenarios:
            print(f"- {name}")
    
    if failed_scenarios:
        print("\nFailed scenarios:")
        for name in failed_scenarios:
            print(f"- {name}")
    
    print("\nResults have been saved in docs/nn_example_results/")

if __name__ == "__main__":
    run_example() 