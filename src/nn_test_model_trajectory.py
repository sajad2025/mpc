#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
import time
from core_solver import EgoConfig, SimConfig, generate_controls
from nn_train_model import train_and_save_duration_predictor, train_and_save_trajectory_predictor
import torch

def test_initialization_performance(num_tests=20, train_first=True):
    """
    Test the performance of neural network initialization vs. default initialization.
    
    Args:
        num_tests: Number of test cases to run
        train_first: Whether to train the model first
    """
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Train models if requested
    if train_first:
        print("Training neural network models...")
        
        # Train duration predictor first
        print("Training duration predictor...")
        train_and_save_duration_predictor(
            num_samples=1000,  # Reduced for faster training
            batch_size=32,
            num_epochs=20,     # Reduced for faster training
            learning_rate=1e-3,
            device=device
        )
        
        # Then train trajectory predictor
        print("Training trajectory predictor...")
        train_and_save_trajectory_predictor(
            num_samples=1000,  # Reduced for faster training
            batch_size=32,
            num_epochs=20,     # Reduced for faster training
            hidden_size=128,
            num_layers=2,
            dropout=0.1,
            learning_rate=1e-3,
            device=device
        )
    
    # Create test results directory
    docs_dir = 'docs'
    results_dir = os.path.join(docs_dir, 'nn_test_results')
    os.makedirs(docs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize results containers
    default_times = []
    nn_times = []
    default_statuses = []
    nn_statuses = []
    
    # Create ego config
    ego = EgoConfig()
    
    # Run tests
    for i in range(num_tests):
        print(f"\nTest {i+1}/{num_tests}")
        
        while True:
            # Generate start position
            x_start = np.random.uniform(-15, 15)  # Increased range to accommodate larger distances
            y_start = np.random.uniform(-15, 15)  # Increased range to accommodate larger distances
            theta_start = np.random.uniform(-np.pi/2, np.pi/2)
            
            # Generate goal position ensuring minimum distance
            angle = np.random.uniform(0, 2*np.pi)  # Random angle for goal position
            min_distance = 10.0  # Minimum 10 meters between start and goal
            max_distance = 30.0  # Maximum 30 meters between start and goal
            distance = np.random.uniform(min_distance, max_distance)
            
            # Calculate goal position based on distance and angle
            x_goal = x_start + distance * np.cos(angle)
            y_goal = y_start + distance * np.sin(angle)
            theta_goal = np.random.uniform(-np.pi/2, np.pi/2)
            
            # Check if goal is within bounds
            if -15 <= x_goal <= 15 and -15 <= y_goal <= 15:  # Increased bounds
                break
        
        # Set start and goal states
        ego.state_start = [x_start, y_start, theta_start, 0, 0]
        ego.state_final = [x_goal, y_goal, theta_goal, 0, 0]
        
        # Calculate duration with more buffer time
        min_duration = distance / ego.velocity_max
        duration = min_duration * 1.5 + 8.0  # 50% more time + 8 seconds buffer
        
        # Ensure minimum duration
        duration = max(duration, 10.0)  # At least 10 seconds for any trajectory
        
        # Create simulation config
        sim_cfg = SimConfig()
        sim_cfg.duration = duration
        sim_cfg.dt = 0.1
        
        # Test default initialization
        print("Testing default initialization...")
        start_time = time.time()
        default_results = generate_controls(ego, sim_cfg, use_nn_init=False)
        default_time = time.time() - start_time
        default_times.append(default_time)
        default_statuses.append(default_results['status'])
        
        print(f"Default initialization: Time = {default_time:.3f}s, Status = {default_results['status']}")
        
        # Test neural network initialization
        print("Testing neural network initialization...")
        start_time = time.time()
        nn_results = generate_controls(ego, sim_cfg, use_nn_init=True)
        nn_time = time.time() - start_time
        nn_times.append(nn_time)
        nn_statuses.append(nn_results['status'])
        
        print(f"Neural network initialization: Time = {nn_time:.3f}s, Status = {nn_results['status']}")
        
        # Plot results
        plt.figure(figsize=(12, 6))
        
        # Plot default initialization results
        plt.subplot(1, 2, 1)
        plt.plot(default_results['x'][:, 0], default_results['x'][:, 1], 'b-', label='Path')
        plt.plot(ego.state_start[0], ego.state_start[1], 'go', label='Start')
        plt.plot(ego.state_final[0], ego.state_final[1], 'ro', label='Goal')
        plt.title(f'Default Init (Time: {default_time:.3f}s, Status: {default_results["status"]})')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.grid(True)
        plt.axis('equal')
        plt.legend()
        
        # Plot neural network initialization results
        plt.subplot(1, 2, 2)
        plt.plot(nn_results['x'][:, 0], nn_results['x'][:, 1], 'b-', label='Path')
        plt.plot(ego.state_start[0], ego.state_start[1], 'go', label='Start')
        plt.plot(ego.state_final[0], ego.state_final[1], 'ro', label='Goal')
        plt.title(f'NN Init (Time: {nn_time:.3f}s, Status: {nn_results["status"]})')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.grid(True)
        plt.axis('equal')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'test_case_{i+1}.png'))
        plt.close()
    
    # Calculate statistics
    default_success_rate = sum(1 for s in default_statuses if s == 0) / len(default_statuses)
    nn_success_rate = sum(1 for s in nn_statuses if s == 0) / len(nn_statuses)
    
    default_avg_time = np.mean(default_times)
    nn_avg_time = np.mean(nn_times)
    
    # Print statistics
    print("\n=== Performance Statistics ===")
    print(f"Default initialization:")
    print(f"  Success rate: {default_success_rate:.2%}")
    print(f"  Average time: {default_avg_time:.3f}s")
    print(f"Neural network initialization:")
    print(f"  Success rate: {nn_success_rate:.2%}")
    print(f"  Average time: {nn_avg_time:.3f}s")
    print(f"Time improvement: {(default_avg_time - nn_avg_time) / default_avg_time:.2%}")
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    
    # Plot computation times
    plt.subplot(1, 2, 1)
    plt.bar(['Default', 'Neural Network'], [default_avg_time, nn_avg_time])
    plt.title('Average Computation Time')
    plt.ylabel('Time (s)')
    plt.grid(True)
    
    # Plot success rates
    plt.subplot(1, 2, 2)
    plt.bar(['Default', 'Neural Network'], [default_success_rate, nn_success_rate])
    plt.title('Success Rate')
    plt.ylabel('Rate')
    plt.ylim([0, 1])
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'test_performance_summary.png'))
    plt.close()
    
    # Save test metrics to a text file
    metrics_file = os.path.join(results_dir, 'test_metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write("=== Performance Statistics ===\n")
        f.write(f"Default initialization:\n")
        f.write(f"  Success rate: {default_success_rate:.2%}\n")
        f.write(f"  Average time: {default_avg_time:.3f}s\n")
        f.write(f"Neural network initialization:\n")
        f.write(f"  Success rate: {nn_success_rate:.2%}\n")
        f.write(f"  Average time: {nn_avg_time:.3f}s\n")
        f.write(f"Time improvement: {(default_avg_time - nn_avg_time) / default_avg_time:.2%}\n")
    
    print(f"\nTest results saved in {results_dir}/")
    
    return {
        'default_times': default_times,
        'nn_times': nn_times,
        'default_statuses': default_statuses,
        'nn_statuses': nn_statuses,
        'default_success_rate': default_success_rate,
        'nn_success_rate': nn_success_rate,
        'default_avg_time': default_avg_time,
        'nn_avg_time': nn_avg_time
    }

if __name__ == "__main__":
    # Test initialization performance
    results = test_initialization_performance(num_tests=20, train_first=True) 