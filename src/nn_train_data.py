#!/usr/bin/env python3

import os
import numpy as np
import random
from tqdm import tqdm
from core_solver import EgoConfig, SimConfig, generate_controls
from plots import plot_results

def generate_training_data(dt=0.1, verbose=False):
    """
    Generate 16 training samples with different goal positions around the origin.
    All trajectories start at the origin (0,0) with zero heading, velocity, and steering.
    Goal positions are at x=[-10,0,10], y=[-10,0,10] (excluding x=y=0),
    with final headings of 0° and 180°.
    
    Args:
        dt: Time step for discretization
        verbose: Whether to print progress
        
    Returns:
        List of dictionaries containing input features and target trajectories
    """
    data = []
    
    # Define grid of goal positions and headings
    x_positions = [-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50]  # x coordinates
    y_positions = [-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50]  # y coordinates
    headings    = [0, np.pi, np.pi/2, 3*np.pi/2]  # 0°, 180°, 90°, 270°
    
    # Calculate total number of samples
    total_samples = (len(x_positions) * len(y_positions) - 1) * len(headings)
    
    # Progress bar
    pbar = tqdm(total=total_samples) if verbose else None
    
    # Sample counter
    sample_id = 1
    
    # Generate samples
    for end_x in x_positions:
        for end_y in y_positions:
            # Skip the origin
            if end_x == 0.0 and end_y == 0.0:
                continue
                
            for final_heading in headings:
                try:
                    # Create EgoConfig
                    ego = EgoConfig()
                    
                    # Set start and end states
                    ego.state_start = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
                    ego.state_final = np.array([end_x, end_y, final_heading, 0.0, 0.0])
                    
                    # Calculate distance to goal
                    distance = np.sqrt(end_x**2 + end_y**2)
                    
                    # Start with a reasonable duration
                    duration = max(10.0, distance / 2.0)  # At least 10 seconds or 0.5 distance
                    
                    # Create simulation config
                    sim_cfg = SimConfig()
                    sim_cfg.dt = dt
                    
                    # Try different durations if needed
                    max_attempts = 5
                    success = False
                    
                    for attempt in range(max_attempts):
                        # Update duration for this attempt
                        sim_cfg.duration = duration
                        
                        # Number of time steps
                        N = int(duration / dt)
                        
                        # Calculate angle to goal
                        angle_to_goal = np.arctan2(end_y, end_x)
                        
                        # Create heuristic initialization for velocity and steering
                        v_init = np.ones(N) * 2.0  # Constant forward velocity
                        
                        # Create steering profile based on goal position
                        steering_init = np.zeros(N)
                        
                        # Determine if we need to turn left or right initially
                        angle_diff = (angle_to_goal - 0.0 + np.pi) % (2*np.pi) - np.pi
                        turn_direction = np.sign(angle_diff)
                        
                        # First half: turn toward goal
                        half_n = N // 2
                        # Use a sinusoidal profile for smooth steering
                        for i in range(half_n):
                            t = i / half_n
                            steering_init[i] = turn_direction * 0.3 * np.sin(np.pi * t)
                        
                        # Second half: turn to achieve final heading
                        # Calculate final turn direction
                        final_angle_diff = (final_heading - angle_to_goal + np.pi) % (2*np.pi) - np.pi
                        final_turn_direction = np.sign(final_angle_diff)
                        
                        for i in range(half_n, N):
                            t = (i - half_n) / (N - half_n)
                            steering_init[i] = final_turn_direction * 0.3 * np.sin(np.pi * t)
                        
                        # Generate optimal controls
                        results = generate_controls(
                            ego=ego, 
                            sim_cfg=sim_cfg, 
                            use_nn_init=False, 
                            v_init=v_init, 
                            steering_init=steering_init
                        )
                        
                        # Check if solver succeeded
                        if results['status'] == 0:
                            success = True
                            break
                        else:
                            # Increase duration for next attempt
                            duration *= 1.5
                            if verbose:
                                print(f"Solver failed for ({end_x}, {end_y}, {final_heading}). Increasing duration to {duration}")
                    
                    if not success:
                        if verbose:
                            print(f"Failed to generate trajectory for ({end_x}, {end_y}, {final_heading}) after {max_attempts} attempts")
                        continue
                    
                    # Extract states from results
                    states = results['x']
                    
                    # Create input features
                    input_features = np.concatenate([
                        ego.state_start,  # 5 elements
                        ego.state_final,  # 5 elements
                        [duration],       # 1 element
                        [dt]              # 1 element
                    ])
                    
                    # Create sample
                    sample = {
                        'input': input_features,
                        'target': states,
                        'duration': duration,
                        'dt': dt,
                        'state_start': ego.state_start,
                        'state_end': ego.state_final,
                        'states': states,
                        'N': N
                    }
                    
                    data.append(sample)
                    
                    if verbose:
                        pbar.update(1)
                    
                    sample_id += 1
                        
                except Exception as e:
                    if verbose:
                        print(f"Error generating sample: {str(e)}")
                    continue
    
    if verbose:
        pbar.close()
        print(f"\nGenerated {len(data)} samples")
    
    return data

if __name__ == "__main__":
    # Set verbose flag for the main script
    verbose = True
    
    # Generate training data
    print("Generating training data...")
    data = generate_training_data(verbose=verbose)
    
    # Create models directory if it doesn't exist
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    
    # Save the training data to a file
    dataset_path = os.path.join(models_dir, 'trajectory_dataset.npz')
    
    # Prepare data for saving
    inputs = np.array([sample['input'] for sample in data])
    
    # Find the maximum trajectory length
    max_length = max(sample['target'].shape[0] for sample in data)
    
    # Pad trajectories to the same length
    padded_targets = []
    for sample in data:
        target = sample['target']
        current_length = target.shape[0]
        
        # Create padded array
        padded_target = np.zeros((max_length, target.shape[1]))
        padded_target[:current_length] = target
        
        padded_targets.append(padded_target)
    
    targets = np.array(padded_targets)
    durations = np.array([sample['duration'] for sample in data])
    
    # Save the dataset with trajectory lengths for later unpadding
    np.savez_compressed(
        dataset_path,
        inputs=inputs,
        targets=targets,
        durations=durations,
        dt=data[0]['dt'] if data else dt,
        trajectory_lengths=np.array([sample['target'].shape[0] for sample in data])
    )
    
    print(f"Dataset saved to {dataset_path}")
    
    # Create plot directory
    plot_dir = 'docs/nn_train_data'
    os.makedirs(plot_dir, exist_ok=True)
    
    # First, clear any existing files in the directory
    for file in os.listdir(plot_dir):
        if file.endswith('.png'):
            os.remove(os.path.join(plot_dir, file))
    
    print("\nGenerating plots...")
    for i, sample in enumerate(tqdm(data), 1):
        # Create EgoConfig for plotting
        ego_config = EgoConfig()
        ego_config.state_start = sample['state_start']
        ego_config.state_final = sample['state_end']
        
        # Create results dict for plotting
        results = {
            't': np.array([i * sample['dt'] for i in range(sample['N']+1)]),
            'x': np.array(sample['states']),
            'u': np.zeros((sample['N'], 2))  # Dummy controls for plotting
        }
        
        # Create plot filename with new naming convention
        # Round coordinates to 1 decimal place and heading to integer degrees
        # Use state_final from ego_config to ensure we're using the correct values
        plot_filename = f'd{i}_x{ego_config.state_final[0]:.1f}_y{ego_config.state_final[1]:.1f}_h{np.rad2deg(ego_config.state_final[2]):.0f}.png'
        plot_path = os.path.join(plot_dir, plot_filename)
        
        # Print debug info
        if verbose:
            print(f"Sample {i}: Goal = ({ego_config.state_final[0]:.1f}, {ego_config.state_final[1]:.1f}, {np.rad2deg(ego_config.state_final[2]):.0f}°)")
        
        # Plot and save
        plot_results(results, ego_config, save_path=plot_path)
    
    print(f"Plots saved in {plot_dir}")

# Add a function to load the dataset for later use
def load_trajectory_dataset(file_path='models/trajectory_dataset.npz'):
    """
    Load the saved trajectory dataset.
    
    Args:
        file_path: Path to the saved dataset file
        
    Returns:
        Dictionary containing the dataset components:
        - inputs: Input features for each sample
        - targets: Target trajectories for each sample (padded to max length)
        - durations: Duration of each trajectory
        - dt: Time step used for discretization
        - trajectory_lengths: Original length of each trajectory before padding
    """
    try:
        data = np.load(file_path)
        return {
            'inputs': data['inputs'],
            'targets': data['targets'],
            'durations': data['durations'],
            'dt': data['dt'].item(),
            'trajectory_lengths': data['trajectory_lengths'] if 'trajectory_lengths' in data else None
        }
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None 