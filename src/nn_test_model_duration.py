#!/usr/bin/env python3

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from nn_train_model import DurationPredictor, load_model, find_best_duration, preprocess_duration_inputs
from core_solver import EgoConfig, SimConfig

def test_duration_predictor_raw(model, input_tensor, distance=None):
    """
    Test the duration predictor directly with a given input tensor.
    
    Args:
        model: The duration predictor model
        input_tensor: Input tensor of shape [batch_size, input_size]
        distance: Optional distance between start and end states (meters)
        
    Returns:
        Predicted duration
    """
    model.eval()
    with torch.no_grad():
        output = model(input_tensor, distance=distance)
    return output.item()

def test_duration_predictor_constrained(model, input_tensor, distance, max_velocity=3.0, max_duration_factor=5.0):
    """
    Test the duration predictor with physical constraints.
    
    Args:
        model: The duration predictor model
        input_tensor: Input tensor of shape [batch_size, input_size]
        distance: Distance between start and end states (meters)
        max_velocity: Maximum velocity of the vehicle (m/s)
        max_duration_factor: Factor to multiply min_duration to get max_duration
        
    Returns:
        Constrained predicted duration
    """
    model.eval()
    with torch.no_grad():
        output = model.forward_with_constraints(
            input_tensor, 
            distance=distance, 
            max_velocity=max_velocity, 
            max_duration_factor=max_duration_factor
        )
    return output.item()

def analyze_model_weights(model):
    """
    Analyze the weights of the duration predictor model to identify potential issues.
    
    Args:
        model: The duration predictor model
    """
    print("\n=== Model Weight Analysis ===")
    
    # Check for NaN or extremely large/small values in weights
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"WARNING: NaN values found in {name}")
        
        if torch.isinf(param).any():
            print(f"WARNING: Infinite values found in {name}")
        
        max_val = param.abs().max().item()
        if max_val > 1000:
            print(f"WARNING: Very large values found in {name}: {max_val}")
        
        if param.numel() > 0 and param.abs().max().item() < 1e-6 and 'bias' not in name:
            print(f"WARNING: Very small values found in {name}: {param.abs().max().item()}")
    
    # Check for dead neurons (LeakyReLU units that are always negative)
    # This is a simplified check and may not catch all cases
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if hasattr(module, 'weight'):
                weight = module.weight.data
                if (weight.mean(dim=1) < -0.1).any():
                    print(f"WARNING: Potential dead neurons in {name}")

def visualize_model_predictions(model, scenarios, max_velocity=3.0, max_duration_factor=5.0):
    """
    Visualize the model predictions for different distances and scenarios.
    
    Args:
        model: The duration predictor model
        scenarios: List of test scenarios
        max_velocity: Maximum velocity of the vehicle (m/s)
        max_duration_factor: Factor to multiply min_duration to get max_duration
    """
    print("\n=== Visualizing Model Predictions ===")
    
    # Create a range of distances to test
    distances = np.linspace(1, 30, 30)
    
    # Test predictions for a simple forward scenario at different distances
    raw_predictions = []
    constrained_predictions = []
    min_durations = []
    max_durations = []
    
    for dist in distances:
        # Create a simple forward scenario
        start_state = np.array([0, 0, 0, 0, 0])
        end_state = np.array([dist, 0, 0, 0, 0])
        
        # Preprocess inputs
        processed_features = preprocess_duration_inputs(start_state, end_state)
        input_tensor = torch.tensor(processed_features, dtype=torch.float32).unsqueeze(0)
        
        # Get raw prediction
        raw_pred = test_duration_predictor_raw(model, input_tensor, distance=dist)
        raw_predictions.append(raw_pred)
        
        # Get constrained prediction
        constrained_pred = test_duration_predictor_constrained(
            model, 
            input_tensor, 
            distance=dist, 
            max_velocity=max_velocity, 
            max_duration_factor=max_duration_factor
        )
        constrained_predictions.append(constrained_pred)
        
        # Calculate min and max durations
        min_duration = dist / max_velocity
        min_durations.append(min_duration)
        
        max_duration = min_duration * max_duration_factor
        max_durations.append(max_duration)
    
    # Plot predictions vs. distances
    plt.figure(figsize=(12, 8))
    plt.plot(distances, raw_predictions, 'o-', label='Raw Model Predictions', alpha=0.7)
    plt.plot(distances, constrained_predictions, 's-', label='Constrained Predictions', color='green')
    plt.plot(distances, min_durations, '--', label=f'Min Duration (distance/{max_velocity})', color='red')
    plt.plot(distances, max_durations, '-.', label=f'Max Duration ({max_duration_factor}x min)', color='orange')
    
    # Plot a reasonable baseline (e.g., distance / avg_speed + buffer)
    baseline = distances / 1.0 + 5.0  # Assuming avg speed of 1.0 m/s and 5s buffer
    plt.plot(distances, baseline, ':', label='Baseline (distance/1.0 + 5.0)', color='purple', alpha=0.7)
    
    plt.xlabel('Distance (m)')
    plt.ylabel('Predicted Duration (s)')
    plt.title('Duration Predictor: Predictions vs. Distance')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save the plot
    os.makedirs('docs/nn_debug', exist_ok=True)
    plt.savefig('docs/nn_debug/duration_vs_distance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Test predictions for all scenarios
    scenario_names = [s['name'] for s in scenarios]
    scenario_raw_predictions = []
    scenario_constrained_predictions = []
    scenario_distances = []
    scenario_min_durations = []
    scenario_max_durations = []
    
    for scenario in scenarios:
        start_state = np.array(scenario['start'])
        end_state = np.array(scenario['goal'])
        
        # Calculate distance
        distance = np.sqrt(np.sum((end_state[:2] - start_state[:2])**2))
        scenario_distances.append(distance)
        
        # Calculate min and max durations
        min_duration = distance / max_velocity
        scenario_min_durations.append(min_duration)
        
        max_duration = min_duration * max_duration_factor
        scenario_max_durations.append(max_duration)
        
        # Preprocess inputs
        processed_features = preprocess_duration_inputs(start_state, end_state)
        input_tensor = torch.tensor(processed_features, dtype=torch.float32).unsqueeze(0)
        
        # Get raw prediction
        raw_pred = test_duration_predictor_raw(model, input_tensor, distance=distance)
        scenario_raw_predictions.append(raw_pred)
        
        # Get constrained prediction
        constrained_pred = test_duration_predictor_constrained(
            model, 
            input_tensor, 
            distance=distance, 
            max_velocity=max_velocity, 
            max_duration_factor=max_duration_factor
        )
        scenario_constrained_predictions.append(constrained_pred)
    
    # Plot predictions for scenarios
    plt.figure(figsize=(14, 8))
    
    x = np.arange(len(scenario_names))
    width = 0.2
    
    plt.bar(x - width*1.5, scenario_raw_predictions, width, label='Raw Predictions', alpha=0.7)
    plt.bar(x - width/2, scenario_constrained_predictions, width, label='Constrained Predictions')
    plt.bar(x + width/2, scenario_min_durations, width, label=f'Min Duration (distance/{max_velocity})', alpha=0.7)
    plt.bar(x + width*1.5, scenario_max_durations, width, label=f'Max Duration ({max_duration_factor}x min)', alpha=0.5)
    
    plt.xlabel('Scenario')
    plt.ylabel('Duration (s)')
    plt.title('Duration Predictor: Scenario Predictions')
    plt.xticks(x, scenario_names, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig('docs/nn_debug/scenario_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print predictions
    print("\nScenario Predictions:")
    for i, name in enumerate(scenario_names):
        print(f"{name}:")
        print(f"  - Distance: {scenario_distances[i]:.2f}m")
        print(f"  - Raw Prediction: {scenario_raw_predictions[i]:.2f}s")
        print(f"  - Constrained Prediction: {scenario_constrained_predictions[i]:.2f}s")
        print(f"  - Min Duration: {scenario_min_durations[i]:.2f}s")
        print(f"  - Max Duration: {scenario_max_durations[i]:.2f}s")
        print(f"  - Default Duration: {scenarios[i]['default_duration']:.2f}s")

def test_find_best_duration_function(scenarios, max_velocity=3.0, max_duration_factor=5.0):
    """
    Test the find_best_duration function with the test scenarios.
    
    Args:
        scenarios: List of test scenarios
        max_velocity: Maximum velocity of the vehicle (m/s)
        max_duration_factor: Factor to multiply min_duration to get max_duration
    """
    print("\n=== Testing find_best_duration Function ===")
    
    for scenario in scenarios:
        start_state = np.array(scenario['start'])
        end_state = np.array(scenario['goal'])
        dt = 0.1  # Still pass dt for API compatibility
        
        # Calculate distance
        distance = np.sqrt(np.sum((end_state[:2] - start_state[:2])**2))
        
        # Calculate min and max durations
        min_duration = distance / max_velocity
        max_duration = min_duration * max_duration_factor
        
        # Call the function
        predicted_duration = find_best_duration(
            state_start=start_state,
            state_end=end_state,
            dt=dt,
            max_velocity=max_velocity,
            max_duration_factor=max_duration_factor
        )
        
        print(f"\n{scenario['name']}:")
        print(f"  - Distance: {distance:.2f}m")
        print(f"  - Predicted Duration: {predicted_duration:.2f}s")
        print(f"  - Min Duration: {min_duration:.2f}s")
        print(f"  - Max Duration: {max_duration:.2f}s")
        print(f"  - Default Duration: {scenario['default_duration']:.2f}s")
        
        # Check if the prediction is reasonable
        if predicted_duration < min_duration:
            print(f"  - WARNING: Predicted duration ({predicted_duration:.2f}s) is less than min duration ({min_duration:.2f}s)")
        
        if predicted_duration > max_duration:
            print(f"  - WARNING: Predicted duration ({predicted_duration:.2f}s) is greater than max duration ({max_duration:.2f}s)")

def inspect_model_architecture(model):
    """
    Inspect the architecture of the duration predictor model.
    
    Args:
        model: The duration predictor model
    """
    print("\n=== Model Architecture ===")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

def test_model_with_random_inputs(model, num_samples=100, max_velocity=3.0, max_duration_factor=5.0):
    """
    Test the model with random inputs to check for stability.
    
    Args:
        model: The duration predictor model
        num_samples: Number of random samples to test
        max_velocity: Maximum velocity of the vehicle (m/s)
        max_duration_factor: Factor to multiply min_duration to get max_duration
    """
    print("\n=== Testing Model with Random Inputs ===")
    
    # Generate random inputs
    raw_predictions = []
    constrained_predictions = []
    min_durations = []
    max_durations = []
    distances = []
    
    zero_count_raw = 0
    negative_count_raw = 0
    very_large_count_raw = 0
    
    zero_count_constrained = 0
    negative_count_constrained = 0
    very_large_count_constrained = 0
    
    for i in range(num_samples):
        # Generate random start and end states
        start_state = np.random.uniform(-10, 10, 5)
        end_state = np.random.uniform(-10, 10, 5)
        
        # Calculate distance
        distance = np.sqrt(np.sum((end_state[:2] - start_state[:2])**2))
        distances.append(distance)
        
        # Calculate min and max durations
        min_duration = distance / max_velocity
        min_durations.append(min_duration)
        
        max_duration = min_duration * max_duration_factor
        max_durations.append(max_duration)
        
        # Preprocess inputs
        processed_features = preprocess_duration_inputs(start_state, end_state)
        input_tensor = torch.tensor(processed_features, dtype=torch.float32).unsqueeze(0)
        
        # Get raw prediction
        raw_pred = test_duration_predictor_raw(model, input_tensor, distance=distance)
        raw_predictions.append(raw_pred)
        
        # Get constrained prediction
        constrained_pred = test_duration_predictor_constrained(
            model, 
            input_tensor, 
            distance=distance, 
            max_velocity=max_velocity, 
            max_duration_factor=max_duration_factor
        )
        constrained_predictions.append(constrained_pred)
        
        # Check for issues in raw predictions
        if raw_pred == 0:
            zero_count_raw += 1
        
        if raw_pred < 0:
            negative_count_raw += 1
        
        if raw_pred > 100:
            very_large_count_raw += 1
            
        # Check for issues in constrained predictions
        if constrained_pred == 0:
            zero_count_constrained += 1
        
        if constrained_pred < 0:
            negative_count_constrained += 1
        
        if constrained_pred > 100:
            very_large_count_constrained += 1
    
    # Print statistics
    print(f"Tested {num_samples} random inputs:")
    print("\nRaw Predictions:")
    print(f"  - Zero predictions: {zero_count_raw} ({zero_count_raw/num_samples*100:.1f}%)")
    print(f"  - Negative predictions: {negative_count_raw} ({negative_count_raw/num_samples*100:.1f}%)")
    print(f"  - Very large predictions (>100s): {very_large_count_raw} ({very_large_count_raw/num_samples*100:.1f}%)")
    
    print("\nConstrained Predictions:")
    print(f"  - Zero predictions: {zero_count_constrained} ({zero_count_constrained/num_samples*100:.1f}%)")
    print(f"  - Negative predictions: {negative_count_constrained} ({negative_count_constrained/num_samples*100:.1f}%)")
    print(f"  - Very large predictions (>100s): {very_large_count_constrained} ({very_large_count_constrained/num_samples*100:.1f}%)")
    
    # Plot histogram of predictions
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.hist(raw_predictions, bins=30, alpha=0.7, label='Raw Predictions')
    plt.xlabel('Predicted Duration (s)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Raw Predictions for Random Inputs')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.hist(constrained_predictions, bins=30, alpha=0.7, label='Constrained Predictions', color='green')
    plt.xlabel('Predicted Duration (s)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Constrained Predictions for Random Inputs')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('docs/nn_debug/random_predictions_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot predictions vs distance
    plt.figure(figsize=(12, 8))
    plt.scatter(distances, raw_predictions, alpha=0.5, label='Raw Predictions')
    plt.scatter(distances, constrained_predictions, alpha=0.5, label='Constrained Predictions', color='green')
    plt.scatter(distances, min_durations, alpha=0.5, label=f'Min Duration (distance/{max_velocity})', color='red')
    plt.scatter(distances, max_durations, alpha=0.5, label=f'Max Duration ({max_duration_factor}x min)', color='orange')
    
    plt.xlabel('Distance (m)')
    plt.ylabel('Duration (s)')
    plt.title('Predictions vs. Distance for Random Inputs')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig('docs/nn_debug/random_predictions_vs_distance.png', dpi=300, bbox_inches='tight')
    plt.close()

def visualize_transformed_inputs(scenarios):
    """
    Visualize the transformed inputs for the test scenarios.
    
    Args:
        scenarios: List of test scenarios
    """
    print("\n=== Visualizing Transformed Inputs ===")
    
    # Collect transformed inputs for all scenarios
    scenario_names = [s['name'] for s in scenarios]
    theta_bars = []
    lateral_errs = []
    longitudinal_errs = []
    
    for scenario in scenarios:
        start_state = np.array(scenario['start'])
        end_state = np.array(scenario['goal'])
        
        # Preprocess inputs
        processed_features = preprocess_duration_inputs(start_state, end_state)
        theta_bars.append(processed_features[0])
        lateral_errs.append(processed_features[1])
        longitudinal_errs.append(processed_features[2])
        
        # Print the transformed inputs
        print(f"\n{scenario['name']}:")
        print(f"  - theta_bar: {processed_features[0]:.4f}")
        print(f"  - lateral_err: {processed_features[1]:.4f}")
        print(f"  - longitudinal_err: {processed_features[2]:.4f}")
    
    # Plot the transformed inputs
    plt.figure(figsize=(12, 8))
    
    x = np.arange(len(scenario_names))
    width = 0.25
    
    plt.bar(x - width, theta_bars, width, label='theta_bar')
    plt.bar(x, lateral_errs, width, label='lateral_err')
    plt.bar(x + width, longitudinal_errs, width, label='longitudinal_err')
    
    plt.xlabel('Scenario')
    plt.ylabel('Value')
    plt.title('Transformed Inputs for Test Scenarios')
    plt.xticks(x, scenario_names, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig('docs/nn_debug/transformed_inputs.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """
    Main function to test and debug the duration predictor.
    """
    print("=== Duration Predictor Debug Tool ===")
    
    # Set physical constraints
    max_velocity = 3.0  # m/s
    max_duration_factor = 5.0
    
    print(f"Using physical constraints:")
    print(f"  - Max velocity: {max_velocity} m/s")
    print(f"  - Max duration factor: {max_duration_factor}x min duration")
    
    # Create directories
    os.makedirs('docs/nn_debug', exist_ok=True)
    
    # Define test scenarios (same as in nn_test_scenarios.py)
    scenarios = [
        {
            'name': 'Simple Forward',
            'start': [0, 0, 0, 0, 0],
            'goal': [10, 0, 0, 0, 0],
            'default_duration': 10.0
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
    
    # Visualize the transformed inputs for the test scenarios
    visualize_transformed_inputs(scenarios)
    
    # Check if model exists
    model_path = 'models/duration_predictor.pt'
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using nn_train_model.py")
        return
    
    # Load the model
    print(f"Loading model from {model_path}...")
    model = load_model('duration', path=model_path)
    
    # Inspect model architecture
    inspect_model_architecture(model)
    
    # Analyze model weights
    analyze_model_weights(model)
    
    # Test the model with the scenarios
    visualize_model_predictions(model, scenarios, max_velocity, max_duration_factor)
    
    # Test the find_best_duration function
    test_find_best_duration_function(scenarios, max_velocity, max_duration_factor)
    
    # Test with random inputs
    test_model_with_random_inputs(model, max_velocity=max_velocity, max_duration_factor=max_duration_factor)
    
    print("\nDebug complete! Results saved in docs/nn_debug/")

if __name__ == "__main__":
    main() 