#!/usr/bin/env python3

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import math
from core_solver import EgoConfig, SimConfig, generate_controls

class DurationPredictor(nn.Module):
    """
    Neural network for predicting trajectory duration based on transformed state features.
    Uses a physics-informed architecture to ensure reasonable predictions.
    """
    def __init__(self, hidden_size=32):
        super(DurationPredictor, self).__init__()
        
        # Input size is 3: [theta_bar, lateral_err, longitudinal_err]
        input_size = 3
        
        # Input normalization
        self.input_norm = nn.BatchNorm1d(input_size)
        
        # Simpler architecture with physics-informed design
        self.base_network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_size),
            
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_size),
            
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()  # Output between 0 and 1 for scaling factor
        )
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with a custom scheme for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier/Glorot initialization for sigmoid/tanh activations
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    # Initialize bias with small positive values
                    nn.init.constant_(m.bias, 0.1)
    
    def forward(self, x, distance=None):
        """
        Forward pass through the network with physics-informed scaling.
        
        Args:
            x: Input tensor of shape [batch_size, input_size] containing 
               [theta_bar, lateral_err, longitudinal_err]
            distance: Optional distance between start and end states (meters)
               If provided, used for physics-based scaling
            
        Returns:
            Predicted duration tensor of shape [batch_size, 1]
        """
        # Apply input normalization
        x_norm = self.input_norm(x)
        
        # Get base complexity factor (0 to 1)
        complexity_factor = self.base_network(x_norm)
        
        # If distance is provided, use it for physics-based scaling
        if distance is not None:
            # Convert scalar distance to tensor if needed
            if not isinstance(distance, torch.Tensor):
                distance = torch.tensor(distance, dtype=torch.float32, device=x.device).unsqueeze(0)
                
            # Base duration is distance-dependent (assuming reasonable speed of 1-3 m/s)
            base_duration = distance / 2.0  # Assuming average speed of 2 m/s
            
            # Scale base duration by complexity factor (1.0 to 3.0)
            # More complex maneuvers (higher complexity_factor) take longer
            scaling = 1.0 + 2.0 * complexity_factor
            
            # Final duration prediction
            duration = base_duration * scaling
        else:
            # If no distance provided, output a reasonable default range (1-10 seconds)
            duration = 1.0 + 9.0 * complexity_factor
        
        return duration
    
    def forward_with_constraints(self, x, distance, max_velocity=3.0, max_duration_factor=5.0):
        """
        Forward pass with physical constraints on the output duration.
        
        Args:
            x: Input tensor of shape [batch_size, input_size]
            distance: Distance between start and end states (meters)
            max_velocity: Maximum velocity of the vehicle (m/s)
            max_duration_factor: Factor to multiply min_duration to get max_duration
            
        Returns:
            Constrained predicted duration tensor
        """
        # Get physics-informed prediction using distance
        raw_duration = self.forward(x, distance)
        
        # Calculate minimum physically possible duration based on distance and max velocity
        min_duration = torch.tensor(distance / max_velocity, device=x.device).unsqueeze(0)
        
        # Calculate maximum reasonable duration
        max_duration = min_duration * max_duration_factor
        
        # Constrain the prediction to be within reasonable bounds
        constrained_duration = torch.clamp(raw_duration, min=min_duration, max=max_duration)
        
        return constrained_duration

def preprocess_duration_inputs(state_start, state_end, dt=None):
    """
    Preprocess the inputs for the duration predictor by transforming raw states
    into more meaningful features for motion planning.
    
    Args:
        state_start: Start state [x, y, theta, velocity, steering]
        state_end: End state [x, y, theta, velocity, steering]
        dt: Time step for discretization (not used for duration prediction, kept for API compatibility)
        
    Returns:
        Preprocessed input features as numpy array
    """
    # Extract position and orientation
    x_start, y_start, theta_start = state_start[0], state_start[1], state_start[2]
    x_end, y_end, theta_end = state_end[0], state_end[1], state_end[2]
    
    # Calculate relative position in global frame
    x_bar = x_start - x_end
    y_bar = y_start - y_end
    
    # Calculate orientation difference (theta_bar)
    theta_bar = theta_start - theta_end
    # Normalize to [-pi, pi]
    theta_bar = np.arctan2(np.sin(theta_bar), np.cos(theta_bar))
    
    # Calculate lateral and longitudinal errors in the end state's frame
    lateral_err = y_bar * np.cos(theta_end) - x_bar * np.sin(theta_end)
    longitudinal_err = -y_bar * np.sin(theta_end) - x_bar * np.cos(theta_end)
    
    # Create input features (dt is no longer included)
    input_features = np.array([theta_bar, lateral_err, longitudinal_err])
    
    return input_features

class TrajectoryLSTM(nn.Module):
    """
    LSTM-based model for predicting trajectory initialization values.
    
    The model takes as input the start state, end state, duration, and dt,
    and outputs a sequence of state vectors for initializing the MPC solver.
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.1):
        super(TrajectoryLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Encoder for processing input features
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # LSTM for sequence generation
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Decoder for producing state vectors
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x, seq_length):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape [batch_size, input_size]
            seq_length: Length of the output sequence
            
        Returns:
            Trajectory output tensor of shape [batch_size, seq_length, output_size]
        """
        batch_size = x.size(0)
        
        # Encode input features
        encoded = self.encoder(x)
        
        # Repeat encoded features for each time step
        encoded = encoded.unsqueeze(1).repeat(1, seq_length, 1)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # Forward pass through LSTM
        lstm_out, _ = self.lstm(encoded, (h0, c0))
        
        # Decode LSTM output to state vectors
        trajectory = self.decoder(lstm_out)
        
        return trajectory

class TrajectoryDataset(Dataset):
    """
    Dataset for training the trajectory initialization model.
    
    Each sample consists of:
    - Input: start state, end state, dt
    - Output: sequence of state vectors for initialization and duration
    """
    def __init__(self, data, max_seq_length=None):
        self.data = data
        
        # Find the maximum sequence length if not provided
        if max_seq_length is None:
            self.max_seq_length = max(sample['target'].shape[0] for sample in data)
        else:
            self.max_seq_length = max_seq_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Extract input features for duration predictor (excluding duration)
        duration_input = np.concatenate([
            sample['input'][:5],   # start state
            sample['input'][5:10], # end state
            [sample['input'][-1]]  # dt
        ])
        duration_input = torch.tensor(duration_input, dtype=torch.float32)
        
        # Extract input features for trajectory predictor (including duration)
        trajectory_input = np.concatenate([
            sample['input'][:5],   # start state
            sample['input'][5:10], # end state
            [sample['duration']],  # duration
            [sample['input'][-1]]  # dt
        ])
        trajectory_input = torch.tensor(trajectory_input, dtype=torch.float32)
        
        target_trajectory = torch.tensor(sample['target'], dtype=torch.float32)
        target_duration = torch.tensor([sample['duration']], dtype=torch.float32)
        
        # Get the actual sequence length
        seq_length = target_trajectory.shape[0]
        
        # Pad or truncate the target trajectory to the maximum sequence length
        if seq_length < self.max_seq_length:
            # Pad with zeros
            padded_trajectory = torch.zeros((self.max_seq_length, target_trajectory.shape[1]), dtype=torch.float32)
            padded_trajectory[:seq_length] = target_trajectory
            target_trajectory = padded_trajectory
        elif seq_length > self.max_seq_length:
            # Truncate
            target_trajectory = target_trajectory[:self.max_seq_length]
        
        return duration_input, trajectory_input, target_trajectory, target_duration, seq_length

def collate_trajectories(batch):
    """
    Custom collate function for handling variable-length trajectories.
    
    Args:
        batch: List of (duration_input, trajectory_input, target_trajectory, target_duration, seq_length) tuples
        
    Returns:
        Batched inputs, targets, durations, and sequence lengths
    """
    duration_inputs, trajectory_inputs, targets, durations, seq_lengths = zip(*batch)
    
    # Stack inputs, targets, and durations
    duration_inputs = torch.stack(duration_inputs, dim=0)
    trajectory_inputs = torch.stack(trajectory_inputs, dim=0)
    targets = torch.stack(targets, dim=0)
    durations = torch.stack(durations, dim=0)
    seq_lengths = torch.tensor(seq_lengths, dtype=torch.long)
    
    return duration_inputs, trajectory_inputs, targets, durations, seq_lengths

def generate_training_data(num_samples=1000, max_duration=50.0, dt=0.1, verbose=False):
    """
    Generate training data by simulating vehicle trajectories.
    
    The approach is:
    1. Start vehicle at origin
    2. Apply smooth control inputs with various patterns (straight, curves, U-turns)
    3. Bring vehicle to smooth stops at random places
    4. Reverse the trajectory to get a valid path from random point to origin
    5. Use these trajectories as training data
    
    Args:
        num_samples: Number of training samples to generate
        max_duration: Maximum duration for trajectories
        dt: Time step for discretization
        verbose: Whether to print progress
        
    Returns:
        List of dictionaries containing input features and target trajectories
    """
    data = []
    ego = EgoConfig()
    
    # Progress bar
    pbar = tqdm(total=num_samples) if verbose else None
    
    # Define different trajectory patterns
    PATTERNS = ['straight', 'curve', 'u_turn', 'spiral', 'zigzag']
    
    while len(data) < num_samples:
        try:
            # Random duration between 5 and max_duration seconds
            duration = random.uniform(5.0, max_duration)
            
            # Create simulation config
            sim_cfg = SimConfig()
            sim_cfg.duration = duration
            sim_cfg.dt = dt
            
            # Number of time steps
            N = int(duration / dt)
            
            # Choose a random pattern
            pattern = random.choice(PATTERNS)
            
            # Generate control parameters based on pattern
            if pattern == 'straight':
                # Mostly straight line with small variations
                acc_freq = random.uniform(0.05, 0.15)
                acc_phase = random.uniform(0, 2*np.pi)
                acc_amp = random.uniform(0.3, 0.7) * ego.acceleration_max
                
                steer_freq = random.uniform(0.05, 0.15)
                steer_phase = random.uniform(0, 2*np.pi)
                steer_amp = random.uniform(0.05, 0.15) * ego.steering_rate_max
                
            elif pattern == 'curve':
                # Constant curvature with smooth entry and exit
                acc_freq = random.uniform(0.1, 0.3)
                acc_phase = random.uniform(0, 2*np.pi)
                acc_amp = random.uniform(0.4, 0.8) * ego.acceleration_max
                
                steer_freq = random.uniform(0.05, 0.15)  # Lower frequency for smoother curves
                steer_phase = random.uniform(0, 2*np.pi)
                steer_amp = random.uniform(0.3, 0.6) * ego.steering_rate_max
                
            elif pattern == 'u_turn':
                # Strong steering in one direction with appropriate speed control
                acc_freq = random.uniform(0.1, 0.2)
                acc_phase = random.uniform(0, 2*np.pi)
                acc_amp = random.uniform(0.3, 0.6) * ego.acceleration_max  # Lower speed for tight turn
                
                steer_freq = random.uniform(0.05, 0.1)  # Very low frequency for sustained turn
                steer_phase = 0  # Start turning immediately
                steer_amp = random.uniform(0.7, 0.9) * ego.steering_rate_max  # Strong steering
                
            elif pattern == 'spiral':
                # Gradually increasing steering with steady speed
                acc_freq = random.uniform(0.1, 0.2)
                acc_phase = random.uniform(0, 2*np.pi)
                acc_amp = random.uniform(0.4, 0.7) * ego.acceleration_max
                
                steer_freq = random.uniform(0.2, 0.4)  # Higher frequency for spiral
                steer_phase = random.uniform(0, 2*np.pi)
                steer_amp = random.uniform(0.4, 0.7) * ego.steering_rate_max
                
            else:  # zigzag
                # Alternating steering with steady speed
                acc_freq = random.uniform(0.1, 0.2)
                acc_phase = random.uniform(0, 2*np.pi)
                acc_amp = random.uniform(0.4, 0.7) * ego.acceleration_max
                
                steer_freq = random.uniform(0.3, 0.5)  # High frequency for zigzag
                steer_phase = random.uniform(0, 2*np.pi)
                steer_amp = random.uniform(0.5, 0.8) * ego.steering_rate_max
            
            # Initialize state
            state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # [x, y, theta, v, steering]
            
            # Simulate forward trajectory
            states = [state.copy()]
            controls = []
            
            for i in range(N):
                t = i * dt
                
                # Smooth sinusoidal control inputs with pattern-specific modulation
                # Taper controls to zero at the end for smooth stopping
                taper = max(0, 1 - (i / N) * 3)  # Taper factor (goes to 0 in the last third)
                
                if pattern == 'u_turn':
                    # Special handling for U-turn: maintain steering direction
                    progress = i / N
                    if progress < 0.4:  # Initial straight
                        steer_mod = 0.2
                    elif 0.4 <= progress < 0.7:  # U-turn phase
                        steer_mod = 1.0
                    else:  # Final straight
                        steer_mod = 0.2
                else:
                    steer_mod = 1.0
                
                acc = acc_amp * np.sin(2 * np.pi * acc_freq * t + acc_phase) * taper
                steer_rate = steer_amp * np.sin(2 * np.pi * steer_freq * t + steer_phase) * taper * steer_mod
                
                # Add some randomness to make paths more diverse
                acc += random.uniform(-0.1, 0.1) * ego.acceleration_max * taper
                steer_rate += random.uniform(-0.1, 0.1) * ego.steering_rate_max * taper
                
                # Clip controls to respect limits
                acc = np.clip(acc, ego.acceleration_min, ego.acceleration_max)
                steer_rate = np.clip(steer_rate, ego.steering_rate_min, ego.steering_rate_max)
                
                controls.append([acc, steer_rate])
                
                # Update state using vehicle dynamics
                x, y, theta, v, steering = state
                
                # State update equations (same as in the MPC model)
                x_new = x + v * np.cos(theta) * dt
                y_new = y + v * np.sin(theta) * dt
                theta_new = theta + v * np.tan(steering) / ego.L * dt
                v_new = v + acc * dt
                steering_new = steering + steer_rate * dt
                
                # Clip velocity and steering to respect limits
                v_new = np.clip(v_new, ego.velocity_min, ego.velocity_max)
                steering_new = np.clip(steering_new, ego.steering_min, ego.steering_max)
                
                # Update state
                state = np.array([x_new, y_new, theta_new, v_new, steering_new])
                states.append(state.copy())
            
            # Get final state (should have near-zero velocity and steering)
            final_state = states[-1]
            
            # Check if final state has sufficiently small velocity and steering
            if abs(final_state[3]) > 0.1 or abs(final_state[4]) > 0.05:
                continue  # Skip this sample if not stopped properly
            
            # Create reversed trajectory (from final state to origin)
            reversed_states = states[::-1]
            
            # Extract start and end states for the reversed trajectory
            start_state = reversed_states[0].copy()
            end_state = reversed_states[-1].copy()
            
            # Ensure final velocity and steering are exactly zero
            start_state[3] = 0.0
            start_state[4] = 0.0
            end_state[3] = 0.0
            end_state[4] = 0.0
            
            # Create input features
            input_features = np.concatenate([
                start_state,  # 5 elements
                end_state,    # 5 elements
                [duration],   # 1 element
                [dt]          # 1 element
            ])
            
            # Create sample
            sample = {
                'input': input_features,
                'target': np.array(reversed_states),
                'duration': duration,
                'dt': dt,
                'pattern': pattern  # Store the pattern for analysis
            }
            
            data.append(sample)
            
            if verbose:
                pbar.update(1)
                
        except Exception as e:
            if verbose:
                print(f"Error generating sample: {str(e)}")
            continue
    
    if verbose:
        pbar.close()
        
        # Print pattern distribution
        pattern_counts = {}
        for sample in data:
            pattern = sample['pattern']
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        print("\nPattern distribution in generated data:")
        for pattern, count in pattern_counts.items():
            print(f"{pattern}: {count} ({count/len(data)*100:.1f}%)")
    
    return data

def train_trajectory_predictor(model, train_loader, val_loader, num_epochs=50, learning_rate=1e-3, device='cuda'):
    """
    Train the trajectory predictor model.
    
    Args:
        model: The trajectory predictor model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on ('cuda' or 'cpu')
        
    Returns:
        Trained model and training history
    """
    # Move model to device
    model = model.to(device)
    
    # Define loss function
    criterion = nn.MSELoss()
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for duration_inputs, trajectory_inputs, targets, durations, seq_lengths in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)"):
            # Move data to device
            trajectory_inputs = trajectory_inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            pred_trajectory = model(trajectory_inputs, targets.size(1))
            
            # Calculate loss
            loss = criterion(pred_trajectory, targets)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update running loss
            train_loss += loss.item() * trajectory_inputs.size(0)
        
        # Calculate average loss
        train_loss /= len(train_loader.dataset)
        history['train_loss'].append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for duration_inputs, trajectory_inputs, targets, durations, seq_lengths in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Val)"):
                # Move data to device
                trajectory_inputs = trajectory_inputs.to(device)
                targets = targets.to(device)
                
                # Forward pass
                pred_trajectory = model(trajectory_inputs, targets.size(1))
                
                # Calculate loss
                loss = criterion(pred_trajectory, targets)
                
                # Update running loss
                val_loss += loss.item() * trajectory_inputs.size(0)
        
        # Calculate average loss
        val_loss /= len(val_loader.dataset)
        history['val_loss'].append(val_loss)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss: {val_loss:.6f}")
    
    return model, history

def save_model(model, model_type, path=None):
    """
    Save a trained model.
    
    Args:
        model: The model to save (either DurationPredictor or TrajectoryLSTM)
        model_type: Type of model ('duration' or 'trajectory')
        path: Optional custom path to save the model. If None, uses default path.
    """
    if path is None:
        # Use default paths
        if model_type == 'duration':
            path = 'models/duration_predictor.pt'
        elif model_type == 'trajectory':
            path = 'models/trajectory_predictor.pt'
        else:
            raise ValueError("model_type must be either 'duration' or 'trajectory'")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save model
    torch.save(model.state_dict(), path)
    print(f"{model_type.capitalize()} predictor saved to {path}")

def load_model(model_type, path=None, device='cpu', **kwargs):
    """
    Load a trained model.
    
    Args:
        model_type: Type of model to load ('duration' or 'trajectory')
        path: Optional custom path to load the model from. If None, uses default path.
        device: Device to load the model on ('cuda' or 'cpu')
        **kwargs: Additional arguments for model initialization
            For trajectory predictor:
                - input_size: Input size of the model (default: 12)
                - hidden_size: Hidden size of the model (default: 128)
                - output_size: Output size of the model (default: 5)
                - num_layers: Number of LSTM layers (default: 2)
                - dropout: Dropout rate (default: 0.1)
            For duration predictor:
                - hidden_size: Hidden size of the model (default: 32)
        
    Returns:
        Loaded model
    """
    if path is None:
        # Use default paths
        if model_type == 'duration':
            path = 'models/duration_predictor.pt'
        elif model_type == 'trajectory':
            path = 'models/trajectory_predictor.pt'
        else:
            raise ValueError("model_type must be either 'duration' or 'trajectory'")
    
    # Create model based on type
    if model_type == 'duration':
        hidden_size = kwargs.get('hidden_size', 32)
        model = DurationPredictor(hidden_size=hidden_size)
    else:  # trajectory
        input_size = kwargs.get('input_size', 12)  # 5 (start) + 5 (end) + 1 (duration) + 1 (dt)
        hidden_size = kwargs.get('hidden_size', 128)
        output_size = kwargs.get('output_size', 5)
        num_layers = kwargs.get('num_layers', 2)
        dropout = kwargs.get('dropout', 0.1)
        model = TrajectoryLSTM(input_size, hidden_size, output_size, num_layers, dropout)
    
    # Load weights if file exists
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"Loaded {model_type} predictor from {path}")
    else:
        print(f"Warning: No saved model found at {path}. Using untrained model.")
    
    # Set model to evaluation mode
    model.eval()
    model = model.to(device)
    
    return model

def find_best_duration(state_start, state_end, dt=0.1, model_path='models', device='cpu', max_velocity=3.0, max_duration_factor=5.0):
    """
    Find the best trajectory duration using the duration predictor model.
    
    Args:
        state_start: Start state [x, y, theta, velocity, steering]
        state_end: End state [x, y, theta, velocity, steering]
        dt: Time step for discretization (kept for API compatibility)
        model_path: Path to the directory containing the models
        device: Device to run inference on
        max_velocity: Maximum velocity of the vehicle (m/s)
        max_duration_factor: Factor to multiply min_duration to get max_duration
        
    Returns:
        Predicted duration in seconds
    """
    duration_path = os.path.join(model_path, 'duration_predictor.pt')
    
    # Calculate distance for fallback and minimum duration
    distance = np.sqrt(
        (state_end[0] - state_start[0])**2 + 
        (state_end[1] - state_start[1])**2
    )
    
    # Calculate minimum physically possible duration based on distance and max velocity
    min_duration = distance / max_velocity
    
    # Calculate maximum reasonable duration
    max_duration = min_duration * max_duration_factor
    
    # Ensure minimum duration is at least dt
    min_duration = max(min_duration, dt)
    
    if not os.path.exists(duration_path):
        print(f"Duration predictor model not found in {model_path}. Using default duration.")
        # Calculate default duration based on distance
        default_duration = min_duration * 1.5  # 1.5x the minimum duration
        return default_duration
    
    try:
        # Load duration predictor
        duration_predictor = load_model('duration', path=duration_path, device=device)
        
        # Preprocess the inputs to get the transformed features
        processed_features = preprocess_duration_inputs(state_start, state_end)
        
        # Create input tensor
        input_tensor = torch.tensor(processed_features, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Predict duration with constraints
        with torch.no_grad():
            # Get physics-informed raw prediction
            raw_prediction = duration_predictor(input_tensor, distance=distance).item()
            print(f"Raw prediction from duration predictor: {raw_prediction:.2f}s")
            
            # Get constrained prediction
            predicted_duration = duration_predictor.forward_with_constraints(
                input_tensor, 
                distance=distance, 
                max_velocity=max_velocity, 
                max_duration_factor=max_duration_factor
            ).item()
        
        print(f"Constrained prediction: {predicted_duration:.2f}s (min: {min_duration:.2f}s, max: {max_duration:.2f}s)")
        return predicted_duration
        
    except Exception as e:
        print(f"Error using duration predictor: {str(e)}. Using fallback duration.")
        fallback_duration = min_duration * 1.5  # 1.5x the minimum duration
        print(f"Fallback duration: {fallback_duration:.2f}s")
        return fallback_duration

def find_best_trajectory(state_start, state_end, duration, dt, model_path='models', device='cpu'):
    """
    Find the best initialization trajectory using the trajectory predictor model.
    
    Args:
        state_start: Start state [x, y, theta, velocity, steering]
        state_end: End state [x, y, theta, velocity, steering]
        duration: Predicted duration from duration predictor
        dt: Time step for discretization
        model_path: Path to the directory containing the models
        device: Device to run inference on
        
    Returns:
        Initialization trajectory as numpy array
    """
    trajectory_path = os.path.join(model_path, 'trajectory_predictor.pt')
    
    if not os.path.exists(trajectory_path):
        print(f"Trajectory predictor model not found in {model_path}. Using default initialization.")
        # Fall back to default initialization
        N = int(duration / dt)
        init_traj = []
        for i in range(N):
            t = float(i) / N
            x_init = state_start[0] + t * (state_end[0] - state_start[0])
            y_init = state_start[1] + t * (state_end[1] - state_start[1])
            theta_init = state_start[2] + t * (state_end[2] - state_start[2])
            v_init = 1.0
            steering_init = 0.0
            init_traj.append([x_init, y_init, theta_init, v_init, steering_init])
        return np.array(init_traj)
    
    # Load trajectory predictor
    trajectory_predictor = load_model('trajectory', path=trajectory_path, device=device)
    
    # Create input features for trajectory prediction
    trajectory_input = np.concatenate([
        state_start,         # 5 elements
        state_end,          # 5 elements
        [duration],         # 1 element
        [dt]                # 1 element
    ])
    trajectory_tensor = torch.tensor(trajectory_input, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Generate trajectory
    with torch.no_grad():
        N = int(duration / dt)
        pred_trajectory = trajectory_predictor(trajectory_tensor, N)
        init_traj = pred_trajectory.squeeze(0).cpu().numpy()
    
    return init_traj

def find_best_init(state_start, state_end, dt, model_path='models', device='cpu'):
    """
    Find the best initialization values using both duration and trajectory predictors.
    This function is maintained for backward compatibility.
    
    Args:
        state_start: Start state [x, y, theta, velocity, steering]
        state_end: End state [x, y, theta, velocity, steering]
        dt: Time step for discretization
        model_path: Path to the directory containing the models
        device: Device to run inference on
        
    Returns:
        Tuple of (init_traj, predicted_duration)
    """
    predicted_duration = find_best_duration(state_start, state_end, dt, model_path, device)
    init_traj = find_best_trajectory(state_start, state_end, predicted_duration, dt, model_path, device)
    
    return init_traj, predicted_duration

def train_duration_predictor(data_loader, model, num_epochs=50, learning_rate=1e-3, device='cuda'):
    """
    Train the duration predictor model with specialized techniques.
    
    Args:
        data_loader: DataLoader containing training data
        model: The duration predictor model
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on
        
    Returns:
        Trained model and training history
    """
    # Move model to device
    model = model.to(device)
    
    # Define loss function - use a combination of MSE and L1 loss
    mse_criterion = nn.MSELoss()
    l1_criterion = nn.L1Loss()
    
    # Define optimizer with weight decay for regularization
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.001)
    
    # Learning rate scheduler with cosine annealing
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5,  # Restart every 5 epochs
        T_mult=2,  # Double the restart period after each restart
        eta_min=learning_rate/20  # Minimum learning rate
    )
    
    # Training history
    history = {
        'train_loss': [],
        'lr': [],
        'batch_losses': []
    }
    
    # Training loop
    best_loss = float('inf')
    best_model_state = None
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        batch_losses = []
        
        for duration_inputs_raw, _, _, durations, _ in tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # Process the raw inputs to get the transformed features
            batch_size = duration_inputs_raw.size(0)
            processed_inputs = torch.zeros((batch_size, 3), dtype=torch.float32).to(device)
            
            for i in range(batch_size):
                # Extract start and end states from the raw inputs
                start_state = duration_inputs_raw[i, :5].cpu().numpy()
                end_state = duration_inputs_raw[i, 5:10].cpu().numpy()
                
                # Preprocess the inputs (dt is no longer needed)
                processed_features = preprocess_duration_inputs(start_state, end_state)
                processed_inputs[i] = torch.tensor(processed_features, dtype=torch.float32)
            
            # Move data to device
            durations = durations.to(device)
            
            # Forward pass
            pred_durations = model(processed_inputs)
            
            # Calculate loss with combined loss functions
            mse_loss = mse_criterion(pred_durations, durations)
            l1_loss = l1_criterion(pred_durations, durations)
            
            # Add a penalty for very small durations to avoid zero predictions
            zero_penalty = torch.mean(torch.exp(-pred_durations * 5.0))
            
            # Weighted combination of losses
            loss = 0.5 * mse_loss + 0.3 * l1_loss + 0.2 * zero_penalty
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update running loss
            batch_loss = loss.item()
            batch_losses.append(batch_loss)
            epoch_loss += batch_loss * duration_inputs_raw.size(0)
        
        # Step the scheduler at the end of each epoch
        scheduler.step()
        
        # Calculate average loss
        epoch_loss /= len(data_loader.dataset)
        history['train_loss'].append(epoch_loss)
        history['lr'].append(scheduler.get_last_lr()[0])
        history['batch_losses'].extend(batch_losses)
        
        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Loss: {epoch_loss:.6f}")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state['model_state_dict'])
        print(f"Restored best model from epoch {best_model_state['epoch']+1} with loss {best_model_state['loss']:.6f}")
    
    return model, history

def train_and_save_duration_predictor(num_samples=1000, batch_size=32, num_epochs=50, learning_rate=1e-3, device='cuda', max_duration=50.0):
    """
    Train and save only the duration predictor model with the new architecture.
    """
    print("Generating training data...")
    data = generate_training_data(num_samples=num_samples, max_duration=max_duration, dt=0.1, verbose=True)
    
    # Split data into training and validation sets
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    
    print(f"Training set size: {len(train_data)}")
    
    # Create dataset
    train_dataset = TrajectoryDataset(train_data)
    
    # Create data loader with custom collate function
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_trajectories)
    
    # Create directories
    docs_dir = 'docs'
    results_dir = os.path.join(docs_dir, 'nn_training_results')
    models_dir = 'models'
    os.makedirs(docs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # Train the duration predictor
    print("\nTraining duration predictor with new architecture...")
    duration_predictor = DurationPredictor(hidden_size=32)
    duration_predictor, duration_history = train_duration_predictor(
        train_loader,
        duration_predictor,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device
    )
    
    # Save duration predictor
    torch.save(duration_predictor.state_dict(), os.path.join(models_dir, 'duration_predictor.pt'))
    print("Duration predictor saved to models/duration_predictor.pt")
    
    # Plot duration predictor training history
    plt.figure(figsize=(12, 10))
    
    # Loss subplot
    plt.subplot(2, 1, 1)
    plt.plot(duration_history['train_loss'], label='Train Loss', linewidth=2)
    plt.title('Duration Predictor Training Loss', fontsize=12)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Loss', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # Learning rate subplot
    plt.subplot(2, 1, 2)
    plt.plot(duration_history['lr'], label='Learning Rate', linewidth=2, color='green')
    plt.title('Learning Rate Schedule', fontsize=12)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Learning Rate', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'duration_training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot batch losses
    plt.figure(figsize=(10, 6))
    plt.plot(duration_history['batch_losses'], alpha=0.3, color='blue')
    # Add smoothed line
    window_size = min(100, len(duration_history['batch_losses']))
    if window_size > 0:
        smoothed = np.convolve(duration_history['batch_losses'], 
                              np.ones(window_size)/window_size, 
                              mode='valid')
        plt.plot(range(window_size-1, len(duration_history['batch_losses'])), 
                smoothed, linewidth=2, color='red', label=f'Moving Avg (window={window_size})')
    plt.title('Batch Losses During Training', fontsize=12)
    plt.xlabel('Batch', fontsize=10)
    plt.ylabel('Loss', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'duration_batch_losses.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save training metrics
    metrics_file = os.path.join(results_dir, 'duration_training_metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write(f"Training Configuration:\n")
        f.write(f"- Number of samples: {num_samples}\n")
        f.write(f"- Batch size: {batch_size}\n")
        f.write(f"- Number of epochs: {num_epochs}\n")
        f.write(f"- Learning rate: {learning_rate}\n")
        f.write(f"- Maximum duration: {max_duration}\n")
        f.write(f"- Architecture: Simplified network with transformed inputs\n")
        f.write(f"- Hidden size: 32\n\n")
        f.write(f"Duration Predictor Results:\n")
        f.write(f"- Final training loss: {duration_history['train_loss'][-1]:.6f}\n")
        f.write(f"- Best training loss: {min(duration_history['train_loss']):.6f}\n")
    
    print("Training completed. Results saved in docs/nn_training_results/")

def train_and_save_trajectory_predictor(num_samples=100, batch_size=32, num_epochs=50, hidden_size=128, num_layers=2, dropout=0.1, learning_rate=1e-3, device='cuda', max_duration=50.0):
    """
    Train and save only the trajectory predictor model.
    """
    print("Generating training data...")
    data = generate_training_data(num_samples=num_samples, max_duration=max_duration, dt=0.1, verbose=True)
    
    # Split data into training and validation sets
    train_size = int(0.8 * len(data))
    val_size = len(data) - train_size
    
    train_data = data[:train_size]
    val_data = data[train_size:]
    
    print(f"Training set size: {len(train_data)}")
    print(f"Validation set size: {len(val_data)}")
    
    # Find the maximum sequence length in the data
    max_seq_length = max(max(sample['target'].shape[0] for sample in train_data),
                        max(sample['target'].shape[0] for sample in val_data))
    
    print(f"Maximum sequence length: {max_seq_length}")
    
    # Create datasets
    train_dataset = TrajectoryDataset(train_data, max_seq_length)
    val_dataset = TrajectoryDataset(val_data, max_seq_length)
    
    # Create data loaders with custom collate function
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_trajectories)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_trajectories)
    
    # Create directories
    docs_dir = 'docs'
    results_dir = os.path.join(docs_dir, 'nn_training_results')
    models_dir = 'models'
    os.makedirs(docs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # Train the trajectory predictor
    print("\nTraining trajectory predictor...")
    input_size = 12  # 5 (start) + 5 (end) + 1 (duration) + 1 (dt)
    output_size = 5  # 5 state dimensions
    
    trajectory_predictor = TrajectoryLSTM(input_size, hidden_size, output_size, num_layers, dropout)
    
    # Train trajectory predictor
    trajectory_predictor, trajectory_history = train_trajectory_predictor(
        model=trajectory_predictor,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device
    )
    
    # Save trajectory predictor
    torch.save(trajectory_predictor.state_dict(), os.path.join(models_dir, 'trajectory_predictor.pt'))
    print("Trajectory predictor saved to models/trajectory_predictor.pt")
    
    # Plot trajectory predictor training history
    plt.figure(figsize=(12, 5))
    
    # Loss subplot
    plt.subplot(1, 2, 1)
    plt.plot(trajectory_history['train_loss'], label='Train', linewidth=2)
    plt.plot(trajectory_history['val_loss'], label='Validation', linewidth=2)
    plt.title('Trajectory Predictor Loss', fontsize=12)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Loss', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # Accuracy subplot
    plt.subplot(1, 2, 2)
    train_acc = [1.0 - loss for loss in trajectory_history['train_loss']]
    val_acc = [1.0 - loss for loss in trajectory_history['val_loss']]
    plt.plot(train_acc, label='Train', linewidth=2)
    plt.plot(val_acc, label='Validation', linewidth=2)
    plt.title('Training Progress', fontsize=12)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Performance Score', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'trajectory_training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save training metrics
    metrics_file = os.path.join(results_dir, 'trajectory_training_metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write(f"Training Configuration:\n")
        f.write(f"- Number of samples: {num_samples}\n")
        f.write(f"- Batch size: {batch_size}\n")
        f.write(f"- Number of epochs: {num_epochs}\n")
        f.write(f"- Hidden size: {hidden_size}\n")
        f.write(f"- Number of LSTM layers: {num_layers}\n")
        f.write(f"- Dropout rate: {dropout}\n")
        f.write(f"- Learning rate: {learning_rate}\n")
        f.write(f"- Maximum duration: {max_duration}\n\n")
        f.write(f"Trajectory Predictor Results:\n")
        f.write(f"- Final training loss: {trajectory_history['train_loss'][-1]:.6f}\n")
        f.write(f"- Final validation loss: {trajectory_history['val_loss'][-1]:.6f}\n")
        f.write(f"- Best validation loss: {min(trajectory_history['val_loss']):.6f}\n")
    
    print("Training completed. Results saved in docs/nn_training_results/")

if __name__ == "__main__":
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Choose which model to train
    import argparse
    parser = argparse.ArgumentParser(description='Train neural network models for trajectory prediction')
    parser.add_argument('--model', type=str, choices=['duration', 'trajectory', 'both'], default='both',
                      help='Which model to train: duration predictor, trajectory predictor, or both')
    parser.add_argument('--samples', type=int, default=1000, 
                      help='Number of training samples to generate')
    parser.add_argument('--epochs', type=int, default=50,
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training')
    
    args = parser.parse_args()
    
    # Common parameters
    params = {
        'num_samples': args.samples,
        'batch_size': args.batch_size,
        'num_epochs': args.epochs,
        'device': device,
        'max_duration': 50.0
    }
    
    if args.model == 'duration':
        # Duration-specific parameters
        params.update({
            'learning_rate': 1e-3,  # Updated learning rate for new architecture
        })
        train_and_save_duration_predictor(**params)
    elif args.model == 'trajectory':
        # Add trajectory-specific parameters
        params.update({
            'hidden_size': 128,
            'num_layers': 2,
            'dropout': 0.1,
            'learning_rate': 1e-3
        })
        train_and_save_trajectory_predictor(**params)
    else:  # both
        # Train duration predictor with its specific parameters
        duration_params = params.copy()
        duration_params.update({
            'learning_rate': 1e-3,  # Updated learning rate for new architecture
        })
        train_and_save_duration_predictor(**duration_params)
        
        # Train trajectory predictor with its specific parameters
        trajectory_params = params.copy()
        trajectory_params.update({
            'hidden_size': 128,
            'num_layers': 2,
            'dropout': 0.1,
            'learning_rate': 1e-3
        })
        train_and_save_trajectory_predictor(**trajectory_params) 