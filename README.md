# Model Predictive Control (MPC) Project

This repository contains examples and implementations of Model Predictive Control using Acados and CasADi, with a focus on path planning for autonomous vehicles.

## Project Structure

```
.
├── README.md
├── requirements.txt
├── setup.sh
├── src/
│   ├── core_solver.py           # Core solver functionality and vehicle configuration
│   ├── geodesic.py              # Geodesic path finding functionality
│   ├── path_finder.py           # Path finding utilities
│   ├── grid_path_plan.py        # Grid-based path planning
│   └── plots.py                 # Visualization functions
└── test_acados/
    ├── test_acados.py           # Basic Acados test
    ├── test_acados_lqr.py       # LQR example using Acados
    ├── test_casadi.py           # MPC example using CasADi
    └── test_casadi_simple.py    # Simple optimization using CasADi
```

## Prerequisites

- Python 3.8 or higher
- Git
- CMake
- C compiler (gcc/clang)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/sajad2025/mpc.git
cd mpc
```

2. Create and activate a Python virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Unix/macOS
# or
.\venv\Scripts\activate  # On Windows
```

3. Clone and build Acados:
```bash
git clone https://github.com/acados/acados.git
cd acados
git submodule update --recursive --init
mkdir build
cd build
cmake -DACADOS_WITH_QPOASES=ON ..
make install -j4
cd ../..
```

4. Set up the environment:
```bash
# Make the setup script executable
chmod +x setup.sh

# Source the setup script
source setup.sh
```

5. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Environment Setup

The `setup.sh` script configures all necessary environment variables and activates the Python virtual environment. It sets:
- ACADOS_SOURCE_DIR
- ACADOS_INSTALL_DIR
- Library paths (DYLD_LIBRARY_PATH, LD_LIBRARY_PATH)
- Python path for Acados templates

To exit the virtual environment:
```bash
deactivate
```

## Running the Examples

The `test_acados` directory contains several example implementations:

1. Simple CasADi Optimization:
```bash
python test_acados/test_casadi_simple.py
```

2. CasADi MPC Example:
```bash
python test_acados/test_casadi.py
```

3. Basic Acados Test:
```bash
python test_acados/test_acados.py
```

4. Acados LQR Example:
```bash
python test_acados/test_acados_lqr.py
```

Each example generates the following in the `test_acados` directory:
- Result plots (*.png files)
- Generated C code (c_generated_code directory)
- Acados OCP configuration files (acados_ocp*.json)

## Path Planning Features

The project includes advanced path planning capabilities in the `src` directory:

### Core Solver (`core_solver.py`)
- Kinematic vehicle model with configurable parameters
- Vehicle and simulation configuration classes (EgoConfig, SimConfig)
- Control generation functionality
- Path duration calculation utilities

### Geodesic Path Finding (`geodesic.py`)
- Simplified `find_geodesic` function with minimal required parameters

### Path Finding (`path_finder.py`)
- Fixed-duration path planning
- Automatic duration calculation based on distance
- Output suppression for cleaner execution
- Utility functions for path finding

### Grid Path Planning (`grid_path_plan.py`)
- Systematic exploration of different starting positions
- Multiple initial heading angles
- Automatic retry with increased duration for failed attempts

### Visualization (`plots.py`)
- Comprehensive visualization of paths and control inputs
- Multi-path visualization for grid-based planning
- Customizable plot saving options

### Configuration Options

The path planning behavior can be customized through the `EgoConfig` class:

```python
ego = EgoConfig()

# Vehicle parameters
ego.L = 2.7  # Wheelbase length (m)

# State constraints
ego.velocity_max = 3.0
ego.steering_max = 0.5

# Cost weights
ego.weight_acceleration = 1.0
ego.weight_steering_rate = 100.0
ego.weight_steering_angle = 1.0

# Terminal weights
ego.weight_terminal_position_x = 100.0
ego.weight_terminal_position_y = 100.0
ego.weight_terminal_heading = 100.0
ego.weight_terminal_velocity = 10.0
ego.weight_terminal_steering = 10.0
```

### Running Path Planning Examples

1. Geodesic path finding:
```bash
python -c "from src.geodesic import find_geodesic, calc_time_range; from src.core_solver import EgoConfig; from src.plots import plot_results; ego = EgoConfig(); _, min_dur, max_dur, _ = calc_time_range(ego); min_duration, min_results = find_geodesic(ego, min_duration=min_dur, max_duration=max_dur, time_steps=0.5); plot_results(min_results, ego) if min_results else print('No feasible path found')"
```

2. Path finding with fixed duration:
```bash
python src/path_finder.py
```

3. Grid-based path planning:
```bash
python src/grid_path_plan.py
```

Generated plots will be saved in the `docs` directory. 

## Generated Files

The following files and directories are git-ignored and will be generated when running the examples:
- `venv/` - Python virtual environment
- `acados/` - Acados library
- `test_acados/c_generated_code/` - Generated C code from Acados
- `test_acados/acados_ocp*.json` - Generated Acados OCP configurations

These files are automatically generated and should not be committed to the repository.

## Troubleshooting

1. Library not found errors:
   - Ensure you've sourced `setup.sh`
   - Check that Acados was built with qpOASES support
   - Verify all environment variables are set correctly

2. Import errors:
   - Make sure the virtual environment is activated
   - Verify all requirements are installed
   - Check Python path includes Acados template directory

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License. See [LICENSE](LICENSE) for more information.

# Neural Network-Based MPC Initialization

This project implements a neural network-based initialization approach for Model Predictive Control (MPC) to improve solver convergence and computation time.

## Overview

Model Predictive Control (MPC) is a powerful optimization-based control technique, but its performance is highly sensitive to the initial guess provided to the solver. Poor initialization can lead to slow convergence, suboptimal solutions, or even solver failure.

This project addresses this challenge by using two neural networks:
1. A Duration Predictor that estimates the optimal trajectory duration
2. A Trajectory Predictor (LSTM) that generates high-quality initialization trajectories

The neural networks learn from successful trajectories and can quickly generate good initial guesses for new start and goal states.

## Components

The project consists of the following main components:

1. **Core MPC Solver (`core_solver.py`)**: Implements the MPC solver using the Acados framework.
2. **Neural Network Models (`nn_init_main.py`)**:
   - Duration Predictor: A residual network that predicts optimal trajectory duration
   - Trajectory Predictor: An LSTM network that generates trajectory initialization
3. **Testing Script (`nn_init_test.py`)**: Compares the performance of neural network initialization vs. default initialization.

## Training the Models

The models can be trained separately using the following functions:

```python
from nn_init_main import train_and_save_duration_predictor, train_and_save_trajectory_predictor

# Train duration predictor
train_and_save_duration_predictor(
    num_samples=1000,
    batch_size=32,
    num_epochs=50,
    learning_rate=1e-3
)

# Train trajectory predictor
train_and_save_trajectory_predictor(
    num_samples=1000,
    batch_size=32,
    num_epochs=50,
    hidden_size=128,
    num_layers=2,
    dropout=0.1,
    learning_rate=1e-3
)
```

The trained models will be saved in the `models` directory as:
- `duration_predictor.pt`
- `trajectory_predictor.pt`

Training metrics and visualizations will be saved in `docs/nn_training_results/`.

## Neural Network Architecture

The neural network architecture consists of:

- An encoder that processes the input features (start state, end state, duration, dt)
- An LSTM network that generates a sequence of state vectors
- A decoder that maps LSTM outputs to state vectors

## Neural Network Initialization

The project uses an LSTM-based neural network to generate initial trajectories for the MPC solver:

**Architecture:**
- Input: 12-dimensional vector [start_state(5), end_state(5), duration(1), dt(1)]
- Output: Sequence of state vectors [x, y, θ, v, steering] for each timestep
- Network: LSTM with fully connected encoder/decoder layers

**Purpose:**
- Learns to predict feasible trajectories from boundary conditions
- Provides better initialization for MPC solver compared to linear interpolation
- Reduces solver computation time and improves convergence

Training data is generated through vehicle dynamics simulation with smooth control inputs. The model is trained using MSE loss to minimize the difference between predicted and simulated trajectories.

## Usage

### Training the Model

To train the neural network model:

```python
from nn_init_main import train_and_save_model

# Train the model
train_and_save_model(
    num_samples=1000,
    batch_size=32,
    num_epochs=50,
    hidden_size=128,
    learning_rate=1e-3
)
```

### Using the Model for Initialization

To use the trained model for MPC initialization:

```python
from core_solver import EgoConfig, SimConfig, generate_controls

# Create ego config and set start/goal states
ego = EgoConfig()
ego.state_start = [0, 0, 0, 0, 0]
ego.state_final = [10, 10, np.pi/4, 0, 0]

# Create simulation config
sim_cfg = SimConfig()
sim_cfg.duration = 10.0
sim_cfg.dt = 0.1

# Generate controls using neural network initialization
results = generate_controls(ego, sim_cfg, use_nn_init=True)
```

### Testing Performance

To compare the performance of neural network initialization vs. default initialization:

```python
from test_nn_init import test_initialization_performance

results = test_initialization_performance(num_tests=20, train_first=True)
```

## Results

The neural network-based initialization approach offers several benefits:

1. **Faster Convergence**: The MPC solver converges more quickly with better initialization.
2. **Higher Success Rate**: Better initialization leads to fewer solver failures.
3. **Better Solution Quality**: The resulting trajectories are smoother and more optimal.

## Dependencies

- Python 3.6+
- PyTorch
- NumPy
- Matplotlib
- Acados
- CasADi
- tqdm

## Installation

1. Install the required dependencies:

```bash
pip install torch numpy matplotlib tqdm
```

2. Install Acados following the instructions at [https://docs.acados.org/installation/](https://docs.acados.org/installation/)

3. Clone this repository:

```bash
git clone https://github.com/yourusername/nn-mpc-init.git
cd nn-mpc-init
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

