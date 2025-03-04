# Model Predictive Control (MPC), Kinematic Car

## Results

**Watch the results:** [https://sajad2025.github.io/mpc/](https://sajad2025.github.io/mpc/)

## Overview

This repository contains implementations of Model Predictive Control using Acados and CasADi, for kinematic car model.

## Project Structure

```
.
├── README.md
├── requirements.txt
├── setup.sh
├── src/
│   ├── core_solver.py           # Core solver functionality and vehicle configuration
│   ├── path_finder.py           # Path finding utilities
│   ├── grid_path_plan.py        # Grid-based path planning
│   ├── plots.py                 # Visualization functions
│   ├── scenarios.py             # Path planning scenarios
│   ├── create_animation.py      # Animation generator for path visualization
│   └── sequential_planning.py   # Sequential path planning implementation
├── docs/
│   ├── index.html               # GitHub Pages website
│   ├── path_animation.mp4       # Vehicle path animation
│   └── grid_path_planning_results.png  # Path planning visualization
└── test_acados/
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

### Running Path Planning Examples

1. Path finding:
```bash
python src/path_finder.py
```

2. Grid-based path planning:
```bash
python src/grid_path_plan.py
```

Generated plots will be saved in the `docs` directory. 

### Configuration Options

The path planning behavior can be customized through the `EgoConfig` class:

```python
ego = EgoConfig()

# Vehicle parameters
ego.L = 2.7  # Wheelbase length (m)

# State constraints
ego.velocity_max = 3.0
ego.velocity_min = -3.0
ego.steering_max = 0.5
ego.steering_min = -0.5

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

## Generated Files

The following files and directories are git-ignored and will be generated when running the examples:
- `venv/` - Python virtual environment
- `acados/` - Acados library
- `test_acados/c_generated_code/` - Generated C code from Acados
- `test_acados/acados_ocp*.json` - Generated Acados OCP configurations

These files are automatically generated and should not be committed to the repository.


## License

MIT License. See [LICENSE](LICENSE) for more information.

