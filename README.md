# Model Predictive Control (MPC) Project

This repository contains examples and implementations of Model Predictive Control using Acados and CasADi, with a focus on path planning for autonomous vehicles.

## Project Structure

```
.
├── README.md
├── requirements.txt
├── setup.sh
├── src/
│   ├── main_path_plan.py        # Main path planning implementation
│   ├── path_finder.py           # Path finding utilities
│   └── grid_path_plan.py        # Grid-based path planning
└── test_acados/
    ├── test_acados.py           # Basic Acados test
    ├── test_acados_lqr.py       # LQR example using Acados
    ├── test_casadi.py           # MPC example using CasADi
    └── test_casadi_simple.py    # Simple optimization using CasADi
```

Generated files (plots, C code, JSON configs) will be saved in the `docs` directory.

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

## Environment Setup

The `setup.sh` script configures all necessary environment variables and activates the Python virtual environment. It sets:
- ACADOS_SOURCE_DIR
- ACADOS_INSTALL_DIR
- Library paths (DYLD_LIBRARY_PATH, LD_LIBRARY_PATH)
- Python path for Acados templates

To use the environment:
```bash
source setup.sh
```

To exit the virtual environment:
```bash
deactivate
```

## Dependencies

Key Python packages:
- acados_template
- casadi
- numpy
- matplotlib
- scipy

See `requirements.txt` for complete list with versions.

## Generated Files

The following files and directories are git-ignored and will be generated when running the examples:
- `venv/` - Python virtual environment
- `acados/` - Acados library
- `test_acados/c_generated_code/` - Generated C code from Acados
- `test_acados/*_results.png` - Generated plot results
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

3. Git-related issues:
   - If you see untracked files that should be ignored, try:
     ```bash
     git rm -r --cached .
     git add .
     ```
   - Make sure to never commit the virtual environment or generated files

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License. See [LICENSE](LICENSE) for more information.

## Path Planning Features

The project includes advanced path planning capabilities in the `src` directory:

### Main Path Planning (`main_path_plan.py`)
- Kinematic vehicle model with configurable parameters
- Customizable cost function weights for:
  - Acceleration smoothness
  - Steering rate
  - Steering angle
  - Terminal state precision
- Automatic duration optimization
- Visualization of paths and control inputs

### Path Finding (`path_finder.py`)
- Fixed-duration path planning
- Automatic duration calculation based on distance
- Output suppression for cleaner execution
- Utility functions for path finding

### Grid Path Planning (`grid_path_plan.py`)
- Systematic exploration of different starting positions
- Multiple initial heading angles
- Visualization of all successful paths
- Automatic retry with increased duration for failed attempts

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

1. Basic path planning:
```bash
python src/main_path_plan.py
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