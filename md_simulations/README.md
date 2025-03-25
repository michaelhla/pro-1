# Molecular Dynamics Simulation Framework

This framework provides a Python interface for running molecular dynamics simulations using GROMACS 5.1.2 with the CHARMM36 force field and SPC/E water model. It implements the exact methodology described in the reference protocol.

## Prerequisites

- GROMACS 5.1.2
- Python 3.7+
- Required Python packages (install via `pip install -r requirements.txt`)

## Installation

1. Ensure GROMACS 5.1.2 is installed and accessible in your PATH
2. Clone this repository
3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

```python
from src.md_simulation import MDSimulation

# Initialize simulation with input structure
sim = MDSimulation(
    structure_path="path/to/protein.pdb",
    output_dir="simulation_output"
)

# Prepare structure (pH 7.0 by default)
sim.prepare_structure()

# Setup simulation box
sim.setup_box(distance=1.2)  # 1.2 nm from protein to box edge

# Add solvent and ions
sim.solvate_and_ions()

# Perform energy minimization
sim.energy_minimization(nsteps=2000)

# Run production simulation at different temperatures
temperatures = [343, 353, 363, 400]
for temp in temperatures:
    sim.run_simulation(
        temperature=temp,
        duration_ns=100,  # 100 ns simulation
        dt=0.002  # 2 fs timestep
    )
    
    # Analyze trajectory
    results = sim.analyze_trajectory(temperature=temp)
    
# Identify flexible regions
flexible_regions = sim.identify_flexible_regions(
    temperatures=[343, 353, 363]
)
```

## Features

- Structure preparation with proper protonation states at pH 7.0
- System setup with triclinic box and SPC/E water model
- Energy minimization using steepest descent
- Production MD with:
  - Particle-mesh Ewald for long-range electrostatics (cutoff 0.8 nm)
  - Van der Waals interactions with twin range potential (0.8/1.4 nm)
  - LINCS algorithm for hydrogen bonds
  - Nose-Hoover thermostat
  - Parrinello-Rahman barostat
  - Multiple temperature support (343K, 353K, 363K, 400K)
  
## Analysis Capabilities

- RMSD calculation
- RMSF analysis
- Radius of gyration
- Solvent accessible surface area
- Hydrogen bonds
- Flexible region identification

## Directory Structure

```
md_simulations/
├── src/
│   └── md_simulation.py
├── structures/
├── config/
├── requirements.txt
└── README.md
```

## Notes

- The implementation follows the exact protocol specified in the reference
- All simulations are run for 100 ns by default
- Flexible regions are identified using RMSF analysis at multiple temperatures
- N-terminal residues are excluded from flexibility analysis
- Analysis tools from GROMACS are used for trajectory analysis

## References

The implementation follows the methodology described in the reference protocol, using:
- GROMACS 5.1.2
- CHARMM36 force field
- SPC/E water model
- H++ server for protonation states 