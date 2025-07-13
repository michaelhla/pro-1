# Carbonic Anhydrase Designer

This module uses OpenAI's o3 reasoning model with function calling to design more stable carbonic anhydrase variants.

## Features

- **o3 Reasoning Model**: Leverages OpenAI's latest reasoning model for systematic protein design
- **Function Calling**: Uses structured function calls to access computational tools
- **Automated Design**: Systematic approach to protein stabilization
- **Interactive Mode**: Chat-based interface for custom design queries

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

3. Test the tools:
```bash
python test_protein_folder.py      # Test ESMFold integration
python test_rosetta_scorer.py      # Test Rosetta scoring
python test_simplified_designer.py # Test complete system
```

## Usage

### Basic Usage

```python
from carbonic_anhydrase_designer import CarbonicAnhydraseDesigner

# Create designer instance
designer = CarbonicAnhydraseDesigner(reasoning_effort="medium")

# Design a stable variant
result = designer.design_stable_carbonic_anhydrase(
    target_pdb="1CA2",
    stability_goals=[
        "Increase thermal stability by 25°C",
        "Improve stability at pH 6-8",
        "Reduce aggregation",
        "Maintain >80% catalytic activity"
    ]
)

print(result)
```

### Interactive Mode

```python
# Start interactive session
designer.interactive_design_session()
```

### Command Line

```bash
python carbonic_anhydrase_designer.py
```

## Available Tools

The system includes the following computational tools for protein design:

1. **fold_protein**: ✅ **FUNCTIONAL** - Fold protein sequences using ESMFold and save structures as PDB files
2. **calculate_rosetta_score**: ✅ **FUNCTIONAL** - Calculate Rosetta energy scores for protein structures (lower = more stable)

## How It Works

1. **Reasoning Phase**: The o3 model analyzes the design challenge and creates a systematic plan
2. **Structure Prediction**: Uses `fold_protein` to predict 3D structures from amino acid sequences
3. **Stability Assessment**: Uses `calculate_rosetta_score` to quantify protein stability
4. **Iterative Design**: Proposes mutations, folds new variants, and scores them for stability
5. **Quantitative Validation**: Compares Rosetta scores to validate design improvements
6. **Final Design**: Provides recommendations with specific mutations and quantitative rationale

## Configuration

### Reasoning Effort Levels

- `"low"`: Faster responses, less thorough analysis
- `"medium"`: Balanced speed and thoroughness (default)
- `"high"`: Most thorough analysis, slower responses

### Model Configuration

```python
designer = CarbonicAnhydraseDesigner(
    reasoning_effort="medium",  # or "low", "high"
    api_key="your-key"  # optional, uses env var if not provided
)
```

## Extending the System

### Adding New Tools

1. Define the tool schema in `_initialize_tools()`
2. Add the implementation function 
3. Update the `tool_mapping` dictionary

### Example Tool Addition

```python
# In _initialize_tools()
{
    "type": "function",
    "name": "my_new_tool",
    "description": "Description of what this tool does",
    "parameters": {
        "type": "object",
        "properties": {
            "param1": {"type": "string", "description": "Parameter description"}
        },
        "required": ["param1"]
    },
    "strict": True
}

# In _create_tool_mapping()
"my_new_tool": self._my_new_tool_implementation
```

## Protein Structure Prediction

The system now includes a fully functional protein folding tool using **ESMFold**:

- **Input**: Amino acid sequences (up to 1024 residues)
- **Output**: High-quality 3D protein structures saved as PDB files
- **Features**: 
  - Automatic model caching for faster loading
  - Structure caching to avoid recomputation
  - GPU acceleration when available
  - Detailed validation and error handling

### Tool Integration

**ESMFold Integration** (`fold_protein`):
- Accepts amino acid sequences (up to 1024 residues)
- Folds them into 3D structures using ESMFold
- Saves structures as standard PDB files
- Returns the file path for further analysis

**Rosetta Integration** (`calculate_rosetta_score`):
- Takes PDB file paths as input
- Performs side-chain optimization and energy minimization
- Calculates stability scores in Rosetta Energy Units (REU)
- Lower scores indicate more stable protein structures

## Notes

- Both computational tools are **fully functional**:
  - ESMFold for structure prediction
  - PyRosetta for stability scoring
- Additional computational tools can be added as separate modules in the tools folder
- The o3 model requires an OpenAI API key with access to reasoning models
- Function calls are processed iteratively to handle complex multi-step reasoning
- System provides quantitative stability assessment for design validation

## Future Enhancements

- Integration with real structural analysis tools (PyMOL, ChimeraX)
- Connection to protein databases (UniProt, PDB, Pfam)
- Molecular dynamics simulation integration
- Experimental validation planning
- Batch processing capabilities 