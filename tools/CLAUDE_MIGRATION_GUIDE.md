# Migration Guide: OpenAI o3 → Claude 4 Sonnet

This guide explains the differences between the original OpenAI o3-based carbonic anhydrase designer and the new Claude 4 Sonnet version.

## Key Differences

### API and Dependencies

**OpenAI Version (`carbonic_anhydrase_designer.py`)**
- Uses `openai` package
- Requires `OPENAI_API_KEY` environment variable
- Uses o3 model with reasoning capabilities
- Supports reasoning effort levels ("low", "medium", "high")

**Claude Version (`carbonic_anhydrase_designer_claude.py`)**
- Uses `anthropic` package  
- Requires `ANTHROPIC_API_KEY` environment variable
- Uses Claude 3.5 Sonnet (latest available model)
- Standard conversation-based interaction

### Model Configuration

**OpenAI o3:**
```python
model_config = {
    "model": "o3",
    "reasoning": {
        "effort": reasoning_effort,
        "summary": "detailed"
    },
    "store": True,
    "max_output_tokens": 100000
}
```

**Claude 4 Sonnet:**
```python
model = "claude-3-5-sonnet-20241022"
max_tokens = 8192
```

### Tool Definition Format

**OpenAI Format:**
```python
{
    "type": "function",
    "name": "fold_protein",
    "description": "...",
    "parameters": {
        "type": "object",
        "properties": {...},
        "required": [...],
        "additionalProperties": False
    },
    "strict": True
}
```

**Anthropic Format:**
```python
{
    "name": "fold_protein",
    "description": "...",
    "input_schema": {
        "type": "object", 
        "properties": {...},
        "required": [...]
    }
}
```

### Response Processing

**OpenAI o3:**
- Uses `previous_response_id` for conversation continuity
- Processes reasoning summaries and items
- Complex response structure with output items

**Claude:**
- Uses messages array for conversation history
- Simpler response structure with content blocks
- Direct tool use and text content

### Usage Examples

**OpenAI Version:**
```python
from carbonic_anhydrase_designer import CarbonicAnhydraseDesigner

designer = CarbonicAnhydraseDesigner(
    reasoning_effort="medium",
    api_key="your-openai-key"
)
result = designer.design_stable_carbonic_anhydrase()
```

**Claude Version:**
```python
from carbonic_anhydrase_designer_claude import CarbonicAnhydraseDesignerClaude

designer = CarbonicAnhydraseDesignerClaude(
    model="claude-3-5-sonnet-20241022",
    max_tokens=8192,
    api_key="your-anthropic-key"
)
result = designer.design_stable_carbonic_anhydrase()
```

## Installation

### For Claude Version:
```bash
pip install -r requirements_claude.txt
export ANTHROPIC_API_KEY="your-api-key-here"
```

### For OpenAI Version:
```bash
pip install -r requirements.txt  # Original requirements
export OPENAI_API_KEY="your-api-key-here"
```

## Feature Parity

Both versions provide identical functionality:
- ✅ Protein folding with ESMFold
- ✅ Rosetta energy scoring
- ✅ RMSD calculation with sequence alignment
- ✅ Web search capabilities
- ✅ Catalytic activity examination with PyMOL
- ✅ Secondary structure analysis
- ✅ Iterative design workflow
- ✅ Comprehensive logging and output files

## Output Differences

**OpenAI Version:**
- `reasoning_data.jsonl` - Contains detailed reasoning summaries
- Iteration-based output files
- Complex conversation tracking

**Claude Version:**
- `conversation_history.json` - Full conversation in messages format
- Simpler iteration tracking
- Direct tool call logging

## Choosing Between Versions

**Use OpenAI o3 Version When:**
- You need advanced reasoning capabilities
- You want detailed reasoning summaries
- You have access to o3 model
- You prefer explicit reasoning effort control

**Use Claude Version When:**
- You prefer Anthropic's API and models
- You want simpler conversation handling
- You need reliable tool use capabilities
- You want to avoid OpenAI's reasoning token costs

Both versions are fully functional and will produce equivalent scientific results for carbonic anhydrase design. 