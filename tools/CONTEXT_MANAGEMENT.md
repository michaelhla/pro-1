# Context Window Management

The Carbonic Anhydrase Designer now includes automatic context window management to prevent hitting Claude's 200k token limit during long design sessions.

## How It Works

### Token Estimation
- The system continuously estimates the total token count in the conversation
- Uses a rough approximation of ~4 characters per token for English text
- Tracks tokens from:
  - User messages
  - Assistant responses
  - Tool calls and results
  - System messages and tool definitions

### Automatic Summarization
- When the estimated token count approaches the configured limit (default: 180k tokens)
- The system automatically calls Claude to summarize the entire conversation
- The summary preserves:
  - Main objectives and goals
  - Key findings from literature searches
  - Specific mutations tested and their results
  - Quantitative data (Rosetta scores, RMSD values, etc.)
  - Current design status and next steps
  - Important conclusions and design decisions

### Context Reset
- After summarization, the conversation history is replaced with just the summary
- This allows the design process to continue within the token limit
- Multiple summarizations can occur during a single session if needed

## Configuration

### Context Limit
Set the token limit before triggering summarization:

```python
designer = CarbonicAnhydraseDesignerClaude(
    context_limit=180000  # Default: 180k tokens (stays under 200k limit)
)
```

### Custom Limits
For testing or special use cases, you can use lower limits:

```python
# Force frequent summarization for testing
designer = CarbonicAnhydraseDesignerClaude(
    context_limit=5000
)
```

## Output Files

When summarization occurs, additional files are generated:

### Summarization Log
- `summarizations.jsonl`: JSON log of all summarization events with metadata

### Summary Files
- `summary_1.txt`, `summary_2.txt`, etc.: Generated conversation summaries
- `pre_summary_1_conversation.txt`, etc.: Full conversation snapshots before summarization

### Session Summary
The final session summary includes:
- Total number of summarizations performed
- Context management statistics
- List of all generated files

## Testing

Use the test script to verify context management:

```bash
python tools/test_context_management.py
```

This creates a designer with a very low context limit to force summarization and demonstrate the functionality.

## Benefits

1. **Prevents Context Overflow**: Long design sessions won't hit the 200k token limit
2. **Maintains Continuity**: Summaries preserve important design information
3. **Transparent Operation**: All summarization events are logged and saved
4. **Configurable**: Token limits can be adjusted based on needs
5. **Robust**: Fallback mechanisms handle summarization errors

## Token Estimation Accuracy

The token estimation uses a simple heuristic (~4 chars/token) which provides reasonable approximations for:
- English text
- Code snippets
- JSON structures
- Tool call arguments

For maximum accuracy, the system errs on the side of caution by triggering summarization before the actual limit.

## Best Practices

1. **Use Default Limit**: The 180k default provides good safety margin
2. **Monitor Logs**: Check `summarizations.jsonl` to understand summarization frequency
3. **Review Summaries**: Verify that important information is preserved in summaries
4. **Adjust if Needed**: Lower limits for frequent summarization, higher for less frequent

## Error Handling

If summarization fails:
- A basic fallback summary is generated
- The error is logged
- The session continues with available context
- Manual intervention may be needed for very long sessions 