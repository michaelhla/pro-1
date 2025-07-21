#!/usr/bin/env python3
"""
Carbonic Anhydrase Designer using Anthropic's Claude 4 Sonnet with tool use.

This module uses Claude 4 Sonnet to design more stable carbonic anhydrase
variants by leveraging tool use capabilities to access various computational
tools and databases.
"""

import json
import os
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Callable, Any, Optional
import anthropic
from dotenv import load_dotenv
load_dotenv()

# Import the tools
from protein_folder import fold_protein
from rosetta_scorer import calculate_rosetta_score
from rmsd_calculator import calculate_rmsd_with_alignment
from websearch_tool import websearch
from catalytic_activity_examiner import examine_catalytic_activity
from secondary_structure_examiner import examine_secondary_structure


REFERENCE_SEQUENCE="MSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSYDQATSLRILNNGHAFNVEFDDSQDKAVLKGGPLDGTYRLIQFHFHWGSLDGQGSEHTVDKKKYAAELHLVHWNTKYGDFGKAVQQPDGLAVLGIFLKVGSAKPGLQKVVDVLDSIKTKGKSADFTNFDPRGLLPESLDYWTYPGSLTTPPLLECVTWIVLKEPISVSSEQVLKFRKLNFNGEGEPEELMVDNWRPAQPLKNRQIKASFK"

class CarbonicAnhydraseDesignerClaude:
    """
    A class that uses Anthropic's Claude 4 Sonnet with tool use to design
    more stable carbonic anhydrase variants.
    
    Features automatic context window management that summarizes conversations
    when approaching the 200k token limit to prevent context overflow.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-5-sonnet-20241022", 
                 output_dir: str = None, max_tokens: int = 8192, context_limit: int = 40000,
                 enable_thinking: bool = False, thinking_budget: int = 2048):
        """
        Initialize the designer with Anthropic client and model configuration.
        
        Args:
            api_key: Anthropic API key (if None, will use ANTHROPIC_API_KEY env var)
            model: Claude model to use (default: claude-3-5-sonnet-20241022)
            output_dir: Directory to save outputs (if None, creates timestamped dir)
            max_tokens: Maximum tokens for responses
            context_limit: Token limit before triggering summarization (default: 180k to stay under 200k)
            enable_thinking: Whether to enable Claude's extended thinking (requires supported model)
            thinking_budget: Number of tokens allocated for thinking (minimum 1024, must be < max_tokens)
        """
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.context_limit = context_limit
        self.enable_thinking = enable_thinking
        self.thinking_budget = thinking_budget
        
        # Set up output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"carbonic_anhydrase_design_claude_{timestamp}"
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tools for Anthropic format
        self.tools = self._initialize_tools()
        self.tool_mapping = self._create_tool_mapping()
        
        # Track session state
        self.iteration_count = 0
        self.messages = []
        self.summarization_count = 0
        
        print(f"🚀 Starting Claude design session - outputs will be saved to: {self.output_dir}")
        print(f"📊 Context limit set to {self.context_limit:,} tokens")
        if self.enable_thinking:
            if self.thinking_budget >= 1024 and self.thinking_budget < self.max_tokens:
                print(f"🧠 Extended thinking enabled with {self.thinking_budget:,} token budget")
            else:
                print(f"⚠️  Extended thinking disabled: invalid budget ({self.thinking_budget}) - must be ≥1024 and <{self.max_tokens}")
        else:
            print("💭 Extended thinking disabled")

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text string.
        Uses a rough approximation of ~4 characters per token for English text.
        """
        return len(text) // 4

    def _estimate_message_tokens(self, message: Dict) -> int:
        """
        Estimate tokens in a message, handling different content types.
        """
        total_tokens = 0
        
        if isinstance(message.get("content"), str):
            total_tokens += self._estimate_tokens(message["content"])
        elif isinstance(message.get("content"), list):
            for content in message["content"]:
                if isinstance(content, dict):
                    if content.get("type") == "text":
                        total_tokens += self._estimate_tokens(content.get("text", ""))
                    elif content.get("type") == "tool_result":
                        total_tokens += self._estimate_tokens(content.get("content", ""))
                    elif content.get("type") == "tool_use":
                        total_tokens += self._estimate_tokens(str(content.get("input", {})))
                elif hasattr(content, 'text'):
                    total_tokens += self._estimate_tokens(content.text)
                elif hasattr(content, 'input'):
                    total_tokens += self._estimate_tokens(str(content.input))
        
        return total_tokens

    def _estimate_total_context_tokens(self) -> int:
        """
        Estimate the total number of tokens in the current conversation context.
        """
        total = 0
        for message in self.messages:
            total += self._estimate_message_tokens(message)
        
        # Add some buffer for system messages and tool definitions
        total += self._estimate_tokens(str(self.tools)) + 1000
        
        return total

    def _summarize_conversation(self) -> str:
        """
        Use Claude to summarize the conversation history to compress the context.
        """
        print(f"📝 Triggering conversation summarization (attempt #{self.summarization_count + 1})")
        
        # Create a text version of the conversation for summarization
        conversation_text = ""
        for i, message in enumerate(self.messages):
            role = message.get("role", "unknown")
            conversation_text += f"\n--- {role.upper()} MESSAGE {i+1} ---\n"
            
            if isinstance(message.get("content"), str):
                conversation_text += message["content"]
            elif isinstance(message.get("content"), list):
                for content in message["content"]:
                    if isinstance(content, dict):
                        if content.get("type") == "text":
                            conversation_text += content.get("text", "")
                        elif content.get("type") == "tool_use":
                            conversation_text += f"\n[TOOL CALL: {content.get('name', 'unknown')}]\n"
                            conversation_text += f"Arguments: {content.get('input', {})}\n"
                        elif content.get("type") == "tool_result":
                            conversation_text += f"\n[TOOL RESULT]\n"
                            # Truncate very long tool results
                            result_content = content.get("content", "")
                            if len(result_content) > 2000:
                                result_content = result_content[:2000] + "... [TRUNCATED]"
                            conversation_text += result_content + "\n"
                    elif hasattr(content, 'text'):
                        conversation_text += content.text
                    elif hasattr(content, 'name') and hasattr(content, 'input'):
                        conversation_text += f"\n[TOOL CALL: {content.name}]\n"
                        conversation_text += f"Arguments: {content.input}\n"
            
            conversation_text += "\n"
        
        # Create summarization prompt
        summarization_prompt = f"""
You are being asked to summarize a protein engineering conversation to compress the context. This is a CRITICAL task to prevent context overflow.

Please provide a comprehensive but concise summary that preserves:
1. The main objective: designing stable carbonic anhydrase variants
2. Key findings from web searches and literature review
3. Specific mutations that were proposed and tested
4. Results from computational tools (fold_protein, calculate_rosetta_score, calculate_rmsd_with_sequences, etc.)
5. Quantitative results (Rosetta scores, RMSD values, structural analysis)
6. Current design iteration status and next planned steps
7. Important conclusions and design decisions made so far

Be specific about:
- Exact mutations tested (e.g., "L143F mutation")
- Quantitative results with numbers (e.g., "Rosetta score: -485.6")
- Structural insights and assessments
- What approaches worked and what didn't

Format your summary to be used as context for continuing the protein design process.

CONVERSATION TO SUMMARIZE:
{conversation_text}

SUMMARY:
"""
        
        try:
            # Save the pre-summarization conversation
            self.summarization_count += 1
            pre_summary_file = self.output_dir / f"pre_summary_{self.summarization_count}_conversation.txt"
            with open(pre_summary_file, "w") as f:
                f.write(conversation_text)
            
            # Call Claude to summarize with retry logic
            summary_response = self._api_call_with_retry(
                model=self.model,
                max_tokens=4096,
                messages=[
                    {
                        "role": "user", 
                        "content": summarization_prompt
                    }
                ]
            )
            
            # Extract the summary text
            summary_text = ""
            for content_block in summary_response.content:
                if content_block.type == "text":
                    summary_text += content_block.text
            
            # Save the summary
            summary_file = self.output_dir / f"summary_{self.summarization_count}.txt"
            with open(summary_file, "w") as f:
                f.write(summary_text)
            
            print(f"✅ Conversation summarized (summary #{self.summarization_count})")
            print(f"   Original length: {len(conversation_text):,} characters")
            print(f"   Summary length: {len(summary_text):,} characters")
            print(f"   Compression ratio: {len(summary_text)/len(conversation_text):.2%}")
            
            return summary_text
            
        except Exception as e:
            print(f"❌ Error during summarization: {e}")
            # Return a basic summary if Claude summarization fails
            return f"""
SUMMARIZATION ERROR - BASIC SUMMARY:
This is iteration {self.iteration_count} of a carbonic anhydrase stability design project.
The goal is to design more stable variants while maintaining catalytic activity.
Reference sequence: {REFERENCE_SEQUENCE}
Key catalytic residues: Y7, N62, H64, N67, Q92
Zinc binding residues: H94, H96, H119

[Error occurred during detailed summarization: {e}]
Please continue with the protein design process.
"""

    def _check_and_manage_context(self):
        """
        Check if we're approaching the context limit and summarize if needed.
        """
        estimated_tokens = self._estimate_total_context_tokens()
        
        print(f"📊 Context check: ~{estimated_tokens:,} tokens (limit: {self.context_limit:,})")
        
        if estimated_tokens > self.context_limit:
            print("⚠️  Approaching context limit - triggering summarization")
            
            # Get summary
            summary = self._summarize_conversation()
            
            # Replace the conversation with just the summary
            self.messages = [
                {
                    "role": "user",
                    "content": f"""
CONVERSATION SUMMARY (to manage context window):
{summary}

Please continue with the carbonic anhydrase design process based on this summary.
Continue using the computational tools extensively and keep iterating on the design.
"""
                }
            ]
            
            print(f"✅ Context reset with summary. New estimated tokens: ~{self._estimate_total_context_tokens():,}")
            
            # Log the summarization event
            summary_log = {
                "timestamp": datetime.now().isoformat(),
                "iteration": self.iteration_count,
                "summarization_number": self.summarization_count,
                "original_tokens_estimate": estimated_tokens,
                "new_tokens_estimate": self._estimate_total_context_tokens(),
                "compression_ratio": self._estimate_total_context_tokens() / estimated_tokens
            }
            
            with open(self.output_dir / "summarizations.jsonl", "a") as f:
                f.write(json.dumps(summary_log) + "\n")

    def _api_call_with_retry(self, max_retries: int = 10, base_delay: float = 1.0, **kwargs):
        """
        Make an API call to Claude with retry logic for overload errors.
        
        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay for exponential backoff
            **kwargs: Arguments to pass to the messages.create call
            
        Returns:
            Response from the API call
        """
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                # Make the API call
                response = self.client.messages.create(**kwargs)
                
                # If successful, log the success if we had previous failures
                if attempt > 0:
                    print(f"✅ API call succeeded on attempt {attempt + 1}")
                
                return response
                
            except Exception as e:
                error_str = str(e).lower()
                last_exception = e
                
                # Check if this is an overload error (status 529)
                is_overload_error = (
                    "overloaded" in error_str or 
                    "529" in error_str or
                    "overloaded_error" in error_str
                )
                
                if is_overload_error and attempt < max_retries - 1:
                    # Calculate wait time with exponential backoff + jitter
                    wait_time = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    
                    print(f"⏳ API overloaded (attempt {attempt + 1}/{max_retries})")
                    print(f"   Error: {e}")
                    print(f"   Waiting {wait_time:.1f}s before retry...")
                    
                    # Log the retry attempt
                    retry_log = {
                        "timestamp": datetime.now().isoformat(),
                        "iteration": self.iteration_count,
                        "retry_attempt": attempt + 1,
                        "max_retries": max_retries,
                        "wait_time": wait_time,
                        "error": str(e)
                    }
                    
                    with open(self.output_dir / "api_retries.jsonl", "a") as f:
                        f.write(json.dumps(retry_log) + "\n")
                    
                    time.sleep(wait_time)
                    continue
                else:
                    # Either not an overload error, or we've exhausted retries
                    if is_overload_error:
                        print(f"❌ API still overloaded after {max_retries} attempts")
                    
                    raise e
        
        # This should never be reached, but just in case
        raise last_exception or Exception("Unexpected error in retry logic")
    
    def _initialize_tools(self) -> List[Dict[str, Any]]:
        """
        Initialize and return the available tools for carbonic anhydrase design in Anthropic format.
        """
        tools = [
            {
                "name": "fold_protein",
                "description": "Fold a protein sequence using ESMFold and save the structure as a PDB file. Use this when you need to predict the 3D structure of a protein from its amino acid sequence.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "sequence": {
                            "type": "string",
                            "description": "Amino acid sequence to fold (single letter code, e.g., 'MKILVS...')"
                        },
                        "filename": {
                            "type": "string",
                            "description": "Filename to save the folded structure (e.g., 'mutant.pdb')"
                        }
                    },
                    "required": ["sequence", "filename"]
                }
            },
            {
                "name": "calculate_rosetta_score",
                "description": "Calculate the Rosetta energy score for a protein structure from a PDB file. Lower scores indicate more stable structures. Use this to evaluate the stability of folded proteins.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "pdb_file_path": {
                            "type": "string",
                            "description": "Path to the PDB file to score (e.g., 'predicted_structures/protein_123.pdb')"
                        }
                    },
                    "required": ["pdb_file_path"]
                }
            },
            {
                "name": "calculate_rmsd_with_sequences",
                "description": "Calculate RMSD between the hardcoded reference structure (hCA2_folded.pdb) and a newly folded protein structure using sliding window sequence alignment. Finds the region of maximum sequence overlap, then calculates RMSD over the aligned core region. Returns RMSD value plus overlap percentage, sequence identity, and detailed alignment information.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "pdb_file2": {
                            "type": "string",
                            "description": "Path to the newly folded PDB file to compare against reference (e.g., 'predicted_structures/mutant.pdb')"
                        }
                    },
                    "required": ["pdb_file2"]
                }
            },
            {
                "name": "websearch",
                "description": "Perform a web search using Perplexity's Sonar API to find current information about protein engineering, research papers, methodologies, and recent advances. Returns natural language responses with citations.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to execute (e.g., 'latest carbonic anhydrase stability research 2024', 'protein thermostability engineering methods')"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "examine_catalytic_activity",
                "description": "Examine the catalytic activity sites of carbonic anhydrase using PyMOL visualization. Takes exact residue dictionaries specifying which residues to examine and their positions. Generates labeled images (returned as base64-encoded data) and assesses catalytic integrity to ensure modifications haven't affected enzyme activity. Images are saved to tools/images/{image_subdir}/.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "pdb_file_path": {
                            "type": "string",
                            "description": "Path to the PDB file to examine (e.g., 'predicted_structures/mutant.pdb')"
                        },
                        "active_site_residues": {
                            "type": "object",
                            "description": "Dictionary of active site residues with format {'Y7': {'name': 'TYR', 'function': 'Proton transfer', 'number': 7}, ...}. For standard hCA II: Y7, N62, H64, N67, Q92"
                        },
                        "zinc_binding_residues": {
                            "type": "object",
                            "description": "Dictionary of zinc binding residues with format {'H94': {'name': 'HIS', 'function': 'Zinc coordination', 'number': 94}, ...}. For standard hCA II: H94, H96, H119"
                        },
                        "image_subdir": {
                            "type": "string",
                            "description": "Subdirectory name within tools/images/ to save visualization images. Choose a descriptive name."
                        }
                    },
                    "required": ["pdb_file_path", "active_site_residues", "zinc_binding_residues", "image_subdir"]
                }
            },
            {
                "name": "examine_secondary_structure",
                "description": "Examine secondary structure and calculate structural properties using PyMOL. Analyzes helix/sheet/loop content, calculates SASA (Solvent Accessible Surface Area), radius of gyration, and generates a colored secondary structure visualization (returned as base64-encoded data). Provides quality assessment and compactness analysis. Images are saved to tools/images/{image_subdir}/.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "pdb_file_path": {
                            "type": "string",
                            "description": "Path to the PDB file to examine (e.g., 'predicted_structures/mutant.pdb')"
                        },
                        "image_subdir": {
                            "type": "string",
                            "description": "Subdirectory name within tools/images/ to save visualization images. Choose a descriptive name."
                        }
                    },
                    "required": ["pdb_file_path", "image_subdir"]
                }
            }
        ]
        
        return tools
    
    def _create_tool_mapping(self) -> Dict[str, Callable]:
        """
        Create a mapping of tool names to their implementation functions.
        """
        return {
            "fold_protein": fold_protein,
            "calculate_rosetta_score": calculate_rosetta_score,
            "calculate_rmsd_with_sequences": calculate_rmsd_with_alignment,
            "websearch": websearch,
            "examine_catalytic_activity": examine_catalytic_activity,
            "examine_secondary_structure": examine_secondary_structure
        }
    
    def _execute_tool_call(self, tool_call) -> str:
        """
        Execute a tool call and return the result.
        
        Args:
            tool_call: Tool call object from Claude's response
            
        Returns:
            String result from the tool execution
        """
        tool_name = tool_call.name
        
        # Get the function from our mapping
        if tool_name not in self.tool_mapping:
            error_msg = f"ERROR: Unknown tool '{tool_name}'"
            print(f"❌ {error_msg}")
            return error_msg
        
        try:
            # Get arguments (already parsed for Anthropic)
            arguments = tool_call.input
            
            print(f"🔧 Executing tool: {tool_name}")
            print(f"   Arguments: {arguments}")
            
            # Execute the function
            result = self.tool_mapping[tool_name](**arguments)
            
            print(f"✅ Tool {tool_name} completed")
            print(result)
            
            # Log tool execution to file
            tool_log = {
                "timestamp": datetime.now().isoformat(),
                "iteration": self.iteration_count,
                "tool_name": tool_name,
                "arguments": arguments,
                "status": "success",
                "result_length": len(str(result))
            }
            
            with open(self.output_dir / "tool_calls.jsonl", "a") as f:
                f.write(json.dumps(tool_log) + "\n")
            
            # Simple feedback for visualization functions
            if tool_name in ['examine_catalytic_activity', 'examine_secondary_structure']:
                print(f"🖼️  Generated visualizations for {tool_name}")
            
            return str(result)
            
        except Exception as e:
            error_msg = f"ERROR executing {tool_name}: {str(e)}"
            print(f"❌ {error_msg}")
            
            # Log error to file
            error_log = {
                "timestamp": datetime.now().isoformat(),
                "iteration": self.iteration_count,
                "tool_name": tool_name,
                "arguments": arguments if 'arguments' in locals() else None,
                "status": "error",
                "error": str(e)
            }
            
            with open(self.output_dir / "tool_calls.jsonl", "a") as f:
                f.write(json.dumps(error_log) + "\n")
            
            return error_msg
    
    def _process_response(self, response) -> tuple[bool, List[Dict[str, Any]]]:
        """
        Process Claude's response and execute any tool calls.
        
        Args:
            response: Response from Anthropic's Messages API
            
        Returns:
            (is_complete, tool_results): 
                - is_complete: True if no more tool calls needed
                - tool_results: List of tool result messages to add to conversation
        """
        tool_results = []
        
        # Print the response content (non-tool parts)
        print("💬 CLAUDE'S RESPONSE:")
        print("=" * 60)
        
        for content_block in response.content:
            if content_block.type == "text":
                print(content_block.text)
                print()
            elif content_block.type == "thinking":
                print("🧠 CLAUDE'S THINKING:")
                print("─" * 40)
                print(content_block.thinking)
                print("─" * 40)
                print()
            elif content_block.type == "tool_use":
                print(f"🔧 TOOL CALL: {content_block.name}")
                print(f"   Tool ID: {content_block.id}")
                print(f"   Input: {content_block.input}")
                
                # Execute the tool
                result = self._execute_tool_call(content_block)
                
                # Create tool result for conversation
                tool_result = {
                    "type": "tool_result",
                    "tool_use_id": content_block.id,
                    "content": str(result)
                }
                tool_results.append(tool_result)
                
                print(f"   Result length: {len(str(result))} characters")
                print("   " + "─" * 50)
            else:
                # Handle any other content types that might be added in the future
                print(f"⚠️  Unknown content block type: {content_block.type}")
                if hasattr(content_block, '__dict__'):
                    print(f"   Content: {content_block.__dict__}")
                print()
        
        print("=" * 60)
        
        # Save this iteration's content
        iteration_content = ""
        for content_block in response.content:
            if content_block.type == "text":
                iteration_content += content_block.text + "\n"
            elif content_block.type == "thinking":
                iteration_content += "=== CLAUDE'S THINKING ===\n"
                iteration_content += content_block.thinking + "\n"
                iteration_content += "=== END THINKING ===\n\n"
        
        if iteration_content.strip():
            iteration_file = self.output_dir / f"iteration_{self.iteration_count}_output.txt"
            with open(iteration_file, "w") as f:
                f.write(iteration_content)
        
        # Log usage if available
        if hasattr(response, 'usage'):
            usage = response.usage
            print(f"📊 TOKEN USAGE:")
            print(f"   Input tokens: {usage.input_tokens:,}")
            print(f"   Output tokens: {usage.output_tokens:,}")
            print()
        
        return len(tool_results) == 0, tool_results
    
    def design_stable_carbonic_anhydrase(self, target_pdb: str = "1CA2", 
                                       stability_goals: List[str] = None) -> str:
        """
        Design a more stable carbonic anhydrase variant using Claude 4 Sonnet.
        
        Args:
            target_pdb: PDB ID of the starting carbonic anhydrase structure
            stability_goals: List of stability improvements to target
            
        Returns:
            Final design recommendations as a string
        """
        if stability_goals is None:
            stability_goals = [
                "Increase thermal stability by 20°C",
                "Improve pH stability range",
                "Reduce aggregation propensity",
                "Maintain catalytic activity"
            ]
        
        print("=" * 80)
        print("CARBONIC ANHYDRASE STABILITY DESIGN SESSION - CLAUDE 4 SONNET")
        print("=" * 80)
        print(f"Target: {target_pdb}")
        print(f"Goals: {', '.join(stability_goals)}")
        print("=" * 80)
        
        # Create the initial prompt for Claude
        design_prompt = f"""
        You are an expert protein engineer tasked with designing a more stable variant of carbonic anhydrase. THIS IS SO WE CAN REDUCE CARBON EMISSIONS AND IMPROVE CARBON CAPTURE. 

        ORIGINAL SEQUENCE: {REFERENCE_SEQUENCE}
        Stability goals: {', '.join(stability_goals)}
        CATALYTIC RESIDUES: Y7, N62, H64, N67, Q92 (proton transfer and activator binding)
        ZINC BINDING RESIDUES: H94, H96, H119 (essential for catalytic activity)

        You have access to six computational tools that you MUST use extensively to accomplish this task:
        1. fold_protein: Predicts 3D structures from amino acid sequences using ESMFold
        2. calculate_rosetta_score: Calculates Rosetta energy scores for PDB structures (lower or more negative = more stable)
        3. calculate_rmsd_with_sequences: Uses sliding window sequence alignment to find maximum overlap between reference (hCA2_folded.pdb) and new structure, then calculates RMSD over aligned core region. Returns RMSD, overlap percentage, and sequence identity.
        4. websearch: Searches the web for current information about protein engineering, research papers, and methodologies
        5. examine_secondary_structure: Analyzes secondary structure content, calculates SASA and structural properties, and provides quality assessment with base64-encoded visualizations you can examine
        
        IMPORTANT: You MUST use these tools multiple times throughout the design process. Do not stop after a single tool call. Keep using tools until you have completed the entire design workflow.

        Please approach this systematically:
        1. Use websearch to find current research on general enzyme stability and recent engineering approaches. Go deep into the literature and Uniprot. THIS SHOULD BE VERY THOROUGH BEFORE CONTINUING TO THE NEXT STEPS. THIS IS A FUNDAMENTALLY IMPORTANT STEP.
        2. Using the findings from your research, propose specific amino acid mutations that could improve stability. These can be simple point mutations, or larger insertions/deletions.
        3. Then modify the original sequence using the mutations you proposed. Make sure you have applied your mutations correctly. 
        4. Use the fold_protein tool to predict the structure of your modified sequence. This will return a filepath to your pdb. 
        5. Use calculate_rosetta_score to get the stability score of your modified sequence.
        6. Use calculate_rmsd_with_sequences to compare with the reference structure.
        7. Use examine_secondary_structure to analyze the new structure's fold, SASA, and structural quality
        8. Compare scores and provide recommendations with quantitative rationale. 
        

        Always preserve the catalytic activity of the enzyme while improving stability.
        Use the Rosetta scores to validate your design decisions quantitatively (more negative = more stable).
        
        RMSD interpretation guidelines:
        - RMSD < 2.0 Å: Very similar structures in aligned region (conservative mutations)
        - RMSD 2.0-5.0 Å: Moderate structural changes in aligned region (acceptable for stability improvements)
        - RMSD > 5.0 Å: Significant structural changes in aligned region (may affect function, use with caution)
        
        Sequence alignment interpretation:
        - Overlap percentage: Shows how much of the shorter sequence was aligned (higher = better coverage)
        - Sequence identity: Percentage of identical amino acids in aligned region (higher = more conservative changes)
        - High overlap (>80%) + High identity (>80%): Conservative design with good coverage
        - High overlap (>80%) + Medium identity (60-80%): Moderate mutations with good coverage
        - Low overlap (<60%): Limited alignment - RMSD may not represent overall similarity
        - Always examine aligned_ref_sequence and aligned_new_sequence to understand what was compared
        - The alignment uses sliding window to find the region of maximum sequence similarity
        

        Remember, protein sequences are 1-indexed, not 0-indexed.
        
        Start by conducting a thorough web search to understand the current state of carbonic anhydrase engineering. The rosetta score of your design must be lower than the wildtype score of ~ -305.
       
        You are not restricted to point mutations. You can propose larger insertions/deletions, or even entire domains. Feel free to fetch other sequences from uniprot and use them in your designs + modifications. 

        The modifications you propose MUST BE NOVEL!!!! 
        """
        
        # Save initial prompt to file
        with open(self.output_dir / "initial_prompt.txt", "w") as f:
            f.write(design_prompt)
        
        # Initialize conversation with system message and user prompt
        self.messages = [
            {
                "role": "user",
                "content": design_prompt
            }
        ]
        
        print("🚀 Starting Claude reasoning loop...")
        
        self.iteration_count = 1
        max_iterations = 200
        all_accumulated_text = ""
        
        while self.iteration_count <= max_iterations:
            print(f"\n{'='*20} ITERATION {self.iteration_count} {'='*20}")
            
            try:
                # Check and manage context before making API call
                self._check_and_manage_context()

                # Make API call to Claude with retry logic
                api_params = {
                    "model": self.model,
                    "max_tokens": self.max_tokens,
                    "tools": self.tools,
                    "messages": self.messages
                }
                
                # Add thinking parameter if enabled and budget is valid
                if self.enable_thinking and self.thinking_budget >= 1024 and self.thinking_budget < self.max_tokens:
                    api_params["thinking"] = {
                        "type": "enabled",
                        "budget_tokens": self.thinking_budget
                    }
                
                response = self._api_call_with_retry(**api_params)
                
                # Process the response
                is_complete, tool_results = self._process_response(response)
                
                # Add Claude's response to the conversation
                self.messages.append({
                    "role": "assistant",
                    "content": response.content
                })
                
                # If there are tool results, add them and continue
                if tool_results:
                    print(f"🔄 Adding {len(tool_results)} tool results and continuing...")
                    
                    # Add all tool results as a single user message
                    self.messages.append({
                        "role": "user",
                        "content": tool_results
                    })
                    
                    self.iteration_count += 1
                    continue
                else:
                    # No tool calls - check if we should prompt for continuation
                    print("🤔 No tool calls in this response.")
                    
                    # Check if the response seems to be concluding prematurely
                    last_text = ""
                    for content_block in response.content:
                        if content_block.type == "text":
                            last_text += content_block.text
                    
                    # If the response is very short or mentions final/conclusion, prompt to continue
                    print("📢 Prompting Claude to continue with more iterations...")
                        
                    continue_prompt = """
Continue with the next design iteration. Please proceed with:

1. Building the next mutant variant
2. Using the computational tools (fold_protein, calculate_rosetta_score, etc.)
3. Testing and evaluating the new design
4. Comparing results and planning further iterations

Remember to use the tools extensively and keep iterating until you have a design that meets all the stability goals.

HINT: You can search for modifications that have made other, similar proteins more stable. THIS IS WHAT YOU SHOULD DO IF THE PREVIOUS MODIFICATIONS DID NOT WORK.
"""
                        
                    self.messages.append({
                        "role": "user",
                        "content": continue_prompt
                    })
                    
                    self.iteration_count += 1
                    time.sleep(2)  # Reduced from 5 seconds since we now have retry logic
                    continue

                        
            except Exception as e:
                print(f"❌ ERROR IN ITERATION {self.iteration_count}: {e}")
                
                # Save error to file
                with open(self.output_dir / "errors.txt", "a") as f:
                    f.write(f"Iteration {self.iteration_count}: {e}\n")
                break
        
        # Collect all text from the conversation
        for message in self.messages:
            if message["role"] == "assistant":
                for content in message["content"]:
                    if hasattr(content, 'text'):
                        all_accumulated_text += content.text + "\n\n"
                    elif hasattr(content, 'thinking'):
                        all_accumulated_text += "=== CLAUDE'S THINKING ===\n"
                        all_accumulated_text += content.thinking + "\n"
                        all_accumulated_text += "=== END THINKING ===\n\n"
                    elif isinstance(content, dict) and content.get("type") == "text":
                        all_accumulated_text += content.get("text", "") + "\n\n"
                    elif isinstance(content, dict) and content.get("type") == "thinking":
                        all_accumulated_text += "=== CLAUDE'S THINKING ===\n"
                        all_accumulated_text += content.get("thinking", "") + "\n"
                        all_accumulated_text += "=== END THINKING ===\n\n"
        
        # Save final conversation to file
        final_output_file = self.output_dir / "final_output.txt"
        with open(final_output_file, "w") as f:
            f.write(all_accumulated_text)
        
        # Save conversation history
        conversation_file = self.output_dir / "conversation_history.json"
        with open(conversation_file, "w") as f:
            # Convert messages to serializable format
            serializable_messages = []
            for msg in self.messages:
                if msg["role"] == "assistant":
                    # Convert anthropic content blocks to dict format
                    content_list = []
                    for content in msg["content"]:
                        if hasattr(content, 'type'):
                            if content.type == "text":
                                content_list.append({"type": "text", "text": content.text})
                            elif content.type == "thinking":
                                content_list.append({
                                    "type": "thinking", 
                                    "thinking": content.thinking,
                                    "signature": getattr(content, 'signature', None)
                                })
                            elif content.type == "tool_use":
                                content_list.append({
                                    "type": "tool_use",
                                    "id": content.id,
                                    "name": content.name,
                                    "input": content.input
                                })
                        elif isinstance(content, dict):
                            content_list.append(content)
                    serializable_messages.append({
                        "role": msg["role"],
                        "content": content_list
                    })
                else:
                    serializable_messages.append(msg)
            
            json.dump(serializable_messages, f, indent=2)
        
        # Save final summary
        summary = f"""
CLAUDE DESIGN SESSION SUMMARY
=============================
Total iterations: {self.iteration_count}
Model used: {self.model}
Context limit: {self.context_limit:,} tokens
Context summarizations performed: {self.summarization_count}
Final output length: {len(all_accumulated_text)} characters
Output directory: {self.output_dir}

Files generated:
- final_output.txt: Complete session output with all text responses
- conversation_history.json: Full conversation including tool calls
- iteration_*.txt: Individual iteration outputs
- tool_calls.jsonl: Record of all tool calls and their results"""

        if self.summarization_count > 0:
            summary += f"""
- summarizations.jsonl: Log of context summarization events
- summary_*.txt: Generated conversation summaries
- pre_summary_*_conversation.txt: Pre-summarization conversation snapshots"""

        # Check if retry log exists
        retry_log_file = self.output_dir / "api_retries.jsonl"
        if retry_log_file.exists():
            summary += f"""
- api_retries.jsonl: Log of API retry attempts due to overload errors"""

        summary += f"""

Session completed after {self.iteration_count} iterations.
"""
        with open(self.output_dir / "session_summary.txt", "w") as f:
            f.write(summary)

        print(f"\n{'='*20} CLAUDE DESIGN SESSION COMPLETE {'='*20}")
        print(f"Total iterations: {self.iteration_count}")
        if self.summarization_count > 0:
            print(f"Context summarizations: {self.summarization_count}")
        
        # Check if there were API retries
        if retry_log_file.exists():
            try:
                retry_count = sum(1 for _ in open(retry_log_file))
                print(f"API retry attempts: {retry_count}")
            except:
                pass
        
        print(f"📁 All outputs saved to: {self.output_dir}")
        print(f"\n📁 Results saved to: {self.output_dir}/")
        print(f"   - final_output.txt")
        print(f"   - conversation_history.json")
        print(f"   - iteration_*.txt")
        print(f"   - tool_calls.jsonl")
        print(f"   - session_summary.txt")
        if self.summarization_count > 0:
            print(f"   - summarizations.jsonl")
            print(f"   - summary_*.txt ({self.summarization_count} files)")
            print(f"   - pre_summary_*_conversation.txt ({self.summarization_count} files)")
        if retry_log_file.exists():
            print(f"   - api_retries.jsonl")
        
        return all_accumulated_text


def main():
    """
    Main function to demonstrate the Claude-based carbonic anhydrase designer.
    """
    # Check for API keys
    if not os.getenv('ANTHROPIC_API_KEY'):
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("You can get an API key from https://console.anthropic.com/")
        return
    
    if not os.getenv('PERPLEXITY_API_KEY'):
        print("Warning: PERPLEXITY_API_KEY environment variable not set")
        print("Web search functionality will not be available")
        print("You can get an API key from https://docs.perplexity.ai/guides/getting-started")
    
    # Check if PyMOL is available
    try:
        import pymol
        print("✓ PyMOL is available for catalytic activity and secondary structure examination")
    except ImportError:
        print("Warning: PyMOL not available")
        print("Catalytic activity and secondary structure examination will not be available")
        print("Install with: conda install -c conda-forge pymol-open-source")
    
    # Create designer instance with Claude 3.7 (thinking model)
    designer = CarbonicAnhydraseDesignerClaude(
        model="claude-3-7-sonnet-20250219",  # Claude 3.7 with extended thinking support
        max_tokens=16384,  # Increased to accommodate thinking + response
        context_limit=40000,  # 180k tokens to stay under 200k limit
        enable_thinking=True,  # Enable extended reasoning display
        thinking_budget=4096  # Generous budget for deep protein engineering reasoning
    )
    
    # Example usage
    print(f"Starting Claude-based carbonic anhydrase design session...")
    print("=" * 80)
    
    # Automated design
    result = designer.design_stable_carbonic_anhydrase(
        target_pdb="1HEA",
        stability_goals=[
            "Increase thermal stability",
            "Improve stability in more acidic conditions", 
            "Maintain catalytic activity"
        ]
    )
    
    print("\n" + "=" * 80)
    print("🎯 CLAUDE DESIGN SESSION COMPLETE")
    print(f"📁 All outputs saved to: {designer.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main() 