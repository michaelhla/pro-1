#!/usr/bin/env python3
"""
Carbonic Anhydrase Designer using OpenAI's o3 model with function calling.

This module uses OpenAI's o3 reasoning model to design more stable carbonic anhydrase
variants by leveraging function calling capabilities to access various computational
tools and databases.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Callable, Any, Optional
from openai import OpenAI
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

class CarbonicAnhydraseDesigner:
    """
    A class that uses OpenAI's o3 model with function calling to design
    more stable carbonic anhydrase variants.
    """
    
    def __init__(self, api_key: Optional[str] = None, reasoning_effort: str = "high", 
                 output_dir: str = None):
        """
        Initialize the designer with OpenAI client and model configuration.
        
        Args:
            api_key: OpenAI API key (if None, will use OPENAI_API_KEY env var)
            reasoning_effort: Level of reasoning effort ("low", "medium", "high")
            output_dir: Directory to save outputs (if None, creates timestamped dir)
        """
        self.client = OpenAI(api_key=api_key)
        self.model_config = {
            "model": "o3",
            "reasoning": {
                "effort": reasoning_effort,
                "summary": "detailed"
            },
            "store": True,  # Required for reasoning items to persist between calls
            "max_output_tokens": 100000
        }
        
        # Set up output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"carbonic_anhydrase_design_{timestamp}"
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tools from the rest of the folder
        self.tools = self._initialize_tools()
        self.tool_mapping = self._create_tool_mapping()
        
        # Track session state
        self.iteration_count = 0
        
        print(f"🚀 Starting design session - outputs will be saved to: {self.output_dir}")
    
    def _initialize_tools(self) -> List[Dict[str, Any]]:
        """
        Initialize and return the available tools for carbonic anhydrase design.
        
        Currently only includes the protein folding tool for simplicity.
        """
        tools = [
            {
                "type": "function",
                "name": "fold_protein",
                "description": "Fold a protein sequence using ESMFold and save the structure as a PDB file. Use this when you need to predict the 3D structure of a protein from its amino acid sequence.",
                "parameters": {
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
                    "required": ["sequence", "filename"],
                    "additionalProperties": False
                },
                "strict": True
            },
            {
                "type": "function",
                "name": "calculate_rosetta_score",
                "description": "Calculate the Rosetta energy score for a protein structure from a PDB file. Lower scores indicate more stable structures. Use this to evaluate the stability of folded proteins.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pdb_file_path": {
                            "type": "string",
                            "description": "Path to the PDB file to score (e.g., 'predicted_structures/protein_123.pdb')"
                        }
                    },
                    "required": ["pdb_file_path"],
                    "additionalProperties": False
                },
                "strict": True
            },
            {
                "type": "function",
                "name": "calculate_rmsd_with_sequences",
                "description": "Calculate RMSD between the hardcoded reference structure (hCA2_folded.pdb) and a newly folded protein structure using sliding window sequence alignment. Finds the region of maximum sequence overlap, then calculates RMSD over the aligned core region. Returns RMSD value plus overlap percentage, sequence identity, and detailed alignment information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pdb_file2": {
                            "type": "string",
                            "description": "Path to the newly folded PDB file to compare against reference (e.g., 'predicted_structures/mutant.pdb')"
                        }
                    },
                    "required": ["pdb_file2"],
                    "additionalProperties": False
                },
                "strict": True
            },
            {
                "type": "function",
                "name": "websearch",
                "description": "Perform a web search using Perplexity's Sonar API to find current information about protein engineering, research papers, methodologies, and recent advances. Returns natural language responses with citations.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to execute (e.g., 'latest carbonic anhydrase stability research 2024', 'protein thermostability engineering methods')"
                        }
                    },
                    "required": ["query"],
                    "additionalProperties": False
                },
                "strict": True
            },
            {
                "type": "function",
                "name": "examine_catalytic_activity",
                "description": "Examine the catalytic activity sites of carbonic anhydrase using PyMOL visualization. Takes exact residue dictionaries specifying which residues to examine and their positions. Generates a single combined catalytic site image (returned as base64-encoded data) showing both active site and zinc binding residues, and assesses catalytic integrity to ensure modifications haven't affected enzyme activity. Images are saved to tools/images/{image_subdir}/.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pdb_file_path": {
                            "type": "string",
                            "description": "Path to the PDB file to examine (e.g., 'predicted_structures/mutant.pdb')"
                        },
                        "active_site_residues": {
                            "type": "object",
                            "description": "Dictionary of active site residues with format {'Y7': {'name': 'TYR', 'function': 'Proton transfer', 'number': 7}, ...}. For standard hCA II: Y7, N62, H64, N67, Q92",
                            "additionalProperties": True
                        },
                        "zinc_binding_residues": {
                            "type": "object",
                            "description": "Dictionary of zinc binding residues with format {'H94': {'name': 'HIS', 'function': 'Zinc coordination', 'number': 94}, ...}. For standard hCA II: H94, H96, H119",
                            "additionalProperties": True
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
                "type": "function",
                "name": "examine_secondary_structure",
                "description": "Examine secondary structure and calculate structural properties using PyMOL. Analyzes helix/sheet/loop content, calculates SASA (Solvent Accessible Surface Area), radius of gyration, and generates a colored secondary structure visualization (returned as base64-encoded data). Provides quality assessment and compactness analysis. Images are saved to tools/images/{image_subdir}/.",
                "parameters": {
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
                    "required": ["pdb_file_path", "image_subdir"],
                    "additionalProperties": False
                },
                "strict": True
            }
        ]
        
        # Add tools to model config
        if tools:
            self.model_config["tools"] = tools
            
        return tools
    
    def _create_tool_mapping(self) -> Dict[str, Callable]:
        """
        Create a mapping of tool names to their implementation functions.
        """
        return {
            "fold_protein": fold_protein,  # Real implementation from protein_folder.py
            "calculate_rosetta_score": calculate_rosetta_score,  # Real implementation from rosetta_scorer.py
            "calculate_rmsd_with_sequences": calculate_rmsd_with_alignment,  # Detailed RMSD with sequence alignment info
            "websearch": websearch,  # Real implementation from websearch_tool.py
            "examine_catalytic_activity": examine_catalytic_activity,  # Real implementation from catalytic_activity_examiner.py
            "examine_secondary_structure": examine_secondary_structure  # Real implementation from secondary_structure_examiner.py
        }
    
    def _execute_function_call(self, function_call) -> str:
        """
        Execute a function call and return the result.
        
        Args:
            function_call: Function call object from the API response
            
        Returns:
            String result from the function execution, with enhanced formatting for base64 images
        """
        function_name = function_call.name
        
        # Get the function from our mapping
        if function_name not in self.tool_mapping:
            error_msg = f"ERROR: Unknown function '{function_name}'"
            print(f"❌ {error_msg}")
            return error_msg
        
        try:
            # Parse arguments
            arguments = json.loads(function_call.arguments)
            
            print(f"🔧 Executing tool: {function_name}")
            
            # Execute the function
            result = self.tool_mapping[function_name](**arguments)
            
            print(f"✅ Tool {function_name} completed")
            
            # Log tool execution to file
            tool_log = {
                "timestamp": datetime.now().isoformat(),
                "iteration": self.iteration_count,
                "function_name": function_name,
                "arguments": arguments,
                "status": "success",
                "result_length": len(str(result))
            }
            
            with open(self.output_dir / "tool_calls.jsonl", "a") as f:
                f.write(json.dumps(tool_log) + "\n")
            
            # Simple feedback for visualization functions
            if function_name in ['examine_catalytic_activity', 'examine_secondary_structure']:
                print(f"🖼️  Generated visualizations for {function_name}")
            
            # Wait 30 seconds between tool calls
            print(f"⏳ Waiting 30 seconds before next tool call...")
            time.sleep(30)
            
            return str(result)
            
        except Exception as e:
            error_msg = f"ERROR executing {function_name}: {str(e)}"
            print(f"❌ {error_msg}")
            
            # Log error to file
            error_log = {
                "timestamp": datetime.now().isoformat(),
                "iteration": self.iteration_count,
                "function_name": function_name,
                "arguments": arguments if 'arguments' in locals() else None,
                "status": "error",
                "error": str(e)
            }
            
            with open(self.output_dir / "tool_calls.jsonl", "a") as f:
                f.write(json.dumps(error_log) + "\n")
            
            return error_msg
    
    def _save_reasoning_data(self, response, iteration: int) -> None:
        """
        Save reasoning summaries and items to files.
        
        Args:
            response: Response from OpenAI API
            iteration: Current iteration number
        """
        # Create a data structure to hold all reasoning information
        reasoning_data = {
            "timestamp": datetime.now().isoformat(),
            "iteration": iteration,
            "summaries": [],
            "items": [],
            "messages": []
        }
        
        # Collect reasoning summaries
        if hasattr(response, 'reasoning') and response.reasoning and response.reasoning.summary:
            for summary in response.reasoning.summary:
                if hasattr(summary, 'text'):
                    reasoning_data["summaries"].append(summary.text)
        
        # Collect reasoning items
        reasoning_items = [item for item in response.output if item.type == 'reasoning']
        for item in reasoning_items:
            item_data = {
                "id": item.id,
                "summaries": []
            }
            if hasattr(item, 'summary') and item.summary:
                for summary in item.summary:
                    if hasattr(summary, 'text'):
                        item_data["summaries"].append(summary.text)
            reasoning_data["items"].append(item_data)
        
        # Collect message content
        message_items = [item for item in response.output if item.type == 'message']
        for item in message_items:
            if hasattr(item, 'content') and item.content:
                for content_item in item.content:
                    if hasattr(content_item, 'text'):
                        reasoning_data["messages"].append(content_item.text)
        
        # Save to JSONL file
        with open(self.output_dir / "reasoning_data.jsonl", "a") as f:
            f.write(json.dumps(reasoning_data) + "\n")

    def _invoke_functions_from_response(self, response) -> List[Dict[str, Any]]:
        """
        Extract all function calls from the response and execute them.
        This follows the pattern from the OpenAI cookbook.
        
        Args:
            response: Response from OpenAI API
            
        Returns:
            List of function call output messages to send back to the API
        """
        function_responses = []
        
        # Save reasoning data
        self._save_reasoning_data(response, self.iteration_count)
        
        # First, let's show any reasoning summaries
        if hasattr(response, 'reasoning') and response.reasoning and response.reasoning.summary:
            print("🧠 REASONING SUMMARY:")
            for summary in response.reasoning.summary:
                if hasattr(summary, 'text'):
                    print(f"   {summary.text}")
            print()
        
        # Show reasoning items from output if available
        reasoning_items = [item for item in response.output if item.type == 'reasoning']
        if reasoning_items:
            print("🔍 REASONING ITEMS:")
            for i, item in enumerate(reasoning_items):
                print(f"   Reasoning {i+1}: {item.id}")
                if hasattr(item, 'summary') and item.summary:
                    for summary in item.summary:
                        if hasattr(summary, 'text'):
                            print(f"      Summary: {summary.text}")
            print()
        
        # Show any message content before function calls
        message_items = [item for item in response.output if item.type == 'message']
        if message_items:
            print("💬 MODEL MESSAGES:")
            for i, item in enumerate(message_items):
                if hasattr(item, 'content') and item.content:
                    for content_item in item.content:
                        if hasattr(content_item, 'text'):
                            print(f"   Message {i+1}: {content_item.text}")
            print()
        
        # Process function calls
        function_calls = [item for item in response.output if item.type == 'function_call']
        if function_calls:
            print(f"🔧 FUNCTION CALLS ({len(function_calls)} total):")
            print("=" * 80)
        
        for i, response_item in enumerate(function_calls):
            try:
                print(f"📞 FUNCTION CALL #{i+1}")
                print(f"   Function: {response_item.name}")
                print(f"   Call ID: {response_item.call_id}")
                
                # Parse and display arguments
                import json
                try:
                    arguments = json.loads(response_item.arguments)
                    print(f"   Arguments:")
                    for key, value in arguments.items():
                        # Truncate very long values for readability
                        if isinstance(value, str) and len(value) > 200:
                            display_value = value[:200] + "..."
                        else:
                            display_value = value
                        print(f"      {key}: {display_value}")
                except json.JSONDecodeError:
                    print(f"   Arguments (raw): {response_item.arguments}")
                
                print(f"   ⚡ Executing function...")
                
                # Execute the function and time it
                import time
                start_time = time.time()
                result = self._execute_function_call(response_item)
                execution_time = time.time() - start_time
                
                print(f"   ✅ Function completed in {execution_time:.2f}s")
                
                print(f"   Result: {result}")

                
                function_responses.append({
                    "type": "function_call_output",
                    "call_id": response_item.call_id,
                    "output": str(result)
                })
                
                print("   " + "─" * 60)
                
            except Exception as e:
                error_msg = f"Error executing {response_item.name}: {str(e)}"
                print(f"   ❌ ERROR: {error_msg}")
                
                function_responses.append({
                    "type": "function_call_output",
                    "call_id": response_item.call_id,
                    "output": error_msg
                })
                
                print("   " + "─" * 60)
        
        if function_calls:
            print("=" * 80)
            print(f"📤 Returning {len(function_responses)} function responses to model")
            print()
        
        return function_responses

    # def _process_response(self, response) -> tuple[bool, List[Dict[str, Any]], str]:
    #     """
    #     DEPRECATED: Use _invoke_functions_from_response instead.
        
    #     Process API response and execute any function calls.
        
    #     Args:
    #         response: Response from OpenAI API
            
    #     Returns:
    #         (is_complete, function_responses, accumulated_text): 
    #             - is_complete: True if reasoning is complete, False if more calls needed
    #             - function_responses: List of function call responses to send back
    #             - accumulated_text: All text content from the response
    #     """
    #     function_responses = []
    #     has_function_calls = False
    #     accumulated_text = ""
        
    #     try:
    #         print("\nProcessing model response...")
            
    #         # Print everything that's NOT tools
    #         print("=" * 60)
    #         print("📋 FULL RESPONSE (non-tool content):")
    #         print("=" * 60)
            
    #         # Check if response has reasoning summary
    #         if hasattr(response, 'reasoning') and response.reasoning:
    #             print(f"\n💭 Reasoning Summary:")
    #             for summary in response.reasoning.summary:
    #                 if hasattr(summary, 'text'):
    #                     reasoning_text = summary.text
    #                     print(reasoning_text)
    #                     accumulated_text += f"[REASONING_SUMMARY]\n{reasoning_text}\n\n"
            
    #         # Check if response has output
    #         if hasattr(response, 'output') and response.output:
    #             print(f"\n📤 Output Items:")
    #             for i, item in enumerate(response.output):
    #                 if item.type == 'function_call':
    #                     print(f"  Item {i}: [TOOL CALL - {item.name}]")
                        
    #                     has_function_calls = True
    #                     result = self._execute_function_call(item)
                        
    #                     # Use the correct format from OpenAI cookbook
    #                     function_response = {
    #                         "type": "function_call_output",
    #                         "call_id": item.call_id,
    #                         "output": result
    #                     }
                        
    #                     print(f"    📤 Function response: call_id={item.call_id}, output_length={len(str(result))}")
    #                     function_responses.append(function_response)
                    
    #                 elif item.type == 'message':
    #                     print(f"  Item {i}: MESSAGE")
    #                     accumulated_text += f"[MESSAGE]\n"
                        
    #                     if hasattr(item, 'content') and item.content:
    #                         for content_item in item.content:
    #                             if hasattr(content_item, 'text'):
    #                                 text_content = content_item.text
    #                                 print(f"    Text: {text_content}")
    #                                 accumulated_text += text_content + "\n"
                    
    #                 elif item.type == 'reasoning':
    #                     print(f"  Item {i}: REASONING")
    #                     # Reasoning items are handled automatically by previous_response_id
                        
    #                 else:
    #                     # Print any other types of output items
    #                     print(f"  Item {i}: {item.type}")
            
    #         print("=" * 60)
            
    #         # Save this iteration's text to a separate file
    #         if accumulated_text.strip():
    #             iteration_file = self.output_dir / f"iteration_{self.iteration_count}_output.txt"
    #             with open(iteration_file, "w") as f:
    #                 f.write(accumulated_text)
        
    #     except Exception as e:
    #         print(f"❌ Error processing response: {e}")
        
    #     return not has_function_calls, function_responses, accumulated_text
    

    
    def design_stable_carbonic_anhydrase(self, target_pdb: str = "1CA2", 
                                       stability_goals: List[str] = None) -> str:
        """
        Design a more stable carbonic anhydrase variant.
        
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
        print("CARBONIC ANHYDRASE STABILITY DESIGN SESSION")
        print("=" * 80)
        print(f"Target: {target_pdb}")
        print(f"Goals: {', '.join(stability_goals)}")
        print("=" * 80)
        
        # Create the initial prompt for o3
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
        5. examine_catalytic_activity: Visualizes and examines catalytic sites by taking exact residue dictionaries specifying which residues to examine. Generates base64-encoded images you can analyze visually to ensure modifications haven't affected enzyme activity. (do not be alarmed if the tool returns poor catalytic integrity, make your own judgement based on the images and the key residues)
        6. examine_secondary_structure: Analyzes secondary structure content, calculates SASA and structural properties, and provides quality assessment with base64-encoded visualizations you can examine
        
        IMPORTANT: You MUST use these tools multiple times throughout the design process. Do not stop after a single function call. Keep using tools until you have completed the entire design workflow.

        Please approach this systematically:
        1. Use websearch to find current research on carbonic anhydrase stability and recent engineering approaches. Go deep into the literature and Uniprot. THIS SHOULD BE VERY THOROUGH BEFORE CONTINUING TO THE NEXT STEPS. THIS IS A FUNDAMENTALLY IMPORTANT STEP.
        2. Using the findings from your research, propose specific amino acid mutations that could improve stability. These can be simple point mutations, or larger insertions/deletions.
        3. Then modify the original sequence using the mutations you proposed. Make sure you have applied your mutations correctly. 
        4. Use the fold_protein tool to predict the structure of your modified sequence. This will return a filepath to your pdb. 
        5. Use calculate_rosetta_score to get the stability score of your modified sequence.
        7. Use examine_secondary_structure to analyze the new structure's fold, SASA, and structural quality
        8. Use examine_catalytic_activity to verify the new structure's catalytic sites are intact. You need to specify the exact residue numbers and types in the active_site_residues and zinc_binding_residues dictionaries based on your mutant sequence. This includes the ZINC BINDING RESIDUES.
        CRITICAL: Use examine_catalytic_activity on each mutant to ensure catalytic residues are preserved. You must provide the correct residue dictionaries with exact numbers and amino acid types for your specific mutant sequence. This includes the ZINC BINDING RESIDUES.
        9. Compare scores and provide recommendations with quantitative rationale. 
        10. REPEAT THE PROCESS UNTIL YOU HAVE A DESIGN THAT YOU BELIEVE MEETS THE GOALS. DO NOT STOP UNTIL YOU HAVE A DESIGN THAT YOU BELIEVE MEETS THE GOALS.
        
        CRITICAL INSTRUCTION: Do NOT provide a final answer or summary until you have completed ALL the above steps with multiple mutations tested. You must call multiple tools in sequence to fully complete this task. If you find yourself wanting to give a final answer without having used all the tools extensively, instead continue with more tool calls.

        Focus on common protein stabilization strategies:
        - Reducing surface loops and increasing rigidity
        - Improving hydrophobic core packing
        - Adding favorable electrostatic interactions
        - Removing destabilizing residues
        - Increasing secondary structure propensity

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
        
        Web search usage guidelines:
        - Use websearch to find current research on specific stability engineering strategies
        - Search for recent papers on carbonic anhydrase modifications and their effects
        - Look up proven mutation strategies for thermostability improvements
        - Find information about specific amino acid substitutions and their structural effects
        - Search for validation methods and experimental approaches used in similar studies
        
        Catalytic activity examination guidelines:
        - Always use examine_catalytic_activity to verify catalytic integrity before and after mutations
        - For standard hCA II, specify these residues in your dictionaries:
          * Active site residues: Y7, N62, H64, N67, Q92 (proton transfer and activator binding)
          * Zinc binding residues: H94, H96, H119 (essential for catalytic activity)
        - You MUST provide exact residue dictionaries with the correct residue numbers for your specific sequence
        - If you've made insertions/deletions, calculate the new positions FOR ALL OF THE CATALYTIC RESIDUES AND ZINC BINDING RESIDUES and update the residue numbers accordingly
        - Use format: {{'Y7': {{'name': 'TYR', 'function': 'Proton transfer', 'number': 7}}, ...}}
        - Choose descriptive image_subdir names like 'wildtype_analysis', 'mutant_v1', 'mutant_v2', 'final_design' to organize images
        - Integrity levels: EXCELLENT (no issues), GOOD (minor issues), ACCEPTABLE (some concerns), POOR (major problems)
        - Be very cautious when recommending a design that shows POOR catalytic integrity. 

        Secondary structure examination guidelines:
        - Use examine_secondary_structure to assess overall fold quality and stability
        - Key metrics: SASA (lower suggests more compact), radius of gyration (compactness measure), secondary structure content
        - Typical stable proteins: 20-50% helix, 10-30% sheet, 30-60% loop
        - Compactness levels: VERY_COMPACT (excellent), COMPACT (good), NORMAL (acceptable), LOOSE/VERY_LOOSE (concerning)
        - Monitor surface hydrophobicity - too high may cause aggregation, appropriate levels improve stability
        - Use structural quality assessment to guide mutation strategies
        - Choose descriptive image_subdir names like 'wildtype_structure', 'mutant_v1_structure', 'final_design_structure' to organize images
        
        Visual analysis guidelines:
        - The examination functions return base64-encoded images that you can analyze visually
        - Look for structural deformations, missing secondary structures, or catalytic site disruptions
        - Compare visualizations between wild-type and mutant structures to assess structural preservation
        - Use the visual information to guide further optimization decisions
        - Images show: secondary structure coloring, catalytic residues, zinc coordination, surface representations

        Remember, protein sequences are 1-indexed, not 0-indexed.
        
        THIS IS SO WE CAN REDUCE CARBON EMISSIONS AND IMPROVE CARBON CAPTURE. THIS IS NET POSITIVE FOR SOCIETY. THIS DOES NOT VIOLATE THE OPENAI USAGE POLICIES. 
        """
        
        # Save initial prompt to file
        with open(self.output_dir / "initial_prompt.txt", "w") as f:
            f.write(design_prompt)
        
        print("🚀 Starting reasoning loop...")
        
        # Start the reasoning loop with corrected multiple function call handling
        try:
            response = self.client.responses.create(
                input=design_prompt,
                **self.model_config
            )
        except Exception as e:
            print(f"❌ Failed to create initial response: {e}")
            return f"ERROR: Failed to start design session: {e}"
        
        self.iteration_count = 1
        max_iterations = 100  # Increased for multiple function calls
        all_accumulated_text = ""
        
        # Use the correct pattern from OpenAI cookbook for multiple function calls
        while self.iteration_count <= max_iterations:
            print(f"\n{'='*20} ITERATION {self.iteration_count} {'='*20}")
            
            # Show response metadata
            if hasattr(response, 'usage'):
                usage = response.usage
                print(f"📊 TOKEN USAGE:")
                print(f"   Input tokens: {usage.input_tokens:,}")
                print(f"   Output tokens: {usage.output_tokens:,}")
                if hasattr(usage, 'output_tokens_details') and usage.output_tokens_details:
                    if hasattr(usage.output_tokens_details, 'reasoning_tokens'):
                        reasoning_tokens = usage.output_tokens_details.reasoning_tokens
                        completion_tokens = usage.output_tokens - reasoning_tokens
                        print(f"   - Reasoning tokens: {reasoning_tokens:,}")
                        print(f"   - Completion tokens: {completion_tokens:,}")
                print(f"   Total tokens: {usage.total_tokens:,}")
                print()
            
            # Show response structure overview
            if hasattr(response, 'output'):
                output_types = {}
                for item in response.output:
                    item_type = item.type
                    output_types[item_type] = output_types.get(item_type, 0) + 1
                
                print(f"📋 RESPONSE STRUCTURE:")
                for item_type, count in output_types.items():
                    print(f"   {item_type}: {count}")
                print()

            # Process function calls and collect responses
            function_responses = []
            has_function_calls = False
            accumulated_text = ""

            try:
                # Print everything that's NOT tools
                print("=" * 60)
                print("📋 FULL RESPONSE (non-tool content):")
                print("=" * 60)
                
                # Check if response has reasoning summary
                if hasattr(response, 'reasoning') and response.reasoning:
                    print(f"\n💭 Reasoning Summary:")
                    for summary in response.reasoning.summary:
                        if hasattr(summary, 'text'):
                            reasoning_text = summary.text
                            print(reasoning_text)
                            accumulated_text += f"[REASONING_SUMMARY]\n{reasoning_text}\n\n"
                
                # Check if response has output
                if hasattr(response, 'output') and response.output:
                    print(f"\n📤 Output Items:")
                    for i, item in enumerate(response.output):
                        if item.type == 'function_call':
                            print(f"  Item {i}: [TOOL CALL - {item.name}]")
                            print(f"    Arguments: {item.arguments}")
                            
                            has_function_calls = True
                            result = self._execute_function_call(item)
                            
                            # Use the correct format from OpenAI cookbook
                            function_response = {
                                "type": "function_call_output",
                                "call_id": item.call_id,
                                "output": result
                            }
                            
                            print(f"    📤 Function response: call_id={item.call_id}, output_length={len(str(result))}")
                            function_responses.append(function_response)
                        
                        elif item.type == 'message':
                            print(f"  Item {i}: MESSAGE")
                            accumulated_text += f"[MESSAGE]\n"
                            
                            if hasattr(item, 'content') and item.content:
                                for content_item in item.content:
                                    if hasattr(content_item, 'text'):
                                        text_content = content_item.text
                                        print(f"    Text: {text_content}")
                                        accumulated_text += text_content + "\n"
                        
                        elif item.type == 'reasoning':
                            print(f"  Item {i}: REASONING")
                            # Reasoning items are handled automatically by previous_response_id
                            
                        else:
                            # Print any other types of output items
                            print(f"  Item {i}: {item.type}")
                
                print("=" * 60)
                
                # Save this iteration's text to a separate file
                if accumulated_text.strip():
                    iteration_file = self.output_dir / f"iteration_{self.iteration_count}_output.txt"
                    with open(iteration_file, "w") as f:
                        f.write(accumulated_text)
                    all_accumulated_text += f"\n\n=== ITERATION {self.iteration_count} ===\n\n" + accumulated_text
                
                # Save reasoning data
                try:
                    reasoning_data = {
                        "iteration": self.iteration_count,
                        "summaries": [],
                        "items": [],
                        "messages": []
                    }
                    
                    if hasattr(response, 'reasoning') and response.reasoning:
                        for summary in response.reasoning.summary:
                            if hasattr(summary, 'text'):
                                reasoning_data["summaries"].append(summary.text)
                    
                    if hasattr(response, 'output'):
                        for item in response.output:
                            if item.type == 'reasoning':
                                item_data = {
                                    "id": item.id if hasattr(item, 'id') else "unknown",
                                    "summaries": []
                                }
                                if hasattr(item, 'summary'):
                                    for summary in item.summary:
                                        if hasattr(summary, 'text'):
                                            item_data["summaries"].append(summary.text)
                                reasoning_data["items"].append(item_data)
                            elif item.type == 'message':
                                if hasattr(item, 'content'):
                                    for content in item.content:
                                        if hasattr(content, 'text'):
                                            reasoning_data["messages"].append(content.text)
                    
                    with open(self.output_dir / "reasoning_data.jsonl", "a") as f:
                        f.write(json.dumps(reasoning_data) + "\n")
                        
                except Exception as e:
                    print(f"Warning: Could not save reasoning data: {e}")
                
                # If there are function responses, continue the conversation
                if function_responses:
                    try:
                        print(f"🔄 CONTINUING TO ITERATION {self.iteration_count + 1}")
                        print(f"   Sending {len(function_responses)} function responses back to model...")
                        
                        # Show what we're sending back (summary)
                        print(f"📤 FUNCTION RESPONSES BEING SENT:")
                        for i, func_resp in enumerate(function_responses):
                            call_id = func_resp.get('call_id', 'unknown')
                            output_len = len(str(func_resp.get('output', '')))
                            print(f"   Response {i+1}: call_id={call_id}, output_length={output_len}")
                        print()
                        
                        # Continue the conversation with function results
                        response = self.client.responses.create(
                            input=function_responses,
                            previous_response_id=response.id,
                            **self.model_config
                        )
                        
                        self.iteration_count += 1
                        continue
                    except Exception as e:
                        print(f"❌ Failed to continue conversation: {e}")
                        break
                else:
                    # No function calls - check if we should continue or if this is truly final
                    print(f"🔄 NO FUNCTION CALLS - CONTINUING TO ITERATION {self.iteration_count + 1}")
                    print("   Prompting model to continue with next design iteration...")
                    
                    # Continue prompting the model to keep working
                    continue_prompt = """
Continue with the next design iteration as planned. Please proceed with:

1. Building the next mutant variant as you outlined
2. Using the computational tools (fold_protein, calculate_rosetta_score, etc.)
3. Testing and evaluating the new design
4. Comparing results and planning further iterations if needed

Remember to use the tools extensively and keep iterating until you have a design that meets all the stability goals. Do not provide a final summary until you have completed multiple design cycles and thoroughly tested your variants.
"""
                    
                    try:
                        # Continue the conversation with a prompt to keep going
                        response = self.client.responses.create(
                            input=continue_prompt,
                            previous_response_id=response.id,
                            **self.model_config
                        )
                        
                        self.iteration_count += 1
                        continue
                    except Exception as e:
                        print(f"❌ Failed to continue conversation: {e}")
                        break

            except Exception as e:
                print(f"❌ ERROR IN ITERATION {self.iteration_count}: {e}")
                
                # Save error to file
                with open(self.output_dir / "errors.txt", "a") as f:
                    f.write(f"Iteration {self.iteration_count}: {e}\n")
                break

        # Save final output when loop completes
        final_output_file = self.output_dir / "final_output.txt"
        with open(final_output_file, "w") as f:
            f.write(all_accumulated_text)
        
        # Save final summary
        summary = f"""
DESIGN SESSION SUMMARY
======================
Total iterations: {self.iteration_count}
Final output length: {len(all_accumulated_text)} characters
Output directory: {self.output_dir}

Files generated:
- final_output.txt: Complete session output with all iterations
- iteration_*.txt: Individual iteration outputs
- reasoning_data.jsonl: Raw reasoning data in JSONL format
- tool_calls.jsonl: Record of all tool calls and their results

Session completed after {self.iteration_count} iterations.
"""
        with open(self.output_dir / "session_summary.txt", "w") as f:
            f.write(summary)

        print(f"\n{'='*20} DESIGN SESSION COMPLETE {'='*20}")
        print(f"Total iterations: {self.iteration_count}")
        print(f"📁 All outputs saved to: {self.output_dir}")
        print(f"\n📁 Results saved to: {self.output_dir}/")
        print(f"   - final_output.txt")
        print(f"   - iteration_*.txt")
        print(f"   - reasoning_data.jsonl")
        print(f"   - tool_calls.jsonl")
        print(f"   - session_summary.txt")
        
        return all_accumulated_text


def main():
    """
    Main function to demonstrate the carbonic anhydrase designer.
    """
    # Check for API keys
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY environment variable not set")
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
    
    # Create designer instance
    designer = CarbonicAnhydraseDesigner(reasoning_effort="medium")
    
    # Example usage
    print(f"Starting carbonic anhydrase design session...")
    print("=" * 80)
    
    # Option 1: Automated design
    result = designer.design_stable_carbonic_anhydrase(
        target_pdb="1HEA",
        stability_goals=[
            "Increase thermal stability",
            "Improve stability in more acidic conditions", 
            "Maintain catalytic activity"
        ]
    )
    
    print("\n" + "=" * 80)
    print("🎯 DESIGN SESSION COMPLETE")
    print(f"📁 All outputs saved to: {designer.output_dir}")
    print("=" * 80)



if __name__ == "__main__":
    main() 