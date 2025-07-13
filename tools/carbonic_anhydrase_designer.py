#!/usr/bin/env python3
"""
Carbonic Anhydrase Designer using OpenAI's o3 model with function calling.

This module uses OpenAI's o3 reasoning model to design more stable carbonic anhydrase
variants by leveraging function calling capabilities to access various computational
tools and databases.
"""

import json
import os
from typing import Dict, List, Callable, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# Import the tools
from protein_folder import fold_protein
from rosetta_scorer import calculate_rosetta_score
from rmsd_calculator import calculate_rmsd, calculate_rmsd_with_sequences
from websearch_tool import websearch
from catalytic_activity_examiner import examine_catalytic_activity
from secondary_structure_examiner import examine_secondary_structure


class CarbonicAnhydraseDesigner:
    """
    A class that uses OpenAI's o3 model with function calling to design
    more stable carbonic anhydrase variants.
    """
    
    def __init__(self, api_key: Optional[str] = None, reasoning_effort: str = "medium"):
        """
        Initialize the designer with OpenAI client and model configuration.
        
        Args:
            api_key: OpenAI API key (if None, will use OPENAI_API_KEY env var)
            reasoning_effort: Level of reasoning effort ("low", "medium", "high")
        """
        self.client = OpenAI(api_key=api_key)
        self.model_config = {
            "model": "o3",
            "reasoning": {
                "effort": reasoning_effort,
                "summary": "auto"
            },
            "store": False,
            "include": ["reasoning.encrypted_content"]  # Preserve reasoning between calls
        }
        
        # Initialize tools from the rest of the folder
        self.tools = self._initialize_tools()
        self.tool_mapping = self._create_tool_mapping()
        
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
                        "protein_id": {
                            "type": "string",
                            "description": "Optional identifier for the protein (will be auto-generated if not provided)"
                        }
                    },
                    "required": ["sequence"],
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
                "name": "calculate_rmsd",
                "description": "Calculate the Root Mean Square Deviation (RMSD) between two protein structures from PDB files. Returns the structural similarity score in Angstroms - lower values indicate more similar structures. Handles proteins of different lengths through structural alignment.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pdb_file1": {
                            "type": "string",
                            "description": "Path to the first PDB file (e.g., 'predicted_structures/original.pdb')"
                        },
                        "pdb_file2": {
                            "type": "string",
                            "description": "Path to the second PDB file (e.g., 'predicted_structures/mutant.pdb')"
                        },
                        "chain_id1": {
                            "type": "string",
                            "description": "Chain ID for first structure (optional, auto-detected if not provided)"
                        },
                        "chain_id2": {
                            "type": "string",
                            "description": "Chain ID for second structure (optional, auto-detected if not provided)"
                        },
                        "alignment_method": {
                            "type": "string",
                            "description": "Method for handling different lengths: 'structural' (default) or 'sequence'",
                            "enum": ["structural", "sequence"]
                        }
                    },
                    "required": ["pdb_file1", "pdb_file2"],
                    "additionalProperties": False
                },
                "strict": True
            },
            {
                "type": "function",
                "name": "calculate_rmsd_with_sequences",
                "description": "Calculate RMSD between two PDB structures with detailed sequence alignment information. Returns the RMSD score plus the actual amino acid subsequences that were aligned, alignment positions, and coverage statistics. This helps interpret the RMSD score by showing exactly which parts of the proteins were compared.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pdb_file1": {
                            "type": "string",
                            "description": "Path to the first PDB file (e.g., 'predicted_structures/original.pdb')"
                        },
                        "pdb_file2": {
                            "type": "string",
                            "description": "Path to the second PDB file (e.g., 'predicted_structures/mutant.pdb')"
                        },
                        "chain_id1": {
                            "type": "string",
                            "description": "Chain ID for first structure (optional, auto-detected if not provided)"
                        },
                        "chain_id2": {
                            "type": "string",
                            "description": "Chain ID for second structure (optional, auto-detected if not provided)"
                        },
                        "alignment_method": {
                            "type": "string",
                            "description": "Method for handling different lengths: 'structural' (default) or 'sequence'",
                            "enum": ["structural", "sequence"]
                        }
                    },
                    "required": ["pdb_file1", "pdb_file2"],
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
                        },
                        "model": {
                            "type": "string",
                            "description": "The Perplexity model to use (default: 'sonar-pro')",
                            "enum": ["sonar-pro", "sonar"]
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
                "description": "Examine the catalytic activity sites of carbonic anhydrase II using PyMOL visualization. Checks active site residues (Y7, N62, H64, N67, Q92) and zinc binding residues (H94, H96, H119). Generates labeled images and assesses catalytic integrity to ensure modifications haven't affected enzyme activity.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pdb_file_path": {
                            "type": "string",
                            "description": "Path to the PDB file to examine (e.g., 'predicted_structures/mutant.pdb')"
                        },
                        "residue_offsets": {
                            "type": "object",
                            "description": "Dictionary mapping residue names to offset values to account for insertions/deletions (e.g., {'H94': 2, 'H96': 2} if 2 residues were inserted before these positions)",
                            "additionalProperties": {
                                "type": "integer"
                            }
                        },
                        "chain_id": {
                            "type": "string",
                            "description": "Chain identifier for the protein (default: 'A')"
                        },
                        "output_dir": {
                            "type": "string",
                            "description": "Directory to save visualization images (optional)"
                        }
                    },
                    "required": ["pdb_file_path"],
                    "additionalProperties": False
                },
                "strict": True
            },
            {
                "type": "function",
                "name": "examine_secondary_structure",
                "description": "Examine secondary structure and calculate structural properties using PyMOL. Analyzes helix/sheet/loop content, calculates SASA (Solvent Accessible Surface Area), radius of gyration, and generates a colored secondary structure visualization. Provides quality assessment and compactness analysis.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pdb_file_path": {
                            "type": "string",
                            "description": "Path to the PDB file to examine (e.g., 'predicted_structures/mutant.pdb')"
                        },
                        "chain_id": {
                            "type": "string",
                            "description": "Chain identifier for the protein (default: 'A')"
                        },
                        "output_dir": {
                            "type": "string",
                            "description": "Directory to save visualization images (optional)"
                        }
                    },
                    "required": ["pdb_file_path"],
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
            "calculate_rmsd": calculate_rmsd,  # Real implementation from rmsd_calculator.py
            "calculate_rmsd_with_sequences": calculate_rmsd_with_sequences,  # Detailed RMSD with sequence alignment info
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
            String result from the function execution
        """
        function_name = function_call.name
        
        # Get the function from our mapping
        if function_name not in self.tool_mapping:
            return f"ERROR: Unknown function '{function_name}'"
        
        try:
            # Parse arguments
            arguments = json.loads(function_call.arguments)
            
            # Execute the function
            result = self.tool_mapping[function_name](**arguments)
            
            print(f"Executed {function_name}({arguments}) -> {result}")
            return str(result)
            
        except Exception as e:
            error_msg = f"ERROR executing {function_name}: {str(e)}"
            print(error_msg)
            return error_msg
    
    def _process_response(self, response) -> tuple[bool, List[Dict[str, Any]]]:
        """
        Process API response and execute any function calls.
        
        Returns:
            (is_complete, function_responses): 
                - is_complete: True if reasoning is complete, False if more calls needed
                - function_responses: List of function call responses to send back
        """
        function_responses = []
        has_function_calls = False
        
        for item in response.output:
            if item.type == 'function_call':
                has_function_calls = True
                result = self._execute_function_call(item)
                function_responses.append({
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": result
                })
            elif item.type == 'reasoning':
                # Print reasoning summary if available
                if hasattr(item, 'summary') and item.summary:
                    for summary in item.summary:
                        if hasattr(summary, 'text'):
                            print(f"Reasoning: {summary.text}")
        
        return not has_function_calls, function_responses
    
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
        
        # Create the initial prompt for o3
        design_prompt = f"""
        You are an expert protein engineer tasked with designing a more stable variant of carbonic anhydrase.

        Target protein: {target_pdb}
        Stability goals: {', '.join(stability_goals)}

        You have access to seven computational tools:
        1. fold_protein: Predicts 3D structures from amino acid sequences using ESMFold
        2. calculate_rosetta_score: Calculates Rosetta energy scores for PDB structures (lower = more stable)
        3. calculate_rmsd: Calculates structural similarity between two PDB files (lower = more similar)
        4. calculate_rmsd_with_sequences: Like calculate_rmsd but also returns the actual amino acid subsequences that were aligned and detailed alignment statistics
        5. websearch: Searches the web for current information about protein engineering, research papers, and methodologies
        6. examine_catalytic_activity: Visualizes and examines catalytic sites (active site and zinc binding residues) to ensure modifications haven't affected enzyme activity
        7. examine_secondary_structure: Analyzes secondary structure content, calculates SASA and structural properties, and provides quality assessment

        Please approach this systematically:
        1. If given a PDB ID, first provide the corresponding amino acid sequence so you can fold it
        2. Use websearch to find current research on carbonic anhydrase stability and recent engineering approaches
        3. Use the fold_protein tool to predict the structure of the original sequence
        4. Use calculate_rosetta_score to get the baseline stability score
        5. Use examine_secondary_structure to analyze the original structure's fold, SASA, and structural quality
        6. Use examine_catalytic_activity to verify the original structure's catalytic sites are intact
        7. Based on your analysis and current research, propose specific amino acid mutations that could improve stability
        8. For promising mutations, modify the sequence and fold the new variants
        9. Score the new variants with calculate_rosetta_score to quantify stability improvements
        10. Use examine_secondary_structure on each mutant to assess structural changes and compactness
        11. Use calculate_rmsd_with_sequences to compare structural similarity and see exactly which subsequences were aligned
        12. CRITICAL: Use examine_catalytic_activity on each mutant to ensure catalytic residues are preserved (account for any sequence length changes with residue_offsets)
        13. Compare scores and provide final recommendations with quantitative rationale

        Focus on common protein stabilization strategies:
        - Reducing surface loops and increasing rigidity
        - Improving hydrophobic core packing
        - Adding favorable electrostatic interactions
        - Removing destabilizing residues
        - Increasing secondary structure propensity

        Always preserve the catalytic activity of the enzyme while improving stability.
        Use the Rosetta scores to validate your design decisions quantitatively.
        
        RMSD interpretation guidelines:
        - RMSD < 2.0 Å: Very similar structures (conservative mutations)
        - RMSD 2.0-5.0 Å: Moderate structural changes (acceptable for stability improvements)
        - RMSD > 5.0 Å: Significant structural changes (may affect function, use with caution)
        
        Sequence alignment interpretation:
        - High coverage (>80%): Most of the protein structure was aligned - RMSD represents global similarity
        - Medium coverage (50-80%): Partial alignment - RMSD represents similarity of the aligned region only
        - Low coverage (<50%): Limited alignment - RMSD may not be representative of overall structural similarity
        - Always examine the aligned_sequence1 and aligned_sequence2 to understand what was actually compared
        
        Web search usage guidelines:
        - Use websearch to find current research on specific stability engineering strategies
        - Search for recent papers on carbonic anhydrase modifications and their effects
        - Look up proven mutation strategies for thermostability improvements
        - Find information about specific amino acid substitutions and their structural effects
        - Search for validation methods and experimental approaches used in similar studies
        
        Catalytic activity examination guidelines:
        - Always use examine_catalytic_activity to verify catalytic integrity before and after mutations
        - Active site residues monitored: Y7, N62, H64, N67, Q92 (proton transfer and activator binding)
        - Zinc binding residues monitored: H94, H96, H119 (essential for catalytic activity)
        - Use residue_offsets parameter if you've made insertions/deletions that shift residue numbering
        - Integrity levels: EXCELLENT (no issues), GOOD (minor issues), ACCEPTABLE (some concerns), POOR (major problems)
        - NEVER recommend a design that shows POOR catalytic integrity
        
        Secondary structure examination guidelines:
        - Use examine_secondary_structure to assess overall fold quality and stability
        - Key metrics: SASA (lower suggests more compact), radius of gyration (compactness measure), secondary structure content
        - Typical stable proteins: 20-50% helix, 10-30% sheet, 30-60% loop
        - Compactness levels: VERY_COMPACT (excellent), COMPACT (good), NORMAL (acceptable), LOOSE/VERY_LOOSE (concerning)
        - Monitor surface hydrophobicity - too high may cause aggregation, appropriate levels improve stability
        - Use structural quality assessment to guide mutation strategies
        """
        
        print("=" * 80)
        print("CARBONIC ANHYDRASE STABILITY DESIGN SESSION")
        print("=" * 80)
        print(f"Target: {target_pdb}")
        print(f"Goals: {', '.join(stability_goals)}")
        print("=" * 80)
        
        # Start the reasoning loop
        response = self.client.responses.create(
            input=design_prompt,
            **self.model_config
        )
        
        iteration = 0
        max_iterations = 20  # Prevent infinite loops
        
        while iteration < max_iterations:
            iteration += 1
            print(f"\n--- Iteration {iteration} ---")
            
            is_complete, function_responses = self._process_response(response)
            
            if is_complete:
                # Final response ready
                final_result = response.output_text
                print("\n" + "=" * 80)
                print("FINAL DESIGN RECOMMENDATIONS")
                print("=" * 80)
                print(final_result)
                return final_result
            else:
                # More reasoning needed, send function results back
                print(f"Continuing reasoning with {len(function_responses)} function results...")
                response = self.client.responses.create(
                    input=function_responses,
                    previous_response_id=response.id,
                    **self.model_config
                )
        
        return "ERROR: Maximum iterations reached. Design process incomplete."
    
    def interactive_design_session(self):
        """
        Start an interactive design session where users can ask questions
        and get real-time assistance with carbonic anhydrase design.
        """
        print("Interactive Carbonic Anhydrase Design Session")
        print("Type 'quit' to exit, 'help' for commands")
        print("-" * 50)
        
        conversation_history = []
        
        while True:
            user_input = input("\nUser: ").strip()
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'help':
                print("\nAvailable commands:")
                print("- design <PDB_ID>: Start automated design for a specific protein")
                print("- quit: Exit the session")
                print("- Or ask any question about carbonic anhydrase design")
                print("- Available tools:")
                print("  * fold_protein (predicts 3D structure from sequence)")
                print("  * calculate_rosetta_score (scores PDB structures for stability)")
                print("  * calculate_rmsd (compares structural similarity between PDB files)")
                print("  * calculate_rmsd_with_sequences (detailed RMSD with sequence alignment info)")
                print("  * websearch (searches for current research and methodologies)")
                print("  * examine_catalytic_activity (visualizes catalytic sites to ensure activity is preserved)")
                print("  * examine_secondary_structure (analyzes fold, SASA, and structural properties)")
                continue
            elif user_input.lower().startswith('design '):
                pdb_id = user_input.split()[1]
                result = self.design_stable_carbonic_anhydrase(pdb_id)
                print(f"\nDesign completed for {pdb_id}")
                continue
            
            # Regular conversation
            conversation_history.append({"role": "user", "content": user_input})
            
            response = self.client.responses.create(
                input=conversation_history,
                **self.model_config
            )
            
            # Process response with function calls
            iteration = 0
            max_iterations = 10
            
            while iteration < max_iterations:
                iteration += 1
                is_complete, function_responses = self._process_response(response)
                
                if is_complete:
                    assistant_response = response.output_text
                    print(f"\nAssistant: {assistant_response}")
                    conversation_history.append({"role": "assistant", "content": assistant_response})
                    break
                else:
                    response = self.client.responses.create(
                        input=function_responses,
                        previous_response_id=response.id,
                        **self.model_config
                    )


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
    print("Starting carbonic anhydrase design session...")
    
    # Option 1: Automated design
    result = designer.design_stable_carbonic_anhydrase(
        target_pdb="1CA2",
        stability_goals=[
            "Increase thermal stability by 25°C",
            "Improve stability at pH 6-8",
            "Reduce aggregation",
            "Maintain >80% catalytic activity"
        ]
    )
    
    # Option 2: Interactive session (uncomment to use)
    # designer.interactive_design_session()


if __name__ == "__main__":
    main() 