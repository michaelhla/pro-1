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
                "name": "calculate_rmsd_with_sequences",
                "description": "Calculate RMSD between the hardcoded reference structure (hCA2_folded.pdb) and a newly folded protein structure using sliding window sequence alignment. Finds the region of maximum sequence overlap, then calculates RMSD over the aligned core region. Returns RMSD value plus overlap percentage, sequence identity, and detailed alignment information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pdb_file2": {
                            "type": "string",
                            "description": "Path to the newly folded PDB file to compare against reference (e.g., 'predicted_structures/mutant.pdb')"
                        },
                        "chain_id1": {
                            "type": "string",
                            "description": "Chain ID for reference structure (optional, auto-detected if not provided)"
                        },
                        "chain_id2": {
                            "type": "string",
                            "description": "Chain ID for new structure (optional, auto-detected if not provided)"
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

        ORIGINAL SEQUENCE: {REFERENCE_SEQUENCE}
        Stability goals: {', '.join(stability_goals)}
        CATALYTIC RESIDUES: Y7, N62, H64, N67, Q92 (proton transfer and activator binding)
        ZINC BINDING RESIDUES: H94, H96, H119 (essential for catalytic activity)

        You have access to six computational tools:
        1. fold_protein: Predicts 3D structures from amino acid sequences using ESMFold
        2. calculate_rosetta_score: Calculates Rosetta energy scores for PDB structures (lower = more stable)
        3. calculate_rmsd_with_sequences: Uses sliding window sequence alignment to find maximum overlap between reference (hCA2_folded.pdb) and new structure, then calculates RMSD over aligned core region. Returns RMSD, overlap percentage, and sequence identity.
        4. websearch: Searches the web for current information about protein engineering, research papers, and methodologies
        5. examine_catalytic_activity: Visualizes and examines catalytic sites (active site and zinc binding residues) to ensure modifications haven't affected enzyme activity
        6. examine_secondary_structure: Analyzes secondary structure content, calculates SASA and structural properties, and provides quality assessment

        Please approach this systematically:
        1. If given a PDB ID, first provide the corresponding amino acid sequence so you can fold it
        2. Use websearch to find current research on carbonic anhydrase stability and recent engineering approaches. Go deep into the literature and Uniprot.
        3. Using the findings from your research, propose specific amino acid mutations that could improve stability. These can be simple point mutations, or larger insertions/deletions.
        4. Then modify the original sequence using the mutations you proposed. Make sure you have applied your mutations correctly. 
        5. Use the fold_protein tool to predict the structure of your modified sequence. This will return a filepath to your pdb. 
        6. Use calculate_rosetta_score to get the stability score of your modified sequence.
        7. Use examine_secondary_structure to analyze the new structure's fold, SASA, and structural quality
        8. Use examine_catalytic_activity to verify the new structure's catalytic sites are intact. You will have to make sure to account for any sequence length changes with residue_offsets.This includes the ZINC BINDING RESIDUES.
        CRITICAL: Use examine_catalytic_activity on each mutant to ensure catalytic residues are preserved (account for any sequence length changes with residue_offsets). This includes the ZINC BINDING RESIDUES.
        9. Compare scores and provide recommendations with quantitative rationale. 
        10. REPEAT THE PROCESS UNTIL YOU HAVE A DESIGN THAT YOU BELIEVE MEETS THE GOALS.

        Focus on common protein stabilization strategies:
        - Reducing surface loops and increasing rigidity
        - Improving hydrophobic core packing
        - Adding favorable electrostatic interactions
        - Removing destabilizing residues
        - Increasing secondary structure propensity

        Always preserve the catalytic activity of the enzyme while improving stability.
        Use the Rosetta scores to validate your design decisions quantitatively.
        
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



if __name__ == "__main__":
    main() 