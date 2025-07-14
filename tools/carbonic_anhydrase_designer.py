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
                "description": "Examine the catalytic activity sites of carbonic anhydrase II using PyMOL visualization. Checks active site residues (Y7, N62, H64, N67, Q92) and zinc binding residues (H94, H96, H119). Generates labeled images (returned as base64-encoded data) and assesses catalytic integrity to ensure modifications haven't affected enzyme activity.",
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
                "description": "Examine secondary structure and calculate structural properties using PyMOL. Analyzes helix/sheet/loop content, calculates SASA (Solvent Accessible Surface Area), radius of gyration, and generates a colored secondary structure visualization (returned as base64-encoded data). Provides quality assessment and compactness analysis.",
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
            String result from the function execution, with enhanced formatting for base64 images
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
            
            print(f"Executed {function_name}({arguments})")
            
            # Check if result contains base64 images (for examination functions)
            if function_name in ['examine_catalytic_activity', 'examine_secondary_structure']:
                try:
                    result_data = json.loads(str(result))
                    
                    # Check for base64 images and provide enhanced feedback
                    image_summary = []
                    
                    if function_name == 'examine_catalytic_activity':
                        if result_data.get('active_site_image_base64'):
                            image_summary.append("✓ Active site visualization generated")
                        if result_data.get('zinc_binding_image_base64'):
                            image_summary.append("✓ Zinc binding site visualization generated")
                        if result_data.get('combined_catalytic_image_base64'):
                            image_summary.append("✓ Combined catalytic site visualization generated")
                    
                    elif function_name == 'examine_secondary_structure':
                        if result_data.get('secondary_structure_image_base64'):
                            image_summary.append("✓ Secondary structure visualization generated")
                    
                    if image_summary:
                        print(f"Generated visualizations: {', '.join(image_summary)}")
                        
                        # Add a summary to the result for the model
                        result_data['visualization_summary'] = {
                            'images_generated': len(image_summary),
                            'image_descriptions': image_summary,
                            'note': 'Base64-encoded images are included in the response for visual analysis'
                        }
                        
                        # Return the enhanced result
                        enhanced_result = json.dumps(result_data, indent=2)
                        print(f"Enhanced result with {len(image_summary)} base64 images prepared for model analysis")
                        return enhanced_result
                        
                except (json.JSONDecodeError, KeyError):
                    # If parsing fails, return original result
                    pass
            
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
                
                # Check if result contains base64 images and format them properly
                visual_content = self._extract_and_format_images(result, item.name)
                
                if visual_content:
                    # If we have images, format the response with visual content
                    function_responses.append({
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": visual_content
                    })
                else:
                    # Standard text response
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
    
    def _extract_and_format_images(self, result: str, function_name: str) -> Optional[List[Dict[str, Any]]]:
        """
        Extract base64 images from function results and format them for visual analysis.
        
        Uses the OpenAI API format for visual inputs:
        [
            {"type": "input_text", "text": "description..."},
            {"type": "input_image", "image_url": "data:image/png;base64,..."}, 
            ...
        ]
        
        Note: The o3 model's visual analysis capabilities may be different from GPT-4V.
        This implementation provides base64 images in the standard format but includes
        fallback handling if the o3 API doesn't support visual inputs in continuation requests.
        
        Args:
            result: JSON string result from function execution
            function_name: Name of the function that was executed
            
        Returns:
            List of content items for visual analysis, or None if no images
        """
        if function_name not in ['examine_catalytic_activity', 'examine_secondary_structure']:
            return None
            
        try:
            result_data = json.loads(str(result))
            content_items = []
            
            # Start with text summary
            summary_text = "Function execution results:\n\n"
            
            if function_name == 'examine_catalytic_activity':
                integrity = result_data.get('catalytic_integrity', {})
                summary_text += f"Catalytic Integrity: {integrity.get('integrity_level', 'UNKNOWN')}\n"
                summary_text += f"Risk Level: {integrity.get('risk_level', 'UNKNOWN')}\n"
                
                if 'active_site_rmsd' in integrity and integrity['active_site_rmsd']:
                    summary_text += f"Active Site RMSD: {integrity['active_site_rmsd']:.2f} Å\n"
                if 'zinc_binding_rmsd' in integrity and integrity['zinc_binding_rmsd']:
                    summary_text += f"Zinc Binding RMSD: {integrity['zinc_binding_rmsd']:.2f} Å\n"
                
                summary_text += "\nGenerated visualizations for analysis:\n"
                
                # Add images
                if result_data.get('active_site_image_base64'):
                    summary_text += "- Active site residues visualization\n"
                    content_items.append({
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{result_data['active_site_image_base64']}"
                    })
                
                if result_data.get('zinc_binding_image_base64'):
                    summary_text += "- Zinc binding site visualization\n"
                    content_items.append({
                        "type": "input_image", 
                        "image_url": f"data:image/png;base64,{result_data['zinc_binding_image_base64']}"
                    })
                
                if result_data.get('combined_catalytic_image_base64'):
                    summary_text += "- Combined catalytic site visualization\n"
                    content_items.append({
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{result_data['combined_catalytic_image_base64']}"
                    })
            
            elif function_name == 'examine_secondary_structure':
                ss_content = result_data.get('secondary_structure_content', {})
                quality = result_data.get('quality_assessment', {})
                
                summary_text += f"Secondary Structure Analysis:\n"
                summary_text += f"Total Residues: {ss_content.get('total_residues', 'N/A')}\n"
                summary_text += f"Helix: {ss_content.get('helix_percentage', 0):.1f}%\n"
                summary_text += f"Sheet: {ss_content.get('sheet_percentage', 0):.1f}%\n"
                summary_text += f"Loop: {ss_content.get('loop_percentage', 0):.1f}%\n"
                summary_text += f"Overall Quality: {quality.get('overall_quality', 'UNKNOWN')}\n"
                summary_text += f"Compactness: {quality.get('compactness', 'UNKNOWN')}\n"
                
                if result_data.get('secondary_structure_image_base64'):
                    summary_text += "\nGenerated secondary structure visualization for analysis:\n"
                    content_items.append({
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{result_data['secondary_structure_image_base64']}"
                    })
            
            # If we have images, return formatted content
            if content_items:
                # Insert text summary at the beginning
                formatted_content = [{"type": "input_text", "text": summary_text}] + content_items
                print(f"Formatted {len(content_items)} images for visual analysis by the model")
                return formatted_content
            
            return None
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error formatting images: {e}")
            return None
    
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
        5. examine_catalytic_activity: Visualizes and examines catalytic sites (active site and zinc binding residues) with base64-encoded images you can analyze visually to ensure modifications haven't affected enzyme activity
        6. examine_secondary_structure: Analyzes secondary structure content, calculates SASA and structural properties, and provides quality assessment with base64-encoded visualizations you can examine

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
        
        Visual analysis guidelines:
        - The examination functions return base64-encoded images that you can analyze visually
        - Look for structural deformations, missing secondary structures, or catalytic site disruptions
        - Compare visualizations between wild-type and mutant structures to assess structural preservation
        - Use the visual information to guide further optimization decisions
        - Images show: secondary structure coloring, catalytic residues, zinc coordination, surface representations
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
                
                # Check if any responses contain visual content
                has_visual_content = any(
                    isinstance(resp.get('output'), list) and 
                    any(item.get('type') == 'input_image' for item in resp.get('output', []))
                    for resp in function_responses
                )
                
                if has_visual_content:
                    print("⚠️  Visual content detected - o3 model visual analysis may be limited")
                    print("   The model will receive visual data but may have limited image analysis capabilities")
                
                try:
                    response = self.client.responses.create(
                        input=function_responses,
                        previous_response_id=response.id,
                        **self.model_config
                    )
                except Exception as e:
                    print(f"Error sending continuation request: {e}")
                    if has_visual_content:
                        print("This may be due to visual content formatting. Falling back to text-only mode...")
                        # Fallback: send text-only versions
                        fallback_responses = []
                        for resp in function_responses:
                            if isinstance(resp.get('output'), list):
                                # Extract just the text content
                                text_content = ""
                                for item in resp.get('output', []):
                                    if item.get('type') == 'input_text':
                                        text_content += item.get('text', '')
                                fallback_responses.append({
                                    "type": "function_call_output",
                                    "call_id": resp['call_id'],
                                    "output": text_content or "Visual analysis completed (images not displayed)"
                                })
                            else:
                                fallback_responses.append(resp)
                        
                        response = self.client.responses.create(
                            input=fallback_responses,
                            previous_response_id=response.id,
                            **self.model_config
                        )
                    else:
                        raise
        
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