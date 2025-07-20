#!/usr/bin/env python3
"""
Secondary Structure Examiner for Protein Analysis.

This module uses PyMOL to visualize secondary structures and calculate
structural properties including SASA, radius of gyration, and secondary
structure content analysis.
"""

import os
import sys
import tempfile
import json
import math
import base64
from typing import Dict, List, Optional, Tuple, Any

# Try to import PyMOL
try:
    import pymol
    from pymol import cmd
    PYMOL_AVAILABLE = True
except ImportError:
    PYMOL_AVAILABLE = False
    print("Warning: PyMOL not available. Install with: conda install -c conda-forge pymol-open-source")


class SecondaryStructureExaminer:
    """
    A class for examining secondary structures and calculating structural properties using PyMOL.
    """
    
    def __init__(self, chain_id: str = 'A'):
        """
        Initialize the secondary structure examiner.
        
        Args:
            chain_id: Chain identifier for the protein (default: 'A')
        """
        if not PYMOL_AVAILABLE:
            raise ImportError("PyMOL is required for secondary structure examination")
        
        self.chain_id = chain_id
        self.temp_dir = tempfile.mkdtemp()
        
    def examine_secondary_structure(self, pdb_file_path: str, 
                                  image_subdir: str = "default") -> Dict[str, Any]:
        """
        Examine secondary structure and calculate structural properties.
        
        Args:
            pdb_file_path: Path to the PDB file to examine
            image_subdir: Subdirectory name within tools/images/ to save images
            
        Returns:
            Dictionary containing structural analysis and image path
        """
        if not os.path.exists(pdb_file_path):
            raise FileNotFoundError(f"PDB file not found: {pdb_file_path}")
        
        # Always save to tools/images/{image_subdir}
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, 'images', image_subdir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize PyMOL
        pymol.finish_launching()
        cmd.reinitialize()
        
        try:
            # Load the structure
            structure_name = "protein_structure"
            cmd.load(pdb_file_path, structure_name)
            
            # Calculate secondary structure
            cmd.dss(f"{structure_name} and chain {self.chain_id}")
            
            # Generate secondary structure visualization
            ss_image_path = self._visualize_secondary_structure(structure_name, output_dir)
            
            # Calculate structural properties
            structural_properties = self._calculate_structural_properties(structure_name)
            
            # Analyze secondary structure content
            ss_content = self._analyze_secondary_structure_content(structure_name)
            
            # Calculate surface properties
            surface_properties = self._calculate_surface_properties(structure_name)
            
            # Assess structural quality
            quality_assessment = self._assess_structural_quality(
                structural_properties, ss_content, surface_properties
            )
            
            results = {
                'pdb_file': pdb_file_path,
                'chain_id': self.chain_id,
                'secondary_structure_image': ss_image_path,
                'structural_properties': structural_properties,
                'secondary_structure_content': ss_content,
                'surface_properties': surface_properties,
                'quality_assessment': quality_assessment,
                'summary': self._generate_summary(structural_properties, ss_content, surface_properties)
            }
            
            return results
            
        finally:
            # Clean up PyMOL
            cmd.reinitialize()
    
    def _visualize_secondary_structure(self, structure_name: str, output_dir: str) -> str:
        """
        Generate secondary structure visualization.
        """
        # Don't reinitialize - structure is already loaded
        # cmd.reinitialize()
        # cmd.load(cmd.get_object_list()[0] if cmd.get_object_list() else structure_name)
        
        # Hide everything first
        cmd.hide('everything')
        
        # Show cartoon representation
        cmd.show('cartoon', f'{structure_name} and chain {self.chain_id}')
        
        # Color by secondary structure
        # Alpha helices - red
        cmd.color('red', f'{structure_name} and chain {self.chain_id} and ss h')
        
        # Beta sheets - yellow
        cmd.color('yellow', f'{structure_name} and chain {self.chain_id} and ss s')
        
        # Loops/coils - green
        cmd.color('green', f'{structure_name} and chain {self.chain_id} and ss l+""')
        
        # Show side chains for important residues (if they exist)
        important_residues = [7, 62, 64, 67, 92, 94, 96, 119]  # CA II catalytic residues
        for res_num in important_residues:
            selection = f"chain {self.chain_id} and resi {res_num}"
            if cmd.count_atoms(f"{structure_name} and {selection}") > 0:
                cmd.show('sticks', f'{structure_name} and {selection} and sidechain')
                cmd.color('cyan', f'{structure_name} and {selection} and sidechain')
        
        # Set nice view
        cmd.orient(f'{structure_name} and chain {self.chain_id}')
        cmd.zoom(f'{structure_name} and chain {self.chain_id}')
        
        # Add labels for secondary structure
        cmd.set('label_size', 12)
        cmd.set('label_color', 'white')
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Save image with proper settings
        output_path = os.path.join(output_dir, 'secondary_structure.png')
        
        # Set rendering options for better quality
        cmd.set('ray_trace_mode', 1)
        cmd.set('ray_shadows', 0)
        cmd.set('antialias', 2)
        
        # Capture the image
        cmd.png(output_path, width=400, height=300, dpi=72, ray=1)
        
        print(f"Secondary structure image saved to: {output_path}")
        
        return output_path
    
    def _calculate_structural_properties(self, structure_name: str) -> Dict[str, Any]:
        """
        Calculate basic structural properties.
        """
        properties = {}
        
        selection = f"{structure_name} and chain {self.chain_id}"
        
        # Count atoms and residues
        total_atoms = cmd.count_atoms(f"{selection}")
        ca_atoms = cmd.count_atoms(f"{selection} and name CA")
        
        properties['total_atoms'] = total_atoms
        properties['total_residues'] = ca_atoms
        
        # Calculate radius of gyration (approximation using CA atoms)
        if ca_atoms > 0:
            try:
                # Get coordinates of CA atoms
                ca_coords = []
                cmd.iterate_state(1, f"{selection} and name CA", 
                                "ca_coords.append([x, y, z])", space={'ca_coords': ca_coords})
                
                if ca_coords:
                    # Calculate center of mass
                    center_x = sum(coord[0] for coord in ca_coords) / len(ca_coords)
                    center_y = sum(coord[1] for coord in ca_coords) / len(ca_coords)
                    center_z = sum(coord[2] for coord in ca_coords) / len(ca_coords)
                    
                    # Calculate radius of gyration
                    rg_squared = sum((coord[0] - center_x)**2 + (coord[1] - center_y)**2 + (coord[2] - center_z)**2 
                                   for coord in ca_coords) / len(ca_coords)
                    radius_of_gyration = math.sqrt(rg_squared)
                    
                    properties['radius_of_gyration'] = round(radius_of_gyration, 2)
                    properties['center_of_mass'] = [round(center_x, 2), round(center_y, 2), round(center_z, 2)]
                else:
                    properties['radius_of_gyration'] = None
                    properties['center_of_mass'] = None
                    
            except Exception as e:
                properties['radius_of_gyration'] = None
                properties['center_of_mass'] = None
                properties['rg_calculation_error'] = str(e)
        else:
            properties['radius_of_gyration'] = None
            properties['center_of_mass'] = None
        
        # Calculate approximate molecular weight (rough estimate)
        # Average amino acid molecular weight ~ 110 Da
        if ca_atoms > 0:
            properties['estimated_molecular_weight'] = ca_atoms * 110
        else:
            properties['estimated_molecular_weight'] = None
        
        return properties
    
    def _analyze_secondary_structure_content(self, structure_name: str) -> Dict[str, Any]:
        """
        Analyze secondary structure content.
        """
        selection = f"{structure_name} and chain {self.chain_id}"
        
        # Count residues in different secondary structures
        total_residues = cmd.count_atoms(f"{selection} and name CA")
        
        if total_residues == 0:
            return {
                'total_residues': 0,
                'helix_residues': 0,
                'sheet_residues': 0,
                'loop_residues': 0,
                'helix_percentage': 0.0,
                'sheet_percentage': 0.0,
                'loop_percentage': 0.0
            }
        
        helix_residues = cmd.count_atoms(f"{selection} and ss h and name CA")
        sheet_residues = cmd.count_atoms(f"{selection} and ss s and name CA")
        loop_residues = cmd.count_atoms(f"{selection} and ss l+'' and name CA")
        
        # Calculate percentages
        helix_percentage = (helix_residues / total_residues) * 100
        sheet_percentage = (sheet_residues / total_residues) * 100
        loop_percentage = (loop_residues / total_residues) * 100
        
        return {
            'total_residues': total_residues,
            'helix_residues': helix_residues,
            'sheet_residues': sheet_residues,
            'loop_residues': loop_residues,
            'helix_percentage': round(helix_percentage, 1),
            'sheet_percentage': round(sheet_percentage, 1),
            'loop_percentage': round(loop_percentage, 1)
        }
    
    def _calculate_surface_properties(self, structure_name: str) -> Dict[str, Any]:
        """
        Calculate surface properties including SASA.
        """
        selection = f"{structure_name} and chain {self.chain_id}"
        surface_properties = {}
        
        try:
            # Calculate SASA (Solvent Accessible Surface Area)
            # Create a temporary object for surface calculation
            temp_obj = "temp_surface"
            cmd.create(temp_obj, selection)
            
            # Calculate surface area
            cmd.set('dot_solvent', 1)
            cmd.set('dot_density', 2)  # Medium density for reasonable speed
            
            # Get total SASA
            total_sasa = cmd.get_area(temp_obj, quiet=1)
            surface_properties['total_sasa'] = round(total_sasa, 2) if total_sasa else None
            
            # Calculate SASA by atom type if possible
            try:
                # Hydrophobic SASA (C atoms)
                hydrophobic_sasa = cmd.get_area(f"{temp_obj} and elem C", quiet=1)
                surface_properties['hydrophobic_sasa'] = round(hydrophobic_sasa, 2) if hydrophobic_sasa else 0
                
                # Polar SASA (N, O atoms)
                polar_sasa = cmd.get_area(f"{temp_obj} and elem N+O", quiet=1)
                surface_properties['polar_sasa'] = round(polar_sasa, 2) if polar_sasa else 0
                
                # Calculate hydrophobic ratio
                if total_sasa and total_sasa > 0:
                    surface_properties['hydrophobic_ratio'] = round((hydrophobic_sasa / total_sasa) * 100, 1)
                else:
                    surface_properties['hydrophobic_ratio'] = None
                    
            except Exception as e:
                surface_properties['sasa_breakdown_error'] = str(e)
            
            # Clean up temporary object
            cmd.delete(temp_obj)
            
        except Exception as e:
            surface_properties['sasa_calculation_error'] = str(e)
            surface_properties['total_sasa'] = None
        
        return surface_properties
    
    def _assess_structural_quality(self, structural_props: Dict[str, Any], 
                                 ss_content: Dict[str, Any], 
                                 surface_props: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess overall structural quality and compactness.
        """
        assessment = {
            'overall_quality': 'UNKNOWN',
            'compactness': 'UNKNOWN',
            'secondary_structure_quality': 'UNKNOWN',
            'issues': [],
            'recommendations': []
        }
        
        issues = []
        recommendations = []
        
        # Assess secondary structure content
        if ss_content['total_residues'] > 0:
            helix_pct = ss_content['helix_percentage']
            sheet_pct = ss_content['sheet_percentage']
            loop_pct = ss_content['loop_percentage']
            
            # Typical protein has 30-35% helix, 15-25% sheet, 40-50% loop
            if helix_pct < 10:
                issues.append("Low helix content may indicate structural instability")
            elif helix_pct > 60:
                issues.append("Very high helix content - check for artificial regularity")
            
            if sheet_pct > 50:
                issues.append("Unusually high beta sheet content")
            
            if loop_pct > 70:
                issues.append("High loop content may indicate flexibility/disorder")
                recommendations.append("Consider stabilizing mutations to reduce loop regions")
            
            # Overall secondary structure assessment
            if 20 <= helix_pct <= 50 and 10 <= sheet_pct <= 30 and 30 <= loop_pct <= 60:
                assessment['secondary_structure_quality'] = 'GOOD'
            elif 10 <= helix_pct <= 60 and 5 <= sheet_pct <= 40 and 20 <= loop_pct <= 70:
                assessment['secondary_structure_quality'] = 'ACCEPTABLE'
            else:
                assessment['secondary_structure_quality'] = 'POOR'
        
        # Assess compactness using radius of gyration
        rg = structural_props.get('radius_of_gyration')
        num_residues = structural_props.get('total_residues', 0)
        
        if rg and num_residues > 0:
            # Empirical relationship: RG ≈ 2.2 * N^0.57 for globular proteins
            expected_rg = 2.2 * (num_residues ** 0.57)
            rg_ratio = rg / expected_rg
            
            if rg_ratio < 0.8:
                assessment['compactness'] = 'VERY_COMPACT'
            elif rg_ratio < 1.1:
                assessment['compactness'] = 'COMPACT'
            elif rg_ratio < 1.3:
                assessment['compactness'] = 'NORMAL'
            elif rg_ratio < 1.5:
                assessment['compactness'] = 'LOOSE'
            else:
                assessment['compactness'] = 'VERY_LOOSE'
                issues.append("Structure appears unusually extended")
                recommendations.append("Consider mutations to improve compactness")
        
        # Assess surface properties
        total_sasa = surface_props.get('total_sasa')
        if total_sasa and num_residues > 0:
            # Typical SASA per residue is ~140-180 Ų for globular proteins
            sasa_per_residue = total_sasa / num_residues
            
            if sasa_per_residue > 200:
                issues.append("High surface area per residue suggests extended conformation")
            elif sasa_per_residue < 100:
                recommendations.append("Very compact structure - good for stability")
        
        # Overall quality assessment
        if not issues and assessment['secondary_structure_quality'] == 'GOOD':
            assessment['overall_quality'] = 'EXCELLENT'
        elif len(issues) <= 1 and assessment['secondary_structure_quality'] != 'POOR':
            assessment['overall_quality'] = 'GOOD'
        elif len(issues) <= 2:
            assessment['overall_quality'] = 'ACCEPTABLE'
        else:
            assessment['overall_quality'] = 'POOR'
        
        assessment['issues'] = issues
        assessment['recommendations'] = recommendations
        
        return assessment
    
    def _generate_summary(self, structural_props: Dict[str, Any], 
                         ss_content: Dict[str, Any], 
                         surface_props: Dict[str, Any]) -> str:
        """
        Generate a human-readable summary of the structural analysis.
        """
        summary_parts = []
        
        # Basic structure info
        num_residues = structural_props.get('total_residues', 0)
        summary_parts.append(f"Structure contains {num_residues} residues")
        
        # Secondary structure
        if ss_content['total_residues'] > 0:
            helix_pct = ss_content['helix_percentage']
            sheet_pct = ss_content['sheet_percentage']
            loop_pct = ss_content['loop_percentage']
            summary_parts.append(
                f"Secondary structure: {helix_pct}% helix, {sheet_pct}% sheet, {loop_pct}% loop"
            )
        
        # Compactness
        rg = structural_props.get('radius_of_gyration')
        if rg:
            summary_parts.append(f"Radius of gyration: {rg} Å")
        
        # Surface area
        sasa = surface_props.get('total_sasa')
        if sasa:
            summary_parts.append(f"Total SASA: {sasa} Ų")
            
            hydrophobic_ratio = surface_props.get('hydrophobic_ratio')
            if hydrophobic_ratio:
                summary_parts.append(f"Surface hydrophobicity: {hydrophobic_ratio}%")
        
        return ". ".join(summary_parts) + "."


def encode_image_to_base64(image_path: str) -> Optional[str]:
    """
    Encode an image file to base64 string.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64-encoded string of the image, or None if encoding fails
    """
    try:
        if os.path.exists(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        else:
            return None
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None


def examine_secondary_structure(pdb_file_path: str, 
                               chain_id: str = 'A',
                               image_subdir: str = "default") -> str:
    """
    Simplified interface for examining secondary structure.
    
    Args:
        pdb_file_path: Path to the PDB file to examine
        chain_id: Chain identifier (default: 'A')
        image_subdir: Subdirectory name within tools/images/ to save images (default: "default")
        
    Returns:
        JSON string containing structural analysis results with base64-encoded images
    """
    try:
        examiner = SecondaryStructureExaminer(chain_id=chain_id)
        results = examiner.examine_secondary_structure(pdb_file_path, image_subdir)
        
        # Encode the image as base64
        image_base64 = None
        if results['secondary_structure_image'] and os.path.exists(results['secondary_structure_image']):
            image_base64 = encode_image_to_base64(results['secondary_structure_image'])
        
        # Convert to JSON-serializable format with base64 image
        serializable_results = {
            'pdb_file': results['pdb_file'],
            'chain_id': results['chain_id'],
            'secondary_structure_image_path': results['secondary_structure_image'],
            'secondary_structure_image_base64': image_base64,
            'structural_properties': results['structural_properties'],
            'secondary_structure_content': results['secondary_structure_content'],
            'surface_properties': results['surface_properties'],
            'quality_assessment': results['quality_assessment'],
            'summary': results['summary']
        }
        
        return json.dumps(serializable_results, indent=2)
        
    except Exception as e:
        error_result = {
            'error': str(e),
            'pdb_file': pdb_file_path,
            'success': False,
            'secondary_structure_image_base64': None
        }
        return json.dumps(error_result, indent=2)


if __name__ == "__main__":
    # Test the secondary structure examiner
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python secondary_structure_examiner.py <pdb_file>")
        sys.exit(1)
    
    pdb_file = sys.argv[1]
    
    if not PYMOL_AVAILABLE:
        print("Error: PyMOL not available")
        print("Install with: conda install -c conda-forge pymol-open-source")
        sys.exit(1)
    
    try:
        print(f"Examining secondary structure for: {pdb_file}")
        print("=" * 60)
        
        result = examine_secondary_structure(pdb_file)
        print(result)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1) 