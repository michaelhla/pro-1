#!/usr/bin/env python3
"""
RMSD Calculator for comparing protein structures.

This module provides functions to calculate Root Mean Square Deviation (RMSD)
between two protein structures from PDB files, with support for structures
of different lengths through structural alignment.
"""

import os
import warnings
from typing import Tuple, Optional, List, Dict, Any
from Bio import PDB
from Bio.PDB import PDBParser, Superimposer, Selection
from Bio.PDB.Structure import Structure
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
import numpy as np

# Suppress PDB parsing warnings
# warnings.filterwarnings("ignore", category=PDB.PDBConstructionWarning)


def calculate_rmsd(pdb_file1: str, pdb_file2: str, 
                  chain_id1: str = None, chain_id2: str = None,
                  alignment_method: str = "structural") -> float:
    """
    Calculate RMSD between two protein structures from PDB files.
    
    Handles structures of different lengths by performing structural alignment
    to find the best matching residues.
    
    Args:
        pdb_file1: Path to first PDB file
        pdb_file2: Path to second PDB file  
        chain_id1: Chain ID for first structure (auto-detect if None)
        chain_id2: Chain ID for second structure (auto-detect if None)
        alignment_method: Method for handling different lengths ("structural" or "sequence")
        
    Returns:
        RMSD value in Angstroms as a float
        
    Raises:
        FileNotFoundError: If PDB files don't exist
        ValueError: If structures can't be aligned or have no common residues
    """
    # Check if files exist
    if not os.path.exists(pdb_file1):
        raise FileNotFoundError(f"PDB file not found: {pdb_file1}")
    if not os.path.exists(pdb_file2):
        raise FileNotFoundError(f"PDB file not found: {pdb_file2}")
    
    try:
        # Parse PDB structures
        parser = PDBParser(QUIET=True)
        structure1 = parser.get_structure("struct1", pdb_file1)
        structure2 = parser.get_structure("struct2", pdb_file2)
        
        # Get chains
        chain1 = _get_chain(structure1, chain_id1)
        chain2 = _get_chain(structure2, chain_id2)
        
        if alignment_method == "structural":
            rmsd = _calculate_structural_rmsd(chain1, chain2)
        else:
            rmsd = _calculate_sequence_rmsd(chain1, chain2)
            
        return round(rmsd, 3)
        
    except Exception as e:
        raise ValueError(f"Error calculating RMSD: {str(e)}")


def _get_chain(structure: Structure, chain_id: Optional[str] = None) -> Chain:
    """
    Get a chain from a structure, auto-detecting if chain_id is None.
    """
    chains = list(structure.get_chains())
    
    if not chains:
        raise ValueError("No chains found in structure")
    
    if chain_id is None:
        # Auto-detect: use first chain
        return chains[0]
    else:
        # Find specified chain
        for chain in chains:
            if chain.id == chain_id:
                return chain
        raise ValueError(f"Chain {chain_id} not found in structure")


def _calculate_structural_rmsd(chain1: Chain, chain2: Chain) -> float:
    """
    Calculate RMSD using structural alignment to handle different lengths.
    
    Uses a sliding window approach to find the best matching segment
    between the two structures.
    """
    # Get CA atoms from both chains
    ca_atoms1 = _get_ca_atoms(chain1)
    ca_atoms2 = _get_ca_atoms(chain2)
    
    if len(ca_atoms1) == 0 or len(ca_atoms2) == 0:
        raise ValueError("No CA atoms found in one or both structures")
    
    # If structures are the same length, do direct alignment
    if len(ca_atoms1) == len(ca_atoms2):
        return _direct_rmsd(ca_atoms1, ca_atoms2)
    
    # For different lengths, find best matching segment
    min_rmsd = float('inf')
    shorter_atoms = ca_atoms1 if len(ca_atoms1) <= len(ca_atoms2) else ca_atoms2
    longer_atoms = ca_atoms2 if len(ca_atoms1) <= len(ca_atoms2) else ca_atoms1
    
    # Try all possible alignments of the shorter structure within the longer one
    window_size = len(shorter_atoms)
    
    for i in range(len(longer_atoms) - window_size + 1):
        segment = longer_atoms[i:i + window_size]
        try:
            if len(ca_atoms1) <= len(ca_atoms2):
                rmsd = _direct_rmsd(shorter_atoms, segment)
            else:
                rmsd = _direct_rmsd(segment, shorter_atoms)
            min_rmsd = min(min_rmsd, rmsd)
        except:
            continue
    
    if min_rmsd == float('inf'):
        raise ValueError("Could not align structures")
    
    return min_rmsd


def _calculate_structural_rmsd_with_alignment(chain1: Chain, chain2: Chain) -> Tuple[float, Dict[str, Any]]:
    """
    Calculate RMSD using structural alignment and return detailed alignment information.
    
    Returns:
        Tuple of (rmsd, alignment_info) where alignment_info contains:
        - aligned_sequences: subsequences that were aligned
        - alignment_indices: indices of aligned residues
        - alignment_method: method used for alignment
    """
    # Get CA atoms and sequences from both chains
    ca_atoms1 = _get_ca_atoms(chain1)
    ca_atoms2 = _get_ca_atoms(chain2)
    sequence1 = _get_sequence_from_chain(chain1)
    sequence2 = _get_sequence_from_chain(chain2)
    residue_info1 = _get_residue_info(chain1)
    residue_info2 = _get_residue_info(chain2)
    
    if len(ca_atoms1) == 0 or len(ca_atoms2) == 0:
        raise ValueError("No CA atoms found in one or both structures")
    
    # If structures are the same length, do direct alignment
    if len(ca_atoms1) == len(ca_atoms2):
        rmsd = _direct_rmsd(ca_atoms1, ca_atoms2)
        alignment_info = {
            'aligned_sequence1': sequence1,
            'aligned_sequence2': sequence2,
            'alignment_start1': 0,
            'alignment_end1': len(sequence1),
            'alignment_start2': 0,
            'alignment_end2': len(sequence2),
            'alignment_length': len(sequence1),
            'alignment_method': 'direct',
            'coverage1': 100.0,
            'coverage2': 100.0,
            'residue_info1': residue_info1,
            'residue_info2': residue_info2
        }
        return rmsd, alignment_info
    
    # For different lengths, find best matching segment
    min_rmsd = float('inf')
    best_alignment_info = None
    
    if len(ca_atoms1) <= len(ca_atoms2):
        # Structure 1 is shorter, slide it along structure 2
        shorter_atoms = ca_atoms1
        longer_atoms = ca_atoms2
        shorter_seq = sequence1
        longer_seq = sequence2
        shorter_residues = residue_info1
        longer_residues = residue_info2
        struct1_is_shorter = True
    else:
        # Structure 2 is shorter, slide it along structure 1
        shorter_atoms = ca_atoms2
        longer_atoms = ca_atoms1
        shorter_seq = sequence2
        longer_seq = sequence1
        shorter_residues = residue_info2
        longer_residues = residue_info1
        struct1_is_shorter = False
    
    window_size = len(shorter_atoms)
    
    for i in range(len(longer_atoms) - window_size + 1):
        segment = longer_atoms[i:i + window_size]
        try:
            if struct1_is_shorter:
                rmsd = _direct_rmsd(shorter_atoms, segment)
            else:
                rmsd = _direct_rmsd(segment, shorter_atoms)
            
            if rmsd < min_rmsd:
                min_rmsd = rmsd
                
                # Calculate alignment info for this best match
                if struct1_is_shorter:
                    alignment_info = {
                        'aligned_sequence1': shorter_seq,
                        'aligned_sequence2': longer_seq[i:i + window_size],
                        'alignment_start1': 0,
                        'alignment_end1': len(shorter_seq),
                        'alignment_start2': i,
                        'alignment_end2': i + window_size,
                        'alignment_length': window_size,
                        'alignment_method': 'structural_sliding',
                        'coverage1': 100.0,
                        'coverage2': (window_size / len(longer_seq)) * 100.0,
                        'residue_info1': shorter_residues,
                        'residue_info2': longer_residues[i:i + window_size]
                    }
                else:
                    alignment_info = {
                        'aligned_sequence1': longer_seq[i:i + window_size],
                        'aligned_sequence2': shorter_seq,
                        'alignment_start1': i,
                        'alignment_end1': i + window_size,
                        'alignment_start2': 0,
                        'alignment_end2': len(shorter_seq),
                        'alignment_length': window_size,
                        'alignment_method': 'structural_sliding',
                        'coverage1': (window_size / len(longer_seq)) * 100.0,
                        'coverage2': 100.0,
                        'residue_info1': longer_residues[i:i + window_size],
                        'residue_info2': shorter_residues
                    }
                
                best_alignment_info = alignment_info
        except:
            continue
    
    if min_rmsd == float('inf'):
        raise ValueError("Could not align structures")
    
    return min_rmsd, best_alignment_info


def _calculate_sequence_rmsd(chain1: Chain, chain2: Chain) -> float:
    """
    Calculate RMSD based on sequence alignment (simpler approach).
    Only uses residues that exist in both structures at the same positions.
    """
    ca_atoms1 = _get_ca_atoms(chain1)
    ca_atoms2 = _get_ca_atoms(chain2)
    
    if len(ca_atoms1) == 0 or len(ca_atoms2) == 0:
        raise ValueError("No CA atoms found in one or both structures")
    
    # Use the minimum length
    min_length = min(len(ca_atoms1), len(ca_atoms2))
    
    if min_length == 0:
        raise ValueError("No common residues found")
    
    # Calculate RMSD for the overlapping region
    return _direct_rmsd(ca_atoms1[:min_length], ca_atoms2[:min_length])


def _get_ca_atoms(chain: Chain) -> List:
    """
    Extract CA atoms from a chain, maintaining order.
    """
    ca_atoms = []
    for residue in chain:
        if residue.has_id('CA'):
            ca_atoms.append(residue['CA'])
    return ca_atoms


def _get_sequence_from_chain(chain: Chain) -> str:
    """
    Extract amino acid sequence from a chain.
    
    Returns:
        String of single-letter amino acid codes
    """
    # Standard amino acid three-letter to one-letter mapping
    aa_dict = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
    }
    
    sequence = []
    for residue in chain:
        if residue.has_id('CA'):  # Only consider residues with CA atoms
            res_name = residue.get_resname()
            if res_name in aa_dict:
                sequence.append(aa_dict[res_name])
            else:
                sequence.append('X')  # Unknown amino acid
    
    return ''.join(sequence)


def _get_residue_info(chain: Chain) -> List[Dict[str, Any]]:
    """
    Extract residue information from a chain.
    
    Returns:
        List of dictionaries containing residue info
    """
    residue_info = []
    for residue in chain:
        if residue.has_id('CA'):
            residue_info.append({
                'residue_id': residue.get_id()[1],  # Residue number
                'residue_name': residue.get_resname(),
                'chain_id': residue.get_parent().get_id()
            })
    return residue_info


def _direct_rmsd(atoms1: List, atoms2: List) -> float:
    """
    Calculate RMSD between two lists of atoms of the same length.
    """
    if len(atoms1) != len(atoms2):
        raise ValueError("Atom lists must have the same length")
    
    if len(atoms1) == 0:
        raise ValueError("Cannot calculate RMSD for empty atom lists")
    
    # Use BioPython's Superimposer for optimal alignment
    superimposer = Superimposer()
    superimposer.set_atoms(atoms1, atoms2)
    
    return superimposer.rms


def calculate_rmsd_with_alignment_info(pdb_file1: str, pdb_file2: str,
                                     chain_id1: str = None, chain_id2: str = None) -> Tuple[float, dict]:
    """
    Calculate RMSD with additional alignment information.
    
    Returns:
        Tuple of (rmsd_value, alignment_info_dict)
    """
    # Check if files exist
    if not os.path.exists(pdb_file1):
        raise FileNotFoundError(f"PDB file not found: {pdb_file1}")
    if not os.path.exists(pdb_file2):
        raise FileNotFoundError(f"PDB file not found: {pdb_file2}")
    
    try:
        # Parse structures
        parser = PDBParser(QUIET=True)
        structure1 = parser.get_structure("struct1", pdb_file1)
        structure2 = parser.get_structure("struct2", pdb_file2)
        
        # Get chains
        chain1 = _get_chain(structure1, chain_id1)
        chain2 = _get_chain(structure2, chain_id2)
        
        # Get CA atoms
        ca_atoms1 = _get_ca_atoms(chain1)
        ca_atoms2 = _get_ca_atoms(chain2)
        
        # Calculate RMSD
        rmsd = calculate_rmsd(pdb_file1, pdb_file2, chain_id1, chain_id2)
        
        # Collect alignment info
        alignment_info = {
            "rmsd": rmsd,
            "structure1_residues": len(ca_atoms1),
            "structure2_residues": len(ca_atoms2),
            "aligned_residues": min(len(ca_atoms1), len(ca_atoms2)),
            "alignment_method": "structural" if len(ca_atoms1) != len(ca_atoms2) else "direct"
        }
        
        return rmsd, alignment_info
        
    except Exception as e:
        raise ValueError(f"Error calculating RMSD with alignment info: {str(e)}")


def calculate_rmsd_with_sequences(pdb_file1: str, pdb_file2: str,
                                chain_id1: str = None, chain_id2: str = None,
                                alignment_method: str = "structural") -> Tuple[float, Dict[str, Any]]:
    """
    Calculate RMSD with detailed sequence alignment information.
    
    This function returns the RMSD score along with the actual amino acid
    subsequences that were aligned and detailed alignment statistics.
    
    Args:
        pdb_file1: Path to first PDB file
        pdb_file2: Path to second PDB file  
        chain_id1: Chain ID for first structure (auto-detect if None)
        chain_id2: Chain ID for second structure (auto-detect if None)
        alignment_method: Method for handling different lengths ("structural" or "sequence")
        
    Returns:
        Tuple of (rmsd_value, detailed_alignment_info) where detailed_alignment_info contains:
        - aligned_sequence1: Amino acid sequence from structure 1 that was aligned
        - aligned_sequence2: Amino acid sequence from structure 2 that was aligned
        - alignment_start1, alignment_end1: Start and end positions in sequence 1
        - alignment_start2, alignment_end2: Start and end positions in sequence 2
        - alignment_length: Number of residues aligned
        - coverage1, coverage2: Percentage of each structure covered by alignment
        - full_sequence1, full_sequence2: Complete sequences of both structures
        - alignment_method: Method used for alignment
        
    Raises:
        FileNotFoundError: If PDB files don't exist
        ValueError: If structures can't be aligned or have no common residues
    """
    # Check if files exist
    if not os.path.exists(pdb_file1):
        raise FileNotFoundError(f"PDB file not found: {pdb_file1}")
    if not os.path.exists(pdb_file2):
        raise FileNotFoundError(f"PDB file not found: {pdb_file2}")
    
    try:
        # Parse PDB structures
        parser = PDBParser(QUIET=True)
        structure1 = parser.get_structure("struct1", pdb_file1)
        structure2 = parser.get_structure("struct2", pdb_file2)
        
        # Get chains
        chain1 = _get_chain(structure1, chain_id1)
        chain2 = _get_chain(structure2, chain_id2)
        
        # Get full sequences
        full_sequence1 = _get_sequence_from_chain(chain1)
        full_sequence2 = _get_sequence_from_chain(chain2)
        
        if alignment_method == "structural":
            rmsd, alignment_info = _calculate_structural_rmsd_with_alignment(chain1, chain2)
        else:
            # For sequence alignment, use simpler approach
            ca_atoms1 = _get_ca_atoms(chain1)
            ca_atoms2 = _get_ca_atoms(chain2)
            
            if len(ca_atoms1) == 0 or len(ca_atoms2) == 0:
                raise ValueError("No CA atoms found in one or both structures")
            
            # Use the minimum length
            min_length = min(len(ca_atoms1), len(ca_atoms2))
            
            if min_length == 0:
                raise ValueError("No common residues found")
            
            rmsd = _direct_rmsd(ca_atoms1[:min_length], ca_atoms2[:min_length])
            
            alignment_info = {
                'aligned_sequence1': full_sequence1[:min_length],
                'aligned_sequence2': full_sequence2[:min_length],
                'alignment_start1': 0,
                'alignment_end1': min_length,
                'alignment_start2': 0,
                'alignment_end2': min_length,
                'alignment_length': min_length,
                'alignment_method': 'sequence',
                'coverage1': (min_length / len(full_sequence1)) * 100.0 if len(full_sequence1) > 0 else 0.0,
                'coverage2': (min_length / len(full_sequence2)) * 100.0 if len(full_sequence2) > 0 else 0.0,
                'residue_info1': _get_residue_info(chain1)[:min_length],
                'residue_info2': _get_residue_info(chain2)[:min_length]
            }
        
        # Add full sequences and additional info
        alignment_info.update({
            'full_sequence1': full_sequence1,
            'full_sequence2': full_sequence2,
            'full_length1': len(full_sequence1),
            'full_length2': len(full_sequence2),
            'rmsd': round(rmsd, 3),
            'pdb_file1': pdb_file1,
            'pdb_file2': pdb_file2,
            'chain_id1': chain1.get_id(),
            'chain_id2': chain2.get_id()
        })
        
        return round(rmsd, 3), alignment_info
        
    except Exception as e:
        raise ValueError(f"Error calculating RMSD with sequences: {str(e)}")


if __name__ == "__main__":
    # Test the RMSD calculator
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python rmsd_calculator.py <pdb_file1> <pdb_file2>")
        sys.exit(1)
    
    pdb1, pdb2 = sys.argv[1], sys.argv[2]
    
    try:
        rmsd = calculate_rmsd(pdb1, pdb2)
        print(f"RMSD between {pdb1} and {pdb2}: {rmsd:.3f} Å")
        
        # Also show detailed info
        rmsd_detailed, info = calculate_rmsd_with_alignment_info(pdb1, pdb2)
        print(f"Basic alignment info: {info}")
        
        # Show sequence alignment info
        print("\n" + "="*50)
        print("DETAILED SEQUENCE ALIGNMENT INFO")
        print("="*50)
        
        rmsd_seq, seq_info = calculate_rmsd_with_sequences(pdb1, pdb2)
        print(f"RMSD: {rmsd_seq:.3f} Å")
        print(f"Alignment method: {seq_info['alignment_method']}")
        print(f"Alignment length: {seq_info['alignment_length']} residues")
        print(f"Coverage: {seq_info['coverage1']:.1f}% (struct1), {seq_info['coverage2']:.1f}% (struct2)")
        print(f"\nFull sequence 1 ({seq_info['full_length1']} residues): {seq_info['full_sequence1']}")
        print(f"Full sequence 2 ({seq_info['full_length2']} residues): {seq_info['full_sequence2']}")
        print(f"\nAligned sequence 1: {seq_info['aligned_sequence1']}")
        print(f"Aligned sequence 2: {seq_info['aligned_sequence2']}")
        
        if seq_info['alignment_method'] == 'structural_sliding':
            print(f"\nAlignment positions:")
            print(f"  Structure 1: residues {seq_info['alignment_start1']}-{seq_info['alignment_end1']}")
            print(f"  Structure 2: residues {seq_info['alignment_start2']}-{seq_info['alignment_end2']}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1) 