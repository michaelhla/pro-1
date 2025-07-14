#!/usr/bin/env python3
"""
RMSD Calculator for comparing protein structures.

This module provides a single function to calculate Root Mean Square Deviation (RMSD)
between a hardcoded reference structure (hCA2_folded.pdb) and a newly folded protein structure.
Uses sliding window sequence alignment to find maximum overlap, then calculates RMSD over 
the aligned core region along with overlap percentage.
"""

import os
import warnings
from typing import Tuple, Dict, Any, List
from Bio import PDB
from Bio.PDB import PDBParser, Superimposer
from Bio.PDB.Structure import Structure
from Bio.PDB.Chain import Chain
import numpy as np

# Suppress PDB parsing warnings
warnings.filterwarnings("ignore", category=PDB.PDBConstructionWarning)

# Hardcoded reference structure path and sequence
REFERENCE_PDB = "predicted_structures/hCA2_folded.pdb"
REFERENCE_SEQUENCE = "MSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSYDQATSLRILNNGHAFNVEFDDSQDKAVLKGGPLDGTYRLIQFHFHWGSLDGQGSEHTVDKKKYAAELHLVHWNTKYGDFGKAVQQPDGLAVLGIFLKVGSAKPGLQKVVDVLDSIKTKGKSADFTNFDPRGLLPESLDYWTYPGSLTTPPLLECVTWIVLKEPISVSSEQVLKFRKLNFNGEGEPEELMVDNWRPAQPLKNRQIKASFK"


def calculate_rmsd_with_alignment(pdb_file2: str, 
                                chain_id1: str = None, chain_id2: str = None) -> Tuple[float, Dict[str, Any]]:
    """
    Calculate RMSD between the hardcoded reference structure and a newly folded protein structure.
    
    Uses sliding window sequence alignment to find the region of maximum overlap between
    the two protein sequences, then calculates RMSD over the aligned core region.
    
    Algorithm:
    1. Extract sequences from both PDB structures
    2. If sequences are same length, align directly
    3. Otherwise, use sliding window to find best sequence overlap
    4. Calculate RMSD over the aligned CA atoms in the overlapping region
    5. Return RMSD value and detailed alignment information including overlap percentage
    
    Args:
        pdb_file2: Path to the newly folded PDB file to compare against reference
        chain_id1: Chain ID for reference structure (auto-detect if None)
        chain_id2: Chain ID for new structure (auto-detect if None)
        
    Returns:
        Tuple of (rmsd_value, alignment_info) where alignment_info contains:
        - rmsd: RMSD value in Angstroms
        - overlap_percentage: Percentage of shorter sequence that overlaps
        - alignment_length: Number of residues in aligned region
        - ref_sequence: Full reference sequence
        - new_sequence: Full new structure sequence
        - aligned_ref_sequence: Reference sequence in aligned region
        - aligned_new_sequence: New sequence in aligned region
        - ref_start, ref_end: Start/end positions in reference sequence
        - new_start, new_end: Start/end positions in new sequence
        - sequence_identity: Percentage of identical residues in aligned region
        
    Raises:
        FileNotFoundError: If PDB files don't exist
        ValueError: If structures can't be aligned or have no common residues
    """
    pdb_file1 = REFERENCE_PDB
    
    # Check if files exist
    if not os.path.exists(pdb_file1):
        raise FileNotFoundError(f"Reference PDB file not found: {pdb_file1}")
    if not os.path.exists(pdb_file2):
        raise FileNotFoundError(f"PDB file not found: {pdb_file2}")
    
    try:
        # Parse PDB structures
        parser = PDBParser(QUIET=True)
        structure1 = parser.get_structure("reference", pdb_file1)
        structure2 = parser.get_structure("new_struct", pdb_file2)
        
        # Get chains
        chain1 = _get_chain(structure1, chain_id1)
        chain2 = _get_chain(structure2, chain_id2)
        
        # Extract sequences and CA atoms
        ref_sequence, ref_atoms = _extract_sequence_and_atoms(chain1)
        new_sequence, new_atoms = _extract_sequence_and_atoms(chain2)
        
        if len(ref_atoms) == 0 or len(new_atoms) == 0:
            raise ValueError("No CA atoms found in one or both structures")
        
        # Find best sequence alignment
        alignment_info = _find_best_sequence_alignment(ref_sequence, new_sequence, ref_atoms, new_atoms)
        
        # Calculate RMSD over aligned region
        rmsd = _calculate_rmsd_for_atoms(alignment_info['aligned_ref_atoms'], alignment_info['aligned_new_atoms'])
        
        # Prepare final result
        result = {
            'rmsd': round(rmsd, 3),
            'overlap_percentage': alignment_info['overlap_percentage'],
            'alignment_length': alignment_info['alignment_length'],
            'ref_sequence': ref_sequence,
            'new_sequence': new_sequence,
            'aligned_ref_sequence': alignment_info['aligned_ref_sequence'],
            'aligned_new_sequence': alignment_info['aligned_new_sequence'],
            'ref_start': alignment_info['ref_start'],
            'ref_end': alignment_info['ref_end'],
            'new_start': alignment_info['new_start'],
            'new_end': alignment_info['new_end'],
            'sequence_identity': alignment_info['sequence_identity'],
            'pdb_file1': pdb_file1,
            'pdb_file2': pdb_file2,
            'chain_id1': chain1.get_id(),
            'chain_id2': chain2.get_id()
        }
        
        return round(rmsd, 3), result
        
    except Exception as e:
        raise ValueError(f"Error calculating RMSD with alignment: {str(e)}")


def _get_chain(structure: Structure, chain_id: str = None) -> Chain:
    """Get a chain from a structure, auto-detecting if chain_id is None."""
    chains = list(structure.get_chains())
    
    if not chains:
        raise ValueError("No chains found in structure")
    
    if chain_id is None:
        return chains[0]  # Use first chain
    else:
        for chain in chains:
            if chain.id == chain_id:
                return chain
        raise ValueError(f"Chain {chain_id} not found in structure")


def _extract_sequence_and_atoms(chain: Chain) -> Tuple[str, List]:
    """
    Extract amino acid sequence and corresponding CA atoms from a chain.
    
    Returns:
        Tuple of (sequence_string, ca_atoms_list)
    """
    # Standard amino acid mapping
    aa_dict = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
    }
    
    sequence = []
    ca_atoms = []
    
    for residue in chain:
        if residue.has_id('CA'):  # Only consider residues with CA atoms
            res_name = residue.get_resname()
            if res_name in aa_dict:
                sequence.append(aa_dict[res_name])
                ca_atoms.append(residue['CA'])
            else:
                sequence.append('X')  # Unknown amino acid
                ca_atoms.append(residue['CA'])
    
    return ''.join(sequence), ca_atoms


def _find_best_sequence_alignment(ref_seq: str, new_seq: str, ref_atoms: List, new_atoms: List) -> Dict[str, Any]:
    """
    Find the best sequence alignment using sliding window approach.
    
    Returns alignment information including the best overlapping region.
    """
    ref_len = len(ref_seq)
    new_len = len(new_seq)
    
    # If sequences are the same length, align directly
    if ref_len == new_len:
        # Calculate sequence identity
        identical = sum(1 for a, b in zip(ref_seq, new_seq) if a == b)
        sequence_identity = (identical / ref_len) * 100.0
        
        return {
            'alignment_length': ref_len,
            'overlap_percentage': 100.0,
            'aligned_ref_sequence': ref_seq,
            'aligned_new_sequence': new_seq,
            'aligned_ref_atoms': ref_atoms,
            'aligned_new_atoms': new_atoms,
            'ref_start': 0,
            'ref_end': ref_len,
            'new_start': 0,
            'new_end': new_len,
            'sequence_identity': sequence_identity
        }
    
    # Find the shorter and longer sequences
    if ref_len <= new_len:
        shorter_seq, longer_seq = ref_seq, new_seq
        shorter_atoms, longer_atoms = ref_atoms, new_atoms
        ref_is_shorter = True
    else:
        shorter_seq, longer_seq = new_seq, ref_seq
        shorter_atoms, longer_atoms = new_atoms, ref_atoms
        ref_is_shorter = False
    
    shorter_len = len(shorter_seq)
    longer_len = len(longer_seq)
    
    # Sliding window to find best alignment
    best_identity = -1
    best_start = 0
    
    for start in range(longer_len - shorter_len + 1):
        segment = longer_seq[start:start + shorter_len]
        
        # Calculate sequence identity for this alignment
        identical = sum(1 for a, b in zip(shorter_seq, segment) if a == b)
        identity = (identical / shorter_len) * 100.0
        
        if identity > best_identity:
            best_identity = identity
            best_start = start
    
    # Extract the best alignment
    best_end = best_start + shorter_len
    aligned_longer_seq = longer_seq[best_start:best_end]
    aligned_longer_atoms = longer_atoms[best_start:best_end]
    
    # Calculate overlap percentage (relative to shorter sequence)
    overlap_percentage = (shorter_len / min(ref_len, new_len)) * 100.0
    
    # Prepare results based on which sequence was shorter
    if ref_is_shorter:
        result = {
            'alignment_length': shorter_len,
            'overlap_percentage': overlap_percentage,
            'aligned_ref_sequence': shorter_seq,
            'aligned_new_sequence': aligned_longer_seq,
            'aligned_ref_atoms': shorter_atoms,
            'aligned_new_atoms': aligned_longer_atoms,
            'ref_start': 0,
            'ref_end': shorter_len,
            'new_start': best_start,
            'new_end': best_end,
            'sequence_identity': best_identity
        }
    else:
        result = {
            'alignment_length': shorter_len,
            'overlap_percentage': overlap_percentage,
            'aligned_ref_sequence': aligned_longer_seq,
            'aligned_new_sequence': shorter_seq,
            'aligned_ref_atoms': aligned_longer_atoms,
            'aligned_new_atoms': shorter_atoms,
            'ref_start': best_start,
            'ref_end': best_end,
            'new_start': 0,
            'new_end': shorter_len,
            'sequence_identity': best_identity
        }
    
    return result


def _calculate_rmsd_for_atoms(atoms1: List, atoms2: List) -> float:
    """Calculate RMSD between two lists of atoms using optimal superposition."""
    if len(atoms1) != len(atoms2):
        raise ValueError("Atom lists must have the same length")
    
    if len(atoms1) == 0:
        raise ValueError("Cannot calculate RMSD for empty atom lists")
    
    # Use BioPython's Superimposer for optimal alignment
    superimposer = Superimposer()
    superimposer.set_atoms(atoms1, atoms2)
    
    return superimposer.rms


if __name__ == "__main__":
    # Test the RMSD calculator
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python rmsd_calculator.py <newly_folded_pdb_file>")
        print(f"Reference structure: {REFERENCE_PDB}")
        print(f"Reference sequence: {REFERENCE_SEQUENCE}")
        sys.exit(1)
    
    pdb2 = sys.argv[1]
    
    try:
        rmsd, alignment_info = calculate_rmsd_with_alignment(pdb2)
        
        print(f"RMSD between reference ({REFERENCE_PDB}) and {pdb2}: {rmsd:.3f} Å")
        print(f"Alignment length: {alignment_info['alignment_length']} residues")
        print(f"Overlap percentage: {alignment_info['overlap_percentage']:.1f}%")
        print(f"Sequence identity: {alignment_info['sequence_identity']:.1f}%")
        
        print(f"\nReference sequence ({len(alignment_info['ref_sequence'])} residues):")
        print(f"{alignment_info['ref_sequence']}")
        print(f"\nNew structure sequence ({len(alignment_info['new_sequence'])} residues):")
        print(f"{alignment_info['new_sequence']}")
        
        print(f"\nAligned reference sequence (positions {alignment_info['ref_start']}-{alignment_info['ref_end']}):")
        print(f"{alignment_info['aligned_ref_sequence']}")
        print(f"\nAligned new structure sequence (positions {alignment_info['new_start']}-{alignment_info['new_end']}):")
        print(f"{alignment_info['aligned_new_sequence']}")
        
        # Verify reference sequence matches
        if alignment_info['ref_sequence'] == REFERENCE_SEQUENCE:
            print(f"\n✓ Reference sequence matches hardcoded sequence")
        else:
            print(f"\n⚠ Warning: Reference sequence doesn't match hardcoded sequence")
            print(f"Expected: {REFERENCE_SEQUENCE}")
            print(f"Found:    {alignment_info['ref_sequence']}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1) 