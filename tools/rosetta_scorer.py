#!/usr/bin/env python3
"""
Rosetta Scoring Tool

Simple tool that scores protein structures using PyRosetta and returns energy scores.
Based on the scoring logic from stability_reward.py.
"""

import os
import time
import pyrosetta_installer
pyrosetta_installer.install_pyrosetta()
import pyrosetta


class RosettaScorer:
    """
    Simple Rosetta scorer for protein structures.
    """
    
    def __init__(self):
        # Initialize PyRosetta
        pyrosetta.init()
        print("PyRosetta initialized for scoring")

    def calculate_rosetta_score(self, pdb_file_path: str) -> float:
        """
        Calculate Rosetta stability score for a given PDB file.
        
        Args:
            pdb_file_path: Path to the PDB file to score
            
        Returns:
            Rosetta energy score (lower is more stable)
        """
        start_time = time.time()
        
        # Validate file exists
        if not os.path.exists(pdb_file_path):
            raise FileNotFoundError(f"PDB file not found: {pdb_file_path}")
        
        try:
            # Load structure into PyRosetta
            pose = pyrosetta.pose_from_pdb(pdb_file_path)

            # Create score function
            scorefxn = pyrosetta.get_fa_scorefxn()
            
            # Setup packer task for side chain optimization
            task = pyrosetta.standard_packer_task(pose)
            task.restrict_to_repacking()  # Only repack side chains, don't change sequence

            # Optimize side chain conformations
            packer = pyrosetta.rosetta.protocols.minimization_packing.PackRotamersMover(scorefxn, task)
            packer.apply(pose)

            # Perform energy minimization
            min_mover = pyrosetta.rosetta.protocols.minimization_packing.MinMover()
            min_mover.score_function(scorefxn)
            min_mover.apply(pose)

            # Calculate final stability score
            stability_score = scorefxn(pose)

            calculation_time = time.time() - start_time
            print(f"Rosetta scoring took {calculation_time:.2f} seconds")
            print(f"Stability score: {stability_score}")
            
            return stability_score
            
        except Exception as e:
            raise RuntimeError(f"Rosetta scoring failed for {pdb_file_path}: {str(e)}")


# Global instance
_rosetta_scorer = None


def get_rosetta_scorer():
    """Get or create the global Rosetta scorer instance."""
    global _rosetta_scorer
    if _rosetta_scorer is None:
        _rosetta_scorer = RosettaScorer()
    return _rosetta_scorer


def calculate_rosetta_score(pdb_file_path: str) -> str:
    """
    Calculate Rosetta energy score for a PDB file.
    
    Args:
        pdb_file_path: Path to the PDB file to score
        
    Returns:
        String describing the score and details
    """
    try:
        scorer = get_rosetta_scorer()
        score = scorer.calculate_rosetta_score(pdb_file_path)
        
        return f"Rosetta energy score: {score:.2f} REU (Rosetta Energy Units). Lower scores indicate more stable structures. File: {pdb_file_path}"
        
    except Exception as e:
        return f"Error calculating Rosetta score for {pdb_file_path}: {str(e)}"


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python rosetta_scorer.py <pdb_file_path>")
        sys.exit(1)
    
    pdb_path = sys.argv[1]
    
    try:
        result = calculate_rosetta_score(pdb_path)
        print(result)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1) 