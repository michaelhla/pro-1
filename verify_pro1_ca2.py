#!/usr/bin/env python3
"""
Verification script for Pro1 CA2 sequences using ESMFold + Rosetta scoring

This script folds and scores a series of protein sequences to verify their
stability using the protein_folder.py and rosetta_scorer.py tools.
"""

import sys
import os
import time

# Add tools directory to path so we can import our modules
sys.path.append('tools')

from protein_folder import fold_protein
from rosetta_scorer import calculate_rosetta_score


def main():
    """Main function to fold and score all sequences"""
    
    # Define sequences with their expected scores for reference
    sequences = {
        "WT": {
            "sequence": "MSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSYDQATSLRILNNGHAFNVEFDDSQDKAVLKGGPLDGTYRLIQFHFHWGSLDGQGSEHTVDKKKYAAELHLVHWNTKYGDFGKAVQQPDGLAVLGIFLKVGSAKPGLQKVVDVLDSIKTKGKSADFTNFDPRGLLPESLDYWTYPGSLTTPPLLECVTWIVLKEPISVSSEQVLKFRKLNFNGEGEPEELMVDNWRPAQPLKNRQIKASFK",
            "expected_score": -300
        },
        "Creative": {
            "sequence": "MSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSYDQATSLRILNNGHAFNVEFDDSQDKAVLKGGPLDGTYRLIQFHFHWGSLDGQGSEHTVDKKKYAAELHLVHWNTKYGDFGKAVQQPDGLAVLGIFLKPGSAKPGLQKVVDILDSIKTKGKSADFTNFDPRGLLPESLDYWTYPGSLTTPPLLECVTWIVLKEPISVSSEQVLKFRKLNFNGEGEPEELMVDNWRPAQPLKNRQIKASFK",
            "expected_score": -314
        },
        "Regular_1": {
            "sequence": "MEEEEEEELEEEEEMSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSYDQATSLRILNNGHAFNVEFDDSQDKAVLKGGPLDGTYRLIQFHFHWGSLDGQGSEHTVDKKKYAAELHLVHWNTKYGDFGKAVQQPDGLAVLGIFLKVGSAKPGLQKVVDVLDSIKTKGKSADFTNFDPRGLLPESLDYWTYPGSLTTPPLLECVTWIVLKEPISVSSEQVLKFRKLNFNGEGEPEELMVDNWRPAQPLKNRQIKASFK",
            "expected_score": -400
        },
        "Regular_2": {
            "sequence": "MSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSYDQATSLRILNNGHAFNVEFDDSQDKAVLKGGPLDGTYRLIQFHFHWGSLDGQGSEHTVDKKKYAAELHLVHWNTKYGDFGKAVQQPDGLAVLGIFLKVGSAKPGLQKVVDVLDSIKTKGKSADFTNFDPRGLLPESLDYWTYPGSLTTPPLLECVTWIVLKEPISVSSEQVLKFRKSNFNGEGEPEELMVDNWRPAQPLKNRQIKASFKG",
            "expected_score": -400
        },
        "Regular_3": {
            "sequence": "MSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSYDQATSLRILNLGHAFNVEFDDSQDKAVLKGGPLDGTYRLIQFHFHWGSLDGQGSEHTVDKKKYAAELHLVHWNTKYGDFGKAVQQPDGLAVLGIFIKVGSAKPGLQKVVDVLDSIKIKGKSADFTNFDPRGLLPESLDYWTYPGSLTTPPLLECVTWIVLKEPISVSSEQVLKFRKLNFNGEGEPEELMVDNWRPAQPLKNRQIKASFK",
            "expected_score": -400
        },
        "Regular_4": {
            "sequence": "MSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSYDQATSLRILAYLHAFNVEFDDSQDKAVLKGGPLDGTYRLIQFHFHWGSLDGQGSEHTVDKKKYAAELHLVHWNTKYGDFGKAVQQPDGLAVLGIFLKVGSAKPGLQKVVDVLDSIKTKGKSADFTNFDPRGLLPESLDYWTYPGSLTTPPLLECVTWIVLKEPISVSSEQVLKFRKLNFNGEGEPEELMVDNWRPAQPLKNRQIKASFK",
            "expected_score": -400
        }
    }
    
    print("=" * 80)
    print("PROTEIN SEQUENCE FOLDING AND SCORING VERIFICATION")
    print("=" * 80)
    print(f"Processing {len(sequences)} sequences using ESMFold + Rosetta scoring")
    print()
    
    results = {}
    
    for seq_name, seq_data in sequences.items():
        print(f"{'='*60}")
        print(f"PROCESSING: {seq_name}")
        print(f"Expected Score: {seq_data['expected_score']} REU")
        print(f"{'='*60}")
        
        sequence = seq_data["sequence"]
        expected_score = seq_data["expected_score"]
        
        try:
            # Step 1: Fold the protein sequence
            print(f"Step 1: Folding {seq_name} sequence ({len(sequence)} amino acids)...")
            start_time = time.time()
            
            pdb_path = fold_protein(sequence, seq_name)
            fold_time = time.time() - start_time
            
            print(f"✓ Folding completed in {fold_time:.2f}s")
            print(f"  PDB file saved: {pdb_path}")
            
            # Step 2: Score the folded structure
            print(f"Step 2: Scoring {seq_name} structure with Rosetta...")
            score_start = time.time()
            
            score_result = calculate_rosetta_score(pdb_path)
            score_time = time.time() - score_start
            
            print(f"✓ Scoring completed in {score_time:.2f}s")
            print(f"  {score_result}")
            
            # Extract numerical score from result string
            try:
                actual_score = float(score_result.split(": ")[1].split(" ")[0])
                score_diff = actual_score - expected_score
                
                results[seq_name] = {
                    "expected": expected_score,
                    "actual": actual_score,
                    "difference": score_diff,
                    "pdb_path": pdb_path,
                    "fold_time": fold_time,
                    "score_time": score_time
                }
                
                print(f"  Expected: {expected_score} REU")
                print(f"  Actual:   {actual_score:.2f} REU")
                print(f"  Difference: {score_diff:+.2f} REU")
                
            except (IndexError, ValueError) as e:
                print(f"  Warning: Could not parse numerical score from result")
                results[seq_name] = {
                    "expected": expected_score,
                    "actual": None,
                    "difference": None,
                    "pdb_path": pdb_path,
                    "fold_time": fold_time,
                    "score_time": score_time,
                    "error": str(e)
                }
            
        except Exception as e:
            print(f"✗ Error processing {seq_name}: {str(e)}")
            results[seq_name] = {
                "expected": expected_score,
                "actual": None,
                "difference": None,
                "pdb_path": None,
                "error": str(e)
            }
        
        print(f"{'='*60}")
        print()
    
    # Print summary
    print("=" * 80)
    print("SUMMARY RESULTS")
    print("=" * 80)
    
    for seq_name, result in results.items():
        if result.get("actual") is not None:
            print(f"{seq_name:12s}: Expected {result['expected']:6.1f} REU, "
                  f"Actual {result['actual']:6.1f} REU, "
                  f"Diff {result['difference']:+6.1f} REU")
        else:
            print(f"{seq_name:12s}: FAILED - {result.get('error', 'Unknown error')}")
    
    print("=" * 80)
    print("Processing complete!")


if __name__ == "__main__":
    main() 