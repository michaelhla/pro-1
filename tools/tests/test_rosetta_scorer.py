#!/usr/bin/env python3
"""
Test script for the Rosetta scorer tool.
"""

import sys
import os
import tempfile
from protein_folder import fold_protein
from rosetta_scorer import calculate_rosetta_score, RosettaScorer

def test_rosetta_scorer():
    """Test the Rosetta scorer functionality."""
    print("Testing Rosetta Scorer Tool")
    print("=" * 50)
    
    # Test sequence (shorter for quick testing)
    test_sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    
    try:
        # Test 1: First fold a protein to get a PDB file
        print("Test 1: Folding protein for scoring")
        pdb_path = fold_protein(test_sequence, "rosetta_test")
        print(f"✓ Folding successful: {pdb_path}")
        
        # Verify file exists
        if not os.path.exists(pdb_path):
            print("✗ PDB file was not created")
            return False
            
    except Exception as e:
        print(f"✗ Folding test failed: {e}")
        return False
    
    try:
        # Test 2: Score the folded protein
        print("\nTest 2: Scoring with Rosetta")
        result = calculate_rosetta_score(pdb_path)
        print(f"✓ Scoring result: {result}")
        
        # Verify the result contains expected information
        if "Rosetta energy score:" in result and "REU" in result:
            print("✓ Score result contains expected format")
        else:
            print("✗ Score result format unexpected")
            return False
            
    except Exception as e:
        print(f"✗ Scoring test failed: {e}")
        return False
    
    try:
        # Test 3: Direct class usage
        print("\nTest 3: Direct RosettaScorer class usage")
        scorer = RosettaScorer()
        score = scorer.calculate_rosetta_score(pdb_path)
        print(f"✓ Direct scoring successful: {score:.2f} REU")
        
        if isinstance(score, float):
            print("✓ Score is a valid float")
        else:
            print("✗ Score is not a float")
            return False
            
    except Exception as e:
        print(f"✗ Direct scorer test failed: {e}")
        return False
    
    try:
        # Test 4: Error handling for non-existent file
        print("\nTest 4: Error handling")
        fake_path = "non_existent_file.pdb"
        result = calculate_rosetta_score(fake_path)
        
        if "Error" in result:
            print("✓ Error handling works correctly")
        else:
            print("✗ Error handling failed")
            return False
            
    except Exception as e:
        print(f"✓ Error handling caught exception as expected: {type(e).__name__}")
    
    print("\nAll tests completed!")
    return True

def test_with_carbonic_anhydrase():
    """Test with an actual carbonic anhydrase sequence."""
    print("\nTesting with Carbonic Anhydrase Sequence")
    print("=" * 50)
    
    # Human carbonic anhydrase II sequence (partial)
    ca_sequence = "SHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSYDQATSLRILNNGHAFNVEFDDSQDKAVLKGGPLDGTYRLIQFHFHWGSLDGQGSEHTVDKKKYAAELHLVHWNTKYGDFGKAVQQPDGLAVLGIFLKVGSAKPGLQKVVDVLDSIKTKGKSADFTNFDPRGLLPESLDYWTYPGSLTTPPLLECVTWIVLKEPISVSSEQVLKFRKLNFNGEGEPEELMVDNWRPAQPLKNRQIKASFK"
    
    try:
        # Fold the carbonic anhydrase
        print("Folding carbonic anhydrase...")
        pdb_path = fold_protein(ca_sequence, "carbonic_anhydrase_rosetta_test")
        print(f"✓ Folding successful: {pdb_path}")
        
        # Score it with Rosetta
        print("Scoring with Rosetta...")
        result = calculate_rosetta_score(pdb_path)
        print(f"✓ Scoring result: {result}")
        
        # Extract score for analysis
        if "Rosetta energy score:" in result:
            score_part = result.split("Rosetta energy score:")[1].split("REU")[0].strip()
            try:
                score_value = float(score_part)
                print(f"✓ Extracted score: {score_value:.2f} REU")
                
                # Typical protein scores are usually negative and in range -200 to +200
                if -500 < score_value < 500:
                    print("✓ Score is in reasonable range")
                else:
                    print(f"? Score {score_value} is outside typical range (-500 to +500)")
                    
            except ValueError:
                print(f"✗ Could not parse score from: {score_part}")
                return False
        
    except Exception as e:
        print(f"✗ Carbonic anhydrase test failed: {e}")
        return False
    
    return True

def performance_test():
    """Test performance of scoring."""
    print("\nPerformance Test")
    print("-" * 30)
    
    # Use a small sequence for speed
    small_sequence = "MKILVS"
    
    try:
        import time
        
        # Fold
        start_time = time.time()
        pdb_path = fold_protein(small_sequence, "performance_test")
        fold_time = time.time() - start_time
        
        # Score
        start_time = time.time()
        result = calculate_rosetta_score(pdb_path)
        score_time = time.time() - start_time
        
        print(f"✓ Folding time: {fold_time:.2f} seconds")
        print(f"✓ Scoring time: {score_time:.2f} seconds")
        print(f"✓ Total time: {fold_time + score_time:.2f} seconds")
        
    except Exception as e:
        print(f"✗ Performance test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    # Check if we have required dependencies
    try:
        import torch
        import transformers
        import pyrosetta
        print(f"PyTorch version: {torch.__version__}")
        print(f"Transformers version: {transformers.__version__}")
        print(f"PyRosetta available: Yes")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print()
    except ImportError as e:
        print(f"Missing dependencies: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        sys.exit(1)
    
    # Run tests
    success = True
    success &= test_rosetta_scorer()
    success &= test_with_carbonic_anhydrase()
    success &= performance_test()
    
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1) 