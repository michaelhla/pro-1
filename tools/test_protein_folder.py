#!/usr/bin/env python3
"""
Test script for the protein folder tool integration.
"""

import sys
import os
from protein_folder import ProteinFolder, fold_protein

def test_protein_folder():
    """Test the protein folder functionality."""
    print("Testing Protein Folder Tool")
    print("=" * 50)
    
    # Test sequence (shorter for quick testing)
    test_sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    
    try:
        # Test 1: Basic folding
        print("Test 1: Basic protein folding")
        folder = ProteinFolder()
        pdb_path = folder.predict_structure(test_sequence, "test_protein")
        print(f"✓ Folding successful: {pdb_path}")
        
        # Verify file exists
        if os.path.exists(pdb_path):
            print(f"✓ PDB file created: {pdb_path}")
            
            # Check file content
            with open(pdb_path, 'r') as f:
                content = f.read()
                if content.startswith('ATOM'):
                    print("✓ PDB file contains valid structure data")
                else:
                    print("✗ PDB file content appears invalid")
        else:
            print("✗ PDB file was not created")
            
    except Exception as e:
        print(f"✗ Basic folding test failed: {e}")
        return False
    
    try:
        # Test 2: Integration with fold_protein function
        print("\nTest 2: Integration with fold_protein function")
        result = fold_protein(test_sequence, "integration_test")
        print(f"✓ Folding result: {result}")
        
    except Exception as e:
        print(f"✗ Folding integration test failed: {e}")
        return False
    
    try:
        # Test 3: Caching behavior
        print("\nTest 3: Caching behavior")
        import time
        
        start_time = time.time()
        pdb_path1 = folder.predict_structure(test_sequence, "test_cache")
        first_time = time.time() - start_time
        
        start_time = time.time()
        pdb_path2 = folder.predict_structure(test_sequence, "test_cache")
        second_time = time.time() - start_time
        
        if pdb_path1 == pdb_path2 and second_time < first_time:
            print(f"✓ Caching works: {first_time:.2f}s vs {second_time:.2f}s")
        else:
            print(f"? Caching behavior: {first_time:.2f}s vs {second_time:.2f}s")
            
    except Exception as e:
        print(f"✗ Caching test failed: {e}")
        
    print("\nAll tests completed!")
    return True

def test_with_carbonic_anhydrase():
    """Test with an actual carbonic anhydrase sequence."""
    print("\nTesting with Carbonic Anhydrase Sequence")
    print("=" * 50)
    
    # Human carbonic anhydrase II sequence (partial)
    ca_sequence = "SHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSYDQATSLRILNNGHAFNVEFDDSQDKAVLKGGPLDGTYRLIQFHFHWGSLDGQGSEHTVDKKKYAAELHLVHWNTKYGDFGKAVQQPDGLAVLGIFLKVGSAKPGLQKVVDVLDSIKTKGKSADFTNFDPRGLLPESLDYWTYPGSLTTPPLLECVTWIVLKEPISVSSEQVLKFRKLNFNGEGEPEELMVDNWRPAQPLKNRQIKASFK"
    
    try:
        folder = ProteinFolder()
        pdb_path = folder.predict_structure(ca_sequence, "carbonic_anhydrase_test")
        print(f"✓ Carbonic anhydrase folding successful")
        print(f"  Length: {len(ca_sequence)} residues")
        print(f"  File: {pdb_path}")
        
        # Test fold_protein function
        result = fold_protein(ca_sequence, "carbonic_anhydrase_fold_test")
        print(f"✓ Folding function result: {result}")
        
    except Exception as e:
        print(f"✗ Carbonic anhydrase test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    # Check if we have required dependencies
    try:
        import torch
        import transformers
        print(f"PyTorch version: {torch.__version__}")
        print(f"Transformers version: {transformers.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print()
    except ImportError as e:
        print(f"Missing dependencies: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        sys.exit(1)
    
    # Run tests
    success = True
    success &= test_protein_folder()
    success &= test_with_carbonic_anhydrase()
    
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1) 