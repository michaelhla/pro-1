#!/usr/bin/env python3
"""
Test script for the simplified carbonic anhydrase designer with only fold_protein tool.
"""

import sys
import os
from carbonic_anhydrase_designer import CarbonicAnhydraseDesigner

def test_simplified_designer():
    """Test the simplified designer functionality."""
    print("Testing Simplified Carbonic Anhydrase Designer")
    print("=" * 60)
    
    # Check if API key is available
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY not set. Cannot test o3 integration.")
        print("✅ Testing tool configuration only...")
        
        # Test tool initialization without calling o3
        try:
            designer = CarbonicAnhydraseDesigner()
            print(f"✅ Designer initialized successfully")
            print(f"✅ Available tools: {list(designer.tool_mapping.keys())}")
            print(f"✅ Tool count: {len(designer.tools)}")
            
            # Verify we have fold_protein and calculate_rosetta_score
            tool_names = [tool["name"] for tool in designer.tools]
            expected_tools = ["fold_protein", "calculate_rosetta_score"]
            
            if len(designer.tools) == 2 and all(tool in tool_names for tool in expected_tools):
                print("✅ Correctly includes fold_protein and calculate_rosetta_score tools")
            else:
                print(f"❌ Expected {expected_tools}, got: {tool_names}")
                return False
                
        except Exception as e:
            print(f"❌ Designer initialization failed: {e}")
            return False
    else:
        print("✅ OPENAI_API_KEY found. Testing full functionality...")
        
        # Test with API key available
        try:
            designer = CarbonicAnhydraseDesigner(reasoning_effort="low")
            print("✅ Designer initialized with API key")
            
            # Test tool mapping
            print(f"✅ Available tools: {list(designer.tool_mapping.keys())}")
            
            # Test function call execution (without calling o3)
            from protein_folder import fold_protein
            from rosetta_scorer import calculate_rosetta_score
            
            # Test sequence
            test_sequence = "MKILVS"
            pdb_path = fold_protein(test_sequence, "test_simple")
            print(f"✅ Direct fold_protein call successful: {pdb_path}")
            
            # Test scoring
            score_result = calculate_rosetta_score(pdb_path)
            print(f"✅ Direct calculate_rosetta_score call successful: {score_result[:100]}...")
            
        except Exception as e:
            print(f"❌ Full functionality test failed: {e}")
            return False
    
    print("\n🎉 All tests passed! Designer with 2 tools is working correctly.")
    return True

def show_designer_info():
    """Show information about the simplified designer."""
    print("\nSimplified Designer Information")
    print("-" * 40)
    
    try:
        designer = CarbonicAnhydraseDesigner()
        
        print(f"Model: {designer.model_config['model']}")
        print(f"Reasoning effort: {designer.model_config['reasoning']['effort']}")
        print(f"Number of tools: {len(designer.tools)}")
        
        for i, tool in enumerate(designer.tools, 1):
            print(f"\nTool {i}: {tool['name']}")
            print(f"  Description: {tool['description']}")
            print(f"  Required params: {tool['parameters']['required']}")
            print(f"  All params: {list(tool['parameters']['properties'].keys())}")
        
    except Exception as e:
        print(f"Error getting designer info: {e}")

if __name__ == "__main__":
    # Check dependencies
    try:
        import torch
        import transformers
        from openai import OpenAI
        print(f"✅ Dependencies available")
        print(f"   PyTorch: {torch.__version__}")
        print(f"   Transformers: {transformers.__version__}")
        print(f"   CUDA: {torch.cuda.is_available()}")
        print()
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        sys.exit(1)
    
    # Run tests
    success = test_simplified_designer()
    show_designer_info()
    
    if not success:
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("🚀 Ready to use carbonic anhydrase designer with 2 tools!")
    print("   Tools: fold_protein + calculate_rosetta_score")
    print("   Run: python carbonic_anhydrase_designer.py")
    print("=" * 60) 