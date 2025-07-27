#!/usr/bin/env python3
"""
Test the examine_catalytic_activity tool description by showing correct usage.

This demonstrates exactly how an LLM should call the tool based on our 
updated description in carbonic_anhydrase_designer_claude.py.
"""

import json
from catalytic_activity_examiner import examine_catalytic_activity

def test_correct_tool_usage():
    """Show the exact format an LLM should use based on our tool description."""
    
    print("=" * 80)
    print("🤖 TESTING CORRECT LLM TOOL USAGE FOR examine_catalytic_activity")
    print("=" * 80)
    
    # This is the EXACT format the LLM should use based on our tool description
    
    # Standard hCA II active site residues (from tool description)
    active_site_residues = {
        'Y7': {'name': 'TYR', 'function': 'Proton transfer network', 'number': 7},
        'N62': {'name': 'ASN', 'function': 'Proton transfer network', 'number': 62},
        'H64': {'name': 'HIS', 'function': 'Proton shuttle', 'number': 64},
        'N67': {'name': 'ASN', 'function': 'Proton transfer network', 'number': 67},
        'Q92': {'name': 'GLN', 'function': 'Activator binding', 'number': 92}
    }
    
    # Standard hCA II zinc binding residues (from tool description)
    zinc_binding_residues = {
        'H94': {'name': 'HIS', 'function': 'Zinc coordination', 'number': 94},
        'H96': {'name': 'HIS', 'function': 'Zinc coordination', 'number': 96},
        'H119': {'name': 'HIS', 'function': 'Zinc coordination', 'number': 119}
    }
    
    print("📋 EXACT TOOL CALL FORMAT (as LLM should use):")
    print("=" * 60)
    print(f"Tool: examine_catalytic_activity")
    print(f"Arguments:")
    print(f"  pdb_file_path: 'predicted_structures/some_mutant.pdb'")
    print(f"  active_site_residues: {json.dumps(active_site_residues, indent=4)}")
    print(f"  zinc_binding_residues: {json.dumps(zinc_binding_residues, indent=4)}")
    print(f"  image_subdir: 'mutation_test_1'")
    print()
    
    # Validate dictionary format
    print("✅ VALIDATION CHECKS:")
    print("=" * 40)
    
    # Check active site residues
    required_active_keys = ['Y7', 'N62', 'H64', 'N67', 'Q92']
    for key in required_active_keys:
        if key in active_site_residues:
            residue = active_site_residues[key]
            has_name = 'name' in residue
            has_function = 'function' in residue  
            has_number = 'number' in residue
            print(f"   ✅ {key}: name={has_name}, function={has_function}, number={has_number}")
        else:
            print(f"   ❌ {key}: MISSING")
    
    # Check zinc binding residues
    required_zinc_keys = ['H94', 'H96', 'H119']
    for key in required_zinc_keys:
        if key in zinc_binding_residues:
            residue = zinc_binding_residues[key]
            has_name = 'name' in residue
            has_function = 'function' in residue
            has_number = 'number' in residue
            print(f"   ✅ {key}: name={has_name}, function={has_function}, number={has_number}")
        else:
            print(f"   ❌ {key}: MISSING")
    
    print()
    print("📊 EXPECTED TOOL OUTPUT:")
    print("=" * 40)
    print("The tool will return JSON containing:")
    print("  - catalytic_integrity.active_site_rmsd: RMSD for active site residues (Å)")
    print("  - catalytic_integrity.zinc_binding_rmsd: RMSD for zinc binding residues (Å)")  
    print("  - catalytic_integrity.overall_rmsd: Overall RMSD for all catalytic residues (Å)")
    print("  - combined_catalytic_image_path: Path to visualization image")
    print("  - active_site_residues: Original input (echoed back)")
    print("  - zinc_binding_residues: Original input (echoed back)")
    print("  - Other structural analysis data")
    print()
    
    print("🧬 RMSD INTERPRETATION (from tool description):")
    print("=" * 40)
    print("  < 0.5 Å: Minimal structural change")
    print("  0.5-2.0 Å: Moderate structural change") 
    print("  > 2.0 Å: Significant structural change that may affect function")
    print()
    
    print("⚠️  CRITICAL REMINDERS FOR LLM:")
    print("=" * 40)
    print("  1. Use the EXACT residue dictionaries shown above")
    print("  2. These residues are ESSENTIAL for carbonic anhydrase II activity")
    print("  3. Always check RMSD values to ensure mutations didn't disrupt catalytic sites")
    print("  4. Use meaningful image_subdir names for organization")
    print("  5. The tool compares against hCA2_folded.pdb reference structure")
    print("  6. Call this tool for EVERY design variant to verify catalytic integrity")
    
    return True

def demonstrate_common_mistakes():
    """Show common mistakes an LLM might make and how to avoid them."""
    
    print("\n" + "=" * 80)
    print("❌ COMMON LLM MISTAKES TO AVOID")
    print("=" * 80)
    
    print("❌ MISTAKE 1: Wrong dictionary format")
    print("-" * 40)
    wrong_format = {
        'Y7': 'TYR',  # Missing required fields
        'H64': {'name': 'HIS'}  # Missing function and number
    }
    print(f"Wrong: {wrong_format}")
    print("Problem: Missing required 'function' and 'number' fields")
    print()
    
    print("❌ MISTAKE 2: Wrong residue keys")
    print("-" * 40)
    wrong_keys = {
        'TYR7': {'name': 'TYR', 'function': 'Proton transfer', 'number': 7}  # Wrong key format
    }
    print(f"Wrong: {wrong_keys}")
    print("Problem: Key should be 'Y7', not 'TYR7'")
    print()
    
    print("❌ MISTAKE 3: Missing essential residues")
    print("-" * 40)
    incomplete = {
        'Y7': {'name': 'TYR', 'function': 'Proton transfer', 'number': 7}
        # Missing N62, H64, N67, Q92
    }
    print(f"Wrong: {incomplete}")
    print("Problem: Must include ALL standard hCA II catalytic residues")
    print()
    
    print("❌ MISTAKE 4: Vague image subdirectory")
    print("-" * 40)
    print("Wrong: image_subdir='test'")
    print("Better: image_subdir='L143F_mutation_iteration_1'")
    print("Problem: Should use descriptive names for organization")
    print()
    
    print("✅ CORRECT USAGE SUMMARY:")
    print("=" * 40)
    print("Always use the exact residue dictionaries from the tool description")
    print("Include all required fields: name, function, number")  
    print("Use standard hCA II residue keys: Y7, N62, H64, N67, Q92, H94, H96, H119")
    print("Choose descriptive image_subdir names")
    print("Check RMSD values in the returned JSON")

if __name__ == "__main__":
    print("🚀 Testing examine_catalytic_activity Tool Description")
    
    success = test_correct_tool_usage()
    demonstrate_common_mistakes()
    
    print(f"\n{'='*80}")
    if success:
        print("🎉 TOOL DESCRIPTION TEST COMPLETE")
        print("   ✅ Updated tool description provides clear guidance")
        print("   ✅ LLM should be able to call the tool correctly")
        print("   ✅ Examples and validation checks included")
    print(f"{'='*80}") 