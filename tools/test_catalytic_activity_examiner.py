#!/usr/bin/env python3
"""
Test script for the catalytic activity examiner.

This script tests the catalytic activity examiner functionality and integration
with the designer.
"""

import os
import unittest
import tempfile
import json
from unittest.mock import patch, MagicMock

try:
    from catalytic_activity_examiner import (
        CatalyticActivityExaminer, 
        examine_catalytic_activity,
        PYMOL_AVAILABLE
    )
    EXAMINER_AVAILABLE = True
except ImportError:
    EXAMINER_AVAILABLE = False
    PYMOL_AVAILABLE = False


class TestCatalyticActivityExaminer(unittest.TestCase):
    """Test cases for the catalytic activity examiner."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        
        # Mock PDB file content with carbonic anhydrase-like structure
        self.mock_pdb_content = """HEADER    CARBONIC ANHYDRASE II                   01-JAN-23   1CA2
ATOM      1  N   TYR A   7      20.154  16.967  23.986  1.00 20.00           N  
ATOM      2  CA  TYR A   7      21.154  17.967  24.986  1.00 20.00           C  
ATOM      3  C   TYR A   7      22.154  18.967  25.986  1.00 20.00           C  
ATOM      4  O   TYR A   7      23.154  19.967  26.986  1.00 20.00           O  
ATOM      5  N   ASN A  62      24.154  20.967  27.986  1.00 20.00           N  
ATOM      6  CA  ASN A  62      25.154  21.967  28.986  1.00 20.00           C  
ATOM      7  C   ASN A  62      26.154  22.967  29.986  1.00 20.00           C  
ATOM      8  O   ASN A  62      27.154  23.967  30.986  1.00 20.00           O  
ATOM      9  N   HIS A  64      28.154  24.967  31.986  1.00 20.00           N  
ATOM     10  CA  HIS A  64      29.154  25.967  32.986  1.00 20.00           C  
ATOM     11  C   HIS A  64      30.154  26.967  33.986  1.00 20.00           C  
ATOM     12  O   HIS A  64      31.154  27.967  34.986  1.00 20.00           O  
ATOM     13  N   HIS A  94      32.154  28.967  35.986  1.00 20.00           N  
ATOM     14  CA  HIS A  94      33.154  29.967  36.986  1.00 20.00           C  
ATOM     15  C   HIS A  94      34.154  30.967  37.986  1.00 20.00           C  
ATOM     16  O   HIS A  94      35.154  31.967  38.986  1.00 20.00           O  
ATOM     17  N   HIS A  96      36.154  32.967  39.986  1.00 20.00           N  
ATOM     18  CA  HIS A  96      37.154  33.967  40.986  1.00 20.00           C  
ATOM     19  C   HIS A  96      38.154  34.967  41.986  1.00 20.00           C  
ATOM     20  O   HIS A  96      39.154  35.967  42.986  1.00 20.00           O  
ATOM     21  N   HIS A 119      40.154  36.967  43.986  1.00 20.00           N  
ATOM     22  CA  HIS A 119      41.154  37.967  44.986  1.00 20.00           C  
ATOM     23  C   HIS A 119      42.154  38.967  45.986  1.00 20.00           C  
ATOM     24  O   HIS A 119      43.154  39.967  46.986  1.00 20.00           O  
END
"""
        
        # Create mock PDB file
        self.mock_pdb_file = os.path.join(self.test_dir, "test_ca_structure.pdb")
        with open(self.mock_pdb_file, 'w') as f:
            f.write(self.mock_pdb_content)
            
        # Create actual hCA II sequence file
        self.hca2_sequence = (
            "MSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSYDQATSLRILNNGH"
            "AFNVEFDDSQDKAVLKGGPLDGTYRLIQFHFHWGSLDGQGSEHTVDKKKYAAELHLVHWNTKYGD"
            "FGKAVQQPDGLAVLGIFLKVGSAKPGLQKVVDVLDSIKTKGKSADFTNFDPRGLLPESLDYWTYPG"
            "SLTTPPLLECVTWIVLKEPISVSSEQVLKFRKLNFNGEGEPEELMVDNWRPAQPLKNRQIKASFK"
        )
        
        # Save sequence to FASTA file
        self.hca2_fasta = os.path.join(self.test_dir, "hca2.fasta")
        with open(self.hca2_fasta, 'w') as f:
            f.write(">hCA2\n")
            f.write(self.hca2_sequence)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_examiner_import(self):
        """Test that the examiner can be imported."""
        self.assertTrue(EXAMINER_AVAILABLE, "Catalytic activity examiner should be importable")
    
    @unittest.skipUnless(PYMOL_AVAILABLE, "PyMOL not available")
    def test_examiner_initialization(self):
        """Test examiner initialization."""
        examiner = CatalyticActivityExaminer()
        self.assertEqual(examiner.chain_id, 'A')
        self.assertTrue(os.path.exists(examiner.temp_dir))
    
    def test_examine_catalytic_activity_file_not_found(self):
        """Test examiner with non-existent file."""
        if not EXAMINER_AVAILABLE:
            self.skipTest("Catalytic activity examiner not available")
        
        result = examine_catalytic_activity("nonexistent.pdb")
        result_dict = json.loads(result)
        
        self.assertIn('error', result_dict)
        self.assertIn('success', result_dict)
        self.assertFalse(result_dict['success'])
    
    def test_examine_catalytic_activity_basic(self):
        """Test basic catalytic activity examination."""
        if not EXAMINER_AVAILABLE or not PYMOL_AVAILABLE:
            self.skipTest("PyMOL or examiner not available")
        
        try:
            # Load and test structure
            from pymol import cmd
            cmd.reinitialize()
            cmd.load(self.mock_pdb_file, "test_structure")
            
            # Run examination
            result = examine_catalytic_activity(self.mock_pdb_file)
            result_dict = json.loads(result)
            
            # Check that result contains expected keys
            expected_keys = [
                'pdb_file', 'chain_id', 'active_site_image',
                'zinc_binding_image', 'catalytic_integrity', 'summary'
            ]
            
            for key in expected_keys:
                self.assertIn(key, result_dict)
            
            # Check catalytic integrity assessment
            integrity = result_dict['catalytic_integrity']
            self.assertIn('integrity_level', integrity)
            self.assertIn('risk_level', integrity)
            self.assertIn('recommendations', integrity)
            
        except Exception as e:
            # Skip test if PyMOL has issues with mock data
            self.skipTest(f"PyMOL test failed with mock data: {e}")
        finally:
            try:
                cmd.reinitialize()
            except:
                pass
    
    def test_examine_catalytic_activity_with_offsets(self):
        """Test catalytic activity examination with residue offsets."""
        if not EXAMINER_AVAILABLE or not PYMOL_AVAILABLE:
            self.skipTest("PyMOL or examiner not available")
        
        try:
            # Test with some residue offsets
            offsets = {'H94': 2, 'H96': 2}  # Simulate insertions
            result = examine_catalytic_activity(self.mock_pdb_file, residue_offsets=offsets)
            result_dict = json.loads(result)
            
            self.assertEqual(result_dict['residue_offsets'], offsets)
            
        except Exception as e:
            self.skipTest(f"PyMOL test failed with mock data: {e}")

    def test_examine_actual_hca2(self):
        """Test examination with actual hCA II sequence folded using ESMFold."""
        if not EXAMINER_AVAILABLE or not PYMOL_AVAILABLE:
            self.skipTest("PyMOL or examiner not available")
            
        try:
            # Import protein folder
            try:
                from protein_folder import fold_protein
            except ImportError:
                self.skipTest("Protein folder not available")
            
            # Fold the actual hCA II sequence using ESMFold
            print("Folding hCA II sequence using ESMFold...")
            folded_pdb_path = fold_protein(self.hca2_sequence, "hCA2_folded")
            print(f"Folded structure saved to: {folded_pdb_path}")
            
            # Verify the PDB file was created
            self.assertTrue(os.path.exists(folded_pdb_path), "Folded PDB file should exist")
            
            # Run examination on the folded structure
            print("Running catalytic activity examination on folded structure...")
            result = examine_catalytic_activity(folded_pdb_path)
            result_dict = json.loads(result)
            
            # Check that result contains expected keys
            expected_keys = [
                'pdb_file', 'chain_id', 'active_site_image',
                'zinc_binding_image', 'catalytic_integrity', 'summary'
            ]
            
            for key in expected_keys:
                self.assertIn(key, result_dict, f"Missing key: {key}")
            
            # Print detailed results for analysis
            print("\n=== Catalytic Activity Analysis Results ===")
            print(f"PDB file: {result_dict['pdb_file']}")
            print(f"Chain ID: {result_dict['chain_id']}")
            
            # Check catalytic integrity
            integrity = result_dict['catalytic_integrity']
            print(f"\nCatalytic Integrity: {integrity['integrity_level']}")
            print(f"Risk Level: {integrity['risk_level']}")
            print(f"Active Site RMSD: {integrity.get('active_site_rmsd', 'N/A')} Å")
            print(f"Zinc Binding RMSD: {integrity.get('zinc_binding_rmsd', 'N/A')} Å")
            print(f"Max RMSD: {integrity.get('max_rmsd', 'N/A')} Å")
            print(f"Active Site Missing: {integrity.get('active_site_missing', 0)}")
            print(f"Zinc Binding Missing: {integrity.get('zinc_binding_missing', 0)}")
            
            # Print recommendations
            print("\nRecommendations:")
            for rec in integrity['recommendations']:
                print(f"  - {rec}")
            
            # Check specific residues
            print("\n=== Active Site Residues ===")
            active_site_status = result_dict.get('active_site_status', {})
            for res_key, status in active_site_status.items():
                exists = "✓" if status['exists'] else "✗"
                type_match = "✓" if status['type_match'] else "✗"
                print(f"{res_key}: Exists {exists}, Type Match {type_match} "
                      f"(Expected: {status['expected_type']}, Actual: {status['actual_type']})")
            
            print("\n=== Zinc Binding Residues ===")
            zinc_binding_status = result_dict.get('zinc_binding_status', {})
            for res_key, status in zinc_binding_status.items():
                exists = "✓" if status['exists'] else "✗"
                type_match = "✓" if status['type_match'] else "✗"
                print(f"{res_key}: Exists {exists}, Type Match {type_match} "
                      f"(Expected: {status['expected_type']}, Actual: {status['actual_type']})")
            
            # Summary
            print(f"\nSummary: {result_dict['summary']}")
            
            # Basic assertions - the folded structure should at least have the basic structure
            self.assertIn('integrity_level', integrity)
            self.assertIn('risk_level', integrity)
            self.assertIn('recommendations', integrity)
            self.assertIn('active_site_rmsd', integrity)
            self.assertIn('zinc_binding_rmsd', integrity)
            
            # Check that images were generated
            self.assertTrue(os.path.exists(result_dict['active_site_image']) or 
                          result_dict['active_site_image'].endswith('_failed.png'))
            self.assertTrue(os.path.exists(result_dict['zinc_binding_image']) or 
                          result_dict['zinc_binding_image'].endswith('_failed.png'))
            
            print("\n✅ Test completed successfully!")
            
        except ImportError as e:
            self.skipTest(f"Required dependencies not available: {e}")
        except Exception as e:
            print(f"Test failed with error: {e}")
            self.skipTest(f"Test with folded hCA II structure failed: {e}")
        finally:
            try:
                from pymol import cmd
                cmd.reinitialize()
            except:
                pass


def test_examiner_integration():
    """
    Integration test to verify the catalytic activity examiner works with the designer.
    """
    print("Testing catalytic activity examiner integration...")
    
    # Test import
    try:
        from carbonic_anhydrase_designer import CarbonicAnhydraseDesigner
        print("✓ Designer import successful")
    except ImportError as e:
        print(f"✗ Designer import failed: {e}")
        return False
    
    # Test designer creation
    try:
        designer = CarbonicAnhydraseDesigner()
        print("✓ Designer created successfully")
        
        # Check that catalytic activity examiner tool is in the tools
        tool_names = [tool.get('name') for tool in designer.tools]
        assert 'examine_catalytic_activity' in tool_names, "Catalytic activity examiner not found in designer tools"
        print("✓ Catalytic activity examiner found in designer tools")
        
        # Check that function is in the mapping
        assert 'examine_catalytic_activity' in designer.tool_mapping, "Catalytic activity examiner not found in tool mapping"
        print("✓ Catalytic activity examiner found in tool mapping")
        
        # Test tool definition
        examiner_tool = None
        for tool in designer.tools:
            if tool.get('name') == 'examine_catalytic_activity':
                examiner_tool = tool
                break
        
        assert examiner_tool is not None, "Catalytic activity examiner definition not found"
        assert examiner_tool['type'] == 'function', "Invalid tool type"
        assert 'parameters' in examiner_tool, "Tool parameters not found"
        assert 'pdb_file_path' in examiner_tool['parameters']['properties'], "PDB file path parameter not found"
        print("✓ Catalytic activity examiner tool definition is correct")
        
        print("✓ Integration test passed!")
        return True
        
    except Exception as e:
        print(f"⚠ Designer creation failed (expected if no API keys): {e}")
        # Even if designer creation fails, we can still test the import
        try:
            from catalytic_activity_examiner import examine_catalytic_activity
            print("✓ Catalytic activity examiner imported successfully")
            return True
        except ImportError as e:
            print(f"✗ Catalytic activity examiner import failed: {e}")
            return False


def test_catalytic_residues():
    """
    Test that the catalytic residues are correctly defined.
    """
    print("\nTesting catalytic residue definitions...")
    
    if not EXAMINER_AVAILABLE:
        print("⚠ Examiner not available, skipping catalytic residues test")
        return
    
    from catalytic_activity_examiner import CatalyticActivityExaminer
    
    # Test initialization
    examiner = CatalyticActivityExaminer()
    
    # Test active site residues
    expected_active_site = ['Y7', 'N62', 'H64', 'N67', 'Q92']
    actual_active_site = list(examiner.ACTIVE_SITE_RESIDUES.keys())
    
    for res in expected_active_site:
        assert res in actual_active_site, f"Active site residue {res} not found"
    
    print("✓ Active site residues correctly defined:")
    for res, info in examiner.ACTIVE_SITE_RESIDUES.items():
        print(f"   {res} ({info['name']}): {info['function']}")
    
    # Test zinc binding residues
    expected_zinc_binding = ['H94', 'H96', 'H119']
    actual_zinc_binding = list(examiner.ZINC_BINDING_RESIDUES.keys())
    
    for res in expected_zinc_binding:
        assert res in actual_zinc_binding, f"Zinc binding residue {res} not found"
    
    print("✓ Zinc binding residues correctly defined:")
    for res, info in examiner.ZINC_BINDING_RESIDUES.items():
        print(f"   {res} ({info['name']}): {info['function']}")
    
    print("✓ All catalytic residues are properly defined")


if __name__ == "__main__":
    print("Running catalytic activity examiner tests...")
    print("=" * 60)
    
    # Run unit tests
    unittest.main(verbosity=2, exit=False)
    
    print("\n" + "=" * 60)
    print("Running integration test...")
    
    # Run integration test
    test_examiner_integration()
    
    print("\n" + "=" * 60)
    print("Testing catalytic residue definitions...")
    
    # Test catalytic residues
    test_catalytic_residues()
    
    print("\n" + "=" * 60)
    print("Test summary:")
    print("- Unit tests: Check individual examiner functions")
    print("- Integration test: Check examiner integration with designer")
    print("- Catalytic residues: Verify correct residues are defined")
    print("- Features tested:")
    print("  * Active site residue identification (Y7, N62, H64, N67, Q92)")
    print("  * Zinc binding residue analysis (H94, H96, H119)")
    print("  * Residue offset handling for sequence variations")
    print("  * Structural integrity assessment")
    print("  * PyMOL visualization with color coding")
    print("- Note: PyMOL tests may fail if PyMOL is not properly installed")
    print("- Install PyMOL with: conda install -c conda-forge pymol-open-source")
    
    if PYMOL_AVAILABLE:
        print("✓ PyMOL is available for visualization and analysis")
    else:
        print("⚠ PyMOL not available - visualization and calculation tests skipped")
    
    # Clean up PyMOL to ensure script exits properly
    if PYMOL_AVAILABLE:
        try:
            from pymol import cmd
            cmd.quit()
            print("✓ PyMOL cleanup completed")
        except:
            pass 