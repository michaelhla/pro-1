#!/usr/bin/env python3
"""
Test suite for RMSD calculation with rotation using Kabsch algorithm.

This module tests the _kabsch_rmsd method from CatalyticActivityExaminer
to ensure proper implementation of the Kabsch algorithm for optimal
superposition and RMSD calculation.
"""

import numpy as np
import unittest
import sys
import os

# Add the tools directory to path to import the examiner
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from catalytic_activity_examiner import CatalyticActivityExaminer
except ImportError:
    print("Error: Could not import CatalyticActivityExaminer")
    sys.exit(1)


class TestKabschRMSD(unittest.TestCase):
    """Test cases for Kabsch RMSD calculation."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create an examiner instance for testing
        # We don't need PyMOL for just testing the Kabsch algorithm
        self.examiner = CatalyticActivityExaminer.__new__(CatalyticActivityExaminer)
        
    def test_identical_coordinates(self):
        """Test RMSD calculation for identical coordinate sets."""
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        
        rmsd = self.examiner._kabsch_rmsd(coords, coords)
        self.assertAlmostEqual(rmsd, 0.0, places=10, 
                              msg="RMSD of identical coordinates should be 0")
    
    def test_simple_translation(self):
        """Test RMSD calculation for coordinates differing only by translation."""
        coords1 = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        
        # Translate by (2, 3, 4)
        coords2 = coords1 + np.array([2.0, 3.0, 4.0])
        
        rmsd = self.examiner._kabsch_rmsd(coords1, coords2)
        self.assertAlmostEqual(rmsd, 0.0, places=10,
                              msg="RMSD after pure translation should be 0")
    
    def test_simple_rotation(self):
        """Test RMSD calculation for coordinates differing only by rotation."""
        coords1 = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0]
        ])
        
        # Rotate 90 degrees around z-axis
        rotation_matrix = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
        
        coords2 = coords1 @ rotation_matrix.T
        
        rmsd = self.examiner._kabsch_rmsd(coords1, coords2)
        self.assertAlmostEqual(rmsd, 0.0, places=10,
                              msg="RMSD after pure rotation should be 0")
    
    def test_rotation_and_translation(self):
        """Test RMSD calculation for coordinates with both rotation and translation."""
        coords1 = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, -1.0, 0.0]
        ])
        
        # Rotate 90 degrees around z-axis then translate
        rotation_matrix = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
        
        coords2 = coords1 @ rotation_matrix.T + np.array([5.0, -3.0, 2.0])
        
        rmsd = self.examiner._kabsch_rmsd(coords1, coords2)
        self.assertAlmostEqual(rmsd, 0.0, places=10,
                              msg="RMSD after rotation + translation should be 0")
    
    def test_known_rmsd_case(self):
        """Test RMSD calculation for a case with known expected RMSD after optimal alignment."""
        # Create coordinates with actual structural differences
        coords1 = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ])
        
        coords2 = np.array([
            [0.0, 0.0, 0.0],
            [1.1, 0.0, 0.0],  # 0.1 Å difference
            [0.0, 1.1, 0.0]   # 0.1 Å difference
        ])
        
        # After centering and optimal alignment, calculate expected RMSD manually:
        # centroid1 = [1/3, 1/3, 0], centroid2 = [11/30, 11/30, 0]
        # coords1_centered = [[-1/3, -1/3, 0], [2/3, -1/3, 0], [-1/3, 2/3, 0]]
        # coords2_centered = [[-11/30, -11/30, 0], [22/30, -11/30, 0], [-11/30, 22/30, 0]]
        # After optimal alignment, RMSD ≈ 0.0667
        rmsd = self.examiner._kabsch_rmsd(coords1, coords2)
        expected_rmsd = 2.0/30.0  # ≈ 0.0667
        self.assertAlmostEqual(rmsd, expected_rmsd, places=4,
                              msg=f"RMSD should be {expected_rmsd:.4f} after optimal alignment")
    
    def test_simple_structural_difference(self):
        """Test RMSD for a simple case where we can manually verify the result."""
        # Square configuration
        coords1 = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0]
        ])
        
        # Same square but one corner moved slightly
        coords2 = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.1, 1.0, 0.0]  # One corner moved by 0.1 Å in x
        ])
        
        rmsd = self.examiner._kabsch_rmsd(coords1, coords2)
        
        # Since only one point differs and it's a small perturbation,
        # RMSD should be small (< 0.1) but > 0
        self.assertGreater(rmsd, 0.0, msg="RMSD should be > 0 for different structures")
        self.assertLess(rmsd, 0.1, msg="RMSD should be < 0.1 for small structural change")
    
    def test_uniform_shift_case(self):
        """Test that uniform shift gives RMSD≈0 after optimal alignment."""
        coords1 = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ])
        
        coords2 = np.array([
            [0.1, 0.0, 0.0],
            [1.1, 0.0, 0.0],
            [0.1, 1.0, 0.0]
        ])
        
        # All points shifted by 0.1 in x direction (pure translation)
        # After centering and optimal alignment, RMSD should be ≈0
        rmsd = self.examiner._kabsch_rmsd(coords1, coords2)
        self.assertAlmostEqual(rmsd, 0.0, places=10,
                              msg="RMSD should be ≈0 for uniform translation after optimal alignment")
    
    def test_protein_like_coordinates(self):
        """Test with protein-like coordinate ranges."""
        # Simulate some CA coordinates from a small protein segment
        coords1 = np.array([
            [10.632, -9.681, -8.932],   # Y7
            [2.990, 0.056, -14.663],    # N62
            [5.058, -4.080, -11.420],   # H64
            [-2.171, 1.681, -9.957],    # N67
            [-2.603, 6.613, -5.686]     # Q92
        ])
        
        # Create a slightly perturbed version
        np.random.seed(42)  # For reproducible results
        noise = np.random.normal(0, 0.1, coords1.shape)
        coords2 = coords1 + noise
        
        rmsd = self.examiner._kabsch_rmsd(coords1, coords2)
        
        # RMSD should be small but non-zero
        self.assertGreater(rmsd, 0.0, msg="RMSD should be > 0 for noisy coordinates")
        self.assertLess(rmsd, 0.5, msg="RMSD should be < 0.5 for small noise")
    
    def test_reflection_handling(self):
        """Test that the algorithm correctly handles reflections."""
        coords1 = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        
        # Create reflection (determinant = -1)
        coords2 = np.array([
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        
        rmsd = self.examiner._kabsch_rmsd(coords1, coords2)
        
        # Should handle reflection properly (RMSD might not be 0 due to chirality)
        self.assertGreater(rmsd, 0.0, msg="RMSD should be > 0 for reflection")
        self.assertTrue(np.isfinite(rmsd), msg="RMSD should be finite")
    
    def test_minimum_points(self):
        """Test with minimum number of points (3)."""
        coords1 = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ])
        
        coords2 = coords1 + np.array([1.0, 1.0, 1.0])
        
        rmsd = self.examiner._kabsch_rmsd(coords1, coords2)
        self.assertAlmostEqual(rmsd, 0.0, places=10,
                              msg="RMSD should be 0 for translated coordinates")
    
    def test_centering_calculation(self):
        """Test that coordinates are properly centered."""
        coords = np.array([
            [10.0, 20.0, 30.0],
            [11.0, 21.0, 31.0],
            [12.0, 22.0, 32.0],
            [13.0, 23.0, 33.0]
        ])
        
        centroid = np.mean(coords, axis=0)
        centered = coords - centroid
        
        # Check that centered coordinates have zero mean
        centered_mean = np.mean(centered, axis=0)
        np.testing.assert_array_almost_equal(centered_mean, [0.0, 0.0, 0.0], decimal=10,
                                           err_msg="Centered coordinates should have zero mean")
    
    def test_svd_properties(self):
        """Test SVD decomposition properties."""
        # Create a random 3x3 matrix
        np.random.seed(42)
        H = np.random.rand(3, 3)
        
        U, S, Vt = np.linalg.svd(H)
        
        # Check SVD properties
        self.assertAlmostEqual(np.linalg.det(U), 1.0, places=10, 
                              msg="U should have determinant 1")
        self.assertAlmostEqual(np.linalg.det(Vt), 1.0, places=10,
                              msg="Vt should have determinant 1")
        
        # Reconstruct matrix
        H_reconstructed = U @ np.diag(S) @ Vt
        np.testing.assert_array_almost_equal(H, H_reconstructed, decimal=10,
                                           err_msg="SVD should reconstruct original matrix")


class TestRMSDIntegration(unittest.TestCase):
    """Integration tests for RMSD calculation in the context of the examiner."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.examiner = CatalyticActivityExaminer.__new__(CatalyticActivityExaminer)
        
    def test_reference_coordinates_format(self):
        """Test that reference coordinates are in the expected format."""
        # This tests the reference coordinates used in _calculate_aligned_rmsd
        reference_coords = {
            'Y7': [10.632, -9.681, -8.932],
            'N62': [2.990, 0.056, -14.663],
            'H64': [5.058, -4.080, -11.420],
            'N67': [-2.171, 1.681, -9.957],
            'Q92': [-2.603, 6.613, -5.686],
            'H94': [-2.053, -0.242, -5.673],
            'H96': [0.517, -6.624, -3.940],
            'H119': [-1.588, 0.345, -0.294]
        }
        
        for res_key, coords in reference_coords.items():
            self.assertEqual(len(coords), 3, 
                           f"Reference coordinates for {res_key} should have 3 dimensions")
            self.assertTrue(all(isinstance(x, (int, float)) for x in coords),
                           f"Reference coordinates for {res_key} should be numeric")
    
    def test_kabsch_with_reference_coords(self):
        """Test Kabsch RMSD with actual reference coordinates."""
        # Use actual reference coordinates
        ref_coords = np.array([
            [10.632, -9.681, -8.932],   # Y7
            [2.990, 0.056, -14.663],    # N62
            [5.058, -4.080, -11.420],   # H64
            [-2.171, 1.681, -9.957],    # N67
            [-2.603, 6.613, -5.686]     # Q92
        ])
        
        # Create slightly perturbed coordinates
        np.random.seed(42)
        current_coords = ref_coords + np.random.normal(0, 0.05, ref_coords.shape)
        
        rmsd = self.examiner._kabsch_rmsd(ref_coords, current_coords)
        
        self.assertGreater(rmsd, 0.0, msg="RMSD should be > 0 for perturbed coordinates")
        self.assertLess(rmsd, 0.2, msg="RMSD should be < 0.2 for small perturbations")


def run_debug_analysis():
    """Run detailed debug analysis of the Kabsch algorithm."""
    examiner = CatalyticActivityExaminer.__new__(CatalyticActivityExaminer)
    
    print("=== DEBUG ANALYSIS OF KABSCH RMSD CALCULATION ===\n")
    
    # Test case 1: Simple translation
    print("Test 1: Simple Translation")
    coords1 = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    coords2 = coords1 + np.array([2.0, 3.0, 4.0])
    
    print(f"Original coordinates:\n{coords1}")
    print(f"Translated coordinates:\n{coords2}")
    
    try:
        rmsd = examiner._kabsch_rmsd(coords1, coords2)
        print(f"RMSD: {rmsd}")
        print(f"Expected: 0.0")
        print(f"Status: {'PASS' if abs(rmsd) < 1e-10 else 'FAIL'}\n")
    except Exception as e:
        print(f"ERROR: {e}\n")
    
    # Test case 2: Simple rotation
    print("Test 2: Simple Rotation")
    coords1 = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    
    # 90-degree rotation around z-axis
    rotation_matrix = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])
    coords2 = coords1 @ rotation_matrix.T
    
    print(f"Original coordinates:\n{coords1}")
    print(f"Rotated coordinates:\n{coords2}")
    
    try:
        rmsd = examiner._kabsch_rmsd(coords1, coords2)
        print(f"RMSD: {rmsd}")
        print(f"Expected: 0.0")
        print(f"Status: {'PASS' if abs(rmsd) < 1e-10 else 'FAIL'}\n")
    except Exception as e:
        print(f"ERROR: {e}\n")
    
    # Test case 3: Step-by-step analysis
    print("Test 3: Step-by-step Kabsch Algorithm Analysis")
    
    coords1 = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    coords2 = coords1 + np.array([1.0, 1.0, 1.0])  # Simple translation
    
    print(f"coords1:\n{coords1}")
    print(f"coords2:\n{coords2}")
    
    # Step 1: Center coordinates
    centroid1 = np.mean(coords1, axis=0)
    centroid2 = np.mean(coords2, axis=0)
    print(f"centroid1: {centroid1}")
    print(f"centroid2: {centroid2}")
    
    coords1_centered = coords1 - centroid1
    coords2_centered = coords2 - centroid2
    print(f"coords1_centered:\n{coords1_centered}")
    print(f"coords2_centered:\n{coords2_centered}")
    
    # Step 2: Cross-covariance matrix
    H = coords2_centered.T @ coords1_centered
    print(f"Cross-covariance matrix H:\n{H}")
    
    # Step 3: SVD
    U, S, Vt = np.linalg.svd(H)
    print(f"U:\n{U}")
    print(f"S: {S}")
    print(f"Vt:\n{Vt}")
    
    # Step 4: Rotation matrix
    R = Vt.T @ U.T
    print(f"Rotation matrix R:\n{R}")
    print(f"det(R): {np.linalg.det(R)}")
    
    # Step 5: Apply rotation and calculate RMSD
    coords2_rotated = coords2_centered @ R.T
    print(f"coords2_rotated:\n{coords2_rotated}")
    
    diff = coords1_centered - coords2_rotated
    print(f"difference:\n{diff}")
    
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    print(f"Final RMSD: {rmsd}")
    print(f"Expected: 0.0")
    print(f"Status: {'PASS' if abs(rmsd) < 1e-10 else 'FAIL'}")


if __name__ == "__main__":
    print("Running RMSD Calculation Tests...")
    print("=" * 50)
    
    # Run unit tests
    unittest.main(verbosity=2, exit=False)
    
    print("\n" + "=" * 50)
    print("Running Debug Analysis...")
    run_debug_analysis() 