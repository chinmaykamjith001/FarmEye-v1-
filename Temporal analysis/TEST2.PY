#!/usr/bin/env python3
"""
Pattern Overlay Test Script
Quick test to verify if the pattern overlay functionality works correctly
without running the full FarmEye pipeline.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

def create_test_data():
    """Create synthetic test data to simulate FarmEye outputs"""
    print("[TEST] Creating synthetic test data...")
    
    # Create test image dimensions
    h, w = 300, 400  # Small test image
    
    # Create synthetic NDVI background (simulates normalized NDVI 2025)
    np.random.seed(42)
    ndvi_background = np.random.random((h, w)) * 0.8 + 0.1  # Range 0.1-0.9
    
    # Add some realistic NDVI patterns (higher values in center, lower at edges)
    y_indices, x_indices = np.ogrid[:h, :w]
    center_y, center_x = h//2, w//2
    distance_from_center = np.sqrt((y_indices - center_y)**2 + (x_indices - center_x)**2)
    max_distance = np.sqrt(center_y**2 + center_x**2)
    ndvi_background = ndvi_background + 0.3 * (1 - distance_from_center / max_distance)
    ndvi_background = np.clip(ndvi_background, 0, 1)
    
    # Create synthetic pattern clusters (simulates actionable_patterns.csv)
    test_patterns = [
        # High quality positive correlation pattern
        {
            "center_x": 100, "center_y": 80,
            "avg_corr": 0.85, "quality_score": 0.75,
            "is_linear": False, "band1": "ndvi", "band2": "evi"
        },
        # High quality negative correlation pattern  
        {
            "center_x": 250, "center_y": 120,
            "avg_corr": -0.78, "quality_score": 0.82,
            "is_linear": False, "band1": "ndvi", "band2": "ndwi"
        },
        # Medium quality linear pattern (positive)
        {
            "center_x": 320, "center_y": 200,
            "avg_corr": 0.68, "quality_score": 0.55,
            "is_linear": True, "band1": "evi", "band2": "ndwi",
            "max_dimension": 45
        },
        # Low quality pattern (should be small marker)
        {
            "center_x": 50, "center_y": 250,
            "avg_corr": -0.45, "quality_score": 0.48,
            "is_linear": False, "band1": "ndvi", "band2": "evi"
        },
        # Edge case - near boundary
        {
            "center_x": 5, "center_y": 5,
            "avg_corr": 0.92, "quality_score": 0.88,
            "is_linear": False, "band1": "ndvi", "band2": "ndwi"
        },
        # Edge case - near other boundary
        {
            "center_x": 395, "center_y": 295,
            "avg_corr": -0.67, "quality_score": 0.71,
            "is_linear": True, "band1": "evi", "band2": "ndwi",
            "max_dimension": 30
        }
    ]
    
    return ndvi_background, test_patterns, (h, w)

def create_ndvi_background_rgb(ndvi_normalized):
    """Convert NDVI to RGB background (simulates the background creation logic)"""
    h, w = ndvi_normalized.shape
    
    # Create RGB overlay (matches FarmEye logic)
    background_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    ndvi_enhanced = np.power(ndvi_normalized, 0.7)  # Gamma correction
    background_rgb[:, :, 1] = (ndvi_enhanced * 200).astype(np.uint8)  # Green channel
    background_rgb[:, :, 0] = (ndvi_enhanced * 100).astype(np.uint8)  # Red channel  
    background_rgb[:, :, 2] = (ndvi_enhanced * 50).astype(np.uint8)   # Blue channel
    
    return background_rgb

def test_pattern_overlay(clusters, shape, background_rgb):
    """Test the pattern overlay logic (extracted from FarmEye)"""
    print("[TEST] Testing pattern overlay logic...")
    
    h, w = shape
    overlay = background_rgb.copy()
    
    # Draw patterns on overlay (EXACT COPY FROM FARMEYE)
    sorted_clusters = sorted(clusters, key=lambda x: x["quality_score"])
    
    patterns_drawn = 0
    patterns_out_of_bounds = 0
    
    for c in sorted_clusters:
        quality = c["quality_score"]
        base_marker_size = 12
        marker_size = max(base_marker_size, int(25 * quality))
        
        # Colors
        if c["avg_corr"] > 0:
            color = (0, 255, 0)      # Pure green
        else:
            color = (255, 0, 0)      # Pure red
        border_color = (255, 255, 255)  # White border
        
        border_size = marker_size + 4
        cy, cx = c["center_y"], c["center_x"]
        
        print(f"[DEBUG] Pattern: center=({cx}, {cy}), quality={quality:.3f}, "
              f"corr={c['avg_corr']:.3f}, marker_size={marker_size}, "
              f"linear={c.get('is_linear', False)}")
        
        # Check if coordinates are within reasonable bounds before adjustment
        original_in_bounds = (0 <= cy < h and 0 <= cx < w)
        
        # Ensure coordinates are within bounds
        cy = max(border_size, min(h - border_size - 1, cy))
        cx = max(border_size, min(w - border_size - 1, cx))
        
        adjusted_in_bounds = (border_size <= cy <= h - border_size - 1 and 
                            border_size <= cx <= w - border_size - 1)
        
        print(f"[DEBUG] Bounds check: original_in_bounds={original_in_bounds}, "
              f"adjusted_in_bounds={adjusted_in_bounds}, "
              f"adjusted_coords=({cx}, {cy})")
        
        if not adjusted_in_bounds:
            patterns_out_of_bounds += 1
            print(f"[WARN] Pattern still out of bounds after adjustment!")
            continue
        
        try:
            # Draw white border
            border_pixels_drawn = 0
            for dy in range(-border_size, border_size + 1):
                for dx in range(-border_size, border_size + 1):
                    py, px = cy + dy, cx + dx
                    if 0 <= py < h and 0 <= px < w:
                        if (dy*dy + dx*dx) <= border_size*border_size:
                            overlay[py, px] = border_color
                            border_pixels_drawn += 1
            
            # Draw main marker
            main_pixels_drawn = 0
            if c.get("is_linear", False):
                # Linear patterns: draw thick lines
                line_length = max(marker_size, int(c.get("max_dimension", 20) * 0.5))
                for dy in range(-line_length, line_length + 1):
                    for dx in range(-3, 4):
                        py, px = cy + dy, cx + dx
                        if 0 <= py < h and 0 <= px < w:
                            overlay[py, px] = color
                            main_pixels_drawn += 1
            else:
                # Area patterns: draw filled circles
                for dy in range(-marker_size, marker_size + 1):
                    for dx in range(-marker_size, marker_size + 1):
                        py, px = cy + dy, cx + dx
                        if 0 <= py < h and 0 <= px < w:
                            if (dy*dy + dx*dx) <= marker_size*marker_size:
                                overlay[py, px] = color
                                main_pixels_drawn += 1
            
            patterns_drawn += 1
            print(f"[SUCCESS] Drew pattern: border_pixels={border_pixels_drawn}, "
                  f"main_pixels={main_pixels_drawn}")
            
        except Exception as e:
            print(f"[ERROR] Failed to draw pattern: {e}")
            patterns_out_of_bounds += 1
    
    print(f"[SUMMARY] Patterns drawn successfully: {patterns_drawn}")
    print(f"[SUMMARY] Patterns out of bounds: {patterns_out_of_bounds}")
    
    return overlay

def analyze_overlay_result(original_background, final_overlay):
    """Analyze the overlay result to detect if patterns were actually drawn"""
    print("[TEST] Analyzing overlay result...")
    
    # Convert background to same format for comparison
    background_for_comparison = original_background.copy()
    
    # Find pixels that changed from background to overlay
    differences = np.any(final_overlay != background_for_comparison, axis=2)
    changed_pixels = np.sum(differences)
    
    # Find specific colors (red and green patterns, white borders)
    red_pixels = np.sum(np.all(final_overlay == [255, 0, 0], axis=2))
    green_pixels = np.sum(np.all(final_overlay == [0, 255, 0], axis=2))
    white_pixels = np.sum(np.all(final_overlay == [255, 255, 255], axis=2))
    
    print(f"[ANALYSIS] Total pixels changed: {changed_pixels}")
    print(f"[ANALYSIS] Red pattern pixels: {red_pixels}")
    print(f"[ANALYSIS] Green pattern pixels: {green_pixels}")
    print(f"[ANALYSIS] White border pixels: {white_pixels}")
    
    # Determine if overlay worked
    patterns_visible = (red_pixels > 0 or green_pixels > 0) and white_pixels > 0
    
    return {
        'patterns_visible': patterns_visible,
        'changed_pixels': changed_pixels,
        'red_pixels': red_pixels,
        'green_pixels': green_pixels,
        'white_pixels': white_pixels
    }

def save_test_results(background, overlay, analysis_result, output_dir=r"C:\Users\Faster\Downloads\FarmEyeOutputs\Test1"):
    """Save test results for visual inspection"""
    print(f"[TEST] Saving test results to {output_dir}/...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save individual images
    plt.figure(figsize=(15, 5))
    
    # Original NDVI background
    plt.subplot(1, 3, 1)
    plt.imshow(background, cmap='Greens')
    plt.title('Original NDVI Background')
    plt.colorbar(shrink=0.8)
    plt.axis('off')
    
    # RGB background
    rgb_background = create_ndvi_background_rgb(background)
    plt.subplot(1, 3, 2)
    plt.imshow(rgb_background)
    plt.title('RGB Background')
    plt.axis('off')
    
    # Final overlay
    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title('Pattern Overlay Result')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pattern_overlay_test.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save analysis report
    with open(os.path.join(output_dir, 'test_analysis_report.txt'), 'w', encoding='utf-8') as f:
        f.write("PATTERN OVERLAY TEST ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Test Status: {'PASS' if analysis_result['patterns_visible'] else 'FAIL'}\n\n")
        f.write("PIXEL ANALYSIS:\n")
        f.write(f"- Total pixels changed: {analysis_result['changed_pixels']}\n")
        f.write(f"- Red pattern pixels: {analysis_result['red_pixels']}\n")
        f.write(f"- Green pattern pixels: {analysis_result['green_pixels']}\n")
        f.write(f"- White border pixels: {analysis_result['white_pixels']}\n\n")
        
        if analysis_result['patterns_visible']:
            f.write("SUCCESS: Patterns are visible in the overlay!\n")
            f.write("The pattern overlay functionality is working correctly.\n")
        else:
            f.write("FAILURE: No patterns visible in the overlay!\n")
            f.write("There may be issues with:\n")
            f.write("- Coordinate system mismatches\n")
            f.write("- Bounds checking problems\n")
            f.write("- Color assignment issues\n")
            f.write("- Marker size calculations\n")
    
    print(f"✅ Test results saved to {output_dir}/")
    return output_dir

def main():
    """Run the complete pattern overlay test"""
    print("PATTERN OVERLAY FUNCTIONALITY TEST")
    print("=" * 50)
    
    # Step 1: Create test data
    ndvi_background, test_patterns, shape = create_test_data()
    print(f"✅ Created test data: {len(test_patterns)} patterns, image size {shape}")
    
    # Step 2: Create RGB background
    rgb_background = create_ndvi_background_rgb(ndvi_background)
    print(f"✅ Created RGB background: {rgb_background.shape}")
    
    # Step 3: Test pattern overlay
    overlay_result = test_pattern_overlay(test_patterns, shape, rgb_background)
    print(f"✅ Pattern overlay test completed")
    
    # Step 4: Analyze results
    analysis = analyze_overlay_result(rgb_background, overlay_result)
    print(f"✅ Analysis completed")
    
    # Step 5: Save results
    output_dir = save_test_results(ndvi_background, overlay_result, analysis)
    
    # Step 6: Final report
    print("\n" + "=" * 50)
    print("FINAL TEST REPORT")
    print("=" * 50)
    print("Status: {} {}".format('PASS' if analysis['patterns_visible'] else 'FAIL', 
                               '[SUCCESS]' if analysis['patterns_visible'] else '[ERROR]'))
    print(f"Patterns visible: {analysis['patterns_visible']}")
    print(f"Total pixel changes: {analysis['changed_pixels']}")
    
    if analysis['patterns_visible']:
        print("\n[SUCCESS] GREAT! The pattern overlay functionality is working!")
        print("You should see colored patterns with white borders in the result image.")
        print("The main FarmEye pipeline overlay should work correctly.")
    else:
        print("\n[WARNING] Pattern overlay may have issues!")
        print("Check the saved images to see what went wrong.")
        print("You may need to debug the coordinate system or bounds checking.")
    
    print(f"\n[FILES] Check results in: {output_dir}/")
    print("   - pattern_overlay_test.png (visual results)")
    print("   - test_analysis_report.txt (detailed analysis)")

if __name__ == "__main__":
    main()