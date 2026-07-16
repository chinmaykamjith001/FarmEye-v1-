import os
import re
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import DBSCAN
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------------------------
# CONFIG - More stringent filtering parameters
# ---------------------------
INPUT_DIR = r"C:\Users\Faster\Downloads\FarmEyeInputs\heatmaps_chunked"
OUTPUT_CSV = "FarmEye_patterns_strict.csv"
OUTPUT_PNG = "FarmEye_patterns_map_strict.png"
BANDS = ["evi", "ndvi", "ndwi"]

# Temporal filtering - must be recent AND recurring
RECENT_YEAR_THRESHOLD = 2024   
MIN_RECUR_YEARS = 4             # Increased: must appear in at least 4 distinct years
MIN_TEMPORAL_SPAN = 3           # Must span at least 3 years (not just consecutive)

# Correlation filtering - much more stringent
CORR_THRESHOLD = 0.95          # Very strong correlation (was 0.9)
MIN_CORR_CONSISTENCY = 0.85    # Correlation must be consistent across time windows

# Spatial filtering - clusters must be meaningful size
CLUSTER_EPS = 12               # More generous clustering for initial detection
CLUSTER_MIN_SAMPLES = 3        # Lower threshold to start with
CHUNK_STEP = 6                 # Finer sampling for better precision

# Actionability filters - adapted for linear and area patterns
MIN_CLUSTER_LINEAR_LENGTH = 20  # Minimum length for linear patterns (pixels)
MIN_CLUSTER_AREA_SIZE = 10      # Minimum width/height for area patterns
MIN_CLUSTER_SIZE = 1           # Allow single high-quality patterns
MIN_SPATIAL_DENSITY = 0.03     # More lenient density requirement
MAX_CLUSTER_BBOX_SIZE = 200    # Not too large to be meaningless
MIN_LINEAR_ASPECT_RATIO = 3    # Length:width ratio to consider it linear

# Signal quality filters
MIN_SIGNAL_VARIANCE = 0.05     # Minimum variance in time series
MIN_SIGNAL_RANGE = 0.1         # Minimum range (max - min) in time series
MAX_NOISE_RATIO = 0.3          # Maximum noise-to-signal ratio

# ---------------------------
# STEP 1: Load heatmaps
# ---------------------------
def load_heatmaps(input_dir):
    pattern = re.compile(r"heatmap_(?P<band>[A-Za-z]+)_(?P<year>\d{4})\.png")
    data = {}
    for fname in os.listdir(input_dir):
        m = pattern.match(fname)
        if m:
            band = m.group("band").lower()
            year = int(m.group("year"))
            path = os.path.join(input_dir, fname)
            arr = np.array(Image.open(path).convert("L"), dtype=np.float32)
            arr /= 255.0
            data.setdefault(year, {})[band] = arr
    return data

# ---------------------------
# STEP 2: Filter complete years
# ---------------------------
def filter_complete_years(data):
    return {
        year: bands for year, bands in data.items()
        if all(b in bands for b in BANDS)
    }

# ---------------------------
# STEP 3: Enhanced signal quality assessment
# ---------------------------
def assess_signal_quality(ts1, ts2):
    """Assess if time series have sufficient signal quality for meaningful analysis"""
    # Check variance
    if np.var(ts1) < MIN_SIGNAL_VARIANCE or np.var(ts2) < MIN_SIGNAL_VARIANCE:
        return False, "Low variance"
    
    # Check range
    if (np.max(ts1) - np.min(ts1)) < MIN_SIGNAL_RANGE or (np.max(ts2) - np.min(ts2)) < MIN_SIGNAL_RANGE:
        return False, "Low range"
    
    # Check for excessive noise (using rolling correlation if we have enough points)
    if len(ts1) >= 5:
        # Split into windows and check correlation consistency
        mid = len(ts1) // 2
        try:
            corr1, _ = pearsonr(ts1[:mid+1], ts2[:mid+1]) if mid > 1 else (0, 1)
            if np.isnan(corr1):
                corr1 = 0
        except:
            corr1 = 0
            
        try:
            corr2, _ = pearsonr(ts1[mid:], ts2[mid:]) if len(ts1) - mid > 2 else (0, 1)
            if np.isnan(corr2):
                corr2 = 0
        except:
            corr2 = 0
        
        if abs(corr1) > 0.1 and abs(corr2) > 0.1:  # Both windows have some correlation
            consistency = 1 - abs(corr1 - corr2) / (abs(corr1) + abs(corr2) + 1e-6)
            if consistency < MIN_CORR_CONSISTENCY:
                return False, f"Inconsistent correlation: {consistency:.3f}"
    
    return True, "Good signal"

# ---------------------------
# STEP 4: Detect correlations with enhanced filtering
# ---------------------------
def detect_pixel_patterns(data):
    years_sorted = sorted(data.keys())
    h, w = next(iter(data.values()))[BANDS[0]].shape
    patterns = []
    
    print(f"[INFO] Analyzing {len(years_sorted)} years of data")
    print(f"[INFO] Signal quality thresholds: var>{MIN_SIGNAL_VARIANCE}, range>{MIN_SIGNAL_RANGE}")

    for y in tqdm(range(0, h, CHUNK_STEP), desc="Scanning rows"):
        for x in range(0, w, CHUNK_STEP):
            for i in range(len(BANDS)):
                for j in range(i + 1, len(BANDS)):
                    band1, band2 = BANDS[i], BANDS[j]
                    ts1 = [data[yr][band1][y, x] for yr in years_sorted]
                    ts2 = [data[yr][band2][y, x] for yr in years_sorted]

                    # Enhanced signal quality check
                    is_quality, reason = assess_signal_quality(ts1, ts2)
                    if not is_quality:
                        continue

                    corr, p_value = pearsonr(ts1, ts2)
                    
                    # Stricter correlation threshold and significance test
                    if abs(corr) >= CORR_THRESHOLD and p_value < 0.05:
                        patterns.append({
                            "x": x, "y": y,
                            "band1": band1, "band2": band2,
                            "corr": corr,
                            "p_value": p_value,
                            "years": years_sorted,
                            "ts1_var": np.var(ts1),
                            "ts2_var": np.var(ts2),
                            "ts1_range": np.max(ts1) - np.min(ts1),
                            "ts2_range": np.max(ts2) - np.min(ts2)
                        })
    
    print(f"[INFO] Found {len(patterns)} high-quality pixel patterns")
    return patterns

# ---------------------------
# STEP 5: Enhanced clustering with quality metrics
# ---------------------------
def cluster_patterns(patterns):
    if not patterns:
        return []

    coords = np.array([[p["x"], p["y"]] for p in patterns])
    
    # Debug: Print spatial distribution
    print(f"[DEBUG] Pattern coordinates range: X({np.min(coords[:,0])}-{np.max(coords[:,0])}), Y({np.min(coords[:,1])}-{np.max(coords[:,1])})")
    
    # Try multiple clustering parameters if first attempt fails
    clustering_params = [
        (CLUSTER_EPS, CLUSTER_MIN_SAMPLES),
        (CLUSTER_EPS * 1.5, CLUSTER_MIN_SAMPLES),
        (CLUSTER_EPS * 2, max(2, CLUSTER_MIN_SAMPLES - 1)),
        (CLUSTER_EPS * 3, 2)
    ]
    
    clustering = None
    labels = None
    
    for eps, min_samples in clustering_params:
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
        labels = clustering.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        print(f"[DEBUG] DBSCAN(eps={eps}, min_samples={min_samples}): {n_clusters} clusters, {n_noise} noise points")
        
        if n_clusters > 0:
            break
    
    if labels is None or len(set(labels)) <= 1:
        print("[WARNING] No clusters found with any parameter combination. Using individual patterns.")
        # Create individual "clusters" from high-correlation patterns
        labels = list(range(len(patterns)))
        n_clusters = len(patterns)
        print(f"[DEBUG] Created {n_clusters} individual pattern clusters")

    clustered = []
    unique_labels = set(labels)
    for lbl in tqdm(unique_labels, desc="Processing clusters"):
        if lbl == -1:  # Skip noise points
            continue
            
        indices = np.where(np.array(labels) == lbl)[0]
        pts = [patterns[i] for i in indices]

        xs = [p["x"] for p in pts]
        ys = [p["y"] for p in pts]
        bbox_w = max(xs) - min(xs) if len(xs) > 1 else 1
        bbox_h = max(ys) - min(ys) if len(ys) > 1 else 1
        bbox_area = max(bbox_w * bbox_h, 1)

        years_union = sorted(set(y for p in pts for y in p["years"]))
        avg_corr = np.mean([p["corr"] for p in pts])
        corr_std = np.std([p["corr"] for p in pts]) if len(pts) > 1 else 0
        cx, cy = np.mean(xs), np.mean(ys)
        
        # Calculate spatial metrics
        spatial_density = len(pts) / bbox_area
        
        # Determine if this is a linear or area pattern
        max_dim = max(bbox_w, bbox_h)
        min_dim = max(min(bbox_w, bbox_h), 1)  # Avoid division by zero
        aspect_ratio = max_dim / min_dim
        is_linear = aspect_ratio >= MIN_LINEAR_ASPECT_RATIO

        clustered.append({
            "id": lbl,
            "band1": pts[0]["band1"],
            "band2": pts[0]["band2"],
            "avg_corr": avg_corr,
            "corr_std": corr_std,
            "center_x": int(cx),
            "center_y": int(cy),
            "years_observed": years_union,
            "first_year": years_union[0],
            "last_year": years_union[-1],
            "temporal_span": years_union[-1] - years_union[0],
            "bbox_width": bbox_w,
            "bbox_height": bbox_h,
            "bbox_area": bbox_area,
            "cluster_size": len(pts),
            "spatial_density": spatial_density,
            "avg_p_value": np.mean([p["p_value"] for p in pts]),
            "is_linear": is_linear,
            "aspect_ratio": aspect_ratio,
            "max_dimension": max_dim,
            "min_dimension": min_dim
        })
    
    print(f"[INFO] Created {len(clustered)} initial clusters")
    return clustered

# ---------------------------
# STEP 6: Much more stringent actionability filtering
# ---------------------------
def filter_actionable_clusters(clusters):
    """Apply multiple strict filters to identify truly actionable patterns"""
    filtered = []
    rejection_reasons = {}
    
    print(f"[DEBUG] Starting filter with {len(clusters)} clusters")
    
    for c in clusters:
        rejection_reason = None
        
        # Temporal filters
        if c["last_year"] < RECENT_YEAR_THRESHOLD:
            rejection_reason = "Not recent enough"
        elif len(c["years_observed"]) < MIN_RECUR_YEARS:
            rejection_reason = f"Insufficient recurrence ({len(c['years_observed'])} < {MIN_RECUR_YEARS} years)"
        elif c["temporal_span"] < MIN_TEMPORAL_SPAN:
            rejection_reason = f"Insufficient temporal span ({c['temporal_span']} < {MIN_TEMPORAL_SPAN} years)"
        
        # Basic size filter
        elif c["cluster_size"] < MIN_CLUSTER_SIZE:
            rejection_reason = f"Cluster too small ({c['cluster_size']} < {MIN_CLUSTER_SIZE} points)"
        
        # Size and spatial filters - different criteria for linear vs area patterns
        elif c["cluster_size"] > 1:
            if c["is_linear"]:
                # For linear patterns: check minimum length
                if c["max_dimension"] < MIN_CLUSTER_LINEAR_LENGTH:
                    rejection_reason = f"Linear pattern too short ({c['max_dimension']:.1f} < {MIN_CLUSTER_LINEAR_LENGTH} pixels)"
                elif c["spatial_density"] < MIN_SPATIAL_DENSITY:
                    rejection_reason = f"Linear pattern too sparse ({c['spatial_density']:.3f} < {MIN_SPATIAL_DENSITY})"
            else:
                # For area patterns: check minimum width and height
                if c["bbox_width"] < MIN_CLUSTER_AREA_SIZE or c["bbox_height"] < MIN_CLUSTER_AREA_SIZE:
                    rejection_reason = f"Area pattern too small ({c['bbox_width']:.1f}x{c['bbox_height']:.1f} < {MIN_CLUSTER_AREA_SIZE})"
                elif c["spatial_density"] < MIN_SPATIAL_DENSITY:
                    rejection_reason = f"Area pattern too sparse ({c['spatial_density']:.3f} < {MIN_SPATIAL_DENSITY})"
        
        # Maximum size filter (applies to both types)
        if not rejection_reason and (c["bbox_width"] > MAX_CLUSTER_BBOX_SIZE or c["bbox_height"] > MAX_CLUSTER_BBOX_SIZE):
            rejection_reason = f"Pattern too large ({c['bbox_width']:.1f}x{c['bbox_height']:.1f} > {MAX_CLUSTER_BBOX_SIZE})"
        
        # Signal quality filters - more lenient for single high-quality patterns
        if not rejection_reason:
            if c["cluster_size"] > 5 and c["corr_std"] > 0.15:  # Only check consistency for larger clusters
                rejection_reason = f"Inconsistent correlation (std={c['corr_std']:.3f})"
            elif c["avg_p_value"] > 0.05:  # Standard significance level
                rejection_reason = f"Not statistically significant (p={c['avg_p_value']:.3f})"
        
        if rejection_reason:
            rejection_reasons[rejection_reason] = rejection_reasons.get(rejection_reason, 0) + 1
        else:
            # Quality score calculation - different for linear vs area patterns
            if c["cluster_size"] == 1:
                quality_score = abs(c["avg_corr"]) * 0.8 + (1 - c["avg_p_value"]) * 0.2
            elif c["is_linear"]:
                # For linear patterns: emphasize length and correlation strength
                length_score = min(c["max_dimension"] / (MIN_CLUSTER_LINEAR_LENGTH * 2), 1.0)
                quality_score = (
                    abs(c["avg_corr"]) * 0.4 +  # Correlation strength
                    (1 - c["corr_std"]) * 0.2 +   # Correlation consistency
                    length_score * 0.2 +         # Linear length
                    min(len(c["years_observed"]) / (MIN_RECUR_YEARS * 1.5), 1.0) * 0.2  # Temporal recurrence
                )
            else:
                # For area patterns: emphasize area and density
                quality_score = (
                    abs(c["avg_corr"]) * 0.4 +  # Correlation strength
                    (1 - c["corr_std"]) * 0.2 +   # Correlation consistency
                    min(c["spatial_density"] / MIN_SPATIAL_DENSITY, 1.0) * 0.2 +  # Spatial density
                    min(len(c["years_observed"]) / (MIN_RECUR_YEARS * 1.5), 1.0) * 0.2  # Temporal recurrence
                )
            
            c["quality_score"] = quality_score
            
            # Quality threshold - slightly more lenient
            if quality_score > 0.45:
                filtered.append(c)
            else:
                rejection_reasons[f"Low quality score ({quality_score:.3f})"] = rejection_reasons.get(f"Low quality score ({quality_score:.3f})", 0) + 1
    
    # Print detailed rejection statistics
    print(f"\n[FILTERING RESULTS]")
    print(f"Total clusters: {len(clusters)}")
    print(f"Actionable clusters: {len(filtered)}")
    
    if filtered:
        linear_count = sum(1 for c in filtered if c["is_linear"])
        area_count = len(filtered) - linear_count
        print(f"  - Linear patterns: {linear_count}")
        print(f"  - Area patterns: {area_count}")
    
    print(f"Rejection breakdown:")
    for reason, count in sorted(rejection_reasons.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {reason}: {count}")
    
    return filtered

# ---------------------------
# STEP 7: Enhanced CSV output with quality metrics
# ---------------------------
def save_csv(clusters, path):
    df = pd.DataFrame(clusters)
    df["human_rule"] = df.apply(
        lambda r: f"{'Linear' if r.get('is_linear', False) else 'Area'} pattern: "
                 f"{r['band1'].upper()} and {r['band2'].upper()} show {'strong positive' if r['avg_corr'] > 0 else 'strong negative'} correlation "
                 f"(r={r['avg_corr']:.3f}, quality={r['quality_score']:.3f}) across {len(r['years_observed'])} years. "
                 f"{'Length: ' + str(int(r['max_dimension'])) + ' pixels' if r.get('is_linear', False) else 'Size: ' + str(int(r['bbox_width'])) + 'x' + str(int(r['bbox_height'])) + ' pixels'}",
        axis=1
    )
    
    # Sort by quality score
    df = df.sort_values("quality_score", ascending=False)
    df.to_csv(path, index=False)
    print(f"[OK] CSV saved to {path}")

# ---------------------------
# STEP 8: Enhanced map overlay with background satellite image
# ---------------------------
def save_map_overlay(clusters, shape, path):
    h, w = shape
    
    # Try to load the background satellite image
    background_path = r"C:\Users\Faster\Downloads\FarmEyeInputs\2025.tif"
    background = None
    
    try:
        from PIL import Image
        # Load and resize background image to match heatmap dimensions
        bg_img = Image.open(background_path)
        bg_img = bg_img.resize((w, h), Image.Resampling.LANCZOS)
        
        # Convert to RGB if needed
        if bg_img.mode != 'RGB':
            bg_img = bg_img.convert('RGB')
        
        background = np.array(bg_img)
        print(f"[INFO] Loaded background satellite image: {background.shape}")
        
    except Exception as e:
        print(f"[WARN] Could not load background image: {e}")
        print("[INFO] Using default black background")
        background = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Create the overlay
    overlay = background.copy() if background is not None else np.zeros((h, w, 3), dtype=np.uint8)
    
    # Add pattern markers with high visibility
    for c in clusters:
        # Quality-based intensity and size
        quality = c["quality_score"]
        intensity = int(255 * min(quality, 1.0))  # Brightness based on quality
        marker_size = max(3, int(5 * quality))  # Size based on quality
        
        # Color coding
        if c["avg_corr"] > 0:
            color = (0, intensity, 0)  # Green for positive correlation
        else:
            color = (intensity, 0, 0)  # Red for negative correlation
        
        # Different shapes for linear vs area patterns
        cy, cx = c["center_y"], c["center_x"]
        
        if c.get("is_linear", False):
            # Draw a line-like marker for linear patterns
            for dy in range(-marker_size, marker_size + 1):
                for dx in range(-1, 2):  # Thin line
                    py, px = cy + dy, cx + dx
                    if 0 <= py < h and 0 <= px < w:
                        overlay[py, px] = color
        else:
            # Draw a cross/plus for area patterns
            for i in range(-marker_size, marker_size + 1):
                # Horizontal line
                px = cx + i
                if 0 <= cy < h and 0 <= px < w:
                    overlay[cy, px] = color
                # Vertical line  
                py = cy + i
                if 0 <= py < h and 0 <= cx < w:
                    overlay[py, cx] = color
    
    # Create the plot
    plt.figure(figsize=(15, 10))
    
    # Display the overlay
    plt.imshow(overlay)
    
    # Add title and legend
    linear_count = sum(1 for c in clusters if c.get("is_linear", False))
    area_count = len(clusters) - linear_count
    
    plt.title(f"Actionable Agricultural Patterns Overlay\n"
             f"Total: {len(clusters)} patterns ({linear_count} linear, {area_count} area)\n"
             f"Green=Positive Correlation, Red=Negative Correlation",
             fontsize=14, pad=20)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='+', color='w', markerfacecolor='green', markersize=10, label='Area Pattern (Positive Corr.)'),
        Line2D([0], [0], marker='|', color='w', markerfacecolor='green', markersize=10, label='Linear Pattern (Positive Corr.)'),
        Line2D([0], [0], marker='+', color='w', markerfacecolor='red', markersize=10, label='Area Pattern (Negative Corr.)'),
        Line2D([0], [0], marker='|', color='w', markerfacecolor='red', markersize=10, label='Linear Pattern (Negative Corr.)')
    ]
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()  # Close to free memory
    
    print(f"[OK] Map overlay with satellite background saved to {path}")
    
    # Also save a version with transparency overlay for better visibility
    overlay_path = path.replace('.png', '_transparent.png')
    
    if background is not None:
        plt.figure(figsize=(15, 10))
        
        # Show background
        plt.imshow(background)
        
        # Add semi-transparent overlay
        pattern_overlay = np.zeros((h, w, 4), dtype=np.uint8)  # RGBA
        
        for c in clusters:
            quality = c["quality_score"]
            alpha = int(200 * min(quality, 1.0))  # Transparency based on quality
            marker_size = max(3, int(5 * quality))
            
            if c["avg_corr"] > 0:
                color = (0, 255, 0, alpha)  # Green with alpha
            else:
                color = (255, 0, 0, alpha)  # Red with alpha
            
            cy, cx = c["center_y"], c["center_x"]
            
            # Draw marker
            for dy in range(-marker_size, marker_size + 1):
                for dx in range(-marker_size, marker_size + 1):
                    py, px = cy + dy, cx + dx
                    if 0 <= py < h and 0 <= px < w:
                        if c.get("is_linear", False):
                            # Line pattern
                            if abs(dy) <= 1 or abs(dx) <= 1:
                                pattern_overlay[py, px] = color
                        else:
                            # Cross pattern
                            if abs(dy) + abs(dx) <= marker_size:
                                pattern_overlay[py, px] = color
        
        plt.imshow(pattern_overlay)
        plt.title(f"Semi-Transparent Pattern Overlay\n"
                 f"Total: {len(clusters)} patterns ({linear_count} linear, {area_count} area)",
                 fontsize=14, pad=20)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(overlay_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] Semi-transparent overlay saved to {overlay_path}")

# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    print("[INFO] Starting FarmEye pattern detection with strict filtering...")
    
    raw_data = load_heatmaps(INPUT_DIR)
    complete_data = filter_complete_years(raw_data)
    print(f"[INFO] Complete years found: {list(complete_data.keys())}")

    if len(complete_data) < MIN_RECUR_YEARS:
        print(f"[ERROR] Insufficient data years ({len(complete_data)} < {MIN_RECUR_YEARS}). Cannot proceed.")
        exit(1)

    patterns = detect_pixel_patterns(complete_data)
    print(f"[INFO] High-quality pixel patterns found: {len(patterns)}")

    if not patterns:
        print("[WARN] No patterns found. Consider relaxing thresholds.")
        exit(0)

    clusters = cluster_patterns(patterns)
    print(f"[INFO] Clustered zones: {len(clusters)}")

    actionable_clusters = filter_actionable_clusters(clusters)
    print(f"[INFO] Final actionable clusters: {len(actionable_clusters)}")

    if actionable_clusters:
        save_csv(actionable_clusters, OUTPUT_CSV)
        h, w = next(iter(complete_data.values()))[BANDS[0]].shape
        save_map_overlay(actionable_clusters, (h, w), OUTPUT_PNG)
        
        # Summary statistics
        print(f"\n[SUMMARY]")
        print(f"Average quality score: {np.mean([c['quality_score'] for c in actionable_clusters]):.3f}")
        print(f"Average correlation strength: {np.mean([abs(c['avg_corr']) for c in actionable_clusters]):.3f}")
        print(f"Average temporal span: {np.mean([c['temporal_span'] for c in actionable_clusters]):.1f} years")
        print(f"Average cluster size: {np.mean([c['cluster_size'] for c in actionable_clusters]):.1f} points")
    else:
        print("[RESULT] No actionable clusters found with current strict criteria.")
        print("Consider adjusting thresholds if this seems too restrictive for your data.")