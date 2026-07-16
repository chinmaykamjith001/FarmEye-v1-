#Full pipeline
#!/usr/bin/env python3
"""
FarmEye Complete Agricultural Monitoring Pipeline
Combines trend analysis, anomaly detection, pattern recognition, and predictive modeling
into a single comprehensive workflow.
"""

import os
import re
import math
import numpy as np
import pandas as pd
import csv
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Core scientific libraries
from sklearn.linear_model import TheilSenRegressor, Ridge
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from scipy.stats import pearsonr, zscore
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter

# Optional libraries
try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    print("[WARN] rasterio not available - some features disabled")

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("[WARN] catboost not available - using alternative models")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[WARN] xgboost not available - using alternative models")

try:
    import torch
    import torch.nn as nn
    from torchvision.models import resnet18, resnet50
    import torchvision.transforms as T
    TORCH_AVAILABLE = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] PyTorch available, using device: {DEVICE}")
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARN] PyTorch not available - CNN features disabled")

# ==================== CONFIGURATION ====================
class FarmEyeConfig:
    """Centralized configuration for the entire pipeline"""
    
    # Input/Output paths
    INPUT_FOLDER = r"C:\Users\Faster\Downloads\FarmEyeInputs\Test2"
    OUTPUT_FOLDER = r"C:\Users\Faster\Downloads\FarmEyeOutputs\Test2"
    
    # Years and bands configuration
    YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
    BANDS = ['NDVI', 'EVI', 'NDWI']
    NUM_BANDS = len(BANDS)
    NUM_YEARS = len(YEARS)

    CHUNK_SIZE = 24
    CHUNK_STEP = 24         # DOUBLED: Skip even more chunks
    
    TREND_BATCH_SIZE = 64   # DOUBLED: Larger batches
    
    # Pipeline 2: Slightly relaxed for faster processing
    RECENT_YEAR_THRESHOLD = 2023  # RELAXED: Still recent but includes more data
    MIN_RECUR_YEARS = 2           # RELAXED: Faster filtering
    MIN_TEMPORAL_SPAN = 2         # RELAXED: Faster filtering
    CORR_THRESHOLD = 0.85         # RELAXED: Still high quality but faster
    MIN_CORR_CONSISTENCY = 0.85   # RELAXED
    CLUSTER_EPS = 12              # RELAXED: Larger clusters = fewer to process
    CLUSTER_MIN_SAMPLES = 2
    MIN_CLUSTER_LINEAR_LENGTH = 10  # RELAXED
    MIN_CLUSTER_AREA_SIZE = 6       # RELAXED
    MIN_CLUSTER_SIZE = 1
    MIN_SPATIAL_DENSITY = 0.015     # RELAXED
    MAX_CLUSTER_BBOX_SIZE = 200     # SMALLER: Skip very large patterns
    MIN_LINEAR_ASPECT_RATIO = 3     # RELAXED
    MIN_SIGNAL_VARIANCE = 0.03      # RELAXED
    MIN_SIGNAL_RANGE = 0.08         # RELAXED
    MAX_NOISE_RATIO = 0.4           # RELAXED
    
    # Pipeline 3: MAJOR SPEED OPTIMIZATIONS
    PATCH_SIZE = 16         # LARGER: 60% fewer patches (10→16)
    STRIDE = 32             # DOUBLED: 75% fewer patches total
    BATCH_SIZE = 64         # TRIPLED: Much faster GPU utilization
    PCA_DIM = 16            # REDUCED: Faster dimensionality reduction
    
    # Pipeline 4: MAJOR SPEED OPTIMIZATIONS  
    MIN_YEARS_REQ = 2
    N_ESTIMATORS = 50       # 1/3 LESS: Much faster training
    MAX_DEPTH = 6           # REDUCED: Faster trees
    LEARNING_RATE = 0.05    # HIGHER: Faster convergence
    N_SPLITS = 3            # REDUCED: Less cross-validation
    TEST_SIZE = 0.25        # LARGER: Less training data = faster


# ==================== PIPELINE 1: TREND ANALYSIS ====================
class TrendAnalyzer:
    """Handles multi-year trend analysis using Theil-Sen regression"""
    
    def __init__(self, config):
        self.config = config
        
    def load_multiyear_stack(self):
        """Load multi-year satellite data stack"""
        print("[PIPELINE 1] Loading multi-year satellite data...")
        
        if not RASTERIO_AVAILABLE:
            raise ImportError("rasterio required for loading satellite data")
            
        first_path = os.path.join(self.config.INPUT_FOLDER, f"{self.config.YEARS[0]}.tif")
        with rasterio.open(first_path) as src:
            rows, cols = src.height, src.width

        arr = np.full((self.config.NUM_BANDS, rows, cols, self.config.NUM_YEARS), np.nan, dtype=np.float32)

        for t, year in enumerate(self.config.YEARS):
            path = os.path.join(self.config.INPUT_FOLDER, f"{year}.tif")
            print(f"Loading {path}...")
            with rasterio.open(path) as src:
                data = src.read().astype(np.float32)
                if src.nodata is not None:
                    data[data == src.nodata] = np.nan
                if data.shape != (self.config.NUM_BANDS, rows, cols):
                    raise ValueError(f"Shape mismatch in {path}")
                arr[:, :, :, t] = data

        print(f"Loaded data stack shape: {arr.shape} (bands, rows, cols, years)")
        return arr
    
    def compute_theil_sen_slopes(self, stack, output_csv):
        """Compute Theil-Sen slopes for each chunk"""
        print("[PIPELINE 1] Computing Theil-Sen regression slopes...")
        
        bands, rows, cols, times = stack.shape
        X = np.array(self.config.YEARS).reshape(-1, 1)

        os.makedirs(os.path.dirname(output_csv), exist_ok=True)

        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['chunk_row', 'chunk_col', 'band', 'slope', 'intercept'])
            writer.writeheader()

            for b in range(bands):
                band_name = self.config.BANDS[b]
                print(f"Computing Theil-Sen for band {band_name}...")
                batch = []

                for r0 in tqdm(range(0, rows, self.config.CHUNK_SIZE), desc=f"Band {band_name} chunks"):
                    for c0 in range(0, cols, self.config.CHUNK_SIZE):
                        r1 = min(r0 + self.config.CHUNK_SIZE, rows)
                        c1 = min(c0 + self.config.CHUNK_SIZE, cols)

                        chunk_data = stack[b, r0:r1, c0:c1, :]
                        with np.errstate(invalid='ignore'):
                            chunk_mean_ts = np.nanmean(chunk_data.reshape(-1, times), axis=0)

                        if np.isnan(chunk_mean_ts).all() or np.count_nonzero(~np.isnan(chunk_mean_ts)) < 3:
                            continue

                        y_valid = chunk_mean_ts[~np.isnan(chunk_mean_ts)]
                        X_valid = X[~np.isnan(chunk_mean_ts)]

                        try:
                            model = TheilSenRegressor(random_state=42)
                            model.fit(X_valid, y_valid)
                            batch.append({
                                'chunk_row': r0,
                                'chunk_col': c0,
                                'band': band_name,
                                'slope': model.coef_[0],
                                'intercept': model.intercept_
                            })
                        except Exception:
                            continue

                        if len(batch) >= self.config.TREND_BATCH_SIZE:
                            writer.writerows(batch)
                            batch.clear()

                if batch:
                    writer.writerows(batch)
                    batch.clear()
        
        print(f"✅ Slopes saved to {output_csv}")
    
    def compute_residuals(self, stack, slope_csv, output_csv):
        """Compute residuals from trend model"""
        print("[PIPELINE 1] Computing residuals from trend model...")
        
        # Load slopes/intercepts
        slope_intercept_map = {}
        with open(slope_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (int(row['chunk_row']), int(row['chunk_col']), row['band'])
                slope = float(row['slope'])
                intercept = float(row['intercept'])
                slope_intercept_map[key] = (slope, intercept)

        bands, rows, cols, times = stack.shape

        with open(output_csv, 'w', newline='') as f_out:
            writer = csv.DictWriter(f_out, fieldnames=['chunk_row', 'chunk_col', 'band', 'year', 'residual'])
            writer.writeheader()

            buffer = []

            for b in range(bands):
                band_name = self.config.BANDS[b]
                print(f"Computing residuals for band {band_name}...")
                for r0 in tqdm(range(0, rows, self.config.CHUNK_SIZE), desc=f"Band {band_name} residuals"):
                    for c0 in range(0, cols, self.config.CHUNK_SIZE):
                        key = (r0, c0, band_name)
                        if key not in slope_intercept_map:
                            continue

                        r1 = min(r0 + self.config.CHUNK_SIZE, rows)
                        c1 = min(c0 + self.config.CHUNK_SIZE, cols)
                        chunk_data = stack[b, r0:r1, c0:c1, :]
                        with np.errstate(invalid='ignore'):
                            chunk_mean_ts = np.nanmean(chunk_data.reshape(-1, times), axis=0)

                        if np.isnan(chunk_mean_ts).all():
                            continue

                        slope, intercept = slope_intercept_map[key]

                        for t, year in enumerate(self.config.YEARS):
                            val = chunk_mean_ts[t]
                            if np.isnan(val):
                                continue
                            predicted = slope * year + intercept
                            residual = val - predicted
                            buffer.append({
                                'chunk_row': r0,
                                'chunk_col': c0,
                                'band': band_name,
                                'year': year,
                                'residual': residual
                            })

                            if len(buffer) >= 100000:
                                writer.writerows(buffer)
                                buffer.clear()

            if buffer:
                writer.writerows(buffer)
        
        print(f"✅ Residuals saved to {output_csv}")
    
    def compute_slope_zscores(self, slope_csv, output_csv):
        """Compute z-scores for slope analysis"""
        print("[PIPELINE 1] Computing slope z-scores...")
        
        # First pass: collect slopes per band
        slope_by_band = {band: [] for band in self.config.BANDS}
        with open(slope_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                slope = row['slope']
                band = row['band']
                if slope != '' and slope.lower() != 'nan':
                    slope_by_band[band].append(float(slope))

        # Compute mean/std per band
        stats = {}
        for band in self.config.BANDS:
            slopes = np.array(slope_by_band[band])
            mean = slopes.mean()
            std = slopes.std()
            stats[band] = (mean, std if std > 0 else 1)

        # Second pass: write z-scores
        with open(slope_csv, 'r') as f_in, open(output_csv, 'w', newline='') as f_out:
            reader = csv.DictReader(f_in)
            writer = csv.DictWriter(f_out, fieldnames=['chunk_row', 'chunk_col', 'band', 'zscore'])
            writer.writeheader()

            for row in reader:
                slope = row['slope']
                band = row['band']
                if slope == '' or slope.lower() == 'nan':
                    z = 0
                else:
                    mean, std = stats[band]
                    z = (float(slope) - mean) / std
                writer.writerow({
                    'chunk_row': row['chunk_row'],
                    'chunk_col': row['chunk_col'],
                    'band': band,
                    'zscore': z
                })

        print(f"✅ Slope z-scores saved to {output_csv}")
    
    def compute_residual_zscores(self, residual_csv, output_csv):
        """Compute z-scores for residual analysis"""
        print("[PIPELINE 1] Computing residual z-scores...")
        
        residuals_map = {}
        rows_buffer = []

        with open(residual_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                band = row['band']
                year = int(row['year'])
                residual = float(row['residual'])
                key = (band, year)
                residuals_map.setdefault(key, []).append(residual)
                rows_buffer.append(row)

        stats = {}
        for key, values in residuals_map.items():
            arr = np.array(values)
            mean = arr.mean()
            std = arr.std()
            stats[key] = (mean, std if std > 0 else 1)

        with open(output_csv, 'w', newline='') as f_out:
            writer = csv.DictWriter(f_out, fieldnames=['chunk_row', 'chunk_col', 'band', 'year', 'zscore'])
            writer.writeheader()

            for row in rows_buffer:
                band = row['band']
                year = int(row['year'])
                residual = float(row['residual'])
                mean, std = stats[(band, year)]
                z = (residual - mean) / std
                writer.writerow({
                    'chunk_row': row['chunk_row'],
                    'chunk_col': row['chunk_col'],
                    'band': band,
                    'year': year,
                    'zscore': z
                })

        print(f"✅ Residual z-scores saved to {output_csv}")

    def generate_composite_heatmap(self, zscore_csv, output_path):
        """Generate composite impact heatmap"""
        print("[PIPELINE 1] Generating composite impact heatmap...")
        
        # Compute dynamic band weights
        band_values = {band: [] for band in self.config.BANDS}
        with open(zscore_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                zscore = row['zscore']
                band = row['band']
                if zscore != '' and zscore.lower() != 'nan':
                    band_values[band].append(float(zscore))

        std_devs = {}
        for band, values in band_values.items():
            arr = np.array(values)
            std_devs[band] = arr.std() if arr.size > 0 else 0

        total_std = sum(std_devs.values())
        if total_std == 0:
            band_weights = {band: 1/len(self.config.BANDS) for band in self.config.BANDS}
        else:
            band_weights = {band: std_devs[band] / total_std for band in self.config.BANDS}
        
        print(f"Dynamic band weights: {band_weights}")

        # Generate heatmap
        pixel_scores = {}
        with open(zscore_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                r = int(row['chunk_row'])
                c = int(row['chunk_col'])
                band = row['band']
                zscore = float(row['zscore'])
                if (r,c) not in pixel_scores:
                    pixel_scores[(r,c)] = {}
                pixel_scores[(r,c)][band] = zscore

        if not pixel_scores:
            print("No valid scores found for heatmap generation")
            return

        all_rows = [k[0] for k in pixel_scores.keys()]
        all_cols = [k[1] for k in pixel_scores.keys()]
        max_row, max_col = max(all_rows), max(all_cols)

        grid_rows = max_row // self.config.CHUNK_SIZE + 1
        grid_cols = max_col // self.config.CHUNK_SIZE + 1
        composite_array = np.full((grid_rows, grid_cols), np.nan, dtype=np.float32)

        for (r,c), band_dict in pixel_scores.items():
            chunk_r_idx = r // self.config.CHUNK_SIZE
            chunk_c_idx = c // self.config.CHUNK_SIZE
            composite_score = 0
            weight_sum = 0
            for band, z in band_dict.items():
                w = band_weights.get(band, 0)
                composite_score += z * w
                weight_sum += w
            if weight_sum > 0:
                composite_array[chunk_r_idx, chunk_c_idx] = composite_score / weight_sum

        finite_vals = composite_array[np.isfinite(composite_array)]
        if finite_vals.size == 0:
            print("No valid composite scores found")
            return

        vmin, vmax = np.percentile(finite_vals, 2), np.percentile(finite_vals, 98)

        plt.figure(figsize=(12, 10))
        plt.imshow(composite_array, cmap='RdYlGn_r', vmin=vmin, vmax=vmax)
        plt.colorbar(label='Composite z-score (impact)')
        plt.title('Multi-year Impact Composite Heatmap')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

        print(f"✅ Composite heatmap saved to {output_path}")
    
    def generate_individual_heatmaps(self, zscore_csv, output_folder):
        """Generate individual heatmaps per band-year"""
        print("[PIPELINE 1] Generating individual heatmaps...")
        
        zscore_data = {}
        chunk_rows = []
        chunk_cols = []

        with open(zscore_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                band = row['band']
                year = int(row['year'])
                r = int(row['chunk_row'])
                c = int(row['chunk_col'])
                z = float(row['zscore'])

                key = (band, year)
                if key not in zscore_data:
                    zscore_data[key] = {}
                zscore_data[key][(r, c)] = z

                chunk_rows.append(r)
                chunk_cols.append(c)

        if not chunk_rows:
            print("No data found for heatmap generation")
            return

        max_row = max(chunk_rows)
        max_col = max(chunk_cols)
        os.makedirs(output_folder, exist_ok=True)

        for (band, year), pix_dict in tqdm(zscore_data.items(), desc="Generating heatmaps"):
            heatmap_array = np.full((max_row // self.config.CHUNK_SIZE + 1, 
                                   max_col // self.config.CHUNK_SIZE + 1), np.nan, dtype=np.float32)

            for (r, c), z in pix_dict.items():
                chunk_r_idx = r // self.config.CHUNK_SIZE
                chunk_c_idx = c // self.config.CHUNK_SIZE
                heatmap_array[chunk_r_idx, chunk_c_idx] = z

            finite_vals = heatmap_array[np.isfinite(heatmap_array)]
            if finite_vals.size == 0:
                continue

            vmin, vmax = np.percentile(finite_vals, 2), np.percentile(finite_vals, 98)

            plt.figure(figsize=(12, 10))
            plt.imshow(heatmap_array, cmap='RdYlGn_r', vmin=vmin, vmax=vmax)
            plt.colorbar(label='Z-score residual')
            """plt.title(f"Anomaly Heatmap - Band: {band} Year: {year}")"""
            plt.axis('off')
            plt.tight_layout()

            heatmap_path = os.path.join(output_folder, f"heatmap_{band}_{year}.png")
            plt.savefig(heatmap_path, dpi=300)
            plt.close()

        print(f"✅ Individual heatmaps saved to {output_folder}")

    def generate_combined_heatmap_grid(self, heatmap_folder, output_path):
        """Generate a single image with all heatmaps in a grid layout"""
        print("[PIPELINE 1] Generating combined heatmap grid...")
        
        # Find all heatmap files
        heatmap_files = []
        pattern = re.compile(r"heatmap_([A-Za-z]+)_(\d{4})\.png")
        
        for fname in os.listdir(heatmap_folder):
            match = pattern.match(fname)
            if match:
                band = match.group(1).upper()
                year = int(match.group(2))
                filepath = os.path.join(heatmap_folder, fname)
                heatmap_files.append({
                    'band': band,
                    'year': year,
                    'path': filepath,
                    'filename': fname
                })
        
        if not heatmap_files:
            print("[WARN] No heatmap files found for grid generation")
            return
        
        # Sort by year, then by band
        heatmap_files.sort(key=lambda x: (x['year'], x['band']))
        
        # Group by year to create a grid layout
        years = sorted(set(f['year'] for f in heatmap_files))
        bands = sorted(set(f['band'] for f in heatmap_files))
        
        print(f"Creating grid for {len(years)} years x {len(bands)} bands = {len(heatmap_files)} total heatmaps")
        
        # Create figure with subplots
        n_rows = len(years)
        n_cols = len(bands)
        
        # Adjust figure size based on grid dimensions
        fig_width = min(20, n_cols * 4)
        fig_height = min(16, n_rows * 3)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
        
        # Ensure axes is always 2D
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        if n_cols == 1:
            axes = axes.reshape(-1, 1)
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        
        # Create a lookup for heatmap files
        file_lookup = {}
        for f in heatmap_files:
            file_lookup[(f['year'], f['band'])] = f['path']
        
        # Populate the grid
        for row, year in enumerate(years):
            for col, band in enumerate(bands):
                if (year, band) in file_lookup:
                    # Load and display heatmap
                    img_path = file_lookup[(year, band)]
                    try:
                        img = Image.open(img_path)
                        img_array = np.array(img)
                        
                        axes[row, col].imshow(img_array, cmap='RdYlGn_r')
                        axes[row, col].set_title(f'{band} {year}', fontsize=10, pad=5)
                        axes[row, col].axis('off')
                        
                    except Exception as e:
                        print(f"[WARN] Failed to load {img_path}: {e}")
                        axes[row, col].text(0.5, 0.5, f'{band} {year}\nLoad Error', 
                                        ha='center', va='center', transform=axes[row, col].transAxes)
                        axes[row, col].axis('off')
                else:
                    # No heatmap available for this year-band combination
                    axes[row, col].text(0.5, 0.5, f'{band} {year}\nNo Data', 
                                    ha='center', va='center', transform=axes[row, col].transAxes, 
                                    fontsize=10, alpha=0.5)
                    axes[row, col].axis('off')
        
        # Add overall title and layout
        plt.suptitle(f'All Heatmaps Grid View\n{len(years)} Years × {len(bands)} Bands = {len(heatmap_files)} Heatmaps', 
                    fontsize=14, y=0.98)
        
        # Adjust layout to prevent overlap
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save the combined grid
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ Combined heatmap grid saved to {output_path}")
        print(f"   📊 Grid dimensions: {n_rows} rows × {n_cols} columns")
        print(f"   📈 Total heatmaps: {len(heatmap_files)}")
        
        # Create a summary text file
        summary_path = output_path.replace('.png', '_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("HEATMAP GRID SUMMARY\n")
            f.write("=" * 30 + "\n\n")
            f.write(f"Grid Layout: {n_rows} rows × {n_cols} columns\n")
            f.write(f"Total Heatmaps: {len(heatmap_files)}\n")
            f.write(f"Years: {', '.join(map(str, years))}\n")
            f.write(f"Bands: {', '.join(bands)}\n\n")
            
            f.write("HEATMAP LIST:\n")
            f.write("-" * 20 + "\n")
            for hf in heatmap_files:
                f.write(f"• {hf['filename']} - {hf['band']} {hf['year']}\n")
        
        print(f"✅ Heatmap summary saved to {summary_path}")
# ==================== PIPELINE 2: ANOMALY DETECTION ====================
class AnomalyDetector:
    """Handles pattern detection and anomaly identification"""

    def validate_pattern_coordinates(self, clusters, shape):
        """Validate and fix pattern coordinates to ensure they're within bounds"""
        h, w = shape
        fixed_clusters = []
        
        for c in clusters:
            # Fix coordinates that might be out of bounds
            cy = max(0, min(h-1, c["center_y"]))
            cx = max(0, min(w-1, c["center_x"]))
            
            # Update the cluster data
            c_fixed = c.copy()
            c_fixed["center_y"] = cy
            c_fixed["center_x"] = cx
            
            # Add validation flags
            c_fixed["coord_valid"] = (0 <= c["center_y"] < h and 0 <= c["center_x"] < w)
            c_fixed["coord_adjusted"] = (cy != c["center_y"] or cx != c["center_x"])
            
            fixed_clusters.append(c_fixed)
        
        invalid_coords = sum(1 for c in fixed_clusters if not c["coord_valid"])
        adjusted_coords = sum(1 for c in fixed_clusters if c["coord_adjusted"])
        
        if invalid_coords > 0:
            print(f"[WARN] {invalid_coords} patterns had invalid coordinates")
        if adjusted_coords > 0:
            print(f"[INFO] {adjusted_coords} pattern coordinates were adjusted to fit within bounds")
        
        return fixed_clusters

    def create_field_navigation_guide(self, clusters, output_path):
        """Create a text file with field navigation instructions - FIXED Unicode version"""
        print("[PIPELINE 2] Creating field navigation guide...")
        
        # Sort by priority
        priority_patterns = sorted([c for c in clusters if c["quality_score"] > 0.5], 
                                key=lambda x: x["quality_score"], reverse=True)
        
        # Open file with UTF-8 encoding to handle Unicode characters
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("FARM PATTERN INVESTIGATION GUIDE\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"Total patterns requiring attention: {len(priority_patterns)}\n\n")
            
            f.write("PRIORITY ORDER INVESTIGATION LIST:\n")
            f.write("-" * 40 + "\n\n")
            
            for i, c in enumerate(priority_patterns[:20]):  # Top 20
                f.write(f"PATTERN #{i+1} - {'HIGH' if c['quality_score'] > 0.7 else 'MEDIUM'} PRIORITY\n")
                f.write(f"Quality Score: {c['quality_score']:.3f}\n")
                f.write(f"Location: Row {c['center_y']}, Column {c['center_x']}\n")
                f.write(f"Type: {'Linear pattern (field boundary/drainage)' if c.get('is_linear', False) else 'Area pattern (soil/crop issue)'}\n")
                f.write(f"Correlation: {c['avg_corr']:.3f} ({'Positive trend' if c['avg_corr'] > 0 else 'Negative trend'})\n")
                f.write(f"Bands affected: {c['band1'].upper()} and {c['band2'].upper()}\n")
                f.write(f"Years observed: {c['first_year']}-{c['last_year']} ({c['temporal_span']} year span)\n")
                f.write(f"Size: {c.get('bbox_width', 1):.0f} x {c.get('bbox_height', 1):.0f} pixels\n")
                
                # Add investigation recommendations - Using text instead of emojis
                if c['quality_score'] > 0.8:
                    f.write("URGENT: Requires immediate field investigation\n")
                elif c['quality_score'] > 0.6:
                    f.write("IMPORTANT: Schedule investigation within 1-2 weeks\n")
                else:
                    f.write("MONITOR: Check during next routine field visit\n")
                
                if c.get('is_linear', False):
                    f.write("CHECK: Drainage, irrigation lines, field boundaries\n")
                else:
                    f.write("CHECK: Soil conditions, pest damage, nutrient deficiency\n")
                
                f.write("\n" + "-" * 40 + "\n\n")
            
            f.write("NAVIGATION NOTES:\n")
            f.write("- Coordinates are in image pixels (Row, Column format)\n")
            f.write("- Use GPS/drone to locate pixel coordinates in actual field\n")
            f.write("- Larger numbers = higher quality/more important patterns\n")
            f.write("- Linear patterns often follow field infrastructure\n")
            f.write("- Area patterns typically indicate localized field issues\n")
        
        print(f"✅ Field navigation guide saved: {output_path}")

        # Also create a simple CSV summary for easier processing
        summary_csv_path = output_path.replace('.txt', '_summary.csv')
        
        summary_data = []
        for i, c in enumerate(priority_patterns[:20]):
            summary_data.append({
                'pattern_id': i + 1,
                'priority': 'HIGH' if c['quality_score'] > 0.7 else 'MEDIUM',
                'quality_score': c['quality_score'],
                'center_row': c['center_y'],
                'center_col': c['center_x'],
                'correlation': c['avg_corr'],
                'trend': 'Positive' if c['avg_corr'] > 0 else 'Negative',
                'band1': c['band1'].upper(),
                'band2': c['band2'].upper(),
                'first_year': c['first_year'],
                'last_year': c['last_year'],
                'temporal_span': c['temporal_span'],
                'bbox_width': c.get('bbox_width', 1),
                'bbox_height': c.get('bbox_height', 1),
                'cluster_size': c['cluster_size'],
                'is_linear': c.get('is_linear', False),
                'pattern_type': 'Linear' if c.get('is_linear', False) else 'Area',
                'investigation_priority': 'Immediate' if c['quality_score'] > 0.8 else 'Within 1-2 weeks' if c['quality_score'] > 0.6 else 'Next routine visit',
                'check_for': 'Drainage/irrigation/boundaries' if c.get('is_linear', False) else 'Soil/pest/nutrient issues'
            })
        
        if summary_data:
            df_summary = pd.DataFrame(summary_data)
            df_summary.to_csv(summary_csv_path, index=False)
            print(f"✅ Field navigation summary CSV saved: {summary_csv_path}")
        
        print(f"   📋 Created guide for {len(priority_patterns)} patterns")
        if priority_patterns:
            high_priority = sum(1 for c in priority_patterns if c['quality_score'] > 0.7)
            medium_priority = sum(1 for c in priority_patterns if 0.5 < c['quality_score'] <= 0.7)
            print(f"   🔴 {high_priority} high priority patterns")
            print(f"   🟡 {medium_priority} medium priority patterns")
    
    def __init__(self, config):
        self.config = config

    def load_ndvi_background(self, data):
        """Load the most recent NDVI data as background for overlay"""
        print("[PIPELINE 2] Loading NDVI background...")
        
        # Get the most recent year available
        latest_year = max(data.keys())
        ndvi_band = 'ndvi'  # Assuming NDVI is always lowercase in the data
        
        if latest_year in data and ndvi_band in data[latest_year]:
            ndvi_background = data[latest_year][ndvi_band].copy()
            
            # Normalize NDVI to 0-1 range for better visualization
            valid_ndvi = ndvi_background[~np.isnan(ndvi_background)]
            if len(valid_ndvi) > 0:
                # Use robust percentile normalization
                p_low, p_high = np.percentile(valid_ndvi, [2, 98])
                ndvi_background = np.clip(ndvi_background, p_low, p_high)
                ndvi_background = (ndvi_background - p_low) / (p_high - p_low)
            else:
                ndvi_background = np.zeros_like(ndvi_background)
            
            print(f"✅ Loaded NDVI background from year {latest_year}")
            return ndvi_background, latest_year
        else:
            print(f"[WARN] No NDVI data found for year {latest_year}")
            # Return a neutral gray background
            h, w = next(iter(data.values()))[list(next(iter(data.values())).keys())[0]].shape
            return np.full((h, w), 0.5), latest_year
        
    def add_geographic_context(self, overlay, shape, output_map_enhanced):
        """Add scale bar, grid, and other geographic context"""
        h, w = shape
        
        # Create enhanced figure with subplots for additional info
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 3, height_ratios=[1, 8, 1], width_ratios=[8, 1, 1])
        
        # Main map
        ax_main = fig.add_subplot(gs[1, 0])
        ax_main.imshow(overlay)
        ax_main.set_title("Agricultural Pattern Detection with NDVI Background", fontsize=16, pad=20)
        
        # Add grid lines every 50 pixels for reference
        for i in range(0, h, 50):
            ax_main.axhline(y=i, color='white', alpha=0.3, linewidth=0.5)
        for i in range(0, w, 50):
            ax_main.axvline(x=i, color='white', alpha=0.3, linewidth=0.5)
        
        # Add scale reference (approximate)
        scale_length = min(w, h) // 10  # 10% of image width
        scale_y = h - 30
        scale_x_start = 20
        scale_x_end = scale_x_start + scale_length
        
        # Draw scale bar
        ax_main.plot([scale_x_start, scale_x_end], [scale_y, scale_y], 'white', linewidth=4)
        ax_main.plot([scale_x_start, scale_x_start], [scale_y-5, scale_y+5], 'white', linewidth=2)
        ax_main.plot([scale_x_end, scale_x_end], [scale_y-5, scale_y+5], 'white', linewidth=2)
        ax_main.text(scale_x_start + scale_length//2, scale_y-15, f'~{scale_length} pixels', 
                    ha='center', va='top', color='white', fontweight='bold', fontsize=10)
        
        # Add north arrow
        arrow_x, arrow_y = w - 50, 50
        ax_main.annotate('N', xy=(arrow_x, arrow_y), xytext=(arrow_x, arrow_y-20),
                        arrowprops=dict(arrowstyle='->', color='white', lw=2),
                        fontsize=14, fontweight='bold', color='white', ha='center')
        
        ax_main.set_xlim(0, w)
        ax_main.set_ylim(h, 0)
        ax_main.axis('off')
        
        # Add colorbar for NDVI
        ax_cbar = fig.add_subplot(gs[1, 1])
        import matplotlib.colors as mcolors
        import matplotlib.cm as cm
        
        # Create NDVI colorbar
        norm = mcolors.Normalize(vmin=0, vmax=1)
        cmap = cm.Greens
        cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax_cbar, orientation='vertical')
        cbar.set_label('NDVI Value', fontsize=12)
        
        # Add pattern statistics
        ax_stats = fig.add_subplot(gs[1, 2])
        ax_stats.axis('off')
        ax_stats.text(0.1, 0.9, 'Pattern Legend:', fontsize=12, fontweight='bold', transform=ax_stats.transAxes)
        ax_stats.text(0.1, 0.8, '● Green = Positive Correlation', fontsize=10, color='green', transform=ax_stats.transAxes)
        ax_stats.text(0.1, 0.75, '● Red = Negative Correlation', fontsize=10, color='red', transform=ax_stats.transAxes)
        ax_stats.text(0.1, 0.65, 'Marker Size = Pattern Quality', fontsize=10, transform=ax_stats.transAxes)
        
        plt.tight_layout()
        plt.savefig(output_map_enhanced, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ Enhanced map with geographic context saved: {output_map_enhanced}")
    def assess_signal_quality(self, ts1, ts2):
        """Assess signal quality for meaningful analysis"""
        if np.var(ts1) < self.config.MIN_SIGNAL_VARIANCE or np.var(ts2) < self.config.MIN_SIGNAL_VARIANCE:
            return False, "Low variance"
        
        if (np.max(ts1) - np.min(ts1)) < self.config.MIN_SIGNAL_RANGE or (np.max(ts2) - np.min(ts2)) < self.config.MIN_SIGNAL_RANGE:
            return False, "Low range"
        
        if len(ts1) >= 5:
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
            
            if abs(corr1) > 0.1 and abs(corr2) > 0.1:
                consistency = 1 - abs(corr1 - corr2) / (abs(corr1) + abs(corr2) + 1e-6)
                if consistency < self.config.MIN_CORR_CONSISTENCY:
                    return False, f"Inconsistent correlation: {consistency:.3f}"
        
        return True, "Good signal"
    
    def load_heatmaps(self, input_dir):
        """Load heatmap data"""
        print("[PIPELINE 2] Loading heatmap data...")
        
        pattern = re.compile(r"heatmap_(?P<band>[A-Za-z]+)_(?P<year>\d{4})\.png")
        data = {}
        for fname in tqdm(os.listdir(input_dir), desc="Loading heatmaps"):
            m = pattern.match(fname)
            if m:
                band = m.group("band").lower()
                year = int(m.group("year"))
                path = os.path.join(input_dir, fname)
                arr = np.array(Image.open(path).convert("L"), dtype=np.float32)
                arr /= 255.0
                data.setdefault(year, {})[band] = arr
        
        # Filter complete years
        complete_data = {
            year: bands for year, bands in data.items()
            if all(b in bands for b in [band.lower() for band in self.config.BANDS])
        }
        
        print(f"Complete years found: {list(complete_data.keys())}")
        return complete_data
    
    def detect_pixel_patterns(self, data):
        """Detect correlation patterns between bands"""
        print("[PIPELINE 2] Detecting pixel patterns...")
        
        years_sorted = sorted(data.keys())
        h, w = next(iter(data.values()))[self.config.BANDS[0].lower()].shape
        patterns = []
        
        band_names = [band.lower() for band in self.config.BANDS]

        for y in tqdm(range(0, h, self.config.CHUNK_STEP), desc="Scanning for patterns"):
            for x in range(0, w, self.config.CHUNK_STEP):
                for i in range(len(band_names)):
                    for j in range(i + 1, len(band_names)):
                        band1, band2 = band_names[i], band_names[j]
                        ts1 = [data[yr][band1][y, x] for yr in years_sorted]
                        ts2 = [data[yr][band2][y, x] for yr in years_sorted]

                        is_quality, reason = self.assess_signal_quality(ts1, ts2)
                        if not is_quality:
                            continue

                        corr, p_value = pearsonr(ts1, ts2)
                        
                        if abs(corr) >= self.config.CORR_THRESHOLD and p_value < 0.05:
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
        
        print(f"Found {len(patterns)} high-quality pixel patterns")
        return patterns
    
    def cluster_patterns(self, patterns):
        """Cluster detected patterns spatially - FIXED VERSION"""
        print("[PIPELINE 2] Clustering patterns...")
        
        if not patterns:
            print("No patterns to cluster")
            return []

        # Extract coordinates
        coords = np.array([[p["x"], p["y"]] for p in patterns])
        print(f"Coordinate range: X={coords[:, 0].min()}-{coords[:, 0].max()}, Y={coords[:, 1].min()}-{coords[:, 1].max()}")
        
        # Try different clustering parameters
        clustering_params = [
            (self.config.CLUSTER_EPS, self.config.CLUSTER_MIN_SAMPLES),
            (self.config.CLUSTER_EPS * 1.5, max(2, self.config.CLUSTER_MIN_SAMPLES - 1)),
            (self.config.CLUSTER_EPS * 2, 2),
            (self.config.CLUSTER_EPS * 3, 2)
        ]
        
        clustering = None
        labels = None
        
        for eps, min_samples in clustering_params:
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
            labels = clustering.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            print(f"DBSCAN(eps={eps}, min_samples={min_samples}): {n_clusters} clusters, {n_noise} noise points")
            
            if n_clusters > 0:
                break
        
        # If no clustering worked, treat each pattern as its own cluster
        if labels is None or len(set(labels)) <= 1:
            print("No valid clustering found, treating each pattern as individual cluster")
            labels = list(range(len(patterns)))

        clustered = []
        unique_labels = set(labels)
        
        for lbl in tqdm(unique_labels, desc="Processing clusters"):
            if lbl == -1:  # Skip noise points
                continue
                
            # Get all patterns in this cluster
            indices = np.where(np.array(labels) == lbl)[0]
            cluster_patterns = [patterns[i] for i in indices]

            # Extract coordinates and correlations
            xs = [p["x"] for p in cluster_patterns]
            ys = [p["y"] for p in cluster_patterns]
            corrs = [p["corr"] for p in cluster_patterns]
            
            # Calculate bounding box dimensions - FIXED
            if len(xs) == 1:
                bbox_w = 1  # Single point clusters have dimension 1
                bbox_h = 1
            else:
                bbox_w = max(xs) - min(xs) + 1  # Add 1 to include the endpoint
                bbox_h = max(ys) - min(ys) + 1
            
            bbox_area = bbox_w * bbox_h
            
            # Get years information
            years_union = []
            for p in cluster_patterns:
                years_union.extend(p["years"])
            years_union = sorted(set(years_union))
            
            # Calculate statistics
            avg_corr = np.mean(corrs)
            corr_std = np.std(corrs) if len(corrs) > 1 else 0.0
            cx, cy = np.mean(xs), np.mean(ys)
            
            # Spatial density calculation - FIXED
            spatial_density = len(cluster_patterns) / max(bbox_area, 1)
            
            # Aspect ratio and linearity - FIXED
            max_dim = max(bbox_w, bbox_h)
            min_dim = min(bbox_w, bbox_h)
            aspect_ratio = max_dim / max(min_dim, 1)  # Prevent division by zero
            is_linear = aspect_ratio >= self.config.MIN_LINEAR_ASPECT_RATIO
            
            # Calculate average p-value
            p_values = [p.get("p_value", 0.05) for p in cluster_patterns]
            avg_p_value = np.mean(p_values)
            
            cluster_info = {
                "id": lbl,
                "band1": cluster_patterns[0]["band1"],
                "band2": cluster_patterns[0]["band2"],
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
                "cluster_size": len(cluster_patterns),
                "spatial_density": spatial_density,
                "avg_p_value": avg_p_value,
                "is_linear": is_linear,
                "aspect_ratio": aspect_ratio,
                "max_dimension": max_dim,
                "min_dimension": min_dim,
                "pattern_points": cluster_patterns  # Keep original patterns for debugging
            }
            
            clustered.append(cluster_info)
        
        print(f"Created {len(clustered)} clusters from {len(patterns)} patterns")
        
        # Debug information
        for i, c in enumerate(clustered[:5]):  # Show first 5 clusters
            print(f"Cluster {i}: size={c['cluster_size']}, bbox={c['bbox_width']}x{c['bbox_height']}, "
                f"density={c['spatial_density']:.4f}, corr={c['avg_corr']:.3f}")
        
        return clustered

    def filter_actionable_clusters(self, clusters):
        """Filter clusters for actionable patterns - FIXED VERSION"""
        print("[PIPELINE 2] Filtering for actionable patterns...")
        
        if not clusters:
            print("No clusters to filter")
            return []
        
        filtered = []
        rejection_stats = {}
        
        print(f"Starting with {len(clusters)} clusters")
        print(f"Filter thresholds:")
        print(f"  Recent year threshold: {self.config.RECENT_YEAR_THRESHOLD}")
        print(f"  Min recurrence years: {self.config.MIN_RECUR_YEARS}")
        print(f"  Min temporal span: {self.config.MIN_TEMPORAL_SPAN}")
        print(f"  Min cluster size: {self.config.MIN_CLUSTER_SIZE}")
        print(f"  Min area size: {self.config.MIN_CLUSTER_AREA_SIZE}")
        print(f"  Min linear length: {self.config.MIN_CLUSTER_LINEAR_LENGTH}")
        print(f"  Min spatial density: {self.config.MIN_SPATIAL_DENSITY}")
        
        for i, c in enumerate(clusters):
            rejection_reason = None
            
            # Debug info for first few clusters
            if i < 3:
                print(f"\nCluster {i} analysis:")
                print(f"  Last year: {c['last_year']} (threshold: {self.config.RECENT_YEAR_THRESHOLD})")
                print(f"  Years observed: {len(c['years_observed'])} (min: {self.config.MIN_RECUR_YEARS})")
                print(f"  Temporal span: {c['temporal_span']} (min: {self.config.MIN_TEMPORAL_SPAN})")
                print(f"  Cluster size: {c['cluster_size']} (min: {self.config.MIN_CLUSTER_SIZE})")
                print(f"  Dimensions: {c['bbox_width']}x{c['bbox_height']} (min area: {self.config.MIN_CLUSTER_AREA_SIZE})")
                print(f"  Is linear: {c['is_linear']}")
                print(f"  Spatial density: {c['spatial_density']:.4f} (min: {self.config.MIN_SPATIAL_DENSITY})")
            
            # Temporal filters - RELAXED
            if c["last_year"] < self.config.RECENT_YEAR_THRESHOLD:
                rejection_reason = f"Not recent enough ({c['last_year']} < {self.config.RECENT_YEAR_THRESHOLD})"
            elif len(c["years_observed"]) < self.config.MIN_RECUR_YEARS:
                rejection_reason = f"Insufficient recurrence ({len(c['years_observed'])} < {self.config.MIN_RECUR_YEARS} years)"
            elif c["temporal_span"] < self.config.MIN_TEMPORAL_SPAN:
                rejection_reason = f"Insufficient temporal span ({c['temporal_span']} < {self.config.MIN_TEMPORAL_SPAN} years)"
            
            # Size filters - FIXED
            elif c["cluster_size"] < self.config.MIN_CLUSTER_SIZE:
                rejection_reason = f"Cluster too small ({c['cluster_size']} < {self.config.MIN_CLUSTER_SIZE} points)"
            
            # Pattern type specific filters - FIXED
            elif c["cluster_size"] == 1:
                # Single point patterns (anomalies) - accept if they meet basic criteria
                pass  # Already passed temporal and size filters
                
            elif c["is_linear"]:
                # Linear patterns - check length and density
                if c["max_dimension"] < self.config.MIN_CLUSTER_LINEAR_LENGTH:
                    rejection_reason = f"Linear pattern too short ({c['max_dimension']:.1f} < {self.config.MIN_CLUSTER_LINEAR_LENGTH} pixels)"
                elif c["spatial_density"] < self.config.MIN_SPATIAL_DENSITY:
                    rejection_reason = f"Linear pattern too sparse ({c['spatial_density']:.4f} < {self.config.MIN_SPATIAL_DENSITY})"
            
            else:
                # Area patterns - check minimum dimensions
                if c["bbox_width"] < self.config.MIN_CLUSTER_AREA_SIZE and c["bbox_height"] < self.config.MIN_CLUSTER_AREA_SIZE:
                    rejection_reason = f"Area pattern too small ({c['bbox_width']}x{c['bbox_height']} < {self.config.MIN_CLUSTER_AREA_SIZE})"
                elif c["spatial_density"] < self.config.MIN_SPATIAL_DENSITY:
                    rejection_reason = f"Area pattern too sparse ({c['spatial_density']:.4f} < {self.config.MIN_SPATIAL_DENSITY})"
            
            # Size limit filter
            if not rejection_reason and (c["bbox_width"] > self.config.MAX_CLUSTER_BBOX_SIZE or c["bbox_height"] > self.config.MAX_CLUSTER_BBOX_SIZE):
                rejection_reason = f"Pattern too large ({c['bbox_width']}x{c['bbox_height']} > {self.config.MAX_CLUSTER_BBOX_SIZE})"
            
            # Statistical significance filters - RELAXED
            if not rejection_reason:
                if c["cluster_size"] > 5 and c["corr_std"] > 0.2:  # Increased threshold
                    rejection_reason = f"Inconsistent correlation (std={c['corr_std']:.3f} > 0.2)"
                elif c["avg_p_value"] > 0.1:  # Relaxed from 0.05 to 0.1
                    rejection_reason = f"Not statistically significant (p={c['avg_p_value']:.3f} > 0.1)"
            
            # If rejected, record reason
            if rejection_reason:
                rejection_stats[rejection_reason] = rejection_stats.get(rejection_reason, 0) + 1
                if i < 3:
                    print(f"  REJECTED: {rejection_reason}")
            else:
                # Calculate quality score - IMPROVED
                if c["cluster_size"] == 1:
                    # Single point anomaly
                    quality_score = (
                        abs(c["avg_corr"]) * 0.6 +  # Correlation strength
                        (1 - c["avg_p_value"]) * 0.2 +  # Statistical significance
                        min(len(c["years_observed"]) / max(self.config.MIN_RECUR_YEARS, 2), 1.0) * 0.2  # Temporal consistency
                    )
                elif c["is_linear"]:
                    # Linear pattern
                    length_score = min(c["max_dimension"] / max(self.config.MIN_CLUSTER_LINEAR_LENGTH, 5), 1.0)
                    density_score = min(c["spatial_density"] / self.config.MIN_SPATIAL_DENSITY, 1.0)
                    quality_score = (
                        abs(c["avg_corr"]) * 0.3 +  # Correlation strength
                        (1 - c["corr_std"]) * 0.2 +  # Correlation consistency
                        length_score * 0.2 +  # Pattern length
                        density_score * 0.1 +  # Spatial density
                        min(len(c["years_observed"]) / max(self.config.MIN_RECUR_YEARS, 2), 1.0) * 0.2  # Temporal consistency
                    )
                else:
                    # Area pattern
                    size_score = min(np.sqrt(c["bbox_area"]) / max(self.config.MIN_CLUSTER_AREA_SIZE, 2), 1.0)
                    density_score = min(c["spatial_density"] / self.config.MIN_SPATIAL_DENSITY, 1.0)
                    quality_score = (
                        abs(c["avg_corr"]) * 0.3 +  # Correlation strength
                        (1 - c["corr_std"]) * 0.2 +  # Correlation consistency
                        size_score * 0.2 +  # Pattern size
                        density_score * 0.1 +  # Spatial density
                        min(len(c["years_observed"]) / max(self.config.MIN_RECUR_YEARS, 2), 1.0) * 0.2  # Temporal consistency
                    )
                
                # Ensure quality score is within valid range
                quality_score = max(0.0, min(1.0, quality_score))
                c["quality_score"] = quality_score
                
                # Accept if quality score is reasonable - LOWERED THRESHOLD
                if quality_score > 0.3:  # Lowered from 0.45
                    filtered.append(c)
                    if i < 3:
                        print(f"  ACCEPTED: quality_score = {quality_score:.3f}")
                else:
                    rejection_reason = f"Low quality score ({quality_score:.3f} <= 0.3)"
                    rejection_stats[rejection_reason] = rejection_stats.get(rejection_reason, 0) + 1
                    if i < 3:
                        print(f"  REJECTED: {rejection_reason}")
        
        print(f"\nFiltering results:")
        print(f"Actionable clusters: {len(filtered)}")
        
        if rejection_stats:
            print("Rejection reasons:")
            for reason, count in sorted(rejection_stats.items(), key=lambda x: x[1], reverse=True):
                print(f"  {reason}: {count}")
        
        # If still no results, try emergency relaxation
        if len(filtered) == 0 and len(clusters) > 0:
            print("\nNo patterns passed filtering. Applying emergency relaxation...")
            
            # Find the best cluster by correlation strength
            best_cluster = max(clusters, key=lambda x: abs(x['avg_corr']))
            best_cluster["quality_score"] = abs(best_cluster['avg_corr']) * 0.8  # Simple quality score
            
            print(f"Emergency acceptance of best cluster:")
            print(f"  Correlation: {best_cluster['avg_corr']:.3f}")
            print(f"  Size: {best_cluster['cluster_size']}")
            print(f"  Dimensions: {best_cluster['bbox_width']}x{best_cluster['bbox_height']}")
            print(f"  Years: {len(best_cluster['years_observed'])}")
            
            filtered = [best_cluster]
        
        return filtered
    
    def save_pattern_results(self, clusters, output_csv, shape, output_map, heatmap_data=None):
        """Save pattern detection results with enhanced visibility for sparse patterns"""
        print("[PIPELINE 2] Creating enhanced pattern visualization...")
        
        if not clusters:
            print("No actionable patterns found")
            return
        
        # Save CSV with enhanced information
        df = pd.DataFrame(clusters)
        df["human_rule"] = df.apply(
            lambda r: f"{'Linear' if r.get('is_linear', False) else 'Area'} pattern: "
                    f"{r['band1'].upper()} and {r['band2'].upper()} show {'strong positive' if r['avg_corr'] > 0 else 'strong negative'} correlation "
                    f"(r={r['avg_corr']:.3f}, quality={r['quality_score']:.3f}) across {len(r['years_observed'])} years. "
                    f"{'Length: ' + str(int(r['max_dimension'])) + ' pixels' if r.get('is_linear', False) else 'Size: ' + str(int(r['bbox_width'])) + 'x' + str(int(r['bbox_height'])) + ' pixels'}. "
                    f"ACTIONABLE: {'High priority' if r['quality_score'] > 0.7 else 'Medium priority' if r['quality_score'] > 0.5 else 'Monitor'}",
            axis=1
        )
        
        df = df.sort_values("quality_score", ascending=False)
        df.to_csv(output_csv, index=False)
        
        h, w = shape
        
        # Create a more subtle NDVI background for better pattern visibility
        ndvi_2025 = None
        if heatmap_data and 2025 in heatmap_data:
            if 'ndvi' in heatmap_data[2025]:
                ndvi_2025 = heatmap_data[2025]['ndvi'].copy()
                print(f"✅ Using NDVI 2025 from heatmap data")
            else:
                print(f"[WARN] NDVI not found in 2025 heatmap data, available bands: {list(heatmap_data[2025].keys())}")
        
        if ndvi_2025 is None and heatmap_data:
            for year in sorted(heatmap_data.keys(), reverse=True):
                if 'ndvi' in heatmap_data[year]:
                    ndvi_2025 = heatmap_data[year]['ndvi'].copy()
                    print(f"✅ Using NDVI from year {year} as background")
                    break
        
        # Create enhanced background with better contrast
        if ndvi_2025 is not None:
            # Normalize NDVI to 0-1 range
            valid_ndvi = ndvi_2025[~np.isnan(ndvi_2025)]
            if len(valid_ndvi) > 0:
                p_low, p_high = np.percentile(valid_ndvi, [5, 95])  # More aggressive clipping
                ndvi_normalized = np.clip(ndvi_2025, p_low, p_high)
                ndvi_normalized = (ndvi_normalized - p_low) / (p_high - p_low)
            else:
                ndvi_normalized = np.full_like(ndvi_2025, 0.5)
        else:
            ndvi_normalized = np.full((h, w), 0.5)
            print(f"[WARN] No NDVI data available, using neutral background")
        
        # Create RGB overlay with MUTED background for better pattern visibility
        background_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        
        # MUCH MORE SUBTLE background - reduce intensity significantly
        ndvi_muted = np.power(ndvi_normalized, 1.5) * 0.3  # Very muted background
        
        background_rgb[:, :, 1] = (ndvi_muted * 120).astype(np.uint8)  # Reduced green
        background_rgb[:, :, 0] = (ndvi_muted * 60).astype(np.uint8)   # Reduced red  
        background_rgb[:, :, 2] = (ndvi_muted * 30).astype(np.uint8)   # Reduced blue
        
        overlay = background_rgb.copy()
        
        # Draw patterns with MUCH LARGER and MORE VISIBLE markers
        sorted_clusters = sorted(clusters, key=lambda x: x["quality_score"])
        
        print(f"Drawing {len(clusters)} patterns on overlay...")
        
        for i, c in enumerate(sorted_clusters):
            quality = c["quality_score"]
            
            # FIXED: Define center coordinates here
            cy = c["center_y"]  # This was missing!
            cx = c["center_x"]  # This was missing!
            
            # MUCH LARGER markers - especially important for sparse patterns
            # Size based on quality (bigger = more important)
            if quality > 0.7:  # High priority
                base_marker_size = 60
                border_color = (255, 215, 0)  # Gold border for high priority
            elif quality > 0.5:  # Medium priority  
                base_marker_size = 45
                border_color = (255, 165, 0)  # Orange border for medium priority
            else:  # Lower priority
                base_marker_size = 30
                border_color = (255, 255, 255)  # White border for lower priority

            marker_size = max(base_marker_size, int(base_marker_size * quality))

            # Color based on correlation type (what the pattern means)
            if c["avg_corr"] > 0:
                color = (0, 200, 0)      # Green = positive correlation
                pattern_meaning = "Normal growth pattern"
            else:
                color = (220, 20, 20)    # Red = negative correlation  
                pattern_meaning = "Potential problem area"

            border_size = marker_size + 6
            pattern_type = "Linear (boundary/drainage)" if c.get("is_linear", False) else "Area (soil/crop issue)"
            print(f"Pattern {i+1}: {pattern_meaning}, Type: {pattern_type}, Quality: {'HIGH' if quality > 0.7 else 'MEDIUM' if quality > 0.5 else 'LOW'}")

            # Draw WHITE BORDER first (thicker)
            for dy in range(-border_size, border_size + 1):
                for dx in range(-border_size, border_size + 1):
                    py, px = cy + dy, cx + dx
                    if 0 <= py < h and 0 <= px < w:
                        if (dy*dy + dx*dx) <= border_size*border_size:
                            overlay[py, px] = border_color

            # Draw the pattern shape based on type
            if c.get("is_linear", False):
                # LINEAR PATTERNS: Draw THICK lines (field boundaries, drainage, irrigation)
                line_length = max(marker_size, int(c.get("max_dimension", 20) * 2))
                line_thickness = 8
                
                # Draw horizontal line
                for dy in range(-line_thickness//2, line_thickness//2 + 1):
                    for dx in range(-line_length, line_length + 1):
                        py, px = cy + dy, cx + dx
                        if 0 <= py < h and 0 <= px < w:
                            overlay[py, px] = color
                
                # Draw vertical line (crosshair effect for linear patterns)
                for dy in range(-line_length//3, line_length//3 + 1):  # Shorter vertical line
                    for dx in range(-line_thickness//2, line_thickness//2 + 1):
                        py, px = cy + dy, cx + dx
                        if 0 <= py < h and 0 <= px < w:
                            overlay[py, px] = color
            else:
                # AREA PATTERNS: Draw FILLED circles (soil issues, pest damage, etc.)
                for dy in range(-marker_size, marker_size + 1):
                    for dx in range(-marker_size, marker_size + 1):
                        py, px = cy + dy, cx + dx
                        if 0 <= py < h and 0 <= px < w:
                            if (dy*dy + dx*dx) <= marker_size*marker_size:
                                overlay[py, px] = color

            # Add YELLOW CROSSHAIR for precise location (both pattern types)
            crosshair_length = marker_size + 15
            crosshair_thickness = 3

            # Horizontal crosshair
            for dx in range(-crosshair_length, crosshair_length + 1):
                for t in range(-crosshair_thickness, crosshair_thickness + 1):
                    py, px = cy + t, cx + dx
                    if 0 <= py < h and 0 <= px < w:
                        overlay[py, px] = (255, 255, 0)  # Yellow crosshair

            # Vertical crosshair
            for dy in range(-crosshair_length, crosshair_length + 1):
                for t in range(-crosshair_thickness, crosshair_thickness + 1):
                    py, px = cy + dy, cx + t
                    if 0 <= py < h and 0 <= px < w:
                        overlay[py, px] = (255, 255, 0)  # Yellow crosshair
        
        # Save with enhanced title and information
        plt.figure(figsize=(20, 16))  # Larger figure
        plt.imshow(overlay)
        
        high_quality = sum(1 for c in clusters if c["quality_score"] > 0.7)
        medium_quality = sum(1 for c in clusters if 0.5 < c["quality_score"] <= 0.7)
        
        # Enhanced title with pattern details
        pattern_details = []
        for i, c in enumerate(sorted_clusters):
            corr_direction = "↗" if c["avg_corr"] > 0 else "↘"
            pattern_details.append(f"P{i+1}: {c['band1'].upper()}-{c['band2'].upper()} {corr_direction}{abs(c['avg_corr']):.2f}")
        
        pattern_summary = " | ".join(pattern_details[:3])  # Show first 3 patterns
        if len(pattern_details) > 3:
            pattern_summary += f" | +{len(pattern_details)-3} more"
        
        # Create a much better title and legend
        plt.title(f'Anomaly Heatmap - Band: NDVI for year 2025\n'
                f'{len(clusters)} patterns found requiring field investigation', 
                fontsize=18, pad=20, fontweight='bold')

        # Add a proper legend box
        legend_elements = []
        from matplotlib.patches import Circle
        from matplotlib.lines import Line2D

        # Create legend symbols for both pattern types
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=15, 
                label='GREEN CIRCLE = POSITIVE CORRELATION\n(Area issue: healthy growth, good soil)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=15,
                label='RED CIRCLE = NEGATIVE CORRELATION\n(Area issue: stress, disease, poor soil)'),
            Line2D([0], [0], marker='_', color='green', markersize=20, markeredgewidth=6,
                label='GREEN LINE = POSITIVE CORRELATION\n(Linear feature: functioning drainage/irrigation)'),
            Line2D([0], [0], marker='_', color='red', markersize=20, markeredgewidth=6,
                label='RED LINE = NEGATIVE CORRELATION\n(Linear feature: blocked drainage, broken irrigation)'),
            Line2D([0], [0], marker='+', color='yellow', markersize=20, markeredgewidth=4,
                label='YELLOW CROSSHAIR = EXACT GPS LOCATION'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='gold', markersize=12,
                label=f'GOLD BORDER = HIGH PRIORITY ({high_quality} patterns)'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='orange', markersize=10,
                label=f'ORANGE BORDER = MEDIUM PRIORITY ({medium_quality} patterns)')
        ]

        # Add legend to the plot
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), 
                fontsize=10, frameon=True, fancybox=True, shadow=True)
        
        # Add explanatory text box
        explanation_text = (
        "PATTERN SHAPES & MEANINGS:\n"
        "• CIRCLES = Area problems (soil, pests, nutrients) at specific locations\n"
        "• LINES = Linear features (drainage, irrigation, field boundaries)\n"
        "• GREEN = Normal/healthy correlation between vegetation indices\n"  
        "• RED = Problematic correlation (investigate for issues)\n"
        "• LARGER = Higher priority (investigate first)\n"
        "• GOLD BORDER = Urgent investigation needed\n\n"
        "FIELD INVESTIGATION:\n"
        "1. Use GPS to find the yellow crosshair coordinates\n"
        "2. RED CIRCLES: Check for soil compaction, pests, disease, nutrients\n"
        "3. RED LINES: Check drainage/irrigation systems for blockages\n"
        "4. GREEN patterns: Confirm good practices are working"
        )

        plt.figtext(0.02, 0.02, explanation_text, fontsize=9, 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8),
                verticalalignment='bottom')
        
        # Add coordinate grid for easier location identification
        grid_spacing = max(50, min(h, w) // 20)
        for i in range(0, h, grid_spacing):
            plt.axhline(y=i, color='lightgray', alpha=0.3, linewidth=0.5)
        for i in range(0, w, grid_spacing):
            plt.axvline(x=i, color='lightgray', alpha=0.3, linewidth=0.5)
        
        # Add scale reference
        scale_length = min(w, h) // 10
        scale_y = h - 50
        scale_x_start = 50
        scale_x_end = scale_x_start + scale_length
        
        # Draw scale bar
        plt.plot([scale_x_start, scale_x_end], [scale_y, scale_y], 'white', linewidth=6)
        plt.plot([scale_x_start, scale_x_start], [scale_y-10, scale_y+10], 'white', linewidth=4)
        plt.plot([scale_x_end, scale_x_end], [scale_y-10, scale_y+10], 'white', linewidth=4)
        plt.text(scale_x_start + scale_length//2, scale_y-25, f'{scale_length} pixels', 
                ha='center', va='top', color='white', fontweight='bold', fontsize=12,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
        
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_map, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ Enhanced pattern overlay map saved: {output_map}")
        print(f"✅ Pattern data saved: {output_csv}")
        print(f"   📍 {len(clusters)} total patterns identified")
        print(f"   🎯 {high_quality} high-priority patterns found")
        
        # Print pattern locations for verification
        print(f"\n🗺️  PATTERN LOCATIONS:")
        for i, c in enumerate(sorted_clusters):
            print(f"   Pattern {i+1}: ({c['center_x']}, {c['center_y']}) - "
                f"{c['band1'].upper()}-{c['band2'].upper()} correlation: {c['avg_corr']:.3f}")
    
    

# ==================== PIPELINE 3: CNN FEATURE EXTRACTION ====================
class CNNFeatureExtractor:
    """Enhanced CNN-based feature extraction"""
    
    def __init__(self, config):
        self.config = config
        if TORCH_AVAILABLE:
            self.embedder = EnhancedResNetEmbedder()
        else:
            self.embedder = None
            print("[WARN] PyTorch not available - CNN features disabled")
    
    def find_heatmaps(self, input_dir):
        """Find and organize heatmap files"""
        pattern = re.compile(r"heatmap_(?P<band>[A-Za-z]+)_(?P<year>\d{4})\.(png|tif|tiff|jpg|jpeg)$", re.IGNORECASE)
        files = [f for f in os.listdir(input_dir) if pattern.match(f)]
        grid = {}
        for f in files:
            m = pattern.match(f)
            band = m.group("band").lower()
            year = int(m.group("year"))
            grid.setdefault(year, {})[band] = os.path.join(input_dir, f)
        
        years = sorted([y for y in grid.keys() if all(b in grid[y] for b in [band.lower() for band in self.config.BANDS])])
        return grid, years
    
    def load_composite_for_year(self, paths_for_year):
        """Load and composite bands for a given year"""
        arrays = []
        for b in [band.lower() for band in self.config.BANDS]:
            p = paths_for_year[b]
            img = Image.open(p).convert("L")
            arr = np.array(img, dtype=np.float32) / 255.0
            arrays.append(arr)
        
        shapes = [a.shape for a in arrays]
        if len(set(shapes)) != 1:
            raise RuntimeError(f"Band shapes differ: {shapes}")
        
        return np.stack(arrays, axis=-1)
    
    def patches_from_image(self, img):
        """Extract patches from image"""
        H, W, _ = img.shape
        patches = []
        coords = []
        for y in range(0, H - self.config.PATCH_SIZE + 1, self.config.STRIDE):
            for x in range(0, W - self.config.PATCH_SIZE + 1, self.config.STRIDE):
                p = img[y:y+self.config.PATCH_SIZE, x:x+self.config.PATCH_SIZE, :]
                patches.append(p)
                coords.append((x,y))
        return np.array(patches), coords

class EnhancedResNetEmbedder:
    """Enhanced embedding with CNN architectures"""
    
    def __init__(self, model_type='resnet50'):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for CNN embeddings")
            
        if model_type == 'resnet50':
            m = resnet50(pretrained=True)
            self.dim = 2048
        else:
            m = resnet18(pretrained=True)
            self.dim = 512
            
        self.model = torch.nn.Sequential(*list(m.children())[:-1]).to(DEVICE).eval()
        
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
        
        self.augment_transform = T.Compose([
            T.ToTensor(),
            T.RandomHorizontalFlip(p=0.3),
            T.RandomVerticalFlip(p=0.3),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
        
    def embed_batch(self, np_patches, augment=False):
        """Extract embeddings from patches"""
        imgs = []
        transform = self.augment_transform if augment else self.transform
        
        for p in np_patches:
            p_filtered = gaussian_filter(p, sigma=0.5)
            pil = Image.fromarray((np.clip(p_filtered, 0, 1)*255).astype(np.uint8))
            t = transform(pil).to(DEVICE)
            imgs.append(t)
            
        batch = torch.stack(imgs)
        with torch.no_grad():
            out = self.model(batch)
            out = out.view(out.size(0), -1).cpu().numpy()
        return out

# ==================== PIPELINE 4: PREDICTIVE MODELING ====================
class PredictiveModeler:
    """Advanced predictive modeling with ensemble methods"""
    
    def __init__(self, config):
        self.config = config
    
    def advanced_temporal_features(self, ts, years):
        """Extract advanced temporal features"""
        ts = np.array(ts)
        years = np.array(years)
        
        features = {}
        
        # Basic statistics
        features['mean'] = np.mean(ts)
        features['std'] = np.std(ts)
        features['min'] = np.min(ts)
        features['max'] = np.max(ts)
        features['range'] = features['max'] - features['min']
        features['cv'] = features['std'] / (features['mean'] + 1e-8)
        
        # Trend analysis
        if len(ts) >= 3:
            linear_coef = np.polyfit(years, ts, 1)
            features['linear_slope'] = linear_coef[0]
            features['linear_intercept'] = linear_coef[1]
            
            if len(ts) >= 4:
                quad_coef = np.polyfit(years, ts, 2)
                features['quad_a'] = quad_coef[0]
                features['quad_b'] = quad_coef[1]
                features['quad_c'] = quad_coef[2]
                
                mid_point = len(ts) // 2
                early_slope = np.polyfit(years[:mid_point+1], ts[:mid_point+1], 1)[0]
                late_slope = np.polyfit(years[mid_point:], ts[mid_point:], 1)[0]
                features['trend_acceleration'] = late_slope - early_slope
            else:
                features['quad_a'] = features['quad_b'] = features['quad_c'] = 0
                features['trend_acceleration'] = 0
        else:
            features.update({
                'linear_slope': 0, 'linear_intercept': np.mean(ts),
                'quad_a': 0, 'quad_b': 0, 'quad_c': 0,
                'trend_acceleration': 0
            })
        
        # Seasonal patterns
        if len(ts) >= 6:
            try:
                smoothed = savgol_filter(ts, min(5, len(ts)//2*2+1), 2)
                features['smoothness'] = np.mean(np.abs(ts - smoothed))
                
                diff1 = np.diff(ts)
                diff2 = np.diff(diff1)
                features['turning_points'] = np.sum(np.abs(diff2) > np.std(diff2))
                features['volatility'] = np.std(diff1)
            except:
                features['smoothness'] = features['turning_points'] = features['volatility'] = 0
        else:
            features['smoothness'] = features['turning_points'] = features['volatility'] = 0
        
        # Recent vs historical
        if len(ts) >= 4:
            recent = ts[-2:]
            historical = ts[:-2]
            features['recent_vs_historical'] = np.mean(recent) - np.mean(historical)
            features['recent_trend'] = np.polyfit(years[-2:], recent, 1)[0] if len(recent) >= 2 else 0
        else:
            features['recent_vs_historical'] = features['recent_trend'] = 0
            
        return features
    
    def compute_cross_band_features(self, ndvi_ts, evi_ts, ndwi_ts, years):
        """Compute cross-band relationships"""
        features = {}
        
        # Cross-correlations at different lags
        for lag in range(-2, 3):
            try:
                if lag == 0:
                    features[f'ndvi_evi_corr_lag{lag}'] = np.corrcoef(ndvi_ts, evi_ts)[0,1]
                    features[f'ndvi_ndwi_corr_lag{lag}'] = np.corrcoef(ndvi_ts, ndwi_ts)[0,1]
                    features[f'evi_ndwi_corr_lag{lag}'] = np.corrcoef(evi_ts, ndwi_ts)[0,1]
                elif lag > 0 and lag < len(ndvi_ts):
                    features[f'ndvi_evi_corr_lag{lag}'] = np.corrcoef(ndvi_ts[:-lag], evi_ts[lag:])[0,1]
                    features[f'ndvi_ndwi_corr_lag{lag}'] = np.corrcoef(ndvi_ts[:-lag], ndwi_ts[lag:])[0,1]
                    features[f'evi_ndwi_corr_lag{lag}'] = np.corrcoef(evi_ts[:-lag], ndwi_ts[lag:])[0,1]
                elif lag < 0 and abs(lag) < len(ndvi_ts):
                    features[f'ndvi_evi_corr_lag{lag}'] = np.corrcoef(ndvi_ts[-lag:], evi_ts[:lag])[0,1]
                    features[f'ndvi_ndwi_corr_lag{lag}'] = np.corrcoef(ndvi_ts[-lag:], ndwi_ts[:lag])[0,1]
                    features[f'evi_ndwi_corr_lag{lag}'] = np.corrcoef(evi_ts[-lag:], ndwi_ts[:lag])[0,1]
                else:
                    features[f'ndvi_evi_corr_lag{lag}'] = 0
                    features[f'ndvi_ndwi_corr_lag{lag}'] = 0
                    features[f'evi_ndwi_corr_lag{lag}'] = 0
            except:
                features[f'ndvi_evi_corr_lag{lag}'] = 0
                features[f'ndvi_ndwi_corr_lag{lag}'] = 0
                features[f'evi_ndwi_corr_lag{lag}'] = 0
        
        # Ratio-based features
        ndvi_arr, evi_arr, ndwi_arr = np.array(ndvi_ts), np.array(evi_ts), np.array(ndwi_ts)
        
        evi_safe = np.where(evi_arr == 0, 1e-8, evi_arr)
        ndwi_safe = np.where(ndwi_arr == 0, 1e-8, ndwi_arr)
        
        features['ndvi_evi_ratio_mean'] = np.mean(ndvi_arr / evi_safe)
        features['ndvi_evi_ratio_std'] = np.std(ndvi_arr / evi_safe)
        features['ndvi_ndwi_ratio_mean'] = np.mean(ndvi_arr / (ndwi_safe + 0.5))
        
        # Composite indices
        features['vegetation_water_balance'] = np.mean((ndvi_arr + evi_arr) - ndwi_arr)
        features['vegetation_stress_proxy'] = np.std((ndvi_arr - evi_arr) + ndwi_arr)
        
        return features

class EnsembleModel:
    """Enhanced ensemble model"""
    
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.weights = {}
        self.scaler = RobustScaler()
        self.feature_selector = None
        
    def fit(self, X, y, feature_names=None):
        """Train ensemble model"""
        print("[PIPELINE 4] Training ensemble model...")
        
        X_scaled = self.scaler.fit_transform(X)
        
        if X_scaled.shape[1] > 100:
            self.feature_selector = SelectKBest(score_func=f_regression, k=min(100, X_scaled.shape[1]//2))
            X_scaled = self.feature_selector.fit_transform(X_scaled, y.ravel() if y.ndim > 1 else y)
            print(f"Selected {X_scaled.shape[1]} features out of {X.shape[1]}")
        
        # Initialize models
        if CATBOOST_AVAILABLE:
            self.models['catboost'] = CatBoostRegressor(
                loss_function='MultiRMSE' if y.ndim > 1 else 'RMSE',
                iterations=self.config.N_ESTIMATORS,
                depth=self.config.MAX_DEPTH,
                learning_rate=self.config.LEARNING_RATE,
                verbose=False,
                random_seed=42
            )
        
        if XGBOOST_AVAILABLE:
            if y.ndim > 1:
                from sklearn.multioutput import MultiOutputRegressor
                self.models['xgboost'] = MultiOutputRegressor(
                    xgb.XGBRegressor(
                        n_estimators=self.config.N_ESTIMATORS,
                        max_depth=self.config.MAX_DEPTH,
                        learning_rate=self.config.LEARNING_RATE,
                        random_state=42
                    )
                )
            else:
                self.models['xgboost'] = xgb.XGBRegressor(
                    n_estimators=self.config.N_ESTIMATORS,
                    max_depth=self.config.MAX_DEPTH,
                    learning_rate=self.config.LEARNING_RATE,
                    random_state=42
                )
        
        # Random Forest
        if y.ndim > 1:
            from sklearn.multioutput import MultiOutputRegressor
            self.models['rf'] = MultiOutputRegressor(
                RandomForestRegressor(
                    n_estimators=self.config.N_ESTIMATORS,
                    max_depth=self.config.MAX_DEPTH,
                    random_state=42,
                    n_jobs=-1
                )
            )
        else:
            self.models['rf'] = RandomForestRegressor(
                n_estimators=self.config.N_ESTIMATORS,
                max_depth=self.config.MAX_DEPTH,
                random_state=42,
                n_jobs=-1
            )
        
        # Gradient Boosting
        if y.ndim > 1:
            from sklearn.multioutput import MultiOutputRegressor
            self.models['gb'] = MultiOutputRegressor(
                GradientBoostingRegressor(
                    n_estimators=self.config.N_ESTIMATORS//2,
                    max_depth=self.config.MAX_DEPTH,
                    learning_rate=self.config.LEARNING_RATE,
                    random_state=42
                )
            )
        else:
            self.models['gb'] = GradientBoostingRegressor(
                n_estimators=self.config.N_ESTIMATORS//2,
                max_depth=self.config.MAX_DEPTH,
                learning_rate=self.config.LEARNING_RATE,
                random_state=42
            )
        
        # Train models and compute weights
        tscv = TimeSeriesSplit(n_splits=self.config.N_SPLITS)
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            try:
                model.fit(X_scaled, y)
                cv_scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='neg_mean_absolute_error')
                self.weights[name] = np.mean(-cv_scores)
                print(f"{name} CV MAE: {self.weights[name]:.4f}")
            except Exception as e:
                print(f"Failed to train {name}: {e}")
                if name in self.weights:
                    del self.weights[name]
        
        # Normalize weights
        if self.weights:
            total_weight = sum(1/w for w in self.weights.values())
            self.weights = {k: (1/v)/total_weight for k, v in self.weights.items()}
            print("Model weights:", self.weights)
        
    def predict(self, X):
        """Make ensemble predictions"""
        X_scaled = self.scaler.transform(X)
        if self.feature_selector:
            X_scaled = self.feature_selector.transform(X_scaled)
        
        if not self.weights:
            return np.zeros((X.shape[0], 3))
        
        predictions = []
        for name, model in self.models.items():
            if name in self.weights:
                pred = model.predict(X_scaled)
                if pred.ndim == 1:
                    pred = pred.reshape(-1, 1)
                predictions.append(pred * self.weights[name])
        
        return np.sum(predictions, axis=0)

# ==================== MAIN PIPELINE CLASS ====================
class FarmEyePipeline:
    """Complete FarmEye pipeline integrating all components"""
    
    def __init__(self, config=None):
        self.config = config or FarmEyeConfig()
        self.trend_analyzer = TrendAnalyzer(self.config)
        self.anomaly_detector = AnomalyDetector(self.config)
        self.cnn_extractor = CNNFeatureExtractor(self.config)
        self.predictive_modeler = PredictiveModeler(self.config)
        
        # Create output directory
        os.makedirs(self.config.OUTPUT_FOLDER, exist_ok=True)
        
    def run_complete_pipeline(self):
        """Execute the complete FarmEye pipeline"""
        print("="*60)
        print("STARTING FARMEYE COMPLETE PIPELINE")
        print("="*60)
        
        # ========== PIPELINE 1: TREND ANALYSIS ==========
        print("\n" + "="*50)
        print("PIPELINE 1: TREND ANALYSIS")
        print("="*50)
        
        # Load satellite data stack
        stack = self.trend_analyzer.load_multiyear_stack()
        
        # Output paths for Pipeline 1
        slope_csv = os.path.join(self.config.OUTPUT_FOLDER, 'slopes_intercepts.csv')
        residual_csv = os.path.join(self.config.OUTPUT_FOLDER, 'residuals.csv')
        slope_zscore_csv = os.path.join(self.config.OUTPUT_FOLDER, 'slope_zscores.csv')
        residual_zscore_csv = os.path.join(self.config.OUTPUT_FOLDER, 'residual_zscores.csv')
        composite_heatmap = os.path.join(self.config.OUTPUT_FOLDER, 'composite_impact_heatmap.png')
        heatmap_folder = os.path.join(self.config.OUTPUT_FOLDER, 'heatmaps_individual')
        
        # Execute trend analysis
        self.trend_analyzer.compute_theil_sen_slopes(stack, slope_csv)
        self.trend_analyzer.compute_residuals(stack, slope_csv, residual_csv)
        self.trend_analyzer.compute_slope_zscores(slope_csv, slope_zscore_csv)
        self.trend_analyzer.compute_residual_zscores(residual_csv, residual_zscore_csv)
        self.trend_analyzer.generate_composite_heatmap(slope_zscore_csv, composite_heatmap)
        self.trend_analyzer.generate_individual_heatmaps(residual_zscore_csv, heatmap_folder)
        # Generate combined heatmap grid - NEW ADDITION
        combined_heatmap_grid = os.path.join(self.config.OUTPUT_FOLDER, 'all_heatmaps_grid.png')
        self.trend_analyzer.generate_combined_heatmap_grid(heatmap_folder, combined_heatmap_grid)
        
        # ========== PIPELINE 2: ANOMALY DETECTION ==========
        print("\n" + "="*50)
        print("PIPELINE 2: ENHANCED ANOMALY DETECTION")
        print("="*50)
        
        # Load heatmaps and detect patterns
        heatmap_data = self.anomaly_detector.load_heatmaps(heatmap_folder)
        
        if not heatmap_data:
            print("[WARN] No heatmap data found, skipping anomaly detection")
        else:
            # Execute pattern detection pipeline
            patterns = self.anomaly_detector.detect_pixel_patterns(heatmap_data)
            clusters = self.anomaly_detector.cluster_patterns(patterns)
            actionable_clusters = self.anomaly_detector.filter_actionable_clusters(clusters)
            
            # Get dimensions for coordinate validation
            h, w = next(iter(heatmap_data.values()))[self.config.BANDS[0].lower()].shape
            
            if actionable_clusters:
                # Validate and fix coordinates
                actionable_clusters = self.anomaly_detector.validate_pattern_coordinates(actionable_clusters, (h, w))
                
                # Output paths for Pipeline 2
                pattern_csv = os.path.join(self.config.OUTPUT_FOLDER, 'actionable_patterns.csv')
                pattern_map = os.path.join(self.config.OUTPUT_FOLDER, 'pattern_overlay_map.png')
                navigation_guide = os.path.join(self.config.OUTPUT_FOLDER, 'field_navigation_guide.txt')
                
                # Save comprehensive results with NDVI background and enhanced visibility
                self.anomaly_detector.save_pattern_results(actionable_clusters, pattern_csv, (h, w), pattern_map, heatmap_data)
                
                # Create field navigation guide for farmers
                self.anomaly_detector.create_field_navigation_guide(actionable_clusters, navigation_guide)
                
                print(f"✅ Enhanced anomaly detection completed:")
                print(f"   📊 {len(actionable_clusters)} actionable patterns identified")
                print(f"   🗺️ Farmer-friendly maps with NDVI background generated")
                print(f"   📍 Field navigation guide created")
                
            else:
                print("No actionable patterns found after filtering")
        
        # ========== PIPELINE 3 & 4: PREDICTIVE MODELING ==========
        print("\n" + "="*50)
        print("PIPELINE 3 & 4: PREDICTIVE MODELING")
        print("="*50)
        
        # Find and load composites
        grid, years = self.cnn_extractor.find_heatmaps(heatmap_folder)
        if not years:
            print("[WARN] No complete years found for prediction")
            return
        
        print(f"Years available for prediction: {years}")
        
        if len(years) < self.config.MIN_YEARS_REQ:
            print(f"[WARN] Only {len(years)} years available, minimum is {self.config.MIN_YEARS_REQ}")
            if len(years) < 3:
                print("Insufficient data for prediction, skipping")
                return
        
        # Load composites
        composites = {}
        for y in tqdm(years, desc="Loading composites"):
            composites[y] = self.cnn_extractor.load_composite_for_year(grid[y])
        
        H, W, _ = next(iter(composites.values())).shape
        print(f"Composite shape: {H} x {W}")
        
        # Extract patches and coordinates
        sample = next(iter(composites.values()))
        patches0, coords = self.cnn_extractor.patches_from_image(sample)
        N_patches = len(coords)
        print(f"Patches per year: {N_patches} (patch size: {self.config.PATCH_SIZE}, stride: {self.config.STRIDE})")
        
        # Extract CNN embeddings if available
        year_embs = {}
        if TORCH_AVAILABLE and self.cnn_extractor.embedder:
            print("Extracting CNN embeddings...")
            D = self.cnn_extractor.embedder.dim
            
            for y in years:
                print(f"Extracting embeddings for year {y}...")
                patches_y, _ = self.cnn_extractor.patches_from_image(composites[y])
                if patches_y.shape[0] != N_patches:
                    raise RuntimeError("Patch count mismatch across years")
                
                embs = np.zeros((N_patches, D), dtype=np.float32)
                for i in tqdm(range(0, N_patches, self.config.BATCH_SIZE), desc=f"CNN embeddings {y}"):
                    batch = patches_y[i:i+self.config.BATCH_SIZE]
                    emb = self.cnn_extractor.embedder.embed_batch(batch, augment=True)
                    embs[i:i+emb.shape[0]] = emb
                year_embs[y] = embs
        else:
            print("[INFO] CNN embeddings disabled, using statistical features only")
            year_embs = {}
        
        # Extract comprehensive features
        print("Extracting comprehensive features...")
        patch_info = []
        
        for idx in tqdm(range(N_patches), desc="Processing patches"):
            yrs = []
            ndvi_ts = []
            evi_ts = []
            ndwi_ts = []
            E_list = []
            
            # Check if this patch has valid data across years
            valid_patch = True
            temp_means = []
            
            for y in years:
                x0, y0 = coords[idx]
                patch = composites[y][y0:y0+self.config.PATCH_SIZE, x0:x0+self.config.PATCH_SIZE, :]
                
                # Extract band statistics
                ndvi_patch = patch[..., [band.lower() for band in self.config.BANDS].index("ndvi")]
                evi_patch = patch[..., [band.lower() for band in self.config.BANDS].index("evi")]
                ndwi_patch = patch[..., [band.lower() for band in self.config.BANDS].index("ndwi")]
                
                # Check for valid data (not all zeros, not all NaN, reasonable values)
                ndvi_mean = float(ndvi_patch.mean())
                evi_mean = float(evi_patch.mean())
                ndwi_mean = float(ndwi_patch.mean())
                
                # Skip patches with invalid data
                if (np.isnan(ndvi_mean) or np.isnan(evi_mean) or np.isnan(ndwi_mean) or
                    (ndvi_mean == 0 and evi_mean == 0 and ndwi_mean == 0) or
                    ndvi_mean < 0.01 or ndvi_mean > 1.0 or  # NDVI should be 0-1
                    evi_mean < 0.01 or evi_mean > 1.0 or    # EVI should be 0-1
                    abs(ndwi_mean) > 1.0):                  # NDWI should be -1 to 1
                    valid_patch = False
                    break
                    
                temp_means.append([ndvi_mean, evi_mean, ndwi_mean])
            
            if not valid_patch:
                continue  # Skip this patch entirely
            
            # If we get here, patch is valid, so process it
            for i, y in enumerate(years):
                yrs.append(y)
                ndvi_ts.append(temp_means[i][0])
                evi_ts.append(temp_means[i][1])
                ndwi_ts.append(temp_means[i][2])
                
                # Add CNN embeddings if available
                if year_embs and y in year_embs:
                    E_list.append(year_embs[y][idx])
                else:
                    E_list.append(np.zeros(512))  # Dummy embedding
            
            if len(yrs) < self.config.MIN_YEARS_REQ and len(yrs) < 3:
                continue
            
            # Extract temporal features for each band
            ndvi_features = self.predictive_modeler.advanced_temporal_features(ndvi_ts, yrs)
            evi_features = self.predictive_modeler.advanced_temporal_features(evi_ts, yrs)
            ndwi_features = self.predictive_modeler.advanced_temporal_features(ndwi_ts, yrs)
            
            # Cross-band features
            cross_features = self.predictive_modeler.compute_cross_band_features(ndvi_ts, evi_ts, ndwi_ts, yrs)
            
            # Embedding features
            if E_list and len(E_list) > 0:
                E = np.stack(E_list, axis=0)
                emb_features = self.predictive_modeler.advanced_temporal_features(np.mean(E, axis=1), yrs)
            else:
                emb_features = {}
            
            # Combine all features
            all_features = {}
            for k, v in ndvi_features.items():
                all_features[f'ndvi_{k}'] = v
            for k, v in evi_features.items():
                all_features[f'evi_{k}'] = v
            for k, v in ndwi_features.items():
                all_features[f'ndwi_{k}'] = v
            for k, v in cross_features.items():
                all_features[k] = v
            for k, v in emb_features.items():
                all_features[f'emb_{k}'] = v
            
            patch_info.append({
                "idx": idx,
                "coord": coords[idx],
                "years": yrs,
                "ndvi_ts": ndvi_ts,
                "evi_ts": evi_ts,
                "ndwi_ts": ndwi_ts,
                "features": all_features
            })
        
        print(f"Patches with features: {len(patch_info)}")
        
        if not patch_info:
            print("[WARN] No patches with sufficient data for modeling")
            return
        
        # Build training data
        print("Building training data...")
        X = []
        y = []
        feature_names = list(patch_info[0]["features"].keys()) if patch_info else []
        
        for p in tqdm(patch_info, desc="Building training data"):
            yrs = p["years"]
            ndvi_ts = p["ndvi_ts"]
            evi_ts = p["evi_ts"]
            ndwi_ts = p["ndwi_ts"]
            
            for t, year in enumerate(yrs):
                feat = list(p["features"].values())
                X.append(feat)
                y.append([ndvi_ts[t], evi_ts[t], ndwi_ts[t]])
        
        X = np.array(X)
        y = np.array(y)
        print(f"Training data shape: X={X.shape}, y={y.shape}")
        
        # Train ensemble model
        model = EnsembleModel(self.config)
        model.fit(X, y, feature_names)
        
        # Generate predictions
        print("Making predictions...")
        predictions = []
        next_year = years[-1] + 1
        
        for p in tqdm(patch_info, desc="Making predictions"):
            feat = list(p["features"].values())
            feat_array = np.array([feat])
            
            pred_vals = model.predict(feat_array)[0]
            
            # Get last observed values
            ndvi_last = p["ndvi_ts"][-1]
            evi_last = p["evi_ts"][-1]
            ndwi_last = p["ndwi_ts"][-1]
            
            predictions.append({
                "patch_idx": p["idx"],
                "center_x": p["coord"][0] + self.config.PATCH_SIZE//2,
                "center_y": p["coord"][1] + self.config.PATCH_SIZE//2,
                "years_observed": p["years"],
                "ndvi_pred_next": pred_vals[0],
                "evi_pred_next": pred_vals[1],
                "ndwi_pred_next": pred_vals[2],
                "ndvi_last": ndvi_last,
                "evi_last": evi_last,
                "ndwi_last": ndwi_last,
                "delta_ndvi": pred_vals[0] - ndvi_last,
                "delta_evi": pred_vals[1] - evi_last,
                "delta_ndwi": pred_vals[2] - ndwi_last,
                "confidence": 1.0
            })
        
        # Save predictions
        prediction_csv = os.path.join(self.config.OUTPUT_FOLDER, 'predictions_enhanced.csv')
        df_preds = pd.DataFrame(predictions)
        df_preds.to_csv(prediction_csv, index=False)
        
        print(f"✅ Predictions saved to {prediction_csv}")
        
        # Generate prediction summary
        print("\n" + "="*50)
        print("PREDICTION SUMMARY")
        print("="*50)
        print(f"Predictions for year: {next_year}")
        print(f"Total patches predicted: {len(predictions)}")
        print(f"Mean NDVI change: {df_preds['delta_ndvi'].mean():.4f} ± {df_preds['delta_ndvi'].std():.4f}")
        print(f"Mean EVI change: {df_preds['delta_evi'].mean():.4f} ± {df_preds['delta_evi'].std():.4f}")
        print(f"Mean NDWI change: {df_preds['delta_ndwi'].mean():.4f} ± {df_preds['delta_ndwi'].std():.4f}")
        
        # Generate prediction map
        self.generate_prediction_map(predictions, H, W)
        
        print("\n" + "="*60)
        print("FARMEYE COMPLETE PIPELINE FINISHED")
        print("="*60)
        print(f"All outputs saved to: {self.config.OUTPUT_FOLDER}")
        print("Pipeline components completed:")
        print("  ✅ Pipeline 1: Trend Analysis")
        print("  ✅ Pipeline 2: Anomaly Detection") 
        print("  ✅ Pipeline 3: CNN Feature Extraction")
        print("  ✅ Pipeline 4: Predictive Modeling")
        print("="*60)
    
    def generate_prediction_map(self, predictions, height, width):
        """Generate visualization map for predictions - FIXED VERSION"""
        print("Generating prediction visualization map...")
        
        # Create prediction maps for each band
        maps = {
            'ndvi': np.full((height, width), np.nan),
            'evi': np.full((height, width), np.nan), 
            'ndwi': np.full((height, width), np.nan)
        }
        
        delta_maps = {
            'ndvi': np.full((height, width), np.nan),
            'evi': np.full((height, width), np.nan),
            'ndwi': np.full((height, width), np.nan)
        }
        
        # Fill prediction maps - FIXED to handle patch-based coordinates
        # Use larger patches for better visibility and reduce density
        # Fill prediction maps - FIXED to handle patch-based coordinates
        patch_radius = max(1, self.config.PATCH_SIZE // 2)  # Radius to fill around each prediction

        for pred in predictions:
            center_y, center_x = pred['center_y'], pred['center_x']
            
            # Fill a small area around each prediction point for better visibility
            for dy in range(-patch_radius, patch_radius + 1):
                for dx in range(-patch_radius, patch_radius + 1):
                    y, x = center_y + dy, center_x + dx
                    if 0 <= y < height and 0 <= x < width:
                        maps['ndvi'][y, x] = pred['ndvi_pred_next']
                        maps['evi'][y, x] = pred['evi_pred_next']
                        maps['ndwi'][y, x] = pred['ndwi_pred_next']
                        
                        delta_maps['ndvi'][y, x] = pred['delta_ndvi']
                        delta_maps['evi'][y, x] = pred['delta_evi']
                        delta_maps['ndwi'][y, x] = pred['delta_ndwi']
        
        # Apply smoothing for better visualization
        # Apply stronger smoothing for better visualization with larger patches
        from scipy.ndimage import gaussian_filter

        for band in ['ndvi', 'evi', 'ndwi']:
            # Only smooth where we have data
            valid_mask = ~np.isnan(maps[band])
            if np.any(valid_mask):
                # Create a temporary array with zeros where NaN
                temp_map = np.where(valid_mask, maps[band], 0)
                temp_mask = valid_mask.astype(float)
                
                # Use stronger smoothing for larger patches
                smoothed_map = gaussian_filter(temp_map, sigma=3.0)  # Increased from 1.0
                smoothed_mask = gaussian_filter(temp_mask, sigma=3.0)
                
                # Restore smoothed values only where we had original data
                maps[band] = np.where(smoothed_mask > 0.2, smoothed_map / (smoothed_mask + 1e-8), np.nan)
            
            # Same for delta maps
            valid_mask = ~np.isnan(delta_maps[band])
            if np.any(valid_mask):
                temp_map = np.where(valid_mask, delta_maps[band], 0)
                temp_mask = valid_mask.astype(float)
                
                smoothed_map = gaussian_filter(temp_map, sigma=1.0)
                smoothed_mask = gaussian_filter(temp_mask, sigma=1.0)
                
                delta_maps[band] = np.where(smoothed_mask > 0.1, smoothed_map / (smoothed_mask + 1e-8), np.nan)
        
        # Generate plots with proper handling of NaN values
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Prediction maps (top row)
        for i, (band, data) in enumerate(maps.items()):
            valid_data = data[~np.isnan(data)]
            if len(valid_data) > 10:  # Need sufficient data points
                vmin, vmax = np.percentile(valid_data, [5, 95])  # Use wider percentile range
                
                # Create a masked array for better visualization
                masked_data = np.ma.masked_where(np.isnan(data), data)
                
                im = axes[0, i].imshow(masked_data, cmap='RdYlGn', vmin=vmin, vmax=vmax, interpolation='bilinear')
                axes[0, i].set_title(f'{band.upper()} Predictions\n(n={len(valid_data)} points)', fontsize=12)
                plt.colorbar(im, ax=axes[0, i], shrink=0.8)
            else:
                axes[0, i].text(0.5, 0.5, f'{band.upper()}\nInsufficient Data\n({len(valid_data)} points)', 
                            ha='center', va='center', transform=axes[0, i].transAxes, fontsize=12)
                axes[0, i].set_title(f'{band.upper()} Predictions', fontsize=12)
            
            axes[0, i].axis('off')
        
        # Delta maps (bottom row)
        for i, (band, data) in enumerate(delta_maps.items()):
            valid_data = data[~np.isnan(data)]
            if len(valid_data) > 10:
                # For delta maps, use symmetric colormap centered on zero
                abs_max = np.percentile(np.abs(valid_data), 90)  # Use 90th percentile for better contrast
                
                masked_data = np.ma.masked_where(np.isnan(data), data)
                
                im = axes[1, i].imshow(masked_data, cmap='RdBu_r', vmin=-abs_max, vmax=abs_max, interpolation='bilinear')
                axes[1, i].set_title(f'{band.upper()} Change\n(Δ = ±{abs_max:.3f})', fontsize=12)
                plt.colorbar(im, ax=axes[1, i], shrink=0.8)
            else:
                axes[1, i].text(0.5, 0.5, f'{band.upper()}\nChange\nInsufficient Data\n({len(valid_data)} points)', 
                            ha='center', va='center', transform=axes[1, i].transAxes, fontsize=12)
                axes[1, i].set_title(f'{band.upper()} Change', fontsize=12)
            
            axes[1, i].axis('off')
        
        # Add overall statistics
        next_year = max([pred['years_observed'][-1] for pred in predictions]) + 1 if predictions else "Unknown"
        plt.suptitle(f'Agricultural Index Predictions for Year {next_year}\n'
                    f'Based on {len(predictions)} patch predictions', fontsize=16, y=0.95)
        
        plt.tight_layout()
        
        prediction_map_path = os.path.join(self.config.OUTPUT_FOLDER, 'prediction_maps.png')
        plt.savefig(prediction_map_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Print statistics
        for band in ['ndvi', 'evi', 'ndwi']:
            valid_preds = [p[f'{band}_pred_next'] for p in predictions if not np.isnan(p[f'{band}_pred_next'])]
            valid_deltas = [p[f'delta_{band}'] for p in predictions if not np.isnan(p[f'delta_{band}'])]
            
            if valid_preds:
                print(f"  {band.upper()}: {len(valid_preds)} predictions, "
                    f"range {np.min(valid_preds):.3f} to {np.max(valid_preds):.3f}, "
                    f"mean change {np.mean(valid_deltas):.4f}")
            else:
                print(f"  {band.upper()}: No valid predictions")
        
        print(f"✅ Prediction maps saved to {prediction_map_path}")
    
# ==================== MAIN EXECUTION ====================
def main():
    """Main function to run the complete FarmEye pipeline"""
    
    # Initialize configuration
    config = FarmEyeConfig()
    
    # Validate input directory
    if not os.path.exists(config.INPUT_FOLDER):
        print(f"[ERROR] Input folder not found: {config.INPUT_FOLDER}")
        print("Please ensure the input folder contains:")
        print("  - Multi-band satellite images (YEAR.tif format)")
        print("  - Years:", config.YEARS)
        print("  - Bands:", config.BANDS)
        return
    
    # Check for required input files
    required_files = [f"{year}.tif" for year in config.YEARS]
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(config.INPUT_FOLDER, file)):
            missing_files.append(file)
    
    if missing_files:
        print(f"[WARN] Missing input files: {missing_files}")
        print("Pipeline will proceed with available years")
        # Update config with available years
        available_years = []
        for year in config.YEARS:
            if os.path.exists(os.path.join(config.INPUT_FOLDER, f"{year}.tif")):
                available_years.append(year)
        
        if len(available_years) < 3:
            print("[ERROR] Need at least 3 years of data")
            return
            
        config.YEARS = available_years
        config.NUM_YEARS = len(available_years)
        print(f"Using available years: {config.YEARS}")
    
    # Initialize and run pipeline
    try:
        pipeline = FarmEyePipeline(config)
        pipeline.run_complete_pipeline()
        
        print(f"\n[SUCCESS] Pipeline completed successfully!")
        print(f"Check outputs in: {config.OUTPUT_FOLDER}")
        
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    

if __name__ == "__main__":
    main()