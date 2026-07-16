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
    
    def __init__(self):
        # Input/Output paths
        self.INPUT_FOLDER = r"C:\Users\Faster\Downloads\FarmEyeInputs"
        self.OUTPUT_FOLDER = r"C:\Users\Faster\Downloads\FarmEyeOutputs"
        
        # Years and bands configuration
        self.YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
        self.BANDS = ['NDVI', 'EVI', 'NDWI']
        self.NUM_BANDS = len(self.BANDS)
        self.NUM_YEARS = len(self.YEARS)
        
        # Chunk processing parameters (optimized for speed)
        self.CHUNK_SIZE = 500
        self.CHUNK_STEP = 50
        
        # Pipeline 1: Trend Analysis
        self.TREND_BATCH_SIZE = 500
        
        # Pipeline 2: Anomaly Detection  
        self.RECENT_YEAR_THRESHOLD = 2024
        self.MIN_RECUR_YEARS = 2
        self.MIN_TEMPORAL_SPAN = 2
        self.CORR_THRESHOLD = 0.7
        self.MIN_CORR_CONSISTENCY = 0.7
        self.CLUSTER_EPS = 50
        self.CLUSTER_MIN_SAMPLES = 2
        self.MIN_CLUSTER_LINEAR_LENGTH = 30
        self.MIN_CLUSTER_AREA_SIZE = 5
        self.MIN_CLUSTER_SIZE = 1
        self.MIN_SPATIAL_DENSITY = 0.01
        self.MAX_CLUSTER_BBOX_SIZE = 500
        self.MIN_LINEAR_ASPECT_RATIO = 2
        self.MIN_SIGNAL_VARIANCE = 0.01
        self.MIN_SIGNAL_RANGE = 0.05
        self.MAX_NOISE_RATIO = 0.5
        
        # Pipeline 3: Pattern Recognition
        self.PATCH_SIZE = 64
        self.STRIDE = 60
        self.BATCH_SIZE = 64
        self.PCA_DIM = 4
        
        # Pipeline 4: Predictive Modeling
        self.MIN_YEARS_REQ = 2
        self.N_ESTIMATORS = 10
        self.MAX_DEPTH = 2
        self.LEARNING_RATE = 0.5
        self.TEST_SIZE = 0.2
        self.N_SPLITS = 2

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
            plt.title(f"Anomaly Heatmap - Band: {band} Year: {year}")
            plt.axis('off')
            plt.tight_layout()

            heatmap_path = os.path.join(output_folder, f"heatmap_{band}_{year}.png")
            plt.savefig(heatmap_path, dpi=300)
            plt.close()

        print(f"✅ Individual heatmaps saved to {output_folder}")

# ==================== PIPELINE 2: ANOMALY DETECTION ====================
class AnomalyDetector:
    """Handles pattern detection and anomaly identification"""
    
    def __init__(self, config):
        self.config = config
    
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
        """Cluster detected patterns spatially"""
        print("[PIPELINE 2] Clustering patterns...")
        
        if not patterns:
            return []

        coords = np.array([[p["x"], p["y"]] for p in patterns])
        
        clustering_params = [
            (self.config.CLUSTER_EPS, self.config.CLUSTER_MIN_SAMPLES),
            (self.config.CLUSTER_EPS * 1.5, self.config.CLUSTER_MIN_SAMPLES),
            (self.config.CLUSTER_EPS * 2, max(2, self.config.CLUSTER_MIN_SAMPLES - 1)),
            (self.config.CLUSTER_EPS * 3, 2)
        ]
        
        clustering = None
        labels = None
        
        for eps, min_samples in clustering_params:
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
            labels = clustering.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
            if n_clusters > 0:
                break
        
        if labels is None or len(set(labels)) <= 1:
            labels = list(range(len(patterns)))

        clustered = []
        unique_labels = set(labels)
        for lbl in tqdm(unique_labels, desc="Processing clusters"):
            if lbl == -1:
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
            
            spatial_density = len(pts) / bbox_area
            max_dim = max(bbox_w, bbox_h)
            min_dim = max(min(bbox_w, bbox_h), 1)
            aspect_ratio = max_dim / min_dim
            is_linear = aspect_ratio >= self.config.MIN_LINEAR_ASPECT_RATIO

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
        
        return clustered
    
    def filter_actionable_clusters(self, clusters):
        """Filter clusters for actionable patterns"""
        print("[PIPELINE 2] Filtering for actionable patterns...")
        
        filtered = []
        rejection_reasons = {}
        
        for c in clusters:
            rejection_reason = None
            
            # Temporal filters
            if c["last_year"] < self.config.RECENT_YEAR_THRESHOLD:
                rejection_reason = "Not recent enough"
            elif len(c["years_observed"]) < self.config.MIN_RECUR_YEARS:
                rejection_reason = f"Insufficient recurrence ({len(c['years_observed'])} < {self.config.MIN_RECUR_YEARS} years)"
            elif c["temporal_span"] < self.config.MIN_TEMPORAL_SPAN:
                rejection_reason = f"Insufficient temporal span ({c['temporal_span']} < {self.config.MIN_TEMPORAL_SPAN} years)"
            elif c["cluster_size"] < self.config.MIN_CLUSTER_SIZE:
                rejection_reason = f"Cluster too small ({c['cluster_size']} < {self.config.MIN_CLUSTER_SIZE} points)"
            elif c["cluster_size"] > 1:
                if c["is_linear"]:
                    if c["max_dimension"] < self.config.MIN_CLUSTER_LINEAR_LENGTH:
                        rejection_reason = f"Linear pattern too short ({c['max_dimension']:.1f} < {self.config.MIN_CLUSTER_LINEAR_LENGTH} pixels)"
                    elif c["spatial_density"] < self.config.MIN_SPATIAL_DENSITY:
                        rejection_reason = f"Linear pattern too sparse ({c['spatial_density']:.3f} < {self.config.MIN_SPATIAL_DENSITY})"
                else:
                    if c["bbox_width"] < self.config.MIN_CLUSTER_AREA_SIZE or c["bbox_height"] < self.config.MIN_CLUSTER_AREA_SIZE:
                        rejection_reason = f"Area pattern too small ({c['bbox_width']:.1f}x{c['bbox_height']:.1f} < {self.config.MIN_CLUSTER_AREA_SIZE})"
                    elif c["spatial_density"] < self.config.MIN_SPATIAL_DENSITY:
                        rejection_reason = f"Area pattern too sparse ({c['spatial_density']:.3f} < {self.config.MIN_SPATIAL_DENSITY})"
            
            if not rejection_reason and (c["bbox_width"] > self.config.MAX_CLUSTER_BBOX_SIZE or c["bbox_height"] > self.config.MAX_CLUSTER_BBOX_SIZE):
                rejection_reason = f"Pattern too large ({c['bbox_width']:.1f}x{c['bbox_height']:.1f} > {self.config.MAX_CLUSTER_BBOX_SIZE})"
            
            if not rejection_reason:
                if c["cluster_size"] > 5 and c["corr_std"] > 0.15:
                    rejection_reason = f"Inconsistent correlation (std={c['corr_std']:.3f})"
                elif c["avg_p_value"] > 0.05:
                    rejection_reason = f"Not statistically significant (p={c['avg_p_value']:.3f})"
            
            if rejection_reason:
                rejection_reasons[rejection_reason] = rejection_reasons.get(rejection_reason, 0) + 1
            else:
                # Quality score calculation
                if c["cluster_size"] == 1:
                    quality_score = abs(c["avg_corr"]) * 0.8 + (1 - c["avg_p_value"]) * 0.2
                elif c["is_linear"]:
                    length_score = min(c["max_dimension"] / (self.config.MIN_CLUSTER_LINEAR_LENGTH * 2), 1.0)
                    quality_score = (
                        abs(c["avg_corr"]) * 0.4 +
                        (1 - c["corr_std"]) * 0.2 +
                        length_score * 0.2 +
                        min(len(c["years_observed"]) / (self.config.MIN_RECUR_YEARS * 1.5), 1.0) * 0.2
                    )
                else:
                    quality_score = (
                        abs(c["avg_corr"]) * 0.4 +
                        (1 - c["corr_std"]) * 0.2 +
                        min(c["spatial_density"] / self.config.MIN_SPATIAL_DENSITY, 1.0) * 0.2 +
                        min(len(c["years_observed"]) / (self.config.MIN_RECUR_YEARS * 1.5), 1.0) * 0.2
                    )
                
                c["quality_score"] = quality_score
                
                if quality_score > 0.45:
                    filtered.append(c)
                else:
                    rejection_reasons[f"Low quality score ({quality_score:.3f})"] = rejection_reasons.get(f"Low quality score ({quality_score:.3f})", 0) + 1
        
        print(f"Actionable clusters: {len(filtered)}")
        for reason, count in sorted(rejection_reasons.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  Rejected - {reason}: {count}")
        
        return filtered
    
    def save_pattern_results(self, clusters, output_csv, shape, output_map):
        """Save pattern detection results"""
        print("[PIPELINE 2] Saving pattern results...")
        
        if not clusters:
            print("No actionable patterns found")
            return
        
        # Save CSV
        df = pd.DataFrame(clusters)
        df["human_rule"] = df.apply(
            lambda r: f"{'Linear' if r.get('is_linear', False) else 'Area'} pattern: "
                     f"{r['band1'].upper()} and {r['band2'].upper()} show {'strong positive' if r['avg_corr'] > 0 else 'strong negative'} correlation "
                     f"(r={r['avg_corr']:.3f}, quality={r['quality_score']:.3f}) across {len(r['years_observed'])} years. "
                     f"{'Length: ' + str(int(r['max_dimension'])) + ' pixels' if r.get('is_linear', False) else 'Size: ' + str(int(r['bbox_width'])) + 'x' + str(int(r['bbox_height'])) + ' pixels'}",
            axis=1
        )
        
        df = df.sort_values("quality_score", ascending=False)
        df.to_csv(output_csv, index=False)
        
        # Generate map overlay
        h, w = shape
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        
        for c in clusters:
            quality = c["quality_score"]
            intensity = int(255 * min(quality, 1.0))
            marker_size = max(3, int(5 * quality))
            
            if c["avg_corr"] > 0:
                color = (0, intensity, 0)  # Green for positive
            else:
                color = (intensity, 0, 0)  # Red for negative
            
            cy, cx = c["center_y"], c["center_x"]
            
            if c.get("is_linear", False):
                for dy in range(-marker_size, marker_size + 1):
                    for dx in range(-1, 2):
                        py, px = cy + dy, cx + dx
                        if 0 <= py < h and 0 <= px < w:
                            overlay[py, px] = color
            else:
                for i in range(-marker_size, marker_size + 1):
                    px = cx + i
                    if 0 <= cy < h and 0 <= px < w:
                        overlay[cy, px] = color
                    py = cy + i
                    if 0 <= py < h and 0 <= cx < w:
                        overlay[py, cx] = color

        plt.figure(figsize=(15, 10))
        plt.imshow(overlay)
        linear_count = sum(1 for c in clusters if c.get("is_linear", False))
        area_count = len(clusters) - linear_count
        plt.title(f"Actionable Agricultural Patterns\n"
                 f"Total: {len(clusters)} patterns ({linear_count} linear, {area_count} area)")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_map, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Pattern results saved: {output_csv}, {output_map}")

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
        
        # ========== PIPELINE 2: ANOMALY DETECTION ==========
        print("\n" + "="*50)
        print("PIPELINE 2: ANOMALY DETECTION")
        print("="*50)
        
        # Load heatmaps and detect patterns
        heatmap_data = self.anomaly_detector.load_heatmaps(heatmap_folder)
        
        if not heatmap_data:
            print("[WARN] No heatmap data found, skipping anomaly detection")
        else:
            patterns = self.anomaly_detector.detect_pixel_patterns(heatmap_data)
            clusters = self.anomaly_detector.cluster_patterns(patterns)
            actionable_clusters = self.anomaly_detector.filter_actionable_clusters(clusters)
            
            # Output paths for Pipeline 2
            pattern_csv = os.path.join(self.config.OUTPUT_FOLDER, 'actionable_patterns.csv')
            pattern_map = os.path.join(self.config.OUTPUT_FOLDER, 'pattern_overlay_map.png')
            
            if actionable_clusters:
                h, w = next(iter(heatmap_data.values()))[self.config.BANDS[0].lower()].shape
                self.anomaly_detector.save_pattern_results(actionable_clusters, pattern_csv, (h, w), pattern_map)
            else:
                print("No actionable patterns found")
        
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
            
            for y in years:
                yrs.append(y)
                x0, y0 = coords[idx]
                patch = composites[y][y0:y0+self.config.PATCH_SIZE, x0:x0+self.config.PATCH_SIZE, :]
                
                # Extract band statistics
                ndvi_patch = patch[..., [band.lower() for band in self.config.BANDS].index("ndvi")]
                evi_patch = patch[..., [band.lower() for band in self.config.BANDS].index("evi")]
                ndwi_patch = patch[..., [band.lower() for band in self.config.BANDS].index("ndwi")]
                
                ndvi_ts.append(float(ndvi_patch.mean()))
                evi_ts.append(float(evi_patch.mean()))
                ndwi_ts.append(float(ndwi_patch.mean()))
                
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
        """Generate visualization map for predictions"""
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
        
        # Fill prediction maps
        for pred in predictions:
            y, x = pred['center_y'], pred['center_x']
            if 0 <= y < height and 0 <= x < width:
                maps['ndvi'][y, x] = pred['ndvi_pred_next']
                maps['evi'][y, x] = pred['evi_pred_next']
                maps['ndwi'][y, x] = pred['ndwi_pred_next']
                
                delta_maps['ndvi'][y, x] = pred['delta_ndvi']
                delta_maps['evi'][y, x] = pred['delta_evi']
                delta_maps['ndwi'][y, x] = pred['delta_ndwi']
        
        # Generate plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Prediction maps
        for i, (band, data) in enumerate(maps.items()):
            valid_data = data[~np.isnan(data)]
            if len(valid_data) > 0:
                vmin, vmax = np.percentile(valid_data, [2, 98])
                im = axes[0, i].imshow(data, cmap='RdYlGn', vmin=vmin, vmax=vmax)
                axes[0, i].set_title(f'{band.upper()} Predictions')
                axes[0, i].axis('off')
                plt.colorbar(im, ax=axes[0, i])
            else:
                axes[0, i].set_title(f'{band.upper()} Predictions (No Data)')
                axes[0, i].axis('off')
        
        # Delta maps
        for i, (band, data) in enumerate(delta_maps.items()):
            valid_data = data[~np.isnan(data)]
            if len(valid_data) > 0:
                abs_max = np.percentile(np.abs(valid_data), 95)
                im = axes[1, i].imshow(data, cmap='RdBu_r', vmin=-abs_max, vmax=abs_max)
                axes[1, i].set_title(f'{band.upper()} Change')
                axes[1, i].axis('off')
                plt.colorbar(im, ax=axes[1, i])
            else:
                axes[1, i].set_title(f'{band.upper()} Change (No Data)')
                axes[1, i].axis('off')
        
        plt.suptitle('Agricultural Index Predictions and Changes', fontsize=16)
        plt.tight_layout()
        
        prediction_map_path = os.path.join(self.config.OUTPUT_FOLDER, 'prediction_maps.png')
        plt.savefig(prediction_map_path, dpi=300, bbox_inches='tight')
        plt.close()
        
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