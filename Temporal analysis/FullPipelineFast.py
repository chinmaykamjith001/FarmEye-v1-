# Change: make correlation map for step 4 make CSV and points be only if its above 0.95 value of "r" and have background o fiamge
 
import rasterio
import numpy as np
import os
import csv
import re
import math
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.linear_model import TheilSenRegressor, Ridge
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.decomposition import PCA
from sklearn.multioutput import MultiOutputRegressor
from scipy.stats import pearsonr, zscore
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    from torchvision.models import resnet18, resnet50
    TORCH_AVAILABLE = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARN] PyTorch not available. CNN features will be skipped.")

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# ================== CONFIGURATION ==================
INPUT_FOLDER = r"C:\Users\Faster\Downloads\FarmEyeInputs"
OUTPUT_FOLDER = r"C:\Users\Faster\Downloads\FarmEyeOutputs"
YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
BANDS = ['NDVI', 'EVI', 'NDWI']
NUM_BANDS = len(BANDS)
NUM_YEARS = len(YEARS)
CHUNK_SIZE = 20

# Analysis parameters
MIN_YEARS_REQ = 4
CORR_THRESHOLD = 0.85
MIN_RECUR_YEARS = 3
MIN_TEMPORAL_SPAN = 2
CLUSTER_EPS = 12
CLUSTER_MIN_SAMPLES = 3
MIN_CLUSTER_SIZE = 1
MIN_SPATIAL_DENSITY = 0.03

# ML parameters
PATCH_SIZE = 24
STRIDE = 12
BATCH_SIZE = 32
N_ESTIMATORS = 200
MAX_DEPTH = 8
LEARNING_RATE = 0.05
N_SPLITS = 5

# Ensure output directory exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

class FarmEyeCompletePipeline:
    def __init__(self):
        self.stack = None
        self.trends_data = {}
        self.anomalies_data = {}
        self.patterns_data = {}
        self.predictions_data = {}
        self.heatmaps_folder = os.path.join(OUTPUT_FOLDER, "heatmaps")
        os.makedirs(self.heatmaps_folder, exist_ok=True)

    # ================== DATA LOADING ==================
    def load_multiyear_stack(self):
        """Load and stack multi-year raster data"""
        print("[INFO] Loading multi-year raster stack...")
        
        first_path = os.path.join(INPUT_FOLDER, f"{YEARS[0]}.tif")
        with rasterio.open(first_path) as src:
            rows, cols = src.height, src.width

        arr = np.full((NUM_BANDS, rows, cols, NUM_YEARS), np.nan, dtype=np.float32)

        for t, year in enumerate(YEARS):
            path = os.path.join(INPUT_FOLDER, f"{year}.tif")
            print(f"Loading {path} ...")
            
            if not os.path.exists(path):
                print(f"[WARN] File not found: {path}")
                continue
                
            with rasterio.open(path) as src:
                data = src.read().astype(np.float32)
                if src.nodata is not None:
                    data[data == src.nodata] = np.nan
                if data.shape != (NUM_BANDS, rows, cols):
                    raise ValueError(f"Shape mismatch in {path}")
                arr[:, :, :, t] = data

        self.stack = arr
        print(f"[INFO] Loaded data stack shape: {self.stack.shape}")
        return arr

    # ================== TREND ANALYSIS ==================
    def compute_theil_sen_trends(self):
        """Compute Theil-Sen trends per chunk"""
        print("[INFO] Computing Theil-Sen trend analysis...")
        
        if self.stack is None:
            self.load_multiyear_stack()
            
        bands, rows, cols, times = self.stack.shape
        X = np.array(YEARS).reshape(-1, 1)
        
        slopes_csv = os.path.join(OUTPUT_FOLDER, 'trend_slopes_chunked.csv')
        zscores_csv = os.path.join(OUTPUT_FOLDER, 'trend_zscores_chunked.csv')
        
        # Compute slopes
        with open(slopes_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['chunk_row', 'chunk_col', 'band', 'slope'])
            writer.writeheader()

            for b in range(bands):
                band_name = BANDS[b]
                print(f"Computing trends for band {band_name}...")
                batch = []

                for r0 in tqdm(range(0, rows, CHUNK_SIZE)):
                    for c0 in range(0, cols, CHUNK_SIZE):
                        r1 = min(r0 + CHUNK_SIZE, rows)
                        c1 = min(c0 + CHUNK_SIZE, cols)

                        chunk_data = self.stack[b, r0:r1, c0:c1, :]
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
                                'slope': model.coef_[0]
                            })
                        except Exception:
                            continue

                        if len(batch) >= 100:
                            writer.writerows(batch)
                            batch.clear()

                if batch:
                    writer.writerows(batch)

        # Compute z-scores
        self.compute_slope_zscores(slopes_csv, zscores_csv)
        
        # Generate trend heatmap
        self.generate_trend_heatmap(zscores_csv)
        
        self.trends_data['slopes_file'] = slopes_csv
        self.trends_data['zscores_file'] = zscores_csv

    def compute_slope_zscores(self, input_csv, output_csv):
        """Compute z-scores of slopes"""
        slope_by_band = {band: [] for band in BANDS}
        
        with open(input_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                slope = row['slope']
                band = row['band']
                if slope != '' and slope.lower() != 'nan':
                    slope_by_band[band].append(float(slope))

        stats = {}
        for band in BANDS:
            slopes = np.array(slope_by_band[band])
            if len(slopes) > 0:
                mean = slopes.mean()
                std = slopes.std()
                stats[band] = (mean, std if std > 0 else 1)
            else:
                stats[band] = (0, 1)

        with open(input_csv, 'r') as f_in, open(output_csv, 'w', newline='') as f_out:
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

    def generate_trend_heatmap(self, zscores_csv):
        """Generate composite trend heatmap"""
        print("[INFO] Generating trend composite heatmap...")
        
        # Dynamic band weights
        band_weights = self.compute_dynamic_weights(zscores_csv)
        
        pixel_scores = {}
        with open(zscores_csv, 'r') as f:
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
            print("[WARN] No pixel scores found for trend heatmap")
            return

        all_rows = [k[0] for k in pixel_scores.keys()]
        all_cols = [k[1] for k in pixel_scores.keys()]
        max_row, max_col = max(all_rows), max(all_cols)

        grid_rows = max_row // CHUNK_SIZE + 1
        grid_cols = max_col // CHUNK_SIZE + 1
        composite_array = np.full((grid_rows, grid_cols), np.nan, dtype=np.float32)

        total_weight = sum(band_weights.values())
        normalized_weights = {k: v / total_weight for k, v in band_weights.items()}

        for (r,c), band_dict in pixel_scores.items():
            chunk_r_idx = r // CHUNK_SIZE
            chunk_c_idx = c // CHUNK_SIZE
            composite_score = 0
            weight_sum = 0
            for band, z in band_dict.items():
                w = normalized_weights.get(band, 0)
                composite_score += z * w
                weight_sum += w
            if weight_sum > 0:
                composite_array[chunk_r_idx, chunk_c_idx] = composite_score

        finite_vals = composite_array[np.isfinite(composite_array)]
        if finite_vals.size == 0:
            print("[WARN] No finite values for trend heatmap")
            return

        vmin, vmax = np.percentile(finite_vals, 2), np.percentile(finite_vals, 98)

        plt.figure(figsize=(12, 10))
        plt.imshow(composite_array, cmap='RdYlGn_r', vmin=vmin, vmax=vmax)
        plt.colorbar(label='Composite Trend Z-score')
        plt.title('Multi-year Trend Analysis Heatmap')
        plt.axis('off')
        plt.tight_layout()
        
        heatmap_path = os.path.join(OUTPUT_FOLDER, 'trend_composite_heatmap.png')
        plt.savefig(heatmap_path, dpi=300)
        plt.close()
        
        print(f"[OK] Trend heatmap saved to {heatmap_path}")

    def compute_dynamic_weights(self, zscores_csv):
        """Compute dynamic band weights based on variance"""
        band_values = {band: [] for band in BANDS}
        
        with open(zscores_csv, 'r') as f:
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
            return {band: 1/len(BANDS) for band in BANDS}

        weights = {band: std_devs[band] / total_std for band in BANDS}
        print(f"[INFO] Dynamic band weights: {weights}")
        return weights

    # ================== ANOMALY DETECTION ==================
    def detect_anomalies(self):
        """Detect temporal anomalies using residual analysis"""
        print("[INFO] Detecting temporal anomalies...")
        
        if not self.trends_data.get('slopes_file'):
            print("[WARN] Trends not computed. Running trend analysis first...")
            self.compute_theil_sen_trends()
            
        slopes_csv = self.trends_data['slopes_file']
        residuals_csv = os.path.join(OUTPUT_FOLDER, 'residuals_chunked.csv')
        anomaly_zscores_csv = os.path.join(OUTPUT_FOLDER, 'anomaly_zscores_chunked.csv')
        
        # Compute residuals
        self.compute_residuals_per_chunk_year(slopes_csv, residuals_csv)
        
        # Compute z-scores of residuals
        self.compute_residual_zscores(residuals_csv, anomaly_zscores_csv)
        
        # Generate anomaly heatmaps per band-year
        self.generate_anomaly_heatmaps(anomaly_zscores_csv)
        
        self.anomalies_data['residuals_file'] = residuals_csv
        self.anomalies_data['zscores_file'] = anomaly_zscores_csv

    def compute_residuals_per_chunk_year(self, slopes_csv, output_csv):
        """Compute residuals for anomaly detection"""
        # Load slopes and intercepts
        slope_intercept_map = {}
        with open(slopes_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (int(row['chunk_row']), int(row['chunk_col']), row['band'])
                slope = float(row['slope'])
                # Estimate intercept from mean
                slope_intercept_map[key] = (slope, 0)  # Simplified

        bands, rows, cols, times = self.stack.shape

        with open(output_csv, 'w', newline='') as f_out:
            writer = csv.DictWriter(f_out, fieldnames=['chunk_row', 'chunk_col', 'band', 'year', 'residual'])
            writer.writeheader()

            buffer = []

            for b in range(bands):
                band_name = BANDS[b]
                print(f"Computing residuals for band {band_name}...")
                
                for r0 in tqdm(range(0, rows, CHUNK_SIZE)):
                    for c0 in range(0, cols, CHUNK_SIZE):
                        key = (r0, c0, band_name)
                        if key not in slope_intercept_map:
                            continue

                        r1 = min(r0 + CHUNK_SIZE, rows)
                        c1 = min(c0 + CHUNK_SIZE, cols)
                        chunk_data = self.stack[b, r0:r1, c0:c1, :]
                        
                        with np.errstate(invalid='ignore'):
                            chunk_mean_ts = np.nanmean(chunk_data.reshape(-1, times), axis=0)

                        if np.isnan(chunk_mean_ts).all():
                            continue

                        slope, intercept = slope_intercept_map[key]
                        
                        # Estimate intercept from first valid point
                        valid_indices = ~np.isnan(chunk_mean_ts)
                        if np.sum(valid_indices) > 0:
                            first_year_idx = np.where(valid_indices)[0][0]
                            first_year = YEARS[first_year_idx]
                            first_val = chunk_mean_ts[first_year_idx]
                            intercept = first_val - slope * first_year

                        for t, year in enumerate(YEARS):
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

                            if len(buffer) >= 10000:
                                writer.writerows(buffer)
                                buffer.clear()

            if buffer:
                writer.writerows(buffer)

    def compute_residual_zscores(self, residuals_csv, output_csv):
        """Compute z-scores of residuals by band-year"""
        residuals_map = {}
        rows_buffer = []

        with open(residuals_csv, 'r') as f:
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

        print(f"[OK] Anomaly z-scores saved to {output_csv}")

    def generate_anomaly_heatmaps(self, zscores_csv):
        """Generate individual heatmaps for each band-year combination"""
        print("[INFO] Generating anomaly heatmaps...")
        
        zscore_data = {}
        chunk_rows = []
        chunk_cols = []

        with open(zscores_csv, 'r') as f:
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
            print("[WARN] No data for anomaly heatmaps")
            return

        max_row = max(chunk_rows)
        max_col = max(chunk_cols)

        for (band, year), pix_dict in tqdm(zscore_data.items(), desc="Generating anomaly heatmaps"):
            heatmap_array = np.full((max_row // CHUNK_SIZE + 1, max_col // CHUNK_SIZE + 1), np.nan, dtype=np.float32)

            for (r, c), z in pix_dict.items():
                chunk_r_idx = r // CHUNK_SIZE
                chunk_c_idx = c // CHUNK_SIZE
                heatmap_array[chunk_r_idx, chunk_c_idx] = z

            finite_vals = heatmap_array[np.isfinite(heatmap_array)]
            if finite_vals.size == 0:
                continue

            vmin, vmax = np.percentile(finite_vals, 2), np.percentile(finite_vals, 98)

            plt.figure(figsize=(12, 10))
            plt.imshow(heatmap_array, cmap='RdYlGn_r', vmin=vmin, vmax=vmax)
            plt.colorbar(label='Anomaly Z-score')
            plt.title(f"Anomaly Detection - Band: {band} Year: {year}")
            plt.axis('off')
            plt.tight_layout()

            heatmap_path = os.path.join(self.heatmaps_folder, f"anomaly_{band}_{year}.png")
            plt.savefig(heatmap_path, dpi=300)
            plt.close()

        print(f"[OK] Anomaly heatmaps saved to {self.heatmaps_folder}")

    # ================== PATTERN RECOGNITION ==================
    def detect_correlation_patterns(self):
        """Detect spatial correlation patterns between bands"""
        print("[INFO] Detecting correlation patterns...")
        
        if not os.path.exists(self.heatmaps_folder) or len(os.listdir(self.heatmaps_folder)) == 0:
            print("[WARN] No heatmaps found. Running anomaly detection first...")
            self.detect_anomalies()
        
        # Load heatmaps
        heatmap_data = self.load_heatmaps_for_patterns()
        
        if not heatmap_data:
            print("[WARN] No heatmap data loaded for pattern detection")
            return
        
        # Filter complete years
        complete_data = self.filter_complete_years(heatmap_data)
        print(f"[INFO] Complete years for patterns: {list(complete_data.keys())}")
        
        if len(complete_data) < MIN_RECUR_YEARS:
            print(f"[WARN] Insufficient data years for patterns ({len(complete_data)} < {MIN_RECUR_YEARS})")
            return
        
        # Detect pixel-level patterns
        patterns = self.detect_pixel_patterns(complete_data)
        
        if not patterns:
            print("[WARN] No correlation patterns detected")
            return
        
        # Cluster patterns
        clusters = self.cluster_patterns(patterns)
        
        # Filter actionable patterns
        actionable_patterns = self.filter_actionable_patterns(clusters)
        
        # Save results
        if actionable_patterns:
            patterns_csv = os.path.join(OUTPUT_FOLDER, 'correlation_patterns.csv')
            patterns_map = os.path.join(OUTPUT_FOLDER, 'correlation_patterns_map.png')
            
            self.save_patterns_csv(actionable_patterns, patterns_csv)
            self.save_patterns_map(actionable_patterns, complete_data, patterns_map)
            
            self.patterns_data['patterns_file'] = patterns_csv
            self.patterns_data['patterns_count'] = len(actionable_patterns)
            
            print(f"[OK] Found {len(actionable_patterns)} actionable correlation patterns")
        else:
            print("[INFO] No actionable correlation patterns found")

    def load_heatmaps_for_patterns(self):
        """Load heatmaps for pattern analysis"""
        pattern = re.compile(r"anomaly_(?P<band>[A-Za-z]+)_(?P<year>\d{4})\.png")
        data = {}
        
        for fname in os.listdir(self.heatmaps_folder):
            m = pattern.match(fname)
            if m:
                band = m.group("band").lower()
                year = int(m.group("year"))
                path = os.path.join(self.heatmaps_folder, fname)
                
                try:
                    arr = np.array(Image.open(path).convert("L"), dtype=np.float32)
                    arr /= 255.0
                    data.setdefault(year, {})[band] = arr
                except Exception as e:
                    print(f"[WARN] Could not load {path}: {e}")
                    
        return data

    def filter_complete_years(self, data):
        """Filter years that have all required bands"""
        band_names = [b.lower() for b in BANDS]
        return {
            year: bands for year, bands in data.items()
            if all(b in bands for b in band_names)
        }

    def detect_pixel_patterns(self, data):
        """Detect correlation patterns at pixel level"""
        years_sorted = sorted(data.keys())
        h, w = next(iter(data.values()))[BANDS[0].lower()].shape
        patterns = []
        
        band_names = [b.lower() for b in BANDS]
        
        print(f"[INFO] Analyzing {len(years_sorted)} years of heatmap data")

        for y in tqdm(range(0, h, 6), desc="Scanning rows for patterns"):
            for x in range(0, w, 6):
                for i in range(len(band_names)):
                    for j in range(i + 1, len(band_names)):
                        band1, band2 = band_names[i], band_names[j]
                        
                        ts1 = [data[yr][band1][y, x] for yr in years_sorted]
                        ts2 = [data[yr][band2][y, x] for yr in years_sorted]

                        # Quality checks
                        if np.var(ts1) < 0.01 or np.var(ts2) < 0.01:
                            continue
                            
                        if (np.max(ts1) - np.min(ts1)) < 0.05 or (np.max(ts2) - np.min(ts2)) < 0.05:
                            continue

                        try:
                            corr, p_value = pearsonr(ts1, ts2)
                            
                            if abs(corr) >= CORR_THRESHOLD and p_value < 0.05:
                                patterns.append({
                                    "x": x, "y": y,
                                    "band1": band1, "band2": band2,
                                    "corr": corr,
                                    "p_value": p_value,
                                    "years": years_sorted,
                                    "ts1_var": np.var(ts1),
                                    "ts2_var": np.var(ts2),
                                })
                        except:
                            continue

        print(f"[INFO] Found {len(patterns)} correlation patterns")
        return patterns

    def cluster_patterns(self, patterns):
        """Cluster similar patterns spatially"""
        if not patterns:
            return []

        coords = np.array([[p["x"], p["y"]] for p in patterns])
        
        clustering = DBSCAN(eps=CLUSTER_EPS, min_samples=CLUSTER_MIN_SAMPLES).fit(coords)
        labels = clustering.labels_
        
        clustered = []
        unique_labels = set(labels)
        
        for lbl in unique_labels:
            if lbl == -1:  # Skip noise
                continue
                
            indices = np.where(np.array(labels) == lbl)[0]
            pts = [patterns[i] for i in indices]

            xs = [p["x"] for p in pts]
            ys = [p["y"] for p in pts]
            
            years_union = sorted(set(y for p in pts for y in p["years"]))
            avg_corr = np.mean([p["corr"] for p in pts])
            
            clustered.append({
                "id": lbl,
                "band1": pts[0]["band1"],
                "band2": pts[0]["band2"],
                "avg_corr": avg_corr,
                "center_x": int(np.mean(xs)),
                "center_y": int(np.mean(ys)),
                "years_observed": years_union,
                "temporal_span": years_union[-1] - years_union[0],
                "cluster_size": len(pts),
                "avg_p_value": np.mean([p["p_value"] for p in pts]),
            })

        return clustered

    def filter_actionable_patterns(self, clusters):
        """Filter for actionable patterns"""
        filtered = []
        
        for c in clusters:
            # Basic filters
            if len(c["years_observed"]) < MIN_RECUR_YEARS:
                continue
            if c["temporal_span"] < MIN_TEMPORAL_SPAN:
                continue
            if c["cluster_size"] < MIN_CLUSTER_SIZE:
                continue
            if c["avg_p_value"] > 0.05:
                continue
                
            # Quality score
            quality_score = (
                abs(c["avg_corr"]) * 0.5 +
                min(len(c["years_observed"]) / MIN_RECUR_YEARS, 1.0) * 0.3 +
                min(c["cluster_size"] / 5, 1.0) * 0.2
            )
            
            if quality_score > 0.4:
                c["quality_score"] = quality_score
                filtered.append(c)
        
        return filtered

    def save_patterns_csv(self, patterns, path):
        """Save patterns to CSV"""
        df = pd.DataFrame(patterns)
        
        df["pattern_description"] = df.apply(
            lambda r: f"Strong {'positive' if r['avg_corr'] > 0 else 'negative'} correlation "
                     f"between {r['band1'].upper()} and {r['band2'].upper()} "
                     f"(r={r['avg_corr']:.3f}, quality={r['quality_score']:.3f}) "
                     f"across {len(r['years_observed'])} years",
            axis=1
        )
        
        df = df.sort_values("quality_score", ascending=False)
        df.to_csv(path, index=False)
        print(f"[OK] Patterns CSV saved to {path}")

    def save_patterns_map(self, patterns, data, path):
        """Save patterns overlay map"""
        h, w = next(iter(data.values()))[BANDS[0].lower()].shape
        
        # Create overlay
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        
        for p in patterns:
            quality = p["quality_score"]
            intensity = int(255 * min(quality, 1.0))
            marker_size = max(3, int(5 * quality))
            
            if p["avg_corr"] > 0:
                color = (0, intensity, 0)  # Green for positive
            else:
                color = (intensity, 0, 0)  # Red for negative
            
            cy, cx = p["center_y"], p["center_x"]
            
            # Draw marker
            for dy in range(-marker_size, marker_size + 1):
                for dx in range(-marker_size, marker_size + 1):
                    py, px = cy + dy, cx + dx
                    if 0 <= py < h and 0 <= px < w:
                        if abs(dy) + abs(dx) <= marker_size:
                            overlay[py, px] = color

        plt.figure(figsize=(15, 10))
        plt.imshow(overlay)
        plt.title(f"Correlation Patterns Map\n"
                 f"Total: {len(patterns)} patterns (Green=Positive, Red=Negative)",
                 fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] Patterns map saved to {path}")

    # ================== ADVANCED TEMPORAL FEATURES ==================
    def advanced_temporal_features(self, ts, years):
        """Extract comprehensive temporal features"""
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
                
                # Trend acceleration
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
        
        # Recent trends
        if len(ts) >= 4:
            recent = ts[-2:]
            historical = ts[:-2]
            features['recent_vs_historical'] = np.mean(recent) - np.mean(historical)
            features['recent_trend'] = np.polyfit(years[-2:], recent, 1)[0] if len(recent) >= 2 else 0
        else:
            features['recent_vs_historical'] = features['recent_trend'] = 0
            
        return features

    def compute_cross_band_features(self, ndvi_ts, evi_ts, ndwi_ts, years):
        """Compute cross-band relationship features"""
        features = {}
        
        # Cross-correlations
        try:
            features['ndvi_evi_corr'] = np.corrcoef(ndvi_ts, evi_ts)[0,1]
            features['ndvi_ndwi_corr'] = np.corrcoef(ndvi_ts, ndwi_ts)[0,1]
            features['evi_ndwi_corr'] = np.corrcoef(evi_ts, ndwi_ts)[0,1]
        except:
            features['ndvi_evi_corr'] = features['ndvi_ndwi_corr'] = features['evi_ndwi_corr'] = 0
        
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

    # ================== CNN FEATURE EXTRACTOR ==================
    class CNNFeatureExtractor:
        def __init__(self, model_type='resnet18'):
            if not TORCH_AVAILABLE:
                self.available = False
                return
                
            self.available = True
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
        
        def extract_features(self, np_patches):
            if not self.available:
                return np.zeros((len(np_patches), 512))
                
            imgs = []
            for p in np_patches:
                p_filtered = gaussian_filter(p, sigma=0.5)
                pil = Image.fromarray((np.clip(p_filtered, 0, 1)*255).astype(np.uint8))
                if pil.mode != 'RGB':
                    pil = pil.convert('RGB')
                t = self.transform(pil).to(DEVICE)
                imgs.append(t)
                
            batch = torch.stack(imgs)
            with torch.no_grad():
                out = self.model(batch)
                out = out.view(out.size(0), -1).cpu().numpy()
            return out

    # ================== ML ENSEMBLE MODEL ==================
    class EnsembleModel:
        def __init__(self):
            self.models = {}
            self.weights = {}
            self.scaler = RobustScaler()
            self.feature_selector = None
            
        def fit(self, X, y):
            print("[INFO] Training ensemble model...")
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Feature selection if too many features
            if X_scaled.shape[1] > 100:
                self.feature_selector = SelectKBest(score_func=f_regression, k=min(100, X_scaled.shape[1]//2))
                X_scaled = self.feature_selector.fit_transform(X_scaled, y.ravel() if y.ndim > 1 else y)
                print(f"[INFO] Selected {X_scaled.shape[1]} features out of {X.shape[1]}")
            
            # Initialize models
            if CATBOOST_AVAILABLE:
                if y.ndim > 1:
                    self.models['catboost'] = CatBoostRegressor(
                        loss_function='MultiRMSE',
                        iterations=N_ESTIMATORS,
                        depth=MAX_DEPTH,
                        learning_rate=LEARNING_RATE,
                        verbose=False,
                        random_seed=42
                    )
                else:
                    self.models['catboost'] = CatBoostRegressor(
                        loss_function='RMSE',
                        iterations=N_ESTIMATORS,
                        depth=MAX_DEPTH,
                        learning_rate=LEARNING_RATE,
                        verbose=False,
                        random_seed=42
                    )
            
            if XGBOOST_AVAILABLE:
                if y.ndim > 1:
                    self.models['xgboost'] = MultiOutputRegressor(
                        xgb.XGBRegressor(
                            n_estimators=N_ESTIMATORS,
                            max_depth=MAX_DEPTH,
                            learning_rate=LEARNING_RATE,
                            random_state=42
                        )
                    )
                else:
                    self.models['xgboost'] = xgb.XGBRegressor(
                        n_estimators=N_ESTIMATORS,
                        max_depth=MAX_DEPTH,
                        learning_rate=LEARNING_RATE,
                        random_state=42
                    )
            
            # Random Forest
            if y.ndim > 1:
                self.models['rf'] = MultiOutputRegressor(
                    RandomForestRegressor(
                        n_estimators=N_ESTIMATORS,
                        max_depth=MAX_DEPTH,
                        random_state=42,
                        n_jobs=-1
                    )
                )
            else:
                self.models['rf'] = RandomForestRegressor(
                    n_estimators=N_ESTIMATORS,
                    max_depth=MAX_DEPTH,
                    random_state=42,
                    n_jobs=-1
                )
            
            # Train models and compute weights
            tscv = TimeSeriesSplit(n_splits=N_SPLITS)
            
            for name, model in self.models.items():
                print(f"[INFO] Training {name}...")
                try:
                    model.fit(X_scaled, y)
                    
                    cv_scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='neg_mean_absolute_error')
                    self.weights[name] = np.mean(-cv_scores)
                    print(f"[INFO] {name} CV MAE: {self.weights[name]:.4f}")
                except Exception as e:
                    print(f"[WARN] Failed to train {name}: {e}")
                    if name in self.weights:
                        del self.weights[name]
            
            # Normalize weights
            if self.weights:
                total_weight = sum(1/w for w in self.weights.values())
                self.weights = {k: (1/v)/total_weight for k, v in self.weights.items()}
                print("[INFO] Model weights:", self.weights)
            
        def predict(self, X):
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

    # ================== ML PREDICTION PIPELINE ==================
    def run_ml_predictions(self):
        """Run ML-based predictions for next year"""
        print("[INFO] Running ML predictions...")
        
        if self.stack is None:
            self.load_multiyear_stack()
        
        # Create composites for each year
        composites = {}
        bands, rows, cols, times = self.stack.shape
        
        for t, year in enumerate(YEARS):
            comp = np.zeros((rows, cols, NUM_BANDS), dtype=np.float32)
            for b in range(NUM_BANDS):
                comp[:, :, b] = np.nan_to_num(self.stack[b, :, :, t], nan=0)
            composites[year] = comp
        
        print(f"[INFO] Created composites for {len(composites)} years")
        
        # Extract patches
        sample = next(iter(composites.values()))
        patches, coords = self.extract_patches_from_image(sample, PATCH_SIZE, STRIDE)
        N_patches = len(coords)
        print(f"[INFO] Extracted {N_patches} patches")
        
        # Initialize CNN feature extractor
        cnn_extractor = self.CNNFeatureExtractor('resnet18')
        
        # Extract features for each year
        year_features = {}
        for year in YEARS:
            print(f"[INFO] Extracting features for year {year}...")
            patches_year, _ = self.extract_patches_from_image(composites[year], PATCH_SIZE, STRIDE)
            
            if cnn_extractor.available:
                cnn_features = cnn_extractor.extract_features(patches_year)
            else:
                cnn_features = np.zeros((len(patches_year), 512))
            
            year_features[year] = cnn_features
        
        # Build training data
        patch_data = []
        for idx in tqdm(range(N_patches), desc="Processing patches"):
            years_data = []
            ndvi_ts = []
            evi_ts = []
            ndwi_ts = []
            
            for year in YEARS:
                x0, y0 = coords[idx]
                patch = composites[year][y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE, :]
                
                ndvi_val = float(patch[..., 0].mean())  # NDVI
                evi_val = float(patch[..., 1].mean())   # EVI  
                ndwi_val = float(patch[..., 2].mean())  # NDWI
                
                ndvi_ts.append(ndvi_val)
                evi_ts.append(evi_val)
                ndwi_ts.append(ndwi_val)
                years_data.append(year)
            
            if len(years_data) < MIN_YEARS_REQ:
                continue
            
            # Extract temporal features
            ndvi_features = self.advanced_temporal_features(ndvi_ts, years_data)
            evi_features = self.advanced_temporal_features(evi_ts, years_data)
            ndwi_features = self.advanced_temporal_features(ndwi_ts, years_data)
            cross_features = self.compute_cross_band_features(ndvi_ts, evi_ts, ndwi_ts, years_data)
            
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
            
            patch_data.append({
                "idx": idx,
                "coord": coords[idx],
                "years": years_data,
                "ndvi_ts": ndvi_ts,
                "evi_ts": evi_ts,
                "ndwi_ts": ndwi_ts,
                "features": all_features
            })
        
        print(f"[INFO] Processed {len(patch_data)} patches for ML")
        
        if not patch_data:
            print("[WARN] No patches available for ML predictions")
            return
        
        # Build training matrices
        X = []
        y = []
        
        for p in tqdm(patch_data, desc="Building training data"):
            years = p["years"]
            ndvi_ts = p["ndvi_ts"]
            evi_ts = p["evi_ts"]
            ndwi_ts = p["ndwi_ts"]
            
            for t, year in enumerate(years):
                feat = list(p["features"].values())
                X.append(feat)
                y.append([ndvi_ts[t], evi_ts[t], ndwi_ts[t]])
        
        X = np.array(X)
        y = np.array(y)
        print(f"[INFO] Training data shape: X={X.shape}, y={y.shape}")
        
        # Train ensemble model
        model = self.EnsembleModel()
        model.fit(X, y)
        
        # Make predictions
        predictions = []
        next_year = YEARS[-1] + 1
        
        for p in tqdm(patch_data, desc="Making predictions"):
            feat = list(p["features"].values())
            feat_array = np.array([feat])
            
            pred_vals = model.predict(feat_array)[0]
            
            ndvi_last = p["ndvi_ts"][-1]
            evi_last = p["evi_ts"][-1]
            ndwi_last = p["ndwi_ts"][-1]
            
            predictions.append({
                "patch_idx": p["idx"],
                "center_x": p["coord"][0] + PATCH_SIZE//2,
                "center_y": p["coord"][1] + PATCH_SIZE//2,
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
            })
        
        # Save predictions
        df_preds = pd.DataFrame(predictions)
        predictions_csv = os.path.join(OUTPUT_FOLDER, 'ml_predictions.csv')
        df_preds.to_csv(predictions_csv, index=False)
        
        # Generate prediction heatmaps
        self.generate_prediction_heatmaps(predictions, composites[YEARS[-1]].shape[:2])
        
        self.predictions_data['predictions_file'] = predictions_csv
        self.predictions_data['predictions_count'] = len(predictions)
        
        print(f"[OK] ML predictions saved to {predictions_csv}")
        print(f"[INFO] Prediction summary:")
        print(f"  - Mean NDVI change: {df_preds['delta_ndvi'].mean():.4f} ± {df_preds['delta_ndvi'].std():.4f}")
        print(f"  - Mean EVI change: {df_preds['delta_evi'].mean():.4f} ± {df_preds['delta_evi'].std():.4f}")
        print(f"  - Mean NDWI change: {df_preds['delta_ndwi'].mean():.4f} ± {df_preds['delta_ndwi'].std():.4f}")

    def extract_patches_from_image(self, img, patch_size, stride):
        """Extract patches from image"""
        H, W, _ = img.shape
        patches = []
        coords = []
        
        for y in range(0, H - patch_size + 1, stride):
            for x in range(0, W - patch_size + 1, stride):
                p = img[y:y+patch_size, x:x+patch_size, :]
                patches.append(p)
                coords.append((x, y))
                
        return np.array(patches), coords

    def generate_prediction_heatmaps(self, predictions, shape):
        """Generate heatmaps for ML predictions"""
        print("[INFO] Generating prediction heatmaps...")
        
        h, w = shape
        
        for band_name, delta_col in [('NDVI', 'delta_ndvi'), ('EVI', 'delta_evi'), ('NDWI', 'delta_ndwi')]:
            heatmap_array = np.full((h//STRIDE, w//STRIDE), np.nan, dtype=np.float32)
            
            for pred in predictions:
                x, y = pred['center_x'], pred['center_y']
                grid_x, grid_y = x // STRIDE, y // STRIDE
                
                if 0 <= grid_y < heatmap_array.shape[0] and 0 <= grid_x < heatmap_array.shape[1]:
                    heatmap_array[grid_y, grid_x] = pred[delta_col]
            
            finite_vals = heatmap_array[np.isfinite(heatmap_array)]
            if finite_vals.size == 0:
                continue
                
            vmin, vmax = np.percentile(finite_vals, 5), np.percentile(finite_vals, 95)
            
            plt.figure(figsize=(12, 10))
            plt.imshow(heatmap_array, cmap='RdYlGn', vmin=vmin, vmax=vmax)
            plt.colorbar(label=f'{band_name} Change Prediction')
            plt.title(f'ML Prediction Heatmap - {band_name} Change for {YEARS[-1] + 1}')
            plt.axis('off')
            plt.tight_layout()
            
            heatmap_path = os.path.join(OUTPUT_FOLDER, f'prediction_{band_name.lower()}_change.png')
            plt.savefig(heatmap_path, dpi=300)
            plt.close()
        
        print(f"[OK] Prediction heatmaps saved to {OUTPUT_FOLDER}")

    # ================== MAIN PIPELINE ==================
    def run_complete_analysis(self):
        """Run the complete FarmEye analysis pipeline"""
        print("="*60)
        print("FARMEYE COMPLETE AGRICULTURAL ANALYSIS PIPELINE")
        print("="*60)
        
        # Step 1: Load data
        print("\n[STEP 1] Loading multi-year raster data...")
        self.load_multiyear_stack()
        
        # Step 2: Trend analysis
        print("\n[STEP 2] Computing trend analysis...")
        self.compute_theil_sen_trends()
        
        # Step 3: Anomaly detection
        print("\n[STEP 3] Detecting temporal anomalies...")
        self.detect_anomalies()
        
        # Step 4: Pattern recognition
        print("\n[STEP 4] Detecting correlation patterns...")
        self.detect_correlation_patterns()
        
        # Step 5: ML predictions
        print("\n[STEP 5] Running ML predictions...")
        self.run_ml_predictions()
        
        # Step 6: Generate summary report
        print("\n[STEP 6] Generating summary report...")
        self.generate_summary_report()
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"All outputs saved to: {OUTPUT_FOLDER}")
        print("="*60)

    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        report_path = os.path.join(OUTPUT_FOLDER, 'analysis_summary.txt')
        
        with open(report_path, 'w') as f:
            f.write("FARMEYE AGRICULTURAL ANALYSIS SUMMARY REPORT\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"Analysis Period: {YEARS[0]} - {YEARS[-1]}\n")
            f.write(f"Bands Analyzed: {', '.join(BANDS)}\n")
            f.write(f"Chunk Size: {CHUNK_SIZE}x{CHUNK_SIZE} pixels\n\n")
            
            # Data info
            if self.stack is not None:
                f.write("DATA INFORMATION:\n")
                f.write(f"Stack Shape: {self.stack.shape} (bands, rows, cols, years)\n")
                f.write(f"Spatial Resolution: {self.stack.shape[1]}x{self.stack.shape[2]} pixels\n\n")
            
            # Trends
            if self.trends_data:
                f.write("TREND ANALYSIS:\n")
                f.write(f"Slopes file: {os.path.basename(self.trends_data.get('slopes_file', 'N/A'))}\n")
                f.write(f"Z-scores file: {os.path.basename(self.trends_data.get('zscores_file', 'N/A'))}\n")
                f.write("Generated: Composite trend heatmap\n\n")
            
            # Anomalies
            if self.anomalies_data:
                f.write("ANOMALY DETECTION:\n")
                f.write(f"Residuals file: {os.path.basename(self.anomalies_data.get('residuals_file', 'N/A'))}\n")
                f.write(f"Z-scores file: {os.path.basename(self.anomalies_data.get('zscores_file', 'N/A'))}\n")
                f.write(f"Generated: Individual heatmaps for each band-year combination\n\n")
            
            # Patterns
            if self.patterns_data:
                f.write("PATTERN RECOGNITION:\n")
                f.write(f"Patterns found: {self.patterns_data.get('patterns_count', 0)}\n")
                f.write(f"Patterns file: {os.path.basename(self.patterns_data.get('patterns_file', 'N/A'))}\n")
                f.write("Generated: Correlation patterns map\n\n")
            
            # Predictions
            if self.predictions_data:
                f.write("ML PREDICTIONS:\n")
                f.write(f"Predictions made: {self.predictions_data.get('predictions_count', 0)}\n")
                f.write(f"Predictions file: {os.path.basename(self.predictions_data.get('predictions_file', 'N/A'))}\n")
                f.write(f"Target year: {YEARS[-1] + 1}\n")
                f.write("Generated: Prediction heatmaps for each band\n\n")
            
            f.write("OUTPUT FILES:\n")
            f.write("- trend_slopes_chunked.csv: Theil-Sen slope values\n")
            f.write("- trend_zscores_chunked.csv: Trend z-scores\n")
            f.write("- trend_composite_heatmap.png: Composite trend visualization\n")
            f.write("- residuals_chunked.csv: Temporal residuals\n")
            f.write("- anomaly_zscores_chunked.csv: Anomaly z-scores\n")
            f.write("- heatmaps/anomaly_*.png: Individual anomaly heatmaps\n")
            f.write("- correlation_patterns.csv: Detected patterns\n")
            f.write("- correlation_patterns_map.png: Patterns visualization\n")
            f.write("- ml_predictions.csv: ML predictions for next year\n")
            f.write("- prediction_*_change.png: Prediction heatmaps\n")
        
        print(f"[OK] Summary report saved to {report_path}")

# ================== MAIN EXECUTION ==================
if __name__ == "__main__":
    try:
        pipeline = FarmEyeCompletePipeline()
        pipeline.run_complete_analysis()
    except Exception as e:
        print(f"[ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()