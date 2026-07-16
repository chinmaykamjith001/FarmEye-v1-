# farmeye_multiband_pipeline_enhanced.py
import os
import re
import math
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter

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

from torchvision.models import resnet18, resnet50
import torchvision.transforms as T

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- ENHANCED CONFIG ----------------
INPUT_DIR = r"C:\Users\Faster\Downloads\FarmEyeInputs\heatmaps_chunked"
BASE_TIF_PATH = r"C:\Users\Faster\Downloads\FarmEyeInputs\2025.tif"
OUT_CSV = "farmeye_enhanced_predictions.csv"
OUT_MAP = "farmeye_enhanced_map.png"

BANDS = ["evi", "ndvi", "ndwi"]
MIN_YEARS_REQ = 4  # Increased for better temporal modeling

# Enhanced patch processing for MAXIMUM TILE DENSITY
PATCH_SIZE = 24  # Much smaller patches = MORE tiles
STRIDE = 13       # Very small stride = HEAVY overlap = MAXIMUM coverage
BATCH = 36     # Smaller batch for memory efficiency with more patches

# Advanced feature engineering
PCA_DIM = 32
USE_SEASONAL_DECOMP = True
USE_WEATHER_PROXY = True
TEMPORAL_WINDOW = 3  # Years to look back for trends

# Model ensemble parameters
USE_ENSEMBLE = True
N_ESTIMATORS = 200
MAX_DEPTH = 8
LEARNING_RATE = 0.05

# Clustering for precision
DBSCAN_EPS = 0.4  # Tighter clusters
DBSCAN_MIN_SAMPLES = 6
USE_HIERARCHICAL_CLUSTERING = True

# Cross-validation
N_SPLITS = 5
TEST_SIZE = 0.2

# ----------------------------------------

class EnhancedResNetEmbedder:
    """Enhanced embedding with multiple CNN architectures and data augmentation"""
    def __init__(self, model_type='resnet50'):
        if model_type == 'resnet50':
            m = resnet50(pretrained=True)
            self.dim = 2048
        else:
            m = resnet18(pretrained=True)
            self.dim = 512
            
        self.model = torch.nn.Sequential(*list(m.children())[:-1]).to(DEVICE).eval()
        
        # Enhanced transforms with augmentation for training robustness
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
        imgs = []
        transform = self.augment_transform if augment else self.transform
        
        for p in np_patches:
            # Enhanced preprocessing
            p_filtered = gaussian_filter(p, sigma=0.5)  # Slight denoising
            pil = Image.fromarray((np.clip(p_filtered, 0, 1)*255).astype(np.uint8))
            t = transform(pil).to(DEVICE)
            imgs.append(t)
            
        batch = torch.stack(imgs)
        with torch.no_grad():
            out = self.model(batch)
            out = out.view(out.size(0), -1).cpu().numpy()
        return out

def advanced_temporal_features(ts, years):
    """Extract advanced temporal features from time series"""
    ts = np.array(ts)
    years = np.array(years)
    
    features = {}
    
    # Basic statistics
    features['mean'] = np.mean(ts)
    features['std'] = np.std(ts)
    features['min'] = np.min(ts)
    features['max'] = np.max(ts)
    features['range'] = features['max'] - features['min']
    features['cv'] = features['std'] / (features['mean'] + 1e-8)  # Coefficient of variation
    
    # Trend analysis (multiple orders)
    if len(ts) >= 3:
        # Linear trend
        linear_coef = np.polyfit(years, ts, 1)
        features['linear_slope'] = linear_coef[0]
        features['linear_intercept'] = linear_coef[1]
        
        # Quadratic trend
        if len(ts) >= 4:
            quad_coef = np.polyfit(years, ts, 2)
            features['quad_a'] = quad_coef[0]
            features['quad_b'] = quad_coef[1]
            features['quad_c'] = quad_coef[2]
        else:
            features['quad_a'] = features['quad_b'] = features['quad_c'] = 0
            
        # Trend acceleration
        if len(ts) >= 4:
            mid_point = len(ts) // 2
            early_slope = np.polyfit(years[:mid_point+1], ts[:mid_point+1], 1)[0]
            late_slope = np.polyfit(years[mid_point:], ts[mid_point:], 1)[0]
            features['trend_acceleration'] = late_slope - early_slope
        else:
            features['trend_acceleration'] = 0
    else:
        features.update({
            'linear_slope': 0, 'linear_intercept': np.mean(ts),
            'quad_a': 0, 'quad_b': 0, 'quad_c': 0,
            'trend_acceleration': 0
        })
    
    # Seasonal/cyclical patterns (if enough data)
    if len(ts) >= 6:
        try:
            # Smooth the series and look for patterns
            smoothed = savgol_filter(ts, min(5, len(ts)//2*2+1), 2)
            features['smoothness'] = np.mean(np.abs(ts - smoothed))
            
            # Detect turning points
            diff1 = np.diff(ts)
            diff2 = np.diff(diff1)
            features['turning_points'] = np.sum(np.abs(diff2) > np.std(diff2))
            features['volatility'] = np.std(diff1)
        except:
            features['smoothness'] = features['turning_points'] = features['volatility'] = 0
    else:
        features['smoothness'] = features['turning_points'] = features['volatility'] = 0
    
    # Recent vs historical comparison
    if len(ts) >= 4:
        recent = ts[-2:]  # Last 2 years
        historical = ts[:-2]  # All but last 2 years
        features['recent_vs_historical'] = np.mean(recent) - np.mean(historical)
        features['recent_trend'] = np.polyfit(years[-2:], recent, 1)[0] if len(recent) >= 2 else 0
    else:
        features['recent_vs_historical'] = features['recent_trend'] = 0
        
    return features

def compute_cross_band_features(ndvi_ts, evi_ts, ndwi_ts, years):
    """Compute advanced cross-band relationships"""
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
    
    # Avoid division by zero
    evi_safe = np.where(evi_arr == 0, 1e-8, evi_arr)
    ndwi_safe = np.where(ndwi_arr == 0, 1e-8, ndwi_arr)
    
    features['ndvi_evi_ratio_mean'] = np.mean(ndvi_arr / evi_safe)
    features['ndvi_evi_ratio_std'] = np.std(ndvi_arr / evi_safe)
    features['ndvi_ndwi_ratio_mean'] = np.mean(ndvi_arr / (ndwi_safe + 0.5))  # Shift to avoid negatives
    
    # Composite indices
    features['vegetation_water_balance'] = np.mean((ndvi_arr + evi_arr) - ndwi_arr)
    features['vegetation_stress_proxy'] = np.std((ndvi_arr - evi_arr) + ndwi_arr)
    
    return features

class EnsembleModel:
    """Enhanced ensemble model with multiple algorithms"""
    def __init__(self):
        self.models = {}
        self.weights = {}
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.feature_selector = None
        
    def fit(self, X, y, feature_names=None):
        print("[INFO] Training ensemble model...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Feature selection
        if X_scaled.shape[1] > 100:  # Only if we have many features
            self.feature_selector = SelectKBest(score_func=f_regression, k=min(100, X_scaled.shape[1]//2))
            X_scaled = self.feature_selector.fit_transform(X_scaled, y.ravel() if y.ndim > 1 else y)
            print(f"[INFO] Selected {X_scaled.shape[1]} features out of {X.shape[1]}")
        
        # Initialize models
        if CATBOOST_AVAILABLE:
            self.models['catboost'] = CatBoostRegressor(
                loss_function='MultiRMSE' if y.ndim > 1 else 'RMSE',
                iterations=N_ESTIMATORS,
                depth=MAX_DEPTH,
                learning_rate=LEARNING_RATE,
                verbose=False,
                random_seed=42
            )
        
        if XGBOOST_AVAILABLE:
            if y.ndim > 1:
                # Multi-output wrapper for XGBoost
                from sklearn.multioutput import MultiOutputRegressor
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
            from sklearn.multioutput import MultiOutputRegressor
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
        
        # Gradient Boosting
        if y.ndim > 1:
            from sklearn.multioutput import MultiOutputRegressor
            self.models['gb'] = MultiOutputRegressor(
                GradientBoostingRegressor(
                    n_estimators=N_ESTIMATORS//2,
                    max_depth=MAX_DEPTH,
                    learning_rate=LEARNING_RATE,
                    random_state=42
                )
            )
        else:
            self.models['gb'] = GradientBoostingRegressor(
                n_estimators=N_ESTIMATORS//2,
                max_depth=MAX_DEPTH,
                learning_rate=LEARNING_RATE,
                random_state=42
            )
        
        # Train models and compute weights via cross-validation
        tscv = TimeSeriesSplit(n_splits=N_SPLITS)
        
        for name, model in self.models.items():
            print(f"[INFO] Training {name}...")
            try:
                model.fit(X_scaled, y)
                
                # Compute CV score for weighting
                cv_scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='neg_mean_absolute_error')
                self.weights[name] = np.mean(-cv_scores)  # Convert back to positive MAE
                print(f"[INFO] {name} CV MAE: {self.weights[name]:.4f}")
            except Exception as e:
                print(f"[WARN] Failed to train {name}: {e}")
                if name in self.weights:
                    del self.weights[name]
        
        # Normalize weights (inverse of error - lower error gets higher weight)
        if self.weights:
            total_weight = sum(1/w for w in self.weights.values())
            self.weights = {k: (1/v)/total_weight for k, v in self.weights.items()}
            print("[INFO] Model weights:", self.weights)
        
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        if self.feature_selector:
            X_scaled = self.feature_selector.transform(X_scaled)
        
        if not self.weights:
            return np.zeros((X.shape[0], 3))  # Fallback
        
        predictions = []
        for name, model in self.models.items():
            if name in self.weights:
                pred = model.predict(X_scaled)
                if pred.ndim == 1:
                    pred = pred.reshape(-1, 1)
                predictions.append(pred * self.weights[name])
        
        return np.sum(predictions, axis=0)

# Enhanced main function with all improvements
def enhanced_main():
    # Load data (same as before but with enhanced parameters)
    grid, years = find_heatmaps(INPUT_DIR)
    if not years:
        raise RuntimeError("No complete years found (all bands per year).")
    print("[INFO] Years available:", years)

    if len(years) < MIN_YEARS_REQ:
        print(f"[WARN] Only {len(years)} years available, minimum is {MIN_YEARS_REQ}")
        print("[INFO] Proceeding with available data but precision may be reduced")

    composites = {}
    for y in tqdm(years, desc="Loading composites per year"):
        composites[y] = load_composite_for_year(grid[y])
    H,W,_ = next(iter(composites.values())).shape
    print(f"[INFO] composite shape: {H} x {W}")

    # Enhanced patch extraction
    sample = next(iter(composites.values()))
    patches0, coords = patches_from_image(sample, PATCH_SIZE, STRIDE)
    N_patches = len(coords)
    print(f"[INFO] patches per year: {N_patches} (patch {PATCH_SIZE} stride {STRIDE})")

    # Enhanced embedder
    embedder = EnhancedResNetEmbedder(model_type='resnet50')
    D = embedder.dim

    year_embs = {}
    for y in years:
        print(f"[INFO] extracting enhanced embeddings for year {y} ...")
        patches_y, _ = patches_from_image(composites[y], PATCH_SIZE, STRIDE)
        if patches_y.shape[0] != N_patches:
            raise RuntimeError("Patch count mismatch across years.")
        embs = np.zeros((N_patches, D), dtype=np.float32)
        for i in tqdm(range(0, N_patches, BATCH), desc=f"Embeddings year {y}"):
            batch = patches_y[i:i+BATCH]
            emb = embedder.embed_batch(batch, augment=True)  # Use augmentation
            embs[i:i+emb.shape[0]] = emb
        year_embs[y] = embs

    # Enhanced feature extraction
    patch_info = []
    feature_names = []
    
    for idx in tqdm(range(N_patches), desc="Processing patches with enhanced features"):
        yrs = []
        ndvi_ts = []
        evi_ts = []
        ndwi_ts = []
        E_list = []
        
        for y in years:
            embs = year_embs[y]
            E_list.append(embs[idx])
            yrs.append(y)
            x0,y0 = coords[idx]
            patch = composites[y][y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE, :]
            
            # Enhanced patch statistics
            ndvi_patch = patch[..., BANDS.index("ndvi")]
            evi_patch = patch[..., BANDS.index("evi")]
            ndwi_patch = patch[..., BANDS.index("ndwi")]
            
            ndvi_ts.append(float(ndvi_patch.mean()))
            evi_ts.append(float(evi_patch.mean()))
            ndwi_ts.append(float(ndwi_patch.mean()))
        
        if len(yrs) < MIN_YEARS_REQ:
            continue

        # Enhanced temporal features for each band
        ndvi_features = advanced_temporal_features(ndvi_ts, yrs)
        evi_features = advanced_temporal_features(evi_ts, yrs)
        ndwi_features = advanced_temporal_features(ndwi_ts, yrs)
        
        # Cross-band features
        cross_features = compute_cross_band_features(ndvi_ts, evi_ts, ndwi_ts, yrs)
        
        # Enhanced embedding features
        E = np.stack(E_list, axis=0)
        emb_features = advanced_temporal_features(np.mean(E, axis=1), yrs)
        
        # Store all features
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
            "E": E,
            "ndvi_ts": ndvi_ts,
            "evi_ts": evi_ts,
            "ndwi_ts": ndwi_ts,
            "features": all_features
        })
    
    print(f"[INFO] patches with enhanced features: {len(patch_info)}")
    
    # Build training data with all enhanced features
    X = []
    y = []
    feature_names = list(patch_info[0]["features"].keys()) if patch_info else []
    
    for p in tqdm(patch_info, desc="Building enhanced training data"):
        yrs = p["years"]
        ndvi_ts = p["ndvi_ts"]
        evi_ts = p["evi_ts"]
        ndwi_ts = p["ndwi_ts"]
        
        for t, year in enumerate(yrs):
            # Use all enhanced features
            feat = list(p["features"].values())
            X.append(feat)
            y.append([ndvi_ts[t], evi_ts[t], ndwi_ts[t]])
    
    X = np.array(X)
    y = np.array(y)
    print(f"[INFO] Training data shape: X={X.shape}, y={y.shape}")
    
    # Train enhanced ensemble model
    model = EnsembleModel()
    model.fit(X, y, feature_names)
    
    # Enhanced predictions
    predictions = []
    next_year = years[-1] + 1
    
    for p in tqdm(patch_info, desc="Making enhanced predictions"):
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
            "confidence": 1.0  # Could be enhanced with prediction intervals
        })
    
    df_preds = pd.DataFrame(predictions)
    df_preds.to_csv(OUT_CSV, index=False)
    print(f"[OK] Enhanced predictions saved to {OUT_CSV}")
    print(f"[INFO] Prediction summary:")
    print(f"  - Mean NDVI change: {df_preds['delta_ndvi'].mean():.4f} ± {df_preds['delta_ndvi'].std():.4f}")
    print(f"  - Mean EVI change: {df_preds['delta_evi'].mean():.4f} ± {df_preds['delta_evi'].std():.4f}")
    print(f"  - Mean NDWI change: {df_preds['delta_ndwi'].mean():.4f} ± {df_preds['delta_ndwi'].std():.4f}")
    
    print("[DONE] Enhanced precision pipeline completed.")

# Import the utility functions from original code
def find_heatmaps(input_dir):
    pattern = re.compile(r"heatmap_(?P<band>[A-Za-z]+)_(?P<year>\d{4})\.(png|tif|tiff|jpg|jpeg)$", re.IGNORECASE)
    files = [f for f in tqdm(os.listdir(input_dir), desc="Listing files") if pattern.match(f)]
    grid = {}
    for f in tqdm(files, desc="Grouping files by year and band"):
        m = pattern.match(f)
        band = m.group("band").lower()
        year = int(m.group("year"))
        grid.setdefault(year, {})[band] = os.path.join(input_dir, f)
    years = sorted([y for y in grid.keys() if all(b in grid[y] for b in BANDS)])
    return grid, years

def load_composite_for_year(paths_for_year):
    arrays = []
    for b in tqdm(BANDS, desc="Loading bands for year"):
        p = paths_for_year[b]
        img = Image.open(p).convert("L")
        arr = np.array(img, dtype=np.float32) / 255.0
        arrays.append(arr)
    shapes = [a.shape for a in arrays]
    if len(set(shapes)) != 1:
        raise RuntimeError(f"Band shapes differ for year: {shapes}")
    comp = np.stack(arrays, axis=-1)  # H,W,3
    return comp

def patches_from_image(img, patch_size=PATCH_SIZE, stride=STRIDE):
    H,W,_ = img.shape
    patches = []
    coords = []
    for y in tqdm(range(0, H - patch_size + 1, stride), desc="Patch rows"):
        for x in range(0, W - patch_size + 1, stride):
            p = img[y:y+patch_size, x:x+patch_size, :]
            patches.append(p)
            coords.append((x,y))
    return np.array(patches), coords

if __name__ == "__main__":
    enhanced_main()