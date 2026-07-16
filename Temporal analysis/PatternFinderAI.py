# farmeye_multiband_pipeline.py
import os
import re
import math
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
import matplotlib.colors as colors

import torch
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

from torchvision.models import resnet18
import torchvision.transforms as T

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- CONFIG ----------------
INPUT_DIR = r"C:\Users\Faster\Downloads\FarmEyeInputs\heatmaps_chunked"
BASE_TIF_PATH = r"C:\Users\Faster\Downloads\FarmEyeInputs\2025.tif"  # Base image for overlay
OUT_CSV = "farmeye_multiband_cluster_predictions.csv"
OUT_MAP = "farmeye_multiband_map.png"

BANDS = ["evi", "ndvi", "ndwi"]   # must exist per year
MIN_YEARS_REQ = 3                # minimal years per patch

PATCH_SIZE = 64
STRIDE = 64
BATCH = 128
PCA_DIM = 16
DBSCAN_EPS = 0.6
DBSCAN_MIN_SAMPLES = 4
RIDGE_ALPHA = 1.0

# ----------------------------------------

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

class ResNetEmbedder:
    def __init__(self):
        m = resnet18(pretrained=True)
        self.model = torch.nn.Sequential(*list(m.children())[:-1]).to(DEVICE).eval()
        self.dim = 512
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
    def embed_batch(self, np_patches):
        imgs = []
        for p in np_patches:
            pil = Image.fromarray((p*255).astype(np.uint8))
            t = self.transform(pil).to(DEVICE)
            imgs.append(t)
        batch = torch.stack(imgs)
        with torch.no_grad():
            out = self.model(batch)  # B x 512 x 1 x 1
            out = out.view(out.size(0), -1).cpu().numpy()
        return out

def compute_slope_and_curvature(ts, years):
    if len(ts) < 3:
        slope = 0.0
        curvature = 0.0
    else:
        p = np.polyfit(years, ts, 2)
        slope = p[1]
        curvature = p[0]
    return slope, curvature

def compute_lagged_corr(ts1, ts2, max_lag=1):
    best_corr = 0
    for lag in range(-max_lag, max_lag+1):
        if lag < 0:
            corr = np.corrcoef(ts1[:lag], ts2[-lag:])[0,1]
        elif lag > 0:
            corr = np.corrcoef(ts1[lag:], ts2[:-lag])[0,1]
        else:
            corr = np.corrcoef(ts1, ts2)[0,1]
        if not np.isnan(corr) and abs(corr) > abs(best_corr):
            best_corr = corr
    return best_corr

def load_base_image(base_tif_path, target_width, target_height):
    """Load base TIF image using multiple methods"""
    
    # Method 1: Try rasterio for GeoTIFF files
    if RASTERIO_AVAILABLE:
        try:
            with rasterio.open(base_tif_path) as src:
                # Read the image data
                img_data = src.read()
                
                # Handle different band configurations
                if img_data.shape[0] == 1:
                    # Single band - convert to grayscale then RGB
                    img_array = img_data[0]
                    img_array = np.stack([img_array, img_array, img_array], axis=-1)
                elif img_data.shape[0] >= 3:
                    # Multi-band - use first 3 bands as RGB
                    img_array = np.transpose(img_data[:3], (1, 2, 0))
                else:
                    # Two bands - duplicate first band
                    img_array = img_data[0]
                    img_array = np.stack([img_array, img_array, img_array], axis=-1)
                
                # Normalize to 0-1 range
                if img_array.max() > 1:
                    img_array = img_array.astype(np.float32)
                    # Handle different data ranges
                    if img_array.max() > 255:
                        img_array = img_array / img_array.max()  # Normalize to max
                    else:
                        img_array = img_array / 255.0
                
                # Resize if needed
                if img_array.shape[:2] != (target_height, target_width):
                    from PIL import Image
                    # Convert to PIL for resizing
                    pil_img = Image.fromarray((img_array * 255).astype(np.uint8))
                    pil_img = pil_img.resize((target_width, target_height), Image.Resampling.LANCZOS)
                    img_array = np.array(pil_img, dtype=np.float32) / 255.0
                
                print(f"[INFO] Successfully loaded TIF using rasterio: {img_array.shape}")
                return img_array
                
        except Exception as e:
            print(f"[WARN] Rasterio failed: {e}")
    
    # Method 2: Try PIL/Pillow
    try:
        base_img = Image.open(base_tif_path)
        # Convert to RGB if not already
        if base_img.mode != 'RGB':
            base_img = base_img.convert('RGB')
        
        # Resize to match our processing dimensions if needed
        if base_img.size != (target_width, target_height):
            print(f"[INFO] Resizing base image from {base_img.size} to {(target_width, target_height)}")
            base_img = base_img.resize((target_width, target_height), Image.Resampling.LANCZOS)
        
        base_array = np.array(base_img, dtype=np.float32) / 255.0
        print(f"[INFO] Successfully loaded TIF using PIL: {base_array.shape}")
        return base_array
        
    except Exception as e:
        print(f"[WARN] PIL failed: {e}")
    
    # Method 3: Fallback to neutral background
    print("[INFO] Creating default neutral background")
    return np.full((target_height, target_width, 3), 0.7, dtype=np.float32)
def create_overlay_visualization(df_preds, base_tif_path, H, W, coords):
    """Create visualization overlaid on base TIF image"""
    
    # Load base image using multiple methods
    base_array = load_base_image(base_tif_path, W, H)
    
    # Create prediction overlays with transparency
    ndvi_overlay = np.zeros((H, W, 4), dtype=np.float32)  # RGBA
    evi_overlay = np.zeros((H, W, 4), dtype=np.float32)
    ndwi_overlay = np.zeros((H, W, 4), dtype=np.float32)
    cluster_overlay = np.zeros((H, W, 4), dtype=np.float32)
    
    # Define color maps
    def get_vegetation_color(delta, alpha=0.7):
        """Get RGBA color for vegetation change"""
        norm_delta = np.clip(delta * 10, -1, 1)  # Scale for visibility
        if norm_delta >= 0:
            # Green for increase
            return [0, norm_delta, 0, alpha]
        else:
            # Red for decrease
            return [-norm_delta, 0, 0, alpha]
    
    def get_water_color(delta, alpha=0.7):
        """Get RGBA color for water content change"""
        norm_delta = np.clip(delta * 10, -1, 1)
        if norm_delta >= 0:
            # Blue for increase
            return [0, 0, norm_delta, alpha]
        else:
            # Red for decrease
            return [-norm_delta, 0, 0, alpha]
    
    # Fill overlay maps
    unique_clusters = df_preds['cluster_id'].unique()
    unique_clusters = unique_clusters[unique_clusters >= 0]
    cluster_colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_clusters), 10)))
    
    for _, row in df_preds.iterrows():
        patch_idx = int(row['patch_idx'])
        if patch_idx < len(coords):
            x, y = coords[patch_idx]
            
            # Get colors for this patch
            ndvi_color = get_vegetation_color(row['delta_ndvi'], 0.6)
            evi_color = get_vegetation_color(row['delta_evi'], 0.6)
            ndwi_color = get_water_color(row['delta_ndwi'], 0.6)
            
            cluster_id = row['cluster_id']
            if cluster_id >= 0:
                # Fix the broadcasting error by ensuring exactly 4 elements (RGBA)
                base_color = cluster_colors[int(cluster_id) % len(cluster_colors)]
                cluster_color = [float(base_color[0]), float(base_color[1]), float(base_color[2]), 0.5]
            else:
                cluster_color = [0.5, 0.5, 0.5, 0.3]  # Gray for noise
            
            # Fill patch areas
            for py in range(y, min(H, y + PATCH_SIZE)):
                for px in range(x, min(W, x + PATCH_SIZE)):
                    ndvi_overlay[py, px] = ndvi_color
                    evi_overlay[py, px] = evi_color
                    ndwi_overlay[py, px] = ndwi_color
                    cluster_overlay[py, px] = cluster_color
    
    # Create composite images (base + overlay)
    def blend_with_base(base, overlay):
        """Blend RGBA overlay with RGB base"""
        alpha = overlay[:, :, 3:4]
        overlay_rgb = overlay[:, :, :3]
        blended = base * (1 - alpha) + overlay_rgb * alpha
        return np.clip(blended, 0, 1)
    
    ndvi_composite = blend_with_base(base_array, ndvi_overlay)
    evi_composite = blend_with_base(base_array, evi_overlay)
    ndwi_composite = blend_with_base(base_array, ndwi_overlay)
    cluster_composite = blend_with_base(base_array, cluster_overlay)
    
    # Create the visualization
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # Base image
    axes[0,0].imshow(base_array)
    axes[0,0].set_title('Base Image (2025.tif)')
    axes[0,0].axis('off')
    
    # NDVI Delta Overlay
    axes[0,1].imshow(ndvi_composite)
    axes[0,1].set_title('NDVI Delta Overlay\n(Green=Increase, Red=Decrease)')
    axes[0,1].axis('off')
    
    # EVI Delta Overlay
    axes[0,2].imshow(evi_composite)
    axes[0,2].set_title('EVI Delta Overlay\n(Green=Increase, Red=Decrease)')
    axes[0,2].axis('off')
    
    # NDWI Delta Overlay
    axes[1,0].imshow(ndwi_composite)
    axes[1,0].set_title('NDWI Delta Overlay\n(Blue=Increase, Red=Decrease)')
    axes[1,0].axis('off')
    
    # Cluster Overlay
    axes[1,1].imshow(cluster_composite)
    axes[1,1].set_title(f'Cluster Overlay\n({len(unique_clusters)} clusters)')
    axes[1,1].axis('off')
    
    # Create a combined overlay showing all predictions
    combined_overlay = np.zeros((H, W, 4), dtype=np.float32)
    for _, row in df_preds.iterrows():
        patch_idx = int(row['patch_idx'])
        if patch_idx < len(coords):
            x, y = coords[patch_idx]
            
            # Combine NDVI and NDWI changes for comprehensive view
            ndvi_delta = row['delta_ndvi']
            ndwi_delta = row['delta_ndwi']
            
            # Green-Red for vegetation, Blue component for water
            r = max(0, -ndvi_delta * 10)  # Red for vegetation decrease
            g = max(0, ndvi_delta * 10)   # Green for vegetation increase  
            b = max(0, ndwi_delta * 5)    # Blue for water increase
            alpha = 0.7
            
            combined_color = [r, g, b, alpha]
            
            for py in range(y, min(H, y + PATCH_SIZE)):
                for px in range(x, min(W, x + PATCH_SIZE)):
                    combined_overlay[py, px] = combined_color
    
    combined_composite = blend_with_base(base_array, combined_overlay)
    axes[1,2].imshow(combined_composite)
    axes[1,2].set_title('Combined Change Overlay\n(Vegetation + Water)')
    axes[1,2].axis('off')
    
    plt.tight_layout()
    return fig, combined_composite

def main():
    grid, years = find_heatmaps(INPUT_DIR)
    if not years:
        raise RuntimeError("No complete years found (all bands per year).")
    print("[INFO] Years available:", years)

    composites = {}
    for y in tqdm(years, desc="Loading composites per year"):
        composites[y] = load_composite_for_year(grid[y])
    H,W,_ = next(iter(composites.values())).shape
    print(f"[INFO] composite shape: {H} x {W}")

    sample = next(iter(composites.values()))
    patches0, coords = patches_from_image(sample, PATCH_SIZE, STRIDE)
    N_patches = len(coords)
    print(f"[INFO] patches per year: {N_patches} (patch {PATCH_SIZE} stride {STRIDE})")

    embedder = ResNetEmbedder()
    D = embedder.dim

    year_embs = {}
    for y in years:
        print(f"[INFO] extracting embeddings for year {y} ...")
        patches_y, _ = patches_from_image(composites[y], PATCH_SIZE, STRIDE)
        if patches_y.shape[0] != N_patches:
            raise RuntimeError("Patch count mismatch across years.")
        embs = np.zeros((N_patches, D), dtype=np.float32)
        for i in tqdm(range(0, N_patches, BATCH), desc=f"Embeddings year {y}"):
            batch = patches_y[i:i+BATCH]
            emb = embedder.embed_batch(batch)
            embs[i:i+emb.shape[0]] = emb
        year_embs[y] = embs

    patch_info = []
    for idx in tqdm(range(N_patches), desc="Processing patches"):
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
            ndvi_ts.append(float(patch[..., BANDS.index("ndvi")].mean()))
            evi_ts.append(float(patch[..., BANDS.index("evi")].mean()))
            ndwi_ts.append(float(patch[..., BANDS.index("ndwi")].mean()))
        if len(yrs) < MIN_YEARS_REQ:
            continue

        E = np.stack(E_list, axis=0)
        slopes = np.polyfit(yrs, E, 1)[0] if len(yrs) >= 2 else np.zeros(D)
        slope_norm = float(np.linalg.norm(slopes))
        emb_mags = np.linalg.norm(E, axis=1)
        if len(yrs) >= 2:
            p = np.polyfit(yrs, emb_mags, 1)
            pred = np.polyval(p, yrs)
            resid = emb_mags - pred
            resid_score = float(np.max(np.abs(zscore(resid))) if np.std(resid) > 0 else 0.0)
        else:
            resid_score = 0.0

        ndvi_slope, ndvi_curv = compute_slope_and_curvature(ndvi_ts, yrs)
        evi_slope, evi_curv = compute_slope_and_curvature(evi_ts, yrs)
        ndwi_slope, ndwi_curv = compute_slope_and_curvature(ndwi_ts, yrs)

        ndvi_evi_corr = compute_lagged_corr(np.array(ndvi_ts), np.array(evi_ts))
        ndvi_ndwi_corr = compute_lagged_corr(np.array(ndvi_ts), np.array(ndwi_ts))
        evi_ndwi_corr = compute_lagged_corr(np.array(evi_ts), np.array(ndwi_ts))

        ndvi_var = np.std(ndvi_ts)
        evi_var = np.std(evi_ts)
        ndwi_var = np.std(ndwi_ts)

        patch_info.append({
            "idx": idx,
            "coord": coords[idx],
            "years": yrs,
            "E": E,
            "slopes": slopes,
            "slope_norm": slope_norm,
            "resid_score": resid_score,
            "ndvi_ts": ndvi_ts,
            "evi_ts": evi_ts,
            "ndwi_ts": ndwi_ts,
            "ndvi_slope": ndvi_slope,
            "ndvi_curv": ndvi_curv,
            "evi_slope": evi_slope,
            "evi_curv": evi_curv,
            "ndwi_slope": ndwi_slope,
            "ndwi_curv": ndwi_curv,
            "ndvi_evi_corr": ndvi_evi_corr,
            "ndvi_ndwi_corr": ndvi_ndwi_corr,
            "evi_ndwi_corr": evi_ndwi_corr,
            "ndvi_var": ndvi_var,
            "evi_var": evi_var,
            "ndwi_var": ndwi_var
        })

    print(f"[INFO] patches with valid timeseries: {len(patch_info)}")

    X = []
    y = []
    for p in tqdm(patch_info, desc="Building training data"):
        yrs = p["years"]
        E = p["E"]
        ndvi_ts = p["ndvi_ts"]
        evi_ts = p["evi_ts"]
        ndwi_ts = p["ndwi_ts"]

        for t, year in enumerate(yrs):
            feat = list(E[t]) + [
                p["ndvi_slope"],
                p["ndvi_curv"],
                p["evi_slope"],
                p["evi_curv"],
                p["ndwi_slope"],
                p["ndwi_curv"],
                p["ndvi_evi_corr"],
                p["ndvi_ndwi_corr"],
                p["evi_ndwi_corr"],
                p["ndvi_var"],
                p["evi_var"],
                p["ndwi_var"],
                p["slope_norm"],
                p["resid_score"]
            ]
            X.append(feat)
            y.append([ndvi_ts[t], evi_ts[t], ndwi_ts[t]])
    X = np.array(X)
    y = np.array(y)

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    if CATBOOST_AVAILABLE:
        print("[INFO] Training multi-output CatBoostRegressor")
        model = CatBoostRegressor(loss_function='MultiRMSE', verbose=False, random_seed=42)
        model.fit(Xs, y)
    else:
        print("[WARN] CatBoost not available, fallback to Ridge multi-output regression")
        from sklearn.multioutput import MultiOutputRegressor
        base = Ridge(alpha=RIDGE_ALPHA)
        model = MultiOutputRegressor(base).fit(Xs, y)

    print("[INFO] Model training completed")

    predictions = []
    next_year = years[-1] + 1
    for p in tqdm(patch_info, desc="Predicting next year per patch"):
        E_last = p["E"][-1]
        slopes = p["slopes"]
        emb_next = E_last + slopes * 1.0
        feat_pred = list(emb_next) + [
            p["ndvi_slope"],
            p["ndvi_curv"],
            p["evi_slope"],
            p["evi_curv"],
            p["ndwi_slope"],
            p["ndwi_curv"],
            p["ndvi_evi_corr"],
            p["ndvi_ndwi_corr"],
            p["evi_ndwi_corr"],
            p["ndvi_var"],
            p["evi_var"],
            p["ndwi_var"],
            p["slope_norm"],
            p["resid_score"]
        ]
        feat_pred_s = scaler.transform([feat_pred])
        pred_vals = model.predict(feat_pred_s)[0]

        ndvi_last = p["ndvi_ts"][-1]
        evi_last = p["evi_ts"][-1]
        ndwi_last = p["ndwi_ts"][-1]
        delta_ndvi = pred_vals[0] - ndvi_last
        delta_evi = pred_vals[1] - evi_last
        delta_ndwi = pred_vals[2] - ndwi_last

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
            "delta_ndvi": delta_ndvi,
            "delta_evi": delta_evi,
            "delta_ndwi": delta_ndwi,
            "slope_norm": p["slope_norm"],
            "resid_score": p["resid_score"]
        })

    df_preds = pd.DataFrame(predictions)

    slopes_mat = np.stack([p["slopes"] for p in patch_info], axis=0)
    pca = PCA(n_components=min(PCA_DIM, slopes_mat.shape[1])).fit(slopes_mat)
    slopes_reduced = pca.transform(slopes_mat)
    db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(slopes_reduced)
    labels = db.labels_

    df_preds["cluster_id"] = labels

    clusters = []
    for lbl in tqdm(sorted(df_preds["cluster_id"].unique()), desc="Aggregating clusters"):
        if lbl == -1:
            continue
        sub = df_preds[df_preds["cluster_id"]==lbl]
        clusters.append({
            "cluster_id": int(lbl),
            "center_x": int(sub["center_x"].mean()),
            "center_y": int(sub["center_y"].mean()),
            "patch_count": int(len(sub)),
            "ndvi_pred_next_mean": float(sub["ndvi_pred_next"].mean()),
            "evi_pred_next_mean": float(sub["evi_pred_next"].mean()),
            "ndwi_pred_next_mean": float(sub["ndwi_pred_next"].mean()),
            "ndvi_delta_mean": float(sub["delta_ndvi"].mean()),
            "evi_delta_mean": float(sub["delta_evi"].mean()),
            "ndwi_delta_mean": float(sub["delta_ndwi"].mean()),
            "slope_norm_mean": float(sub["slope_norm"].mean()),
            "resid_mean": float(sub["resid_score"].mean())
        })
    clusters_df = pd.DataFrame(clusters).sort_values("ndvi_delta_mean")
    clusters_df.to_csv(OUT_CSV, index=False)
    print(f"[OK] Cluster predictions saved to {OUT_CSV}")

    # Create the improved visualization with base TIF overlay
    fig, combined_composite = create_overlay_visualization(df_preds, BASE_TIF_PATH, H, W, coords)
    fig.savefig(OUT_MAP, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Overlay map saved to {OUT_MAP}")
    
    # Save the combined composite as a standalone image
    combined_map_path = OUT_MAP.replace('.png', '_combined.png')
    plt.figure(figsize=(12, 8))
    plt.imshow(combined_composite)
    plt.axis('off')
    plt.title('Agricultural Change Predictions Overlaid on 2025 Base Image')
    plt.savefig(combined_map_path, dpi=200, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"[OK] Combined overlay saved to {combined_map_path}")
    
    # Also create a simple overlay showing just NDVI delta on base image
    try:
        base_array_simple = load_base_image(BASE_TIF_PATH, W, H)
        
        # Create NDVI overlay
        ndvi_overlay = np.zeros((H, W, 4), dtype=np.float32)
        for _, row in df_preds.iterrows():
            patch_idx = int(row['patch_idx'])
            if patch_idx < len(coords):
                x, y = coords[patch_idx]
                delta = row['delta_ndvi']
                norm_delta = np.clip(delta * 10, -1, 1)
                
                if norm_delta >= 0:
                    color = [0, norm_delta, 0, 0.6]  # Green for increase
                else:
                    color = [-norm_delta, 0, 0, 0.6]  # Red for decrease
                
                for py in range(y, min(H, y + PATCH_SIZE)):
                    for px in range(x, min(W, x + PATCH_SIZE)):
                        ndvi_overlay[py, px] = color
        
        # Blend overlay with base
        alpha = ndvi_overlay[:, :, 3:4]
        overlay_rgb = ndvi_overlay[:, :, :3]
        blended = base_array_simple * (1 - alpha) + overlay_rgb * alpha
        blended = np.clip(blended, 0, 1)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(blended)
        plt.title(f'NDVI Change Prediction for {years[-1] + 1} (Overlaid on Base Image)\nGreen=Vegetation Increase, Red=Vegetation Decrease')
        plt.axis('off')
        
    except Exception as e:
        print(f"[WARN] Could not create base overlay: {e}")
        # Fallback to simple map without overlay
        ndvi_delta_map = np.full((H, W), np.nan)
        for _, row in df_preds.iterrows():
            patch_idx = int(row['patch_idx'])
            if patch_idx < len(coords):
                x, y = coords[patch_idx]
                for py in range(y, min(H, y + PATCH_SIZE)):
                    for px in range(x, min(W, x + PATCH_SIZE)):
                        ndvi_delta_map[py, px] = row['delta_ndvi']
        
        plt.figure(figsize=(12, 8))
        plt.imshow(ndvi_delta_map, cmap='RdYlGn', vmin=-0.1, vmax=0.1)
        plt.colorbar(shrink=0.8, label='NDVI Delta')
        plt.title(f'Predicted NDVI Change for {years[-1] + 1}\n(Green=Vegetation Increase, Red=Vegetation Decrease)')
        plt.axis('off')
    
    simple_map_name = OUT_MAP.replace('.png', '_simple.png')
    plt.savefig(simple_map_name, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[OK] Simple overlay map saved to {simple_map_name}")
    
    print("[DONE] Pipeline finished.")

if __name__ == "__main__":
    main()