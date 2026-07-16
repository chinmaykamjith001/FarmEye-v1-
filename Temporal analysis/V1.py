"""
FarmEye pattern discovery pipeline
Inputs:
 - A directory of raster files (GeoTIFF preferred) containing NDVI, NDWI, EVI images for each year.
 - Filenames should contain band name (ndvi, ndwi, evi) and year (like 2020, 2021).
Outputs:
 - zone_segmentation.png
 - zone_cluster_map.png
 - change_maps_{band}.png
 - lag_map_{bandA}_to_{bandB}.png
 - summary_table.csv (zone, cluster, flagged changes, strong lagged relations, textual descriptions)
Notes:
 - With only 5-7 timepoints this pipeline reports strong, spatially coherent patterns only.
 - To run: pip install numpy rasterio scikit-image scikit-learn pandas matplotlib tqdm
"""

import os
import re
import glob
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from skimage.segmentation import slic
from skimage.color import gray2rgb
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")

# CONFIG (edit these if needed)
# Use raw string for Windows paths or double backslashes
IMAGES_DIR = r"C:\Users\Faster\Downloads\FarmEyeInputs\heatmaps_chunked"
BAND_NAMES = {"ndvi": "NDVI", "ndwi": "NDWI", "evi": "EVI"}  # Core bands
BAND_ALIASES = {"nwdi": "ndwi"}  # Accept NWDI typo as alias for NDWI
N_SEGMENTS = 120      # try 120 zones to start; lower if tiny fields
MIN_PIXELS_PER_ZONE = 50
CHANGE_Z_THRESHOLD = 1.5
LAG_CORR_THRESHOLD = 0.6
PERM_TEST_ITER = 500
N_CLUSTERS = 4
OUT_DIR = "out_farmeye"
os.makedirs(OUT_DIR, exist_ok=True)

# utilities: parse filenames for band and year
def parse_band_year(fname):
    base = os.path.basename(fname).lower()
    year_match = re.search(r"(19|20)\d{2}", base)
    year = int(year_match.group(0)) if year_match else None
    band = None
    # Check core bands first
    for key in BAND_NAMES:
        if key in base:
            band = key
            break
    # Check aliases if no core band found
    if band is None:
        for alias, actual in BAND_ALIASES.items():
            if alias in base:
                band = actual
                break
    return band, year

# load rasters robustly, reproject/resample to a common grid if needed
def load_and_stack(files):
    entries = []
    print("Parsing filenames...")
    for f in tqdm(files, desc="Parsing files"):
        band, year = parse_band_year(f)
        if band is None or year is None:
            print("Skipping (no band/year found):", f)
            continue
        band = band.lower()  # force lowercase for consistency
        entries.append((year, band, f))
    if not entries:
        raise RuntimeError("No valid files found. Filenames must include band name and year.")
    # sort by year
    entries.sort(key=lambda x: x[0])
    years = sorted(list({e[0] for e in entries}))
    # Build mapping year->band->path
    grid = {y: {} for y in years}
    for y, b, p in entries:
        grid[y][b] = p  # ensure lowercase keys
    print("Detected files by year and band:")
    for y in years:
        print(f"Year {y} bands: ", list(grid[y].keys()))
    # ensure each year has all bands
    good_years = []
    for y in years:
        # Check if all required bands are present (case-insensitive)
        required_bands = set(k.lower() for k in BAND_NAMES.keys())
        available_bands = set(grid[y].keys())
        if required_bands.issubset(available_bands):
            good_years.append(y)
        else:
            missing = required_bands - available_bands
            print(f"Year {y} missing bands: {missing}, skipping year.")
    if len(good_years) < 3:
        print("Warning: fewer than 3 full years found. Proceeding but results will be weak.")
    if not good_years:
        raise RuntimeError("No years with all bands found.")
    # Use the first raster as the reference grid
    ref_path = grid[good_years[0]][list(BAND_NAMES.keys())[0].lower()]  # Fixed: ensure lowercase
    with rasterio.open(ref_path) as src:
        ref_meta = src.meta.copy()
        ref_transform = src.transform
        ref_crs = src.crs
        ref_shape = (src.height, src.width)
    # read and reproject each band to ref grid when needed
    stack_list = []  # will hold (year, band_array)
    years_used = []
    print("Loading and reprojecting rasters...")
    for y in tqdm(good_years, desc="Processing years"):
        band_arrays = []
        for b in BAND_NAMES:
            p = grid[y][b.lower()]  # Fixed: ensure lowercase key access
            with rasterio.open(p) as src:
                arr = src.read(1).astype(np.float32)
                if src.crs != ref_crs or src.transform != ref_transform or arr.shape != ref_shape:  # Fixed: use arr.shape not src.shape
                    # reproject/resample to reference
                    dest = np.empty(ref_shape, dtype=np.float32)
                    reproject(
                        source=arr,
                        destination=dest,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=ref_transform,
                        dst_crs=ref_crs,
                        resampling=Resampling.bilinear,
                    )
                    arr = dest
            band_arrays.append(arr)
        stack_list.append((y, np.stack(band_arrays, axis=0)))  # shape (bands, H, W)
        years_used.append(y)
    # final stacked tensor shape (T, C, H, W)
    T = len(stack_list)
    C = len(BAND_NAMES)
    H, W = ref_shape
    tensor = np.zeros((T, C, H, W), dtype=np.float32)
    for i, (y, arr) in enumerate(stack_list):
        tensor[i] = arr
    return tensor, years_used, (H, W), list(BAND_NAMES.keys())

# segmentation on mean composite
def segment_image_mean(tensor, n_segments=N_SEGMENTS):
    # create 2D image for segmentation: mean across time and bands
    img = tensor.mean(axis=(0,1))  # shape H,W
    # slic expects multichannel; convert to 3 channel grayscale replicate
    img_norm = (img - np.nanmin(img)) / (np.nanmax(img) - np.nanmin(img) + 1e-9)
    # Handle NaN values
    img_norm = np.nan_to_num(img_norm, nan=0.0)
    rgb = gray2rgb(img_norm)
    segments = slic(rgb, n_segments=n_segments, compactness=10, start_label=1)
    return segments

# compute zone time series
def zones_time_series(tensor, segments):
    T, C, H, W = tensor.shape
    zone_ids = np.unique(segments)
    zone_ts = {}
    zone_pixel_counts = {}
    print("Computing zone time series...")
    for z in tqdm(zone_ids, desc="Processing zones"):
        mask = segments == z
        cnt = mask.sum()
        zone_pixel_counts[z] = int(cnt)
        if cnt < MIN_PIXELS_PER_ZONE:
            # will still compute but flag later
            pass
        vals = tensor[:, :, mask].mean(axis=2)  # shape (T, C)
        zone_ts[z] = vals  # T x C
    return zone_ts, zone_pixel_counts

# change detection per zone & band
def detect_changes(zone_ts, years, thresh=CHANGE_Z_THRESHOLD):
    # zone_ts: dict zone -> (T, C) array
    changes = defaultdict(list)
    for z, arr in zone_ts.items():
        T, C = arr.shape
        # compute year-deltas along time
        deltas = arr[1:] - arr[:-1]  # shape (T-1, C)
        for t in range(deltas.shape[0]):
            for c in range(C):
                val = deltas[t, c]
                if not np.isnan(val) and np.abs(val) >= thresh:  # Fixed: check for NaN
                    changes[z].append({
                        "year_from": years[t],
                        "year_to": years[t+1],
                        "band_idx": c,
                        "delta": float(val),
                        "sign": "decrease" if val < 0 else "increase"
                    })
    return changes

# lagged correlation (lag=1) per zone, band pairs with permutation p-value
def lagged_corr_per_zone(zone_ts, perm_iter=PERM_TEST_ITER):
    relations = defaultdict(list)
    print("Computing lagged correlations...")
    for z in tqdm(zone_ts.keys(), desc="Processing zone correlations"):
        arr = zone_ts[z]
        T, C = arr.shape
        if T < 2: 
            continue
        for i in range(C):
            for j in range(C):
                if i == j: continue
                x = arr[:-1, i]  # t0..t_{T-2}
                y = arr[1:, j]   # t1..t_{T-1}
                # Remove NaN values
                mask = ~(np.isnan(x) | np.isnan(y))
                if mask.sum() < 2:  # need at least 2 valid points
                    continue
                x_clean = x[mask]
                y_clean = y[mask]
                if np.std(x_clean) == 0 or np.std(y_clean) == 0:
                    continue
                r = np.corrcoef(x_clean, y_clean)[0,1]
                if np.isnan(r):  # Fixed: check for NaN correlation
                    continue
                # permutation p-value: shuffle y many times
                perm_r = []
                for _ in range(perm_iter):
                    ys = np.random.permutation(y_clean)
                    corr = np.corrcoef(x_clean, ys)[0,1]
                    if not np.isnan(corr):  # Fixed: only add valid correlations
                        perm_r.append(corr)
                if len(perm_r) == 0:  # Fixed: check if we have valid permutations
                    continue
                perm_r = np.array(perm_r)
                # two-sided p-value
                p = (np.sum(np.abs(perm_r) >= np.abs(r)) + 1) / (len(perm_r) + 1)
                relations[z].append({
                    "from_band": i,
                    "to_band": j,
                    "r": float(r),
                    "p": float(p)
                })
    return relations

# clustering of trajectory shapes
def cluster_zones(zone_ts, n_clusters=N_CLUSTERS):
    # for each zone, flatten T x C into vector
    zones = sorted(zone_ts.keys())
    X = []
    for z in zones:
        arr = zone_ts[z]  # T x C
        vec = arr.flatten()
        X.append(vec)
    X = np.array(X)
    # standardize
    scaler = StandardScaler()
    Xs = scaler.fit_transform(np.nan_to_num(X))
    k = min(n_clusters, max(2, len(zones)//5))
    km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(Xs)  # Fixed: added n_init parameter
    labels = {z: int(l) for z, l in zip(zones, km.labels_)}
    return labels, km, scaler

# build textual descriptions
def build_descriptions(changes, relations, clusters, band_keys):
    rows = []
    all_zones = set(list(clusters.keys()) + list(changes.keys()) + list(relations.keys()))
    for z in sorted(all_zones):
        cluster = clusters.get(z, None)
        ch_list = changes.get(z, [])
        rel_list = relations.get(z, [])
        descs = []
        # summarize changes
        for ch in ch_list:
            band = band_keys[ch["band_idx"]].upper()
            descs.append(f"{band} {ch['sign']} {ch['year_from']}->{ch['year_to']} (Δ={ch['delta']:.2f})")
        # summarize strong relations (filter by threshold)
        strong_rels = []
        for r in rel_list:
            if abs(r["r"]) >= LAG_CORR_THRESHOLD and r["p"] < 0.05:
                from_b = band_keys[r["from_band"]].upper()
                to_b = band_keys[r["to_band"]].upper()
                strong_rels.append(f"{from_b}(t) -> {to_b}(t+1) r={r['r']:.2f} p={r['p']:.3f}")  # Fixed: removed extra braces
        # human translation mapping
        human = []
        for ch in ch_list:
            bname = band_keys[ch["band_idx"]]
            if bname.lower() == "ndvi":
                human.append(f"vegetation {'loss' if ch['sign']=='decrease' else 'gain'}")
            elif bname.lower() == "ndwi":
                human.append(f"water-content {'loss' if ch['sign']=='decrease' else 'gain'}")
            elif bname.lower() == "evi":
                human.append(f"EVI {'drop' if ch['sign']=='decrease' else 'rise'}")
        summary = "; ".join(descs + strong_rels + human) if (descs or strong_rels or human) else "no strong pattern"
        rows.append({
            "zone": int(z),
            "cluster": int(cluster) if cluster is not None else None,
            "n_changes": len(ch_list),
            "n_strong_relations": sum(1 for r in rel_list if abs(r["r"])>=LAG_CORR_THRESHOLD and r["p"]<0.05),
            "summary": summary
        })
    return pd.DataFrame(rows)

# plotting helpers
def plot_segments(segments, outpath):
    plt.figure(figsize=(10,10))
    plt.imshow(segments, cmap='tab20')  # Fixed: added colormap for better visualization
    plt.axis("off")
    plt.title("Zone segments")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')  # Fixed: added bbox_inches
    plt.close()

def plot_cluster_map(segments, clusters, outpath):
    # map cluster label per zone id
    max_zone = segments.max()
    cluster_map = np.zeros_like(segments, dtype=float)
    for z, lab in clusters.items():
        cluster_map[segments==z] = lab
    plt.figure(figsize=(10,10))
    plt.imshow(cluster_map, cmap='viridis')  # Fixed: added colormap
    plt.colorbar(label='Cluster ID')  # Fixed: added colorbar
    plt.axis("off")
    plt.title("Zone clusters")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')  # Fixed: added bbox_inches
    plt.close()

def plot_change_maps(tensor, segments, years, band_keys, changes, outdir):
    # create zone-level change magnitude map for each band for the last delta (year T-1 -> T)
    T, C, H, W = tensor.shape
    # compute final deltas
    last_deltas = tensor[-1] - tensor[-2] if T>=2 else np.zeros((C,H,W))
    print("Creating change maps...")
    for c_idx, b in enumerate(tqdm(band_keys, desc="Generating change maps")):
        mag = np.abs(last_deltas[c_idx])
        plt.figure(figsize=(10,10))
        plt.imshow(mag, cmap='hot')  # Fixed: added colormap
        plt.colorbar(label='Absolute Change')  # Fixed: added colorbar
        plt.title(f"Abs delta last year - {b.upper()}")
        plt.axis("off")
        plt.tight_layout()
        p = os.path.join(outdir, f"change_map_{b}.png")
        plt.savefig(p, dpi=150, bbox_inches='tight')  # Fixed: added bbox_inches
        plt.close()

def plot_lag_map(segments, relations, band_keys, outpath, from_idx, to_idx):
    # map r where relation from_idx->to_idx strong and significant
    arr = np.zeros_like(segments, dtype=float)
    for z, rels in relations.items():
        # find appropriate pair
        for r in rels:
            if r['from_band']==from_idx and r['to_band']==to_idx and abs(r['r'])>=LAG_CORR_THRESHOLD and r['p']<0.05:
                arr[segments==z] = r['r']
    plt.figure(figsize=(10,10))
    im = plt.imshow(arr, cmap='RdBu_r', vmin=-1, vmax=1)  # Fixed: added colormap and limits
    plt.colorbar(im, label='Correlation coefficient')  # Fixed: added colorbar
    plt.title(f"Lagged corr r: {band_keys[from_idx].upper()}(t) -> {band_keys[to_idx].upper()}(t+1)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')  # Fixed: added bbox_inches
    plt.close()

# main pipeline
def main():
    try:  # Fixed: added error handling
        files = glob.glob(os.path.join(IMAGES_DIR, "*"))
        if not files:
            raise RuntimeError(f"No files found in directory: {IMAGES_DIR}")
        
        tensor, years, shape, band_keys = load_and_stack(files)
        print("Loaded tensor shape (T,C,H,W):", tensor.shape)
        
        segments = segment_image_mean(tensor)
        plot_segments(segments, os.path.join(OUT_DIR, "zone_segmentation.png"))
        
        zone_ts, counts = zones_time_series(tensor, segments)
        # filter tiny zones
        zone_ts = {z:ts for z,ts in zone_ts.items() if counts.get(z,0) >= MIN_PIXELS_PER_ZONE}  # Fixed: use proper threshold
        
        if not zone_ts:  # Fixed: check if we have any zones left
            raise RuntimeError("No zones with sufficient pixels found.")
        
        changes = detect_changes(zone_ts, years)
        relations = lagged_corr_per_zone(zone_ts)
        clusters, km, scaler = cluster_zones(zone_ts)
        
        plot_cluster_map(segments, clusters, os.path.join(OUT_DIR, "zone_cluster_map.png"))
        plot_change_maps(tensor, segments, years, band_keys, changes, OUT_DIR)
        
        # plot a couple lag maps for the most interesting band pairs (NDWI->NDVI and NDVI->EVI)
        keymap = {k.lower():i for i,k in enumerate(band_keys)}  # Fixed: ensure lowercase keys
        if "ndwi" in keymap and "ndvi" in keymap:
            plot_lag_map(segments, relations, band_keys, 
                        os.path.join(OUT_DIR, "lag_ndwi_ndvi.png"), 
                        keymap["ndwi"], keymap["ndvi"])
        if "ndvi" in keymap and "evi" in keymap:
            plot_lag_map(segments, relations, band_keys, 
                        os.path.join(OUT_DIR, "lag_ndvi_evi.png"), 
                        keymap["ndvi"], keymap["evi"])
        
        # summary table
        df = build_descriptions(changes, relations, clusters, band_keys)
        df.to_csv(os.path.join(OUT_DIR, "summary_table.csv"), index=False)
        
        print("Outputs written to", OUT_DIR)
        print("\n" + "="*80)
        print("FARMEYE PATTERN DISCOVERY SUMMARY")
        print("="*80)
        
        # Display top patterns in human-readable form
        display_top_patterns(df, changes, relations, band_keys, years)
        
        print("\nDetailed CSV saved to:", os.path.join(OUT_DIR, "summary_table.csv"))
        
    except Exception as e:
        print(f"Error in pipeline: {str(e)}")
        raise

def display_top_patterns(df, changes, relations, band_keys, years):
    """Display top patterns in human-readable format"""
    
    # Overall statistics
    total_zones = len(df)
    zones_with_changes = len(df[df['n_changes'] > 0])
    zones_with_relations = len(df[df['n_strong_relations'] > 0])
    
    print(f"📊 DATASET OVERVIEW")
    print(f"   • Total zones analyzed: {total_zones}")
    print(f"   • Time period: {min(years)}-{max(years)} ({len(years)} years)")
    print(f"   • Zones with significant changes: {zones_with_changes} ({zones_with_changes/total_zones*100:.1f}%)")
    print(f"   • Zones with strong correlations: {zones_with_relations} ({zones_with_relations/total_zones*100:.1f}%)")
    
    # Top zones with most changes
    print(f"\n🔥 TOP ZONES WITH MOST CHANGES")
    top_change_zones = df.nlargest(5, 'n_changes')
    if len(top_change_zones) > 0:
        for i, row in top_change_zones.iterrows():
            if row['n_changes'] > 0:
                print(f"   Zone {row['zone']:3d} (Cluster {row['cluster']}): {row['n_changes']} changes")
                print(f"        → {row['summary']}")
    else:
        print("   No significant changes detected.")
    
    # Most common change patterns
    print(f"\n📈 MOST COMMON CHANGE PATTERNS")
    change_patterns = {}
    for zone_changes in changes.values():
        for change in zone_changes:
            band = band_keys[change['band_idx']].upper()
            pattern = f"{band} {change['sign']}"
            change_patterns[pattern] = change_patterns.get(pattern, 0) + 1
    
    if change_patterns:
        sorted_patterns = sorted(change_patterns.items(), key=lambda x: x[1], reverse=True)
        for pattern, count in sorted_patterns[:5]:
            print(f"   • {pattern}: {count} zones affected")
            # Add human interpretation
            band, direction = pattern.split()
            if band == "NDVI":
                meaning = "vegetation loss" if direction == "decrease" else "vegetation gain"
            elif band == "NDWI":
                meaning = "water/moisture loss" if direction == "decrease" else "water/moisture gain"  
            elif band == "EVI":
                meaning = "plant health decline" if direction == "decrease" else "plant health improvement"
            else:
                meaning = f"{band.lower()} {'decline' if direction == 'decrease' else 'improvement'}"
            print(f"     ({meaning})")
    else:
        print("   No significant change patterns detected.")
    
    # Strongest correlations found
    print(f"\n🔗 STRONGEST LAGGED CORRELATIONS")
    all_correlations = []
    for zone, zone_relations in relations.items():
        for rel in zone_relations:
            if abs(rel['r']) >= LAG_CORR_THRESHOLD and rel['p'] < 0.05:
                all_correlations.append((zone, rel))
    
    if all_correlations:
        # Sort by correlation strength
        all_correlations.sort(key=lambda x: abs(x[1]['r']), reverse=True)
        shown = set()
        count = 0
        for zone, rel in all_correlations:
            from_band = band_keys[rel['from_band']].upper()
            to_band = band_keys[rel['to_band']].upper()
            pattern_key = f"{from_band}->{to_band}"
            
            if pattern_key not in shown and count < 5:
                print(f"   • {from_band}(t) → {to_band}(t+1): r={rel['r']:.3f} (p={rel['p']:.3f})")
                
                # Add interpretation
                if from_band == "NDVI" and to_band == "NDWI":
                    print("     (Vegetation health predicts water/moisture changes)")
                elif from_band == "NDWI" and to_band == "NDVI":
                    print("     (Water/moisture availability predicts vegetation response)")
                elif from_band == "NDVI" and to_band == "EVI":
                    print("     (General vegetation drives enhanced vegetation index)")
                elif from_band == "EVI" and to_band == "NDVI":
                    print("     (Plant health improvements precede general vegetation growth)")
                elif from_band == "NDWI" and to_band == "EVI":
                    print("     (Water availability drives plant health improvements)")
                else:
                    print(f"     ({from_band} changes predict {to_band} changes)")
                    
                shown.add(pattern_key)
                count += 1
    else:
        print("   No strong lagged correlations detected.")
    
    # Cluster analysis
    print(f"\n🎯 ZONE CLUSTERS")
    cluster_stats = df.groupby('cluster').agg({
        'zone': 'count',
        'n_changes': 'mean',
        'n_strong_relations': 'mean'
    }).round(2)
    cluster_stats.columns = ['zones', 'avg_changes', 'avg_relations']
    
    for cluster_id, stats in cluster_stats.iterrows():
        if pd.notna(cluster_id):
            print(f"   Cluster {int(cluster_id)}: {int(stats['zones'])} zones")
            print(f"     • Average changes per zone: {stats['avg_changes']:.1f}")
            print(f"     • Average strong correlations: {stats['avg_relations']:.1f}")
            
            # Get representative zone from this cluster
            cluster_zones = df[df['cluster'] == cluster_id]
            if len(cluster_zones) > 0:
                representative = cluster_zones.iloc[0]
                if representative['summary'] != "no strong pattern":
                    print(f"     • Example: {representative['summary'][:100]}{'...' if len(representative['summary']) > 100 else ''}")
    
    # Time-based insights
    print(f"\n⏰ TEMPORAL INSIGHTS")
    year_changes = {}
    for zone_changes in changes.values():
        for change in zone_changes:
            year_pair = f"{change['year_from']}-{change['year_to']}"
            year_changes[year_pair] = year_changes.get(year_pair, 0) + 1
    
    if year_changes:
        print("   Most active periods:")
        sorted_years = sorted(year_changes.items(), key=lambda x: x[1], reverse=True)
        for year_pair, count in sorted_years[:3]:
            print(f"   • {year_pair}: {count} significant changes detected")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()