import rasterio
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from scipy.stats import median_abs_deviation
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# === CONFIG ===
INPUT_FOLDER = r"C:\Users\Faster\Downloads\FarmEyeInputs\TIF"  # <-- Remove trailing backslash
CHUNK_SIZE = 5
OUTPUT_FOLDER = os.path.join(INPUT_FOLDER, "CSV")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

BASELINE_CSV = os.path.join(OUTPUT_FOLDER, "baseline_chunk_stats.csv")
OUTPUT_ZSCORE_CSV = os.path.join(OUTPUT_FOLDER, "2025_zscore.csv")

USE_PIXEL_LEVEL_ZSCORES = True  # toggle pixel-level z-score computation

Z_SCORE_CSV = OUTPUT_ZSCORE_CSV
OUTPUT_HEATMAP_PNG = r"C:\Users\Faster\Downloads\FarmEyeInputs\2025_impact_heatmap3.png"

SMOOTH_HEATMAP_SIGMA = 1.0  # Gaussian smoothing sigma


def band_name(band_index):
    return ['NDVI', 'EVI', 'NDWI'][band_index]


def chunk_stats(image_array, chunk_size):
    bands, rows, cols = image_array.shape
    stats_list = []

    n_chunks_row = rows // chunk_size
    n_chunks_col = cols // chunk_size

    for r in tqdm(range(n_chunks_row), desc="Chunk rows"):
        for c in range(n_chunks_col):
            r_start = r * chunk_size
            r_end = r_start + chunk_size
            c_start = c * chunk_size
            c_end = c_start + chunk_size

            for b in range(bands):
                chunk = image_array[b, r_start:r_end, c_start:c_end]
                chunk_flat = chunk.flatten()
                valid = chunk_flat[~np.isnan(chunk_flat)]

                if valid.size == 0:
                    mean = median = std = mad = np.nan
                else:
                    mean = valid.mean()
                    median = np.median(valid)
                    std = valid.std()
                    mad = median_abs_deviation(valid, scale='normal')

                stats_list.append({
                    'chunk_row': int(r),
                    'chunk_col': int(c),
                    'band': int(b),
                    'mean': float(mean),
                    'median': float(median),
                    'stddev': float(std),
                    'mad': float(mad)
                })

    return stats_list


def aggregate_baseline(csv_dir):
    from glob import glob

    csv_files = sorted(glob(os.path.join(csv_dir, "*.csv")))
    print(f"Found {len(csv_files)} CSVs for baseline aggregation")

    dfs = []
    for f in tqdm(csv_files, desc="Reading CSVs"):
        year = os.path.basename(f).split(".")[0]
        df = pd.read_csv(f)
        df["year"] = year
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    grouped = combined.groupby(["chunk_row", "chunk_col", "band_name"])

    baseline_stats = []
    for (r, c, band), group in tqdm(grouped, desc="Aggregating baseline stats", total=grouped.ngroups):
        medians = group["median"].dropna()
        mads = group["mad"].dropna()

        if len(medians) == 0 or len(mads) == 0:
            continue

        baseline_stats.append({
            "chunk_row": r,
            "chunk_col": c,
            "band_name": band,
            "median_baseline": medians.median(),
            "mad_baseline": mads.median()
        })

    baseline_df = pd.DataFrame(baseline_stats)
    return baseline_df


def compute_zscores(current_df, baseline_df=None):
    current_df['band_name'] = current_df['band_name'].str.strip().str.upper()
    if baseline_df is not None and not baseline_df.empty:
        baseline_df['band_name'] = baseline_df['band_name'].str.strip().str.upper()

        merged = pd.merge(
            current_df,
            baseline_df,
            on=["chunk_row", "chunk_col", "band_name"],
            how="inner"
        )

        if not merged.empty:
            merged["z_score"] = (merged["median"] - merged["median_baseline"]) / merged["mad_baseline"]
            merged["z_score"] = merged["z_score"].clip(-5, 5)
            return merged[[
                "chunk_row", "chunk_col", "band_name", "median", "median_baseline", "mad_baseline", "z_score"
            ]]

        else:
            print("⚠️ Merge returned no rows. Falling back to intra-image z-scores.")

    print("🔁 Computing robust z-scores from current image only.")
    band_stats = current_df.groupby("band_name")["median"].agg(['median', median_abs_deviation]).rename(
        columns={'median': 'median_band', 'median_abs_deviation': 'mad_band'}
    ).reset_index()

    merged = pd.merge(current_df, band_stats, on="band_name", how="left")
    merged["mad_band"].replace(0, np.nan, inplace=True)

    merged["z_score"] = (merged["median"] - merged["median_band"]) / merged["mad_band"]
    merged["z_score"] = merged["z_score"].clip(-5, 5)

    return merged[[
        "chunk_row", "chunk_col", "band_name", "median", "z_score"
    ]]


def compute_pixel_level_zscores(arr, chunk_size):
    bands, rows, cols = arr.shape
    n_chunks_row = rows // chunk_size
    n_chunks_col = cols // chunk_size

    pixel_zscore_rows = []

    for b in range(bands):
        band_label = band_name(b)
        for r in tqdm(range(n_chunks_row), desc=f"Band {band_label} chunk rows"):
            for c in range(n_chunks_col):
                r_start = r * chunk_size
                r_end = r_start + chunk_size
                c_start = c * chunk_size
                c_end = c_start + chunk_size

                chunk = arr[b, r_start:r_end, c_start:c_end]
                valid_pixels = chunk[np.isfinite(chunk)]

                if valid_pixels.size == 0:
                    continue

                median = np.median(valid_pixels)
                mad = median_abs_deviation(valid_pixels, scale='normal')

                if mad == 0 or np.isnan(mad):
                    continue

                z_scores = (chunk - median) / mad

                for i in range(chunk_size):
                    for j in range(chunk_size):
                        z = z_scores[i, j]
                        if np.isfinite(z):
                            z = np.clip(z, -5, 5)
                            pixel_zscore_rows.append({
                                "chunk_row": r,
                                "chunk_col": c,
                                "pixel_row": i,
                                "pixel_col": j,
                                "band": b,
                                "band_name": band_label,
                                "value": float(chunk[i, j]),
                                "z_score": float(z)
                            })

    return pd.DataFrame(pixel_zscore_rows)


def dynamic_weighted_impact_score(zscore_df, chunk_rows, chunk_cols, smoothing_sigma=SMOOTH_HEATMAP_SIGMA):
    band_groups = zscore_df.groupby('band_name')['z_score']
    mad_per_band = band_groups.apply(lambda x: median_abs_deviation(x.dropna(), scale='normal'))
    total_mad = mad_per_band.sum()

    if total_mad == 0 or np.isnan(total_mad):
        weights = {band: 1.0 / len(mad_per_band) for band in mad_per_band.index}
    else:
        weights = {band: mad_per_band[band] / total_mad for band in mad_per_band.index}

    pivot = zscore_df.pivot_table(
        index=['chunk_row', 'chunk_col'],
        columns='band_name',
        values='z_score',
        aggfunc='first'
    ).reset_index().fillna(0)

    pivot['impact_score'] = sum(
        weights.get(band, 0) * pivot.get(band, 0)
        for band in ['NDVI', 'EVI', 'NDWI']
    )

    impact_grid = np.full((chunk_rows, chunk_cols), np.nan, dtype=np.float32)
    for _, row in pivot.iterrows():
        r = int(row['chunk_row'])
        c = int(row['chunk_col'])
        impact_grid[r, c] = row['impact_score']

    impact_grid = gaussian_filter(impact_grid, sigma=smoothing_sigma, mode='nearest')

    return impact_grid, weights


def save_heatmap(impact_grid, output_path):
    finite_vals = impact_grid[np.isfinite(impact_grid)]
    if finite_vals.size == 0:
        print("❌ No finite values in impact grid for heatmap. Nothing to plot.")
        return

    max_abs = np.nanmax(np.abs(finite_vals))
    vmin, vmax = -max_abs, max_abs

    plt.figure(figsize=(8, 8))
    norm = Normalize(vmin=vmin, vmax=vmax)
    plt.imshow(impact_grid, cmap='RdYlGn_r', norm=norm)
    plt.colorbar(label='Vegetation Health Impact (robust weighted z-score)')
    plt.title('Vegetation Health Impact Map')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Heatmap saved to {output_path}")


def main():
    # Load baseline or compute if missing
    if not os.path.exists(BASELINE_CSV):
        print("Baseline not found, computing baseline...")
        baseline_df = aggregate_baseline(OUTPUT_FOLDER)
        baseline_df.to_csv(BASELINE_CSV, index=False)
        print(f"✅ Baseline saved to: {BASELINE_CSV}")
    else:
        baseline_df = pd.read_csv(BASELINE_CSV)

    # Load current raster and determine chunk grid size automatically
    current_tif = r"C:\Users\Faster\Downloads\FarmEyeInputs\Main.tif"
    with rasterio.open(current_tif) as src:
        arr = src.read().astype(np.float32)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        _, rows, cols = arr.shape

    # Auto-detect chunk counts
    chunk_rows = rows // CHUNK_SIZE
    chunk_cols = cols // CHUNK_SIZE
    print(f"Auto-detected chunk grid size: rows={chunk_rows}, cols={chunk_cols}")

    if USE_PIXEL_LEVEL_ZSCORES:
        print("🧮 Computing pixel-level robust z-scores per chunk...")
        zscore_df = compute_pixel_level_zscores(arr, CHUNK_SIZE)
    else:
        print("🧮 Computing chunk-level robust z-scores...")
        current_stats = chunk_stats(arr, CHUNK_SIZE)
        current_df = pd.DataFrame(current_stats)
        current_df['band_name'] = current_df['band'].map({0: 'NDVI', 1: 'EVI', 2: 'NDWI'})

        zscore_df = compute_zscores(current_df, baseline_df)

    # Save z-score CSV
    zscore_df.to_csv(OUTPUT_ZSCORE_CSV, index=False)
    print(f"✅ Z-score results saved to: {OUTPUT_ZSCORE_CSV}")

    # Compute weighted impact heatmap and save
    impact_grid, weights = dynamic_weighted_impact_score(zscore_df, chunk_rows, chunk_cols)
    print(f"Band weights for impact score: {weights}")
    save_heatmap(impact_grid, OUTPUT_HEATMAP_PNG)


if __name__ == "__main__":
    main()
