import rasterio
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

# === CONFIG ===
INPUT_TIFF_2025 = r"C:\Users\Faster\Downloads\FarmEyeInputs\Main.tif"
BASELINE_CSV = r"C:\Users\Faster\Downloads\FarmEyeInputs\TIF\CSV\baseline_chunk_stats.csv"
OUTPUT_ZSCORE_CSV = r"C:\Users\Faster\Downloads\FarmEyeInputs\TIF\CSV\2025_zscore.csv"

CHUNK_SIZE = 10
USE_PIXEL_LEVEL_ZSCORES = True  # toggle here


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
                    mean = median = std = np.nan
                else:
                    mean = valid.mean()
                    median = np.median(valid)
                    std = valid.std()

                stats_list.append({
                    'chunk_row': int(r),
                    'chunk_col': int(c),
                    'band': int(b),
                    'mean': float(mean),
                    'median': float(median),
                    'stddev': float(std)
                })

    return stats_list


def band_name(band_index):
    return ['NDVI', 'EVI', 'NDWI'][band_index]


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
            merged["z_score"] = (
                (merged["mean"] - merged["mean_baseline"]) /
                merged["stddev_baseline"]
            )

            return merged[[
                "chunk_row", "chunk_col", "band_name", "mean", "mean_baseline", "stddev_baseline", "z_score"
            ]]

        else:
            print("⚠️ Merge returned no rows. Falling back to intra-image z-scores.")

    # === Fallback: Intra-image z-scores ===
    print("🔁 Computing z-scores from within current image only.")

    band_stats = current_df.groupby("band_name")["mean"].agg(['mean', 'std']).rename(
        columns={'mean': 'mean_band', 'std': 'std_band'}
    ).reset_index()

    merged = pd.merge(current_df, band_stats, on="band_name", how="left")

    merged["z_score"] = (merged["mean"] - merged["mean_band"]) / merged["std_band"]

    return merged[[
        "chunk_row", "chunk_col", "band_name", "mean", "z_score"
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

                mean = valid_pixels.mean()
                std = valid_pixels.std()

                # Sanity check std
                if std == 0 or np.isnan(std):
                    # Skip chunk if std is 0 or nan, no variation for z-score
                    continue

                # Compute pixel-level z-scores
                z_scores = (chunk - mean) / std

                for i in range(chunk_size):
                    for j in range(chunk_size):
                        z = z_scores[i, j]
                        if np.isfinite(z):
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


def main():
    with rasterio.open(INPUT_TIFF_2025) as src:
        arr = src.read().astype(np.float32)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan

    if USE_PIXEL_LEVEL_ZSCORES:
        print("🧮 Computing pixel-level z-scores per chunk...")
        zscore_df = compute_pixel_level_zscores(arr, CHUNK_SIZE)
    else:
        print("🧮 Computing chunk-level stats and z-scores...")
        current_stats = chunk_stats(arr, CHUNK_SIZE)
        current_df = pd.DataFrame(current_stats)
        current_df['band_name'] = current_df['band'].map({0: 'NDVI', 1: 'EVI', 2: 'NDWI'})

        # Try loading baseline stats for chunk-level z-score
        try:
            baseline_df = pd.read_csv(BASELINE_CSV)
        except FileNotFoundError:
            print("⚠️ Baseline CSV not found. Falling back to intra-image chunk z-scores only.")
            baseline_df = None

        zscore_df = compute_zscores(current_df, baseline_df)

    # Save output CSV
    output_dir = os.path.dirname(OUTPUT_ZSCORE_CSV)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    zscore_df.to_csv(OUTPUT_ZSCORE_CSV, index=False)
    print(f"✅ Z-score results saved to: {OUTPUT_ZSCORE_CSV}")


if __name__ == "__main__":
    main()
