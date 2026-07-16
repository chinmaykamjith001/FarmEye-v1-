import rasterio
import numpy as np
import os
import csv
from tqdm import tqdm
from sklearn.linear_model import TheilSenRegressor
import matplotlib.pyplot as plt

# CONFIG
INPUT_FOLDER = r"C:\Users\Faster\Downloads\FarmEyeInputs"
YEARS = [2020, 2021, 2023, 2024, 2025]
BANDS = ['NDVI', 'EVI', 'NDWI']
NUM_BANDS = len(BANDS)
NUM_YEARS = len(YEARS)
CHUNK_SIZE = 20  # chunk size (e.g. 20x20 pixels)

def load_multiyear_stack(input_folder, years, num_bands):
    first_path = os.path.join(input_folder, f"{years[0]}.tif")
    with rasterio.open(first_path) as src:
        rows, cols = src.height, src.width

    arr = np.full((num_bands, rows, cols, NUM_YEARS), np.nan, dtype=np.float32)

    for t, year in enumerate(years):
        path = os.path.join(input_folder, f"{year}.tif")
        print(f"Loading {path} ...")
        with rasterio.open(path) as src:
            data = src.read().astype(np.float32)
            if src.nodata is not None:
                data[data == src.nodata] = np.nan
            if data.shape != (num_bands, rows, cols):
                raise ValueError(f"Shape mismatch in {path}")
            arr[:, :, :, t] = data

    return arr

def compute_theil_sen_per_chunk_streaming(stack, years, output_csv, chunk_size=CHUNK_SIZE, batch_size=100):
    bands, rows, cols, times = stack.shape
    X = np.array(years).reshape(-1, 1)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['chunk_row', 'chunk_col', 'band', 'slope'])
        writer.writeheader()

        for b in range(bands):
            band_name = BANDS[b]
            print(f"Computing Theil-Sen for band {band_name} ...")
            batch = []

            for r0 in tqdm(range(0, rows, chunk_size), desc=f"Band {band_name} chunks (rows)"):
                for c0 in range(0, cols, chunk_size):
                    r1 = min(r0 + chunk_size, rows)
                    c1 = min(c0 + chunk_size, cols)

                    chunk_data = stack[b, r0:r1, c0:c1, :]  # (chunk_rows, chunk_cols, times)
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

                    if len(batch) >= batch_size:
                        writer.writerows(batch)
                        batch.clear()

            if batch:
                writer.writerows(batch)
                batch.clear()

def compute_slope_zscores_from_chunk_csv(input_csv_path, output_csv_path):
    # First pass: collect slopes per band
    slope_by_band = {band: [] for band in BANDS}
    with open(input_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            slope = row['slope']
            band = row['band']
            if slope != '' and slope.lower() != 'nan':
                slope_by_band[band].append(float(slope))

    # Compute mean/std per band
    stats = {}
    for band in BANDS:
        slopes = np.array(slope_by_band[band])
        mean = slopes.mean()
        std = slopes.std()
        stats[band] = (mean, std if std > 0 else 1)

    # Second pass: write z-scores
    with open(input_csv_path, 'r') as f_in, open(output_csv_path, 'w', newline='') as f_out:
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

    print(f"✅ Z-scores saved to {output_csv_path}")

def compute_dynamic_band_weights(zscore_csv_path):
    band_values = {band: [] for band in BANDS}
    with open(zscore_csv_path, 'r') as f:
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
    print(f"Dynamic band weights based on z-score std devs: {weights}")
    return weights

def aggregate_zscores_and_generate_heatmap(zscore_csv_path, output_heatmap_path, band_weights=None, chunk_size=CHUNK_SIZE):
    pixel_scores = {}  # (chunk_row, chunk_col) -> {band: zscore}
    with open(zscore_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            r = int(row['chunk_row'])
            c = int(row['chunk_col'])
            band = row['band']
            zscore = float(row['zscore'])
            if (r,c) not in pixel_scores:
                pixel_scores[(r,c)] = {}
            pixel_scores[(r,c)][band] = zscore

    if band_weights is None:
        band_weights = {band: 1.0 for band in BANDS}
    total_weight = sum(band_weights.values())
    band_weights = {k: v / total_weight for k,v in band_weights.items()}

    all_rows = [k[0] for k in pixel_scores.keys()]
    all_cols = [k[1] for k in pixel_scores.keys()]
    max_row, max_col = max(all_rows), max(all_cols)

    # Map chunk coordinates to heatmap grid indices (divide by chunk_size)
    grid_rows = max_row // chunk_size + 1
    grid_cols = max_col // chunk_size + 1
    composite_array = np.full((grid_rows, grid_cols), np.nan, dtype=np.float32)

    for (r,c), band_dict in pixel_scores.items():
        chunk_r_idx = r // chunk_size
        chunk_c_idx = c // chunk_size
        composite_score = 0
        weight_sum = 0
        for band, z in band_dict.items():
            w = band_weights.get(band, 0)
            composite_score += z * w
            weight_sum += w
        if weight_sum > 0:
            composite_array[chunk_r_idx, chunk_c_idx] = composite_score / weight_sum
        else:
            composite_array[chunk_r_idx, chunk_c_idx] = np.nan

    finite_vals = composite_array[np.isfinite(composite_array)]
    if finite_vals.size == 0:
        print("No valid composite scores found. Heatmap will not be generated.")
        return

    vmin, vmax = np.percentile(finite_vals, 2), np.percentile(finite_vals, 98)

    plt.figure(figsize=(12, 10))
    plt.imshow(composite_array, cmap='RdYlGn_r', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Composite z-score (impact)')
    plt.title('Multi-year Impact Composite Heatmap')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_heatmap_path, dpi=300)
    plt.close()

    print(f"✅ Composite heatmap saved to {output_heatmap_path}")

if __name__ == "__main__":
    stack = load_multiyear_stack(INPUT_FOLDER, YEARS, NUM_BANDS)
    print(f"Loaded data stack shape: {stack.shape} (bands, rows, cols, years)")

    output_slope_csv = os.path.join(INPUT_FOLDER, 'slopes_chunked_streaming.csv')
    output_zscore_csv = os.path.join(INPUT_FOLDER, 'slope_zscores_chunked.csv')
    output_heatmap = os.path.join(INPUT_FOLDER, 'composite_impact_heatmap_chunked.png')

    compute_theil_sen_per_chunk_streaming(stack, YEARS, output_slope_csv, chunk_size=CHUNK_SIZE)
    compute_slope_zscores_from_chunk_csv(output_slope_csv, output_zscore_csv)

    band_weights = compute_dynamic_band_weights(output_zscore_csv)
    aggregate_zscores_and_generate_heatmap(output_zscore_csv, output_heatmap, band_weights, chunk_size=CHUNK_SIZE)
