import rasterio
import numpy as np
import os
import csv
from tqdm import tqdm
from sklearn.linear_model import TheilSenRegressor
import matplotlib.pyplot as plt

# CONFIG
INPUT_FOLDER = r"C:\Users\Faster\Downloads\FarmEyeInputs"
YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
BANDS = ['NDVI', 'EVI', 'NDWI']
NUM_BANDS = len(BANDS)
NUM_YEARS = len(YEARS)
CHUNK_SIZE = 20  # chunk dimension (e.g. 20x20 pixels)


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


def compute_theil_sen_per_chunk(stack, years, output_csv, chunk_size=CHUNK_SIZE, batch_size=100):
    """
    Compute Theil-Sen slope/intercept for each band and chunk.
    Chunk aggregated by mean over pixels in the chunk.
    Outputs CSV with: chunk_row, chunk_col, band, slope, intercept
    """

    bands, rows, cols, times = stack.shape
    X = np.array(years).reshape(-1, 1)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['chunk_row', 'chunk_col', 'band', 'slope', 'intercept'])
        writer.writeheader()

        for b in range(bands):
            band_name = BANDS[b]
            print(f"Computing Theil-Sen for band {band_name} ...")
            batch = []

            for r0 in tqdm(range(0, rows, chunk_size), desc=f"Band {band_name} chunks (rows)"):
                for c0 in range(0, cols, chunk_size):
                    r1 = min(r0 + chunk_size, rows)
                    c1 = min(c0 + chunk_size, cols)

                    chunk_data = stack[b, r0:r1, c0:c1, :]  # shape: (chunk_rows, chunk_cols, times)
                    # Aggregate pixel timeseries inside chunk by mean, ignoring NaNs
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

                    if len(batch) >= batch_size:
                        writer.writerows(batch)
                        batch.clear()

            if batch:
                writer.writerows(batch)
                batch.clear()


def compute_residuals_per_chunk_year(stack, years, slope_csv, output_csv, chunk_size=CHUNK_SIZE, buffer_size=100_000):
    """
    Compute residuals per chunk, band, year:
    residual = observed_mean - (slope * year + intercept)
    Output CSV columns: chunk_row, chunk_col, band, year, residual
    """

    # Load slopes/intercepts keyed by chunk coords and band
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
            band_name = BANDS[b]
            print(f"Computing residuals for band {band_name} ...")
            for r0 in tqdm(range(0, rows, chunk_size), desc=f"Band {band_name} chunks (rows)"):
                for c0 in range(0, cols, chunk_size):
                    key = (r0, c0, band_name)
                    if key not in slope_intercept_map:
                        continue

                    r1 = min(r0 + chunk_size, rows)
                    c1 = min(c0 + chunk_size, cols)
                    chunk_data = stack[b, r0:r1, c0:c1, :]  # shape: (chunk_rows, chunk_cols, times)
                    with np.errstate(invalid='ignore'):
                        chunk_mean_ts = np.nanmean(chunk_data.reshape(-1, times), axis=0)

                    if np.isnan(chunk_mean_ts).all():
                        continue

                    slope, intercept = slope_intercept_map[key]

                    for t, year in enumerate(years):
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

                        if len(buffer) >= buffer_size:
                            writer.writerows(buffer)
                            buffer.clear()

        if buffer:
            writer.writerows(buffer)


def compute_zscores_per_band_year(input_csv_path, output_csv_path):
    """
    Compute z-scores of residuals grouped by band and year.
    Input CSV columns: chunk_row, chunk_col, band, year, residual
    Output CSV columns: chunk_row, chunk_col, band, year, zscore
    """

    residuals_map = {}  # (band, year) -> list of residuals
    rows_buffer = []

    with open(input_csv_path, 'r') as f:
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

    with open(output_csv_path, 'w', newline='') as f_out:
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

    print(f"✅ Z-scores per band-year saved to {output_csv_path}")


def generate_heatmaps_per_band_year(zscore_csv_path, output_folder, chunk_size=CHUNK_SIZE):
    """
    Generate heatmaps for each band-year from chunk-based z-score CSV.
    Heatmaps saved as: heatmap_{band}_{year}.png
    Each pixel corresponds to one chunk.
    """

    zscore_data = {}
    chunk_rows = []
    chunk_cols = []

    with open(zscore_csv_path, 'r') as f:
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

    max_row = max(chunk_rows)
    max_col = max(chunk_cols)

    for (band, year), pix_dict in zscore_data.items():
        print(f"Generating heatmap for band {band} year {year} ...")
        # Create an array sized by chunks (not pixels!)
        heatmap_array = np.full((max_row // chunk_size + 1, max_col // chunk_size + 1), np.nan, dtype=np.float32)

        for (r, c), z in pix_dict.items():
            # Map chunk pixel coords to chunk indices
            chunk_r_idx = r // chunk_size
            chunk_c_idx = c // chunk_size
            heatmap_array[chunk_r_idx, chunk_c_idx] = z

        finite_vals = heatmap_array[np.isfinite(heatmap_array)]
        if finite_vals.size == 0:
            print(f"No valid data for heatmap {band} {year}. Skipping.")
            continue

        vmin, vmax = np.percentile(finite_vals, 2), np.percentile(finite_vals, 98)

        plt.figure(figsize=(12, 10))
        plt.imshow(heatmap_array, cmap='RdYlGn_r', vmin=vmin, vmax=vmax)
        plt.colorbar(label='Z-score residual')
        plt.title(f"Anomaly Heatmap - Band: {band} Year: {year}")
        plt.axis('off')
        plt.tight_layout()

        os.makedirs(output_folder, exist_ok=True)
        heatmap_path = os.path.join(output_folder, f"heatmap_{band}_{year}.png")
        plt.savefig(heatmap_path, dpi=300)
        plt.close()

        print(f"✅ Heatmap saved to {heatmap_path}")


if __name__ == "__main__":
    stack = load_multiyear_stack(INPUT_FOLDER, YEARS, NUM_BANDS)
    print(f"Loaded data stack shape: {stack.shape} (bands, rows, cols, years)")

    output_slope_csv = os.path.join(INPUT_FOLDER, 'slopes_intercepts_chunked.csv')
    residuals_csv = os.path.join(INPUT_FOLDER, 'residuals_chunked.csv')
    zscore_csv = os.path.join(INPUT_FOLDER, 'residuals_zscores_chunked.csv')
    heatmap_output_folder = os.path.join(INPUT_FOLDER, 'heatmaps_chunked')

    compute_theil_sen_per_chunk(stack, YEARS, output_slope_csv, chunk_size=CHUNK_SIZE)

    compute_residuals_per_chunk_year(stack, YEARS, output_slope_csv, residuals_csv, chunk_size=CHUNK_SIZE)

    compute_zscores_per_band_year(residuals_csv, zscore_csv)

    generate_heatmaps_per_band_year(zscore_csv, heatmap_output_folder, chunk_size=CHUNK_SIZE)
