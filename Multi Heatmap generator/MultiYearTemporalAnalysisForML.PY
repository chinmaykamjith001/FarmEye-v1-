import rasterio
import numpy as np
import os
import csv
from tqdm import tqdm
from sklearn.linear_model import TheilSenRegressor
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

# CONFIG
INPUT_FOLDER = r"C:\Users\Faster\Downloads\FarmEyeInputs"
YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
BANDS = ['NDVI', 'EVI', 'NDWI']
NUM_BANDS = len(BANDS)
NUM_YEARS = len(YEARS)


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


def fit_pixel_slope(b, r, c, stack, years):
    y = stack[b, r, c, :]
    if np.isnan(y).all() or np.count_nonzero(~np.isnan(y)) < 3:
        return None
    y_valid = y[~np.isnan(y)]
    X_valid = np.array(years).reshape(-1, 1)[~np.isnan(y)]
    try:
        model = TheilSenRegressor(random_state=42)
        model.fit(X_valid, y_valid)

        # Assign the year corresponding to the most central (median) valid observation
        year_idx = np.where(~np.isnan(y))[0]
        year_median_index = year_idx[len(year_idx) // 2]
        year_label = years[year_median_index]

        return {
            'row': r,
            'col': c,
            'band': BANDS[b],
            'slope': model.coef_[0],
            'intercept': model.intercept_,
            'year': year_label
        }
    except Exception as e:
        print(f"Error fitting pixel ({r},{c},{BANDS[b]}): {e}")
        return None



BUFFER_SIZE = 2500000  # Tune this per your available RAM

def compute_theil_sen_parallel(stack, years, output_csv):
    bands, rows, cols, _ = stack.shape
    print(f"Starting parallel Theil-Sen slope computation for {bands} bands, {rows} rows, {cols} cols")

    fieldnames = ['row', 'col', 'band', 'year', 'slope', 'intercept']
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    buffer = []

    def idx_to_brc(idx):
        b = idx // (rows * cols)
        rem = idx % (rows * cols)
        r = rem // cols
        c = rem % cols
        return b, r, c

    total_pixels = bands * rows * cols

    for start in tqdm(range(0, total_pixels, BUFFER_SIZE), desc="Slope batches"):
        end = min(start + BUFFER_SIZE, total_pixels)
        batch_indices = [idx_to_brc(idx) for idx in range(start, end)]

        results = Parallel(n_jobs=-1, prefer="threads")(
            delayed(fit_pixel_slope)(b, r, c, stack, years) for b, r, c in batch_indices
        )
        buffer.extend([r for r in results if r is not None])

        with open(output_csv, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerows(buffer)
        buffer.clear()

    print(f"✅ Saved slopes & intercepts to {output_csv}")



def compute_residuals_for_pixel(b, r, c, stack, years, slope_intercept_map):
    key = (r, c, BANDS[b])
    if key not in slope_intercept_map:
        return []
    slope, intercept = slope_intercept_map[key]
    residual_rows = []
    for t, year in enumerate(years):
        val = stack[b, r, c, t]
        if np.isnan(val):
            continue
        predicted = slope * year + intercept
        residual = val - predicted
        residual_rows.append({
            'row': r,
            'col': c,
            'band': BANDS[b],
            'year': year,
            'residual': residual
        })
    return residual_rows


def compute_residuals_parallel(stack, years, slope_csv, output_csv):
    slope_intercept_map = {}
    with open(slope_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (int(row['row']), int(row['col']), row['band'])
            slope_intercept_map[key] = (float(row['slope']), float(row['intercept']))

    bands, rows, cols, _ = stack.shape
    print(f"Starting parallel residual computation for {bands} bands, {rows} rows, {cols} cols")

    fieldnames = ['row', 'col', 'band', 'year', 'residual']
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    buffer = []

    def idx_to_brc(idx):
        b = idx // (rows * cols)
        rem = idx % (rows * cols)
        r = rem // cols
        c = rem % cols
        return b, r, c

    total_pixels = bands * rows * cols

    for start in tqdm(range(0, total_pixels, BUFFER_SIZE), desc="Residual batches"):
        end = min(start + BUFFER_SIZE, total_pixels)
        batch_indices = [idx_to_brc(idx) for idx in range(start, end)]

        results = Parallel(n_jobs=-1, prefer="threads")(
            delayed(compute_residuals_for_pixel)(b, r, c, stack, years, slope_intercept_map)
            for b, r, c in batch_indices
        )

        for sublist in results:
            buffer.extend(sublist)

        with open(output_csv, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerows(buffer)
        buffer.clear()

    print(f"✅ Saved residuals to {output_csv}")


def compute_zscores_per_band_year(input_csv_path, output_csv_path):
    residuals_map = {}
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
        writer = csv.DictWriter(f_out, fieldnames=['row', 'col', 'band', 'year', 'zscore'])
        writer.writeheader()

        for row in rows_buffer:
            band = row['band']
            year = int(row['year'])
            residual = float(row['residual'])
            mean, std = stats[(band, year)]
            z = (residual - mean) / std
            writer.writerow({
                'row': row['row'],
                'col': row['col'],
                'band': band,
                'year': year,
                'zscore': z
            })

    print(f"✅ Z-scores per band-year saved to {output_csv_path}")


def generate_heatmaps_per_band_year(zscore_csv_path, output_folder):
    zscore_data = {}
    rows_list = []
    cols_list = []

    with open(zscore_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            band = row['band']
            year = int(row['year'])
            r = int(row['row'])
            c = int(row['col'])
            z = float(row['zscore'])

            key = (band, year)
            if key not in zscore_data:
                zscore_data[key] = {}
            zscore_data[key][(r, c)] = z

            rows_list.append(r)
            cols_list.append(c)

    max_row = max(rows_list)
    max_col = max(cols_list)

    for (band, year), pix_dict in zscore_data.items():
        print(f"Generating heatmap for band {band} year {year} ...")
        heatmap_array = np.full((max_row + 1, max_col + 1), np.nan, dtype=np.float32)

        for (r, c), z in pix_dict.items():
            heatmap_array[r, c] = z

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

    output_slope_csv = os.path.join(INPUT_FOLDER, 'slopes_intercepts.csv')
    residuals_csv = os.path.join(INPUT_FOLDER, 'residuals.csv')
    zscore_csv = os.path.join(INPUT_FOLDER, 'residuals_zscores.csv')
    heatmap_output_folder = os.path.join(INPUT_FOLDER, 'heatmaps')

    # Step 1: Parallel slope/intercept computation
    compute_theil_sen_parallel(stack, YEARS, output_slope_csv)

    # Step 2: Parallel residual computation
    compute_residuals_parallel(stack, YEARS, output_slope_csv, residuals_csv)

    # Step 3: Z-score normalization (serial)
    compute_zscores_per_band_year(residuals_csv, zscore_csv)

    # Step 4: Heatmap generation (serial)
    generate_heatmaps_per_band_year(zscore_csv, heatmap_output_folder)
