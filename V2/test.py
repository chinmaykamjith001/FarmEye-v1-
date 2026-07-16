import rasterio
import numpy as np
import pandas as pd
from scipy.stats import linregress
import matplotlib.pyplot as plt
from tqdm import tqdm

# === CONFIG ===
TIFF_PATHS = [
    r"C:\Users\Faster\Downloads\FarmEye2\Jan2020.tif",
    r"C:\Users\Faster\Downloads\FarmEye2\Jan2021.tif",
    r"C:\Users\Faster\Downloads\FarmEye2\Jan2022.tif",
    r"C:\Users\Faster\Downloads\FarmEye2\Jan2023.tif",
    r"C:\Users\Faster\Downloads\FarmEye2\Jan2024.tif",
]
YEARS = [2020, 2021, 2022, 2023, 2024]
CHUNK_SIZE = 10
BAND_NAMES = ['NDVI', 'EVI', 'NDWI']

def chunk_means(image_array, chunk_size):
    bands, rows, cols = image_array.shape
    n_rows = rows // chunk_size
    n_cols = cols // chunk_size
    stats = []

    for r in range(n_rows):
        for c in range(n_cols):
            r_start, r_end = r * chunk_size, (r + 1) * chunk_size
            c_start, c_end = c * chunk_size, (c + 1) * chunk_size
            for b in range(bands):
                chunk = image_array[b, r_start:r_end, c_start:c_end]
                valid = chunk[~np.isnan(chunk)]
                mean_val = valid.mean() if valid.size > 0 else np.nan
                stats.append({
                    'chunk_row': r,
                    'chunk_col': c,
                    'band': BAND_NAMES[b],
                    'mean': mean_val
                })
    return stats

# Step 1 & 2: Extract chunk means per year
all_stats = []
for year, path in zip(YEARS, TIFF_PATHS):
    print(f"Processing year {year} from {path}")
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
    year_stats = chunk_means(arr, CHUNK_SIZE)
    for stat in year_stats:
        stat['year'] = year
    all_stats.extend(year_stats)

df = pd.DataFrame(all_stats)

# Step 3 & 4: Compute slope per chunk per band
def compute_slope(group):
    if len(group) < 2 or group['mean'].isnull().all():
        return np.nan
    x = group['year']
    y = group['mean']
    valid = ~y.isnull()
    if valid.sum() < 2:
        return np.nan
    slope, _, _, _, _ = linregress(x[valid], y[valid])
    return slope

slopes = df.groupby(['chunk_row', 'chunk_col', 'band']).apply(compute_slope).reset_index(name='slope')

# Step 5: Compute variance-based weights per band
band_vars = slopes.groupby('band')['slope'].var()
total_var = band_vars.sum()
weights = (band_vars / total_var).to_dict()
print("Variance-based weights:", weights)

# Step 6: Z-score normalize slopes per band
def zscore(series):
    return (series - series.mean()) / series.std()

slopes['slope_zscore'] = slopes.groupby('band')['slope'].transform(zscore)

# Step 7: Weighted sum across bands
slopes['weighted_score'] = slopes.apply(lambda row: row['slope_zscore'] * weights.get(row['band'], 0), axis=1)
final_scores = slopes.groupby(['chunk_row', 'chunk_col'])['weighted_score'].sum().reset_index()

# Step 8: Plot per-band z-score maps
for band in BAND_NAMES:
    band_data = slopes[slopes['band'] == band]
    n_rows = band_data['chunk_row'].max() + 1
    n_cols = band_data['chunk_col'].max() + 1
    grid = np.full((n_rows, n_cols), np.nan)
    for _, row in band_data.iterrows():
        grid[int(row['chunk_row']), int(row['chunk_col'])] = row['slope_zscore']
    plt.figure(figsize=(8, 6))
    plt.title(f"Z-score slope map for {band}")
    im = plt.imshow(grid, cmap='RdYlGn_r')
    plt.colorbar(im, label='Z-score of slope')
    plt.tight_layout()
    plt.show()

# Step 9: Plot combined weighted impact map
n_rows = final_scores['chunk_row'].max() + 1
n_cols = final_scores['chunk_col'].max() + 1
impact_grid = np.full((n_rows, n_cols), np.nan)
for _, row in final_scores.iterrows():
    impact_grid[int(row['chunk_row']), int(row['chunk_col'])] = row['weighted_score']

plt.figure(figsize=(8, 6))
plt.title('FarmEye Multi-year Weighted Impact Score')
im = plt.imshow(impact_grid, cmap='RdYlGn_r')
plt.colorbar(im, label='Weighted z-score slope')
plt.tight_layout()
plt.show()
