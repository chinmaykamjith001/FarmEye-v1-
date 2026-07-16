import rasterio
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

# CONFIG
INPUT_FOLDER = r"C:\Users\Faster\Downloads\FarmEyeInputs\TIF"
CHUNK_SIZE = 10
OUTPUT_FOLDER = os.path.join(INPUT_FOLDER, "CSV")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def chunk_stats(image_array, chunk_size):
    bands, rows, cols = image_array.shape
    stats_list = []
    n_chunks_row = rows // chunk_size
    n_chunks_col = cols // chunk_size

    for r in tqdm(range(n_chunks_row), desc="Chunking Rows"):
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
                    'chunk_row': r,
                    'chunk_col': c,
                    'band': b,
                    'mean': mean,
                    'median': median,
                    'stddev': std
                })

    return stats_list

def band_name(band_index):
    return ['NDVI', 'EVI', 'NDWI'][band_index]

def process_tif(tif_path):
    filename = os.path.splitext(os.path.basename(tif_path))[0]
    output_csv = os.path.join(OUTPUT_FOLDER, f"{filename}.csv")

    with rasterio.open(tif_path) as src:
        arr = src.read().astype(np.float32)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan

    stats = chunk_stats(arr, CHUNK_SIZE)

    for d in stats:
        d['band_name'] = band_name(d['band'])

    df = pd.DataFrame(stats)
    df = df[['chunk_row', 'chunk_col', 'band_name', 'mean', 'median', 'stddev']]
    df.to_csv(output_csv, index=False)
    print(f"✅ {filename}.csv saved.")

def main():
    for file in os.listdir(INPUT_FOLDER):
        if file.lower().endswith(".tif"):
            tif_path = os.path.join(INPUT_FOLDER, file)
            process_tif(tif_path)

if __name__ == "__main__":
    main()
