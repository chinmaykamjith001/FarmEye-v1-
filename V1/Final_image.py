import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# === CONFIG ===
Z_SCORE_CSV = r"C:\Users\Faster\Downloads\FarmEyeInputs\TIF\CSV\2025_zscore.csv"
OUTPUT_HEATMAP_PNG = r"C:\Users\Faster\Downloads\FarmEyeInputs\2025_impact_heatmap2.png"

# Define chunk grid size explicitly or infer from data
# You must know chunk_rows and chunk_cols from how you created chunks
CHUNK_ROWS = 275  # example, replace with actual chunk count rows
CHUNK_COLS =  316 # example, replace with actual chunk count cols

# print("Unique bands:", zscore_df['band_name'].unique())  # Moved to main()

def dynamic_weighted_impact_score(zscore_df, chunk_rows, chunk_cols):
    # Compute std dev per band
    std_devs = zscore_df.groupby('band_name')['z_score'].std()
    total_std = std_devs.sum()

    # Normalize to get weights
    weights = {band: std_devs[band] / total_std for band in std_devs.index}

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

    return impact_grid, weights  # Return weights too for inspection


def save_heatmap(impact_grid, output_path):
    # Ensure impact_grid is a single ndarray, not a tuple or list
    if isinstance(impact_grid, (tuple, list)):
        impact_grid = impact_grid[0]

    # Only use finite values for color scaling
    finite_vals = impact_grid[np.isfinite(impact_grid)]
    if finite_vals.size == 0:
        print("❌ No finite values in impact grid for heatmap. Nothing to plot.")
        return

    vmin, vmax = np.nanmin(finite_vals), np.nanmax(finite_vals)
    plt.figure(figsize=(8, 8))
    norm = Normalize(vmin=vmin, vmax=vmax)
    plt.imshow(impact_grid, cmap='RdYlGn_r', norm=norm)
    plt.colorbar(label='Vegetation Health Impact (weighted z-score)')
    plt.title('Vegetation Health Impact Map')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Heatmap saved to {output_path}")

def main():
    zscore_df = pd.read_csv(Z_SCORE_CSV)
    print("Unique bands:", zscore_df['band_name'].unique())
    print("z_score stats:\n", zscore_df['z_score'].describe())
    print("Any NaNs in z_score?", zscore_df['z_score'].isna().sum())


    impact_grid = dynamic_weighted_impact_score(zscore_df, CHUNK_ROWS, CHUNK_COLS)

    save_heatmap(impact_grid, OUTPUT_HEATMAP_PNG)

if __name__ == "__main__":
    main()