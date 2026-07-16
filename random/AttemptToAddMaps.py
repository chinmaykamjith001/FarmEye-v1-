import numpy as np
import matplotlib.pyplot as plt
import rasterio
import pandas as pd

# --- Load grayscale satellite image ---
base_image_path = "C:/Users/Faster/Downloads/FarmEyeInputs/2025.tif"
with rasterio.open(base_image_path) as src:
    grayscale = src.read(1).astype(float)
    grayscale = (grayscale - np.min(grayscale)) / (np.max(grayscale) - np.min(grayscale))
    height, width = grayscale.shape

# --- Load Z-score CSV ---
zscore_path = "C:/Users/Faster/Downloads/FarmEyeInputs/slope_zscores.csv"
df = pd.read_csv(zscore_path)

# --- Compute composite Z-score across bands ---
df_composite = df.groupby(['row', 'col'])['zscore'].mean().reset_index()
df_composite.rename(columns={'zscore': 'composite_zscore'}, inplace=True)

# --- Create full-size heatmap initialized with NaN ---
heatmap = np.full((height, width), np.nan)

# Fill in composite z-scores at (row, col)
for _, row in df_composite.iterrows():
    r, c = int(row['row']), int(row['col'])
    if 0 <= r < height and 0 <= c < width:
        heatmap[r, c] = row['composite_zscore']

# --- Normalize heatmap for colormap display ---
heatmap_display = np.copy(heatmap)
vmin = np.nanpercentile(heatmap_display, 2)
vmax = np.nanpercentile(heatmap_display, 98)
heatmap_display = np.clip(heatmap_display, vmin, vmax)

# --- Plot ---
plt.figure(figsize=(12, 12))
plt.imshow(grayscale, cmap='gray')
plt.imshow(heatmap_display, cmap='inferno', alpha=0.6)
plt.colorbar(label='Composite Z-Score')
plt.title("Z-Score Overlay on Grayscale Satellite Image")
plt.axis('off')

# --- Save output ---
output_path = "C:/Users/Faster/Downloads/FarmEyeInputs/zscore_overlay_heatmap.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()
