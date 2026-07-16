import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

# === CONFIG ===
TIFF_PATH = r"C:\Users\Faster\Downloads\FarmEye(5).tif"
OUTPUT_DIR = r"C:\Users\Faster\Downloads\FarmEye_Output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load bands with nodata handling ===
def load_bands_with_nodata(tiff_path):
    with rasterio.open(tiff_path) as src:
        ndvi = src.read(1).astype(np.float32)
        evi = src.read(2).astype(np.float32)
        ndwi = src.read(3).astype(np.float32)
        meta = src.meta.copy()
        nodata = src.nodata
    # Mask nodata as np.nan
    if nodata is not None:
        ndvi[ndvi == nodata] = np.nan
        evi[evi == nodata] = np.nan
        ndwi[ndwi == nodata] = np.nan
    return ndvi, evi, ndwi, meta

# === Adaptive contrast stretching based on percentiles ===
def contrast_stretch(arr, lower_pct=2, upper_pct=98):
    valid = arr[~np.isnan(arr)]
    vmin = np.percentile(valid, lower_pct)
    vmax = np.percentile(valid, upper_pct)
    stretched = np.clip(arr, vmin, vmax)
    return stretched, vmin, vmax

# === Save GeoTIFF with metadata ===
def save_geotiff(array, meta, filename, dtype='uint8', nodata=0):
    meta_out = meta.copy()
    meta_out.update({
        'dtype': dtype,
        'count': 1,
        'nodata': nodata,
        'compress': 'deflate'
    })
    with rasterio.open(filename, 'w', **meta_out) as dst:
        dst.write(array.astype(dtype), 1)

# === Plot with colormap matching QGIS ===
def plot_colormap(arr, outpath, title, cmap_name, vmin, vmax):
    plt.figure(figsize=(8, 8))
    cmap = cm.colormaps[cmap_name] if hasattr(cm, 'colormaps') else cm.get_cmap(cmap_name)
    norm = Normalize(vmin=vmin, vmax=vmax)
    plt.imshow(arr, cmap=cmap, norm=norm)
    plt.colorbar(shrink=0.8, label=title)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

# === Main pipeline ===
ndvi, evi, ndwi, meta = load_bands_with_nodata(TIFF_PATH)

# Contrast stretch NDVI and EVI to simulate QGIS style
ndvi_cs, ndvi_min, ndvi_max = contrast_stretch(ndvi)
evi_cs, evi_min, evi_max = contrast_stretch(evi)

print(f"NDVI stretch range: {ndvi_min:.3f} to {ndvi_max:.3f}")
print(f"EVI stretch range: {evi_min:.3f} to {evi_max:.3f}")

# NDWI plotted linearly with full value range, no contrast stretch
ndwi_min, ndwi_max = np.nanmin(ndwi), np.nanmax(ndwi)

print(f"NDWI value range (no stretch): {ndwi_min:.3f} to {ndwi_max:.3f}")

# Save PNGs
plot_colormap(ndvi_cs, os.path.join(OUTPUT_DIR, 'NDVI_QGISstyle.png'), 'NDVI', 'RdYlGn', ndvi_min, ndvi_max)
plot_colormap(evi_cs, os.path.join(OUTPUT_DIR, 'EVI_QGISstyle.png'), 'EVI', 'viridis', evi_min, evi_max)
plot_colormap(ndwi, os.path.join(OUTPUT_DIR, 'NDWI_Linear.png'), 'NDWI', 'Blues', ndwi_min, ndwi_max)

# === Adaptive thresholding based on NDVI percentiles for masks ===
valid_ndvi = ndvi[~np.isnan(ndvi)]
stress_threshold = np.percentile(valid_ndvi, 20)  # bottom 20% NDVI → stressed
healthy_threshold = np.percentile(valid_ndvi, 80) # top 20% NDVI → healthy
water_threshold = np.percentile(ndwi[~np.isnan(ndwi)], 80)  # top 20% NDWI → water

print(f"Adaptive thresholds:")
print(f" Stress NDVI < {stress_threshold:.3f}")
print(f" Healthy NDVI > {healthy_threshold:.3f}")
print(f" Water NDWI > {water_threshold:.3f}")

# Initialize ndvi_class to MODERATE (Class 2)
ndvi_class = np.full(ndvi.shape, 2, dtype=np.uint8)

# WATER (Class 0) — high greenness NOT required, just low NDWI
water_mask = ((ndwi < 0.1) & (ndvi < 0.2) & (evi < 0.2))
ndvi_class[water_mask] = 0

# HEALTHY (Class 3)
healthy_mask = ((ndvi > 0.6) & (evi > 0.3) & (ndwi > 0.2))  # More positive NDWI = drier
ndvi_class[healthy_mask] = 3

# STRESSED (Class 1)
stress_mask = ((ndvi < 0.4) | 
               ((ndvi < 0.5) & (ndwi > 0.3)) |     # vegetation but too dry
               ((evi < 0.2) & (ndvi < 0.5)))
stress_mask = stress_mask & (~water_mask)
ndvi_class[stress_mask] = 1

# DEFAULT = MODERATE (Class 2)


# Save masks and classification GeoTIFFs
save_geotiff(ndvi_class* 255, meta, os.path.join(OUTPUT_DIR, 'NDVI_Classification.tif'), dtype='uint8', nodata=0)
save_geotiff(stress_mask* 255, meta, os.path.join(OUTPUT_DIR, 'Stress_Mask.tif'), dtype='uint8', nodata=0)
save_geotiff(healthy_mask* 255, meta, os.path.join(OUTPUT_DIR, 'Healthy_Mask.tif'), dtype='uint8', nodata=0)
save_geotiff(water_mask* 255, meta, os.path.join(OUTPUT_DIR, 'Water_Mask.tif'), dtype='uint8', nodata=0)


print(f"✅ FarmEye outputs saved in {OUTPUT_DIR}")
