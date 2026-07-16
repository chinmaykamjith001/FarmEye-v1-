# FarmEye 🌱

FarmEye is a **multi-year, multi-band agricultural monitoring and predictive analytics platform**. It transforms raw satellite imagery (Sentinel-2, Landsat 8/9) into actionable insights for farm management, policy planning, and research. FarmEye computes vegetation indices, detects anomalies, extracts patterns, and predicts future vegetation health using a combination of statistical methods, machine learning, and deep learning.  

---

## 🚀 Features  

- **Multi-Band Vegetation Analysis**: NDVI, EVI, NDWI computation from BOA-corrected satellite imagery.  
- **Chunked Theil–Sen Regression**: Detect trends robustly, even with missing data and outliers.  
- **Anomaly Detection**: Spatial-temporal z-score grids highlight unusual vegetation/water patterns.  
- **Pattern Mining**: Extracts stable, recurring spatiotemporal correlations using DBSCAN clustering.  
- **Predictive Modeling**: Ensemble regression (CatBoost, XGBoost, Random Forest, Gradient Boosting) forecasts next-year vegetation metrics.  
- **Visualization**: Generates heatmaps, time-series plots, and composite impact scores.  
- **Modular & Scalable**: Designed for parallelized execution on large multi-year raster datasets.  

---

---

## ⚙️ Installation  

1. Clone the repository:  
bash
git clone https://github.com/<username>/FarmEye.git
cd FarmEye

## Install dependencies
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows
pip install -r requirements.txt

📊 Visualization

Heatmaps: generated in outputs/heatmaps/ and outputs/anomalies/.
CSV summaries: outputs/patterns/ and outputs/predictions/ for integration with GIS or plotting tools.
Jupyter notebooks demonstrate interactive analysis and visualization.

🧩 Code Design

Modular: Each script is independent but interoperable.
Memory-Efficient: Chunked processing and streaming CSV writes.
Parallelizable: Designed for multi-core and cloud-based execution.
Reproducible: Metadata annotations on all outputs (source sensor, date, processing flags).

⚡ Future Work

GPU acceleration for regression and deep feature extraction.
Distributed processing with Dask or Ray for multi-terabyte datasets.
LLM-assisted natural language querying of results and actionable recommendations.
Multi-source integration (weather, soil, IoT sensors) for context-aware predictions.
Explainable AI modules (SHAP, Grad-CAM) for decision transparency.
