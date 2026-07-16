import pandas as pd
import numpy as np
import os
from glob import glob
from tqdm import tqdm  # <-- add tqdm

# CONFIG
CSV_DIR = r"C:\Users\Faster\Downloads\FarmEyeInputs\TIF\CSV"
OUTPUT_BASELINE_CSV = os.path.join(CSV_DIR, "baseline_chunk_stats.csv")

def pooled_stddev(stddevs):
    stddevs = np.array(stddevs)
    return np.sqrt(np.nanmean(stddevs ** 2))  # Pooled variance estimate

def aggregate_baseline(csv_dir):
    csv_files = sorted(glob(os.path.join(csv_dir, "*.csv")))
    print(f"Found {len(csv_files)} CSVs: {csv_files}")

    dfs = []
    for f in tqdm(csv_files, desc="Reading CSVs"):
        year = os.path.basename(f).split(".")[0]
        df = pd.read_csv(f)
        df["year"] = year
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    grouped = combined.groupby(["chunk_row", "chunk_col", "band_name"])

    baseline_stats = []
    for (r, c, band), group in tqdm(grouped, desc="Aggregating stats", total=grouped.ngroups):
        means = group["mean"].dropna()
        medians = group["median"].dropna()
        stddevs = group["stddev"].dropna()

        if len(means) == 0 or len(stddevs) == 0:
            continue

        baseline_stats.append({
            "chunk_row": r,
            "chunk_col": c,
            "band_name": band,
            "mean_baseline": means.mean(),
            "median_baseline": medians.median() if len(medians) > 0 else np.nan,
            "stddev_baseline": pooled_stddev(stddevs)
        })

    return pd.DataFrame(baseline_stats)

def main():
    baseline_df = aggregate_baseline(CSV_DIR)
    baseline_df.to_csv(OUTPUT_BASELINE_CSV, index=False)
    print(f"✅ Baseline saved to: {OUTPUT_BASELINE_CSV}")

if __name__ == "__main__":
    main()
