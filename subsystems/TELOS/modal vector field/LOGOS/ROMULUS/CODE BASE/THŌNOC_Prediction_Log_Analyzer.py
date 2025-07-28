# 🧙‍♂️ THŌNOC Prediction Analyzer
# Loads, analyzes, filters, exports metaphysical prediction logs

import json
import pandas as pd
import matplotlib.pyplot as plt
import csv
import argparse

# ========== LOAD ==========

def load_predictions(path="prediction_log.jsonl"):
    """Load all prediction logs from a JSONL file."""
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

# ========== STATS ==========

def summarize(preds):
    df = pd.DataFrame(preds)
    modal_counts = df['modal_status'].value_counts()
    avg_coh = df['coherence'].mean()
    print(f"\nLoaded {len(df)} predictions.")
    print("Modal Class Counts:\n", modal_counts)
    print(f"\nAverage Coherence: {avg_coh:.3f}")
    return df

# ========== HISTOGRAM ==========

def plot_coherence(df):
    plt.figure(figsize=(8, 4))
    plt.hist(df['coherence'], bins=20, color='skyblue', edgecolor='black')
    plt.title("Distribution of Prediction Coherence")
    plt.xlabel("Coherence")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ========== FILTERING ==========

def filter_predictions(df, modal=None, min_coherence=None):
    result = df.copy()
    if modal:
        result = result[result['modal_status'] == modal]
    if min_coherence:
        result = result[result['coherence'] >= min_coherence]
    return result

# ========== EXPORT ==========

def export_predictions(df, out_file="filtered_predictions.csv", fmt="csv"):
    if fmt == "json":
        df.to_json(out_file, orient="records", indent=2)
    elif fmt == "csv":
        df.to_csv(out_file, index=False)
    print(f"[✔] Exported {len(df)} predictions to {out_file}")

# ========== DIVERGENCE PREVIEW ==========

def preview_divergence(df):
    if df['divergence_paths'].notnull().any():
        sample = df[df['divergence_paths'].notnull()].iloc[0]
        print("\n→ Base Trinity:", sample['trinity'], "| Modal:", sample['modal_status'])
        div_df = pd.DataFrame(sample['divergence_paths'])
        print("\nTop Divergence Paths:")
        print(div_df.sort_values(by="coherence", ascending=False).head(3))
    else:
        print("[i] No divergence paths found in dataset.")

# ========== CLI ==========

def main():
    parser = argparse.ArgumentParser(description="THŌNOC Prediction Analyzer")
    parser.add_argument("--file", default="prediction_log.jsonl", help="Path to prediction log file")
    parser.add_argument("--summary", action="store_true", help="Show modal and coherence summary")
    parser.add_argument("--histogram", action="store_true", help="Plot histogram of coherence")
    parser.add_argument("--modal", choices=["necessary", "actual", "possible", "impossible"], help="Filter by modal status")
    parser.add_argument("--min-coherence", type=float, help="Filter by minimum coherence")
    parser.add_argument("--divergence", action="store_true", help="Show one divergence path set")
    parser.add_argument("--export", help="Export filtered results to a file (CSV or JSON)")
    parser.add_argument("--format", choices=["csv", "json"], default="csv", help="Export format")

    args = parser.parse_args()

    preds = load_predictions(args.file)
    df = summarize(preds) if args.summary else pd.DataFrame(preds)

    if args.histogram:
        plot_coherence(df)
