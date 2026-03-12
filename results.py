"""
Result Analysis for Emotion Regulation Model
===========================================

Generates:
• Result table (CSV)
• Summary statistics
• Graphs (ECI distribution, variance, magnitude)

Usage:
    python analyze_results.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ------------------------------------------------------------------
# Example results (replace with your real outputs if needed)
# ------------------------------------------------------------------

results = [
    {
        "dataset": "IEMOCAP-Session1-val",
        "samples": 26,
        "eci_mean": 0.9228,
        "eci_std": 0.1811,
        "eci_median": 1.0000,
        "temporal_variance": 0.0154,
        "incongruence_magnitude": 1.0000
    },
    {
        "dataset": "CMU-MOSEI",
        "samples": 500,
        "eci_mean": 0.6774,
        "eci_std": 0.2388,
        "eci_median": 0.4980,
        "temporal_variance": 0.0156,
        "incongruence_magnitude": 0.9953
    },
    {
        "dataset": "YouTube-Simon-Sinek",
        "samples": 1,
        "eci_mean": 0.4981,
        "eci_std": 0.0,
        "eci_median": 0.4981,
        "temporal_variance": 0.0156,
        "incongruence_magnitude": 1.0000
    }
]

# ------------------------------------------------------------------
# Convert to DataFrame
# ------------------------------------------------------------------

df = pd.DataFrame(results)

print("\n==============================")
print("RESULT MATRIX")
print("==============================\n")

print(df)

# Save table
output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

csv_path = output_dir / "results_matrix.csv"
df.to_csv(csv_path, index=False)

print(f"\n✓ Results saved to {csv_path}")

# ------------------------------------------------------------------
# Plot 1 — ECI comparison
# ------------------------------------------------------------------

plt.figure(figsize=(8,5))
sns.barplot(x="dataset", y="eci_mean", data=df)

plt.title("Expressive Control Index Comparison")
plt.ylabel("Mean ECI Score")
plt.xlabel("Dataset")

plt.xticks(rotation=20)
plt.tight_layout()

plot1 = output_dir / "eci_comparison.png"
plt.savefig(plot1)
plt.close()

print(f"✓ Graph saved: {plot1}")

# ------------------------------------------------------------------
# Plot 2 — Temporal variance
# ------------------------------------------------------------------

plt.figure(figsize=(8,5))
sns.barplot(x="dataset", y="temporal_variance", data=df)

plt.title("Temporal Variance Across Datasets")
plt.ylabel("Variance")
plt.xlabel("Dataset")

plt.xticks(rotation=20)
plt.tight_layout()

plot2 = output_dir / "temporal_variance.png"
plt.savefig(plot2)
plt.close()

print(f"✓ Graph saved: {plot2}")

# ------------------------------------------------------------------
# Plot 3 — Incongruence magnitude
# ------------------------------------------------------------------

plt.figure(figsize=(8,5))
sns.barplot(x="dataset", y="incongruence_magnitude", data=df)

plt.title("Cross-Modal Incongruence Magnitude")
plt.ylabel("Magnitude")
plt.xlabel("Dataset")

plt.xticks(rotation=20)
plt.tight_layout()

plot3 = output_dir / "incongruence_magnitude.png"
plt.savefig(plot3)
plt.close()

print(f"✓ Graph saved: {plot3}")

print("\nAnalysis complete.")