import os
import pandas as pd
import matplotlib.pyplot as plt


RESULTS_DIR = "results"
FIGURES_DIR = "results/figures"

os.makedirs(FIGURES_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(RESULTS_DIR, "collection_quality_scores.csv"))

plot_df = df[
    df["dimension"].isin([
        "Completeness",
        "Validity",
        "Uniqueness",
        "Consistency",
        "Overall Data Quality"
    ])
].copy()

pivot_df = plot_df.pivot(
    index="dimension",
    columns="collection",
    values="score_percent"
)

pivot_df = pivot_df.loc[
    ["Completeness", "Validity", "Uniqueness", "Consistency", "Overall Data Quality"]
]

pivot_df.plot(kind="bar", figsize=(10, 6))

plt.title("Data Quality Score Comparison: Raw vs Clean Collection")
plt.xlabel("Data Quality Dimension")
plt.ylabel("Score (%)")
plt.ylim(0, 105)
plt.xticks(rotation=20)
plt.tight_layout()

output_path = os.path.join(FIGURES_DIR, "collection_quality_comparison.png")
plt.savefig(output_path, dpi=300)
plt.close()

print(f"Saved figure: {output_path}")