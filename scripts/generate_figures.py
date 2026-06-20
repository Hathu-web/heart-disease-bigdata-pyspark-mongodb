import os
import pandas as pd
import matplotlib.pyplot as plt


RESULTS_DIR = "results"
FIGURES_DIR = "results/figures"

os.makedirs(FIGURES_DIR, exist_ok=True)


# =========================
# 1. Data Quality Scores
# =========================
quality_path = os.path.join(RESULTS_DIR, "quality_report.csv")
quality_df = pd.read_csv(quality_path)

quality_plot_df = quality_df[
    quality_df["dimension"] != "Overall Data Quality"
].copy()

plt.figure(figsize=(8, 5))
plt.bar(quality_plot_df["dimension"], quality_plot_df["score_percent"])
plt.title("Data Quality Scores by Dimension")
plt.xlabel("Data Quality Dimension")
plt.ylabel("Score (%)")
plt.ylim(0, 105)
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "data_quality_scores.png"), dpi=300)
plt.close()


# =========================
# 2. Cleaning Summary
# =========================
cleaning_path = os.path.join(RESULTS_DIR, "cleaning_summary.csv")
cleaning_df = pd.read_csv(cleaning_path)

raw_records = int(cleaning_df.loc[0, "raw_records"])
clean_records = int(cleaning_df.loc[0, "clean_records"])
removed_records = int(cleaning_df.loc[0, "removed_records"])

plt.figure(figsize=(7, 5))
plt.bar(
    ["Raw Records", "Clean Records", "Removed Records"],
    [raw_records, clean_records, removed_records]
)
plt.title("Data Cleaning Summary")
plt.xlabel("Category")
plt.ylabel("Number of Records")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "cleaning_summary.png"), dpi=300)
plt.close()


# =========================
# 3. Model Metrics Comparison
# =========================
metrics_path = os.path.join(RESULTS_DIR, "model_metrics.csv")
metrics_df = pd.read_csv(metrics_path)

metric_cols = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]

for metric in metric_cols:
    plt.figure(figsize=(7, 5))
    plt.bar(metrics_df["model_name"], metrics_df[metric])
    plt.title(f"Model Comparison by {metric}")
    plt.xlabel("Model")
    plt.ylabel(metric)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"model_{metric}.png"), dpi=300)
    plt.close()


print("Generated figures:")
for file in os.listdir(FIGURES_DIR):
    print(os.path.join(FIGURES_DIR, file))