import os
import pandas as pd
from pymongo import MongoClient


OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

client = MongoClient("mongodb://127.0.0.1:27017")
db = client["heart_disease_db"]

collections = {
    "quality_report": "quality_report.csv",
    "cleaning_summary": "cleaning_summary.csv",
    "model_metrics": "model_metrics.csv",
    "collection_quality_scores": "collection_quality_scores.csv"
}

for collection_name, output_file in collections.items():
    records = list(db[collection_name].find({}, {"_id": 0}))
    df = pd.DataFrame(records)

    output_path = os.path.join(OUTPUT_DIR, output_file)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Exported {collection_name} -> {output_path}")
collections = {
    "quality_report": "quality_report.csv",
    "cleaning_summary": "cleaning_summary.csv",
    "model_metrics": "model_metrics.csv",
    "collection_quality_scores": "collection_quality_scores.csv"
}