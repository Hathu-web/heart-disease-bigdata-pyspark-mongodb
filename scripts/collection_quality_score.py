from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


MONGO_URI = "mongodb://127.0.0.1:27017/"
DATABASE = "heart_disease_db"

COLLECTIONS_TO_CHECK = [
    "raw_patients",
    "clean_patients"
]

OUTPUT_COLLECTION = "collection_quality_scores"


def calculate_score(passed, total):
    if total == 0:
        return 0.0
    return round((passed / total) * 100, 2)


def check_collection_quality(spark, collection_name):
    print(f"\n========== CHECKING COLLECTION: {collection_name} ==========")

    df = (
        spark.read.format("mongodb")
        .option("database", DATABASE)
        .option("collection", collection_name)
        .load()
    )

    total_records = df.count()
    print(f"Total records: {total_records}")

    business_columns = [c for c in df.columns if c != "_id"]

    # =========================================================
    # 1. COMPLETENESS
    # =========================================================
    missing_exprs = [
        F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c)
        for c in business_columns
    ]

    missing_result = df.select(missing_exprs).collect()[0].asDict()

    total_items = total_records * len(business_columns)
    failed_items = sum(missing_result.values())
    passed_items = total_items - failed_items
    completeness_score = calculate_score(passed_items, total_items)

    # =========================================================
    # 2. VALIDITY
    # =========================================================
    validity_condition = (
        (F.col("id").isNotNull()) &
        (F.col("age").between(18 * 365, 100 * 365)) &
        (F.col("height").between(100, 250)) &
        (F.col("weight").between(30, 250)) &
        (F.col("ap_hi").between(70, 250)) &
        (F.col("ap_lo").between(40, 150)) &
        (F.col("gender").isin(1, 2)) &
        (F.col("cholesterol").isin(1, 2, 3)) &
        (F.col("gluc").isin(1, 2, 3)) &
        (F.col("smoke").isin(0, 1)) &
        (F.col("alco").isin(0, 1)) &
        (F.col("active").isin(0, 1)) &
        (F.col("cardio").isin(0, 1))
    )

    valid_records = df.filter(validity_condition).count()
    invalid_records = total_records - valid_records
    validity_score = calculate_score(valid_records, total_records)

    # =========================================================
    # 3. UNIQUENESS
    # =========================================================
    distinct_ids = df.select("id").distinct().count()
    duplicate_records = total_records - distinct_ids
    uniqueness_score = calculate_score(distinct_ids, total_records)

    # =========================================================
    # 4. CONSISTENCY
    # =========================================================
    if "bmi" in df.columns:
        df_check = df
    else:
        df_check = df.withColumn(
            "bmi",
            F.col("weight") / ((F.col("height") / 100) * (F.col("height") / 100))
        )

    if "age_years" in df_check.columns:
        df_check = df_check
    else:
        df_check = df_check.withColumn("age_years", F.col("age") / 365)

    consistency_condition = (
        (F.col("ap_hi") >= F.col("ap_lo")) &
        (F.col("bmi").between(10, 80)) &
        (F.col("age_years") >= 18)
    )

    consistent_records = df_check.filter(consistency_condition).count()
    inconsistent_records = total_records - consistent_records
    consistency_score = calculate_score(consistent_records, total_records)

    # =========================================================
    # 5. OVERALL SCORE
    # =========================================================
    overall_score = round(
        (
            completeness_score +
            validity_score +
            uniqueness_score +
            consistency_score
        ) / 4,
        2
    )

    checked_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    result = [
        {
            "collection": collection_name,
            "dimension": "Completeness",
            "unit": "cell",
            "total_items": int(total_items),
            "passed_items": int(passed_items),
            "failed_items": int(failed_items),
            "score_percent": float(completeness_score),
            "rule": "Check missing/null values across all business columns",
            "checked_at": checked_at
        },
        {
            "collection": collection_name,
            "dimension": "Validity",
            "unit": "record",
            "total_items": int(total_records),
            "passed_items": int(valid_records),
            "failed_items": int(invalid_records),
            "score_percent": float(validity_score),
            "rule": "Check valid ranges and valid categorical values",
            "checked_at": checked_at
        },
        {
            "collection": collection_name,
            "dimension": "Uniqueness",
            "unit": "record",
            "total_items": int(total_records),
            "passed_items": int(distinct_ids),
            "failed_items": int(duplicate_records),
            "score_percent": float(uniqueness_score),
            "rule": "Check duplicate patient IDs",
            "checked_at": checked_at
        },
        {
            "collection": collection_name,
            "dimension": "Consistency",
            "unit": "record",
            "total_items": int(total_records),
            "passed_items": int(consistent_records),
            "failed_items": int(inconsistent_records),
            "score_percent": float(consistency_score),
            "rule": "Check ap_hi >= ap_lo, valid BMI, and adult age",
            "checked_at": checked_at
        },
        {
            "collection": collection_name,
            "dimension": "Overall Data Quality",
            "unit": "score",
            "total_items": int(total_records),
            "passed_items": 0,
            "failed_items": 0,
            "score_percent": float(overall_score),
            "rule": "Average score of Completeness, Validity, Uniqueness, and Consistency",
            "checked_at": checked_at
        }
    ]

    return result


spark = (
    SparkSession.builder
    .appName("Collection Level Data Quality Scoring")
    .config("spark.mongodb.read.connection.uri", MONGO_URI)
    .config("spark.mongodb.write.connection.uri", MONGO_URI)
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

all_results = []

for collection in COLLECTIONS_TO_CHECK:
    all_results.extend(check_collection_quality(spark, collection))

result_df = spark.createDataFrame(all_results)

print("\n========== COLLECTION QUALITY SCORES ==========")
result_df.orderBy("collection", "dimension").show(truncate=False)

(
    result_df.write.format("mongodb")
    .mode("overwrite")
    .option("database", DATABASE)
    .option("collection", OUTPUT_COLLECTION)
    .save()
)

print(f"\nSaved collection quality scores to MongoDB collection: {OUTPUT_COLLECTION}")

spark.stop()