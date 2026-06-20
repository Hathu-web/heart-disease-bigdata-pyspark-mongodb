from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


READ_URI = "mongodb://127.0.0.1:27017/heart_disease_db.raw_patients"
WRITE_URI = "mongodb://127.0.0.1:27017/heart_disease_db.quality_report"


def calculate_score(passed, total):
    if total == 0:
        return 0.0
    return round((passed / total) * 100, 2)


spark = (
    SparkSession.builder
    .appName("Heart Disease Data Quality Assessment")
    .config("spark.mongodb.read.connection.uri", READ_URI)
    .config("spark.mongodb.write.connection.uri", WRITE_URI)
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# Đọc dữ liệu từ MongoDB bằng MongoDB Spark Connector
df = spark.read.format("mongodb").load()

print("===== RAW DATA SCHEMA =====")
df.printSchema()

total_records = df.count()
print(f"Total records: {total_records}")

# Bỏ cột _id của MongoDB, chỉ kiểm tra các cột nghiệp vụ
business_columns = [col for col in df.columns if col != "_id"]

# =========================================================
# 1. COMPLETENESS - Độ đầy đủ
# =========================================================
missing_exprs = [
    F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c)
    for c in business_columns
]

missing_result = df.select(missing_exprs).collect()[0].asDict()

total_cells = total_records * len(business_columns)
missing_cells = sum(missing_result.values())
complete_cells = total_cells - missing_cells

completeness_score = calculate_score(complete_cells, total_cells)

print("\n===== COMPLETENESS =====")
for col_name, missing_count in missing_result.items():
    print(f"{col_name}: missing = {missing_count}")

print(f"Completeness Score: {completeness_score}%")

# =========================================================
# 2. VALIDITY - Độ hợp lệ
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

print("\n===== VALIDITY =====")
print(f"Valid records: {valid_records}")
print(f"Invalid records: {invalid_records}")
print(f"Validity Score: {validity_score}%")

# =========================================================
# 3. UNIQUENESS - Độ duy nhất
# =========================================================
distinct_ids = df.select("id").distinct().count()
duplicate_records = total_records - distinct_ids
uniqueness_score = calculate_score(distinct_ids, total_records)

print("\n===== UNIQUENESS =====")
print(f"Distinct IDs: {distinct_ids}")
print(f"Duplicate records: {duplicate_records}")
print(f"Uniqueness Score: {uniqueness_score}%")

# =========================================================
# 4. CONSISTENCY - Độ nhất quán
# =========================================================
# Quy tắc:
# - Huyết áp tâm thu ap_hi phải >= huyết áp tâm trương ap_lo
# - BMI phải nằm trong khoảng hợp lý
# - Tuổi phải >= 18

df_with_bmi = df.withColumn(
    "bmi",
    F.col("weight") / ((F.col("height") / 100) * (F.col("height") / 100))
).withColumn(
    "age_years",
    F.col("age") / 365
)

consistency_condition = (
    (F.col("ap_hi") >= F.col("ap_lo")) &
    (F.col("bmi").between(10, 80)) &
    (F.col("age_years") >= 18)
)

consistent_records = df_with_bmi.filter(consistency_condition).count()
inconsistent_records = total_records - consistent_records
consistency_score = calculate_score(consistent_records, total_records)

print("\n===== CONSISTENCY =====")
print(f"Consistent records: {consistent_records}")
print(f"Inconsistent records: {inconsistent_records}")
print(f"Consistency Score: {consistency_score}%")

# =========================================================
# 5. TỔNG HỢP ĐIỂM CHẤT LƯỢNG DỮ LIỆU
# =========================================================
overall_score = round(
    (completeness_score + validity_score + uniqueness_score + consistency_score) / 4,
    2
)

checked_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

quality_report = [
    {
        "collection": "raw_patients",
        "dimension": "Completeness",
        "rule": "Check missing/null values across all business columns",
        "total_records": int(total_records),
        "passed_records": int(complete_cells),
        "failed_records": int(missing_cells),
        "score_percent": float(completeness_score),
        "checked_at": checked_at
    },
    {
        "collection": "raw_patients",
        "dimension": "Validity",
        "rule": "Check valid ranges and valid categorical values",
        "total_records": int(total_records),
        "passed_records": int(valid_records),
        "failed_records": int(invalid_records),
        "score_percent": float(validity_score),
        "checked_at": checked_at
    },
    {
        "collection": "raw_patients",
        "dimension": "Uniqueness",
        "rule": "Check duplicate patient IDs",
        "total_records": int(total_records),
        "passed_records": int(distinct_ids),
        "failed_records": int(duplicate_records),
        "score_percent": float(uniqueness_score),
        "checked_at": checked_at
    },
    {
        "collection": "raw_patients",
        "dimension": "Consistency",
        "rule": "Check ap_hi >= ap_lo, valid BMI, and adult age",
        "total_records": int(total_records),
        "passed_records": int(consistent_records),
        "failed_records": int(inconsistent_records),
        "score_percent": float(consistency_score),
        "checked_at": checked_at
    },
    {
        "collection": "raw_patients",
        "dimension": "Overall Data Quality",
        "rule": "Average score of Completeness, Validity, Uniqueness, and Consistency",
        "total_records": int(total_records),
        "passed_records": 0,
        "failed_records": 0,
        "score_percent": float(overall_score),
        "checked_at": checked_at
    }
]

quality_df = spark.createDataFrame(quality_report)

print("\n===== QUALITY REPORT =====")
quality_df.show(truncate=False)

# Ghi kết quả Data Quality về MongoDB
quality_df.write.format("mongodb").mode("overwrite").save()

print("\nSaved quality report to MongoDB collection: quality_report")

spark.stop()