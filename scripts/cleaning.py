from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


MONGO_URI = "mongodb://127.0.0.1:27017/"
DATABASE = "heart_disease_db"

RAW_COLLECTION = "raw_patients"
CLEAN_COLLECTION = "clean_patients"
SUMMARY_COLLECTION = "cleaning_summary"


def calculate_rate(part, total):
    if total == 0:
        return 0.0
    return round((part / total) * 100, 2)


spark = (
    SparkSession.builder
    .appName("Heart Disease Data Cleaning")
    .config("spark.mongodb.read.connection.uri", MONGO_URI)
    .config("spark.mongodb.write.connection.uri", MONGO_URI)
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# Đọc dữ liệu raw từ MongoDB bằng MongoDB Spark Connector
df = (
    spark.read.format("mongodb")
    .option("database", DATABASE)
    .option("collection", RAW_COLLECTION)
    .load()
)

print("===== RAW DATA =====")
df.printSchema()

raw_count = df.count()
print(f"Raw records: {raw_count}")

# Tạo thêm biến age_years và BMI
df_transformed = (
    df.withColumn("age_years", F.round(F.col("age") / 365, 2))
      .withColumn(
          "bmi",
          F.round(
              F.col("weight") / ((F.col("height") / 100) * (F.col("height") / 100)),
              2
          )
      )
)

# Quy tắc Validity
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

# Quy tắc Consistency
consistency_condition = (
    (F.col("ap_hi") >= F.col("ap_lo")) &
    (F.col("bmi").between(10, 80)) &
    (F.col("age_years") >= 18)
)

# Làm sạch dữ liệu
clean_df = (
    df_transformed
    .filter(validity_condition & consistency_condition)
    .dropDuplicates(["id"])
)

clean_count = clean_df.count()
removed_count = raw_count - clean_count
retention_rate = calculate_rate(clean_count, raw_count)

print("\n===== CLEANING RESULT =====")
print(f"Raw records: {raw_count}")
print(f"Clean records: {clean_count}")
print(f"Removed records: {removed_count}")
print(f"Retention rate: {retention_rate}%")

print("\n===== CLEAN DATA SAMPLE =====")
clean_df.show(5, truncate=False)

# Bỏ _id cũ để MongoDB tự sinh _id mới
if "_id" in clean_df.columns:
    clean_df = clean_df.drop("_id")

# Ghi dữ liệu sạch vào đúng collection clean_patients
(
    clean_df.write.format("mongodb")
    .mode("overwrite")
    .option("database", DATABASE)
    .option("collection", CLEAN_COLLECTION)
    .save()
)

print("\nSaved clean data to MongoDB collection: clean_patients")

# Tạo cleaning summary
checked_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

summary_data = [
    {
        "collection_source": RAW_COLLECTION,
        "collection_target": CLEAN_COLLECTION,
        "raw_records": int(raw_count),
        "clean_records": int(clean_count),
        "removed_records": int(removed_count),
        "retention_rate_percent": float(retention_rate),
        "rules_applied": (
            "Remove invalid age, height, weight, blood pressure values; "
            "remove invalid categorical values; "
            "remove inconsistent blood pressure records where ap_hi < ap_lo; "
            "remove records with unreasonable BMI; "
            "remove duplicate patient IDs"
        ),
        "checked_at": checked_at
    }
]

summary_df = spark.createDataFrame(summary_data)

print("\n===== CLEANING SUMMARY =====")
summary_df.show(truncate=False)

# Ghi summary vào đúng collection cleaning_summary
(
    summary_df.write.format("mongodb")
    .mode("overwrite")
    .option("database", DATABASE)
    .option("collection", SUMMARY_COLLECTION)
    .save()
)

print("\nSaved cleaning summary to MongoDB collection: cleaning_summary")

spark.stop()