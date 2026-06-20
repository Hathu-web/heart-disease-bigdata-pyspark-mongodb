from pyspark.sql import SparkSession

MONGO_URI = "mongodb://127.0.0.1:27017/heart_disease_db.raw_patients"

spark = (
    SparkSession.builder
    .appName("MongoDB Spark Connector Test")
    .config("spark.mongodb.read.connection.uri", MONGO_URI)
    .config("spark.mongodb.write.connection.uri", MONGO_URI)
    .getOrCreate()
)

df = spark.read.format("mongodb").load()

print("===== SCHEMA =====")
df.printSchema()

print("===== SAMPLE DATA =====")
df.show(5, truncate=False)

print("===== TOTAL RECORDS =====")
print(df.count())

spark.stop()