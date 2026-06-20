from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator


MONGO_URI = "mongodb://127.0.0.1:27017/"
DATABASE = "heart_disease_db"

READ_COLLECTION = "clean_patients"
PREDICTION_COLLECTION = "prediction_results"
METRICS_COLLECTION = "model_metrics"


spark = (
    SparkSession.builder
    .appName("Heart Disease Prediction with Spark MLlib")
    .config("spark.mongodb.read.connection.uri", MONGO_URI)
    .config("spark.mongodb.write.connection.uri", MONGO_URI)
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# Đọc dữ liệu sạch từ MongoDB
df = (
    spark.read.format("mongodb")
    .option("database", DATABASE)
    .option("collection", READ_COLLECTION)
    .load()
)

print("===== CLEAN DATA SCHEMA =====")
df.printSchema()

print("===== TOTAL CLEAN RECORDS =====")
print(df.count())

# Chọn đặc trưng đầu vào
feature_cols = [
    "age_years",
    "gender",
    "height",
    "weight",
    "ap_hi",
    "ap_lo",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active",
    "bmi"
]

# Chọn các cột cần thiết và đổi cardio thành label
model_df = df.select(
    "id",
    *feature_cols,
    F.col("cardio").cast("double").alias("label")
).dropna()

# Chia train/test
train_df, test_df = model_df.randomSplit([0.8, 0.2], seed=42)

print("===== TRAIN / TEST SPLIT =====")
print(f"Train records: {train_df.count()}")
print(f"Test records: {test_df.count()}")

# Evaluators
auc_evaluator = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

accuracy_evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)

precision_evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="weightedPrecision"
)

recall_evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="weightedRecall"
)

f1_evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="f1"
)


def evaluate_model(model_name, predictions):
    auc = round(auc_evaluator.evaluate(predictions), 4)
    accuracy = round(accuracy_evaluator.evaluate(predictions), 4)
    precision = round(precision_evaluator.evaluate(predictions), 4)
    recall = round(recall_evaluator.evaluate(predictions), 4)
    f1 = round(f1_evaluator.evaluate(predictions), 4)

    confusion = (
        predictions.groupBy("label", "prediction")
        .count()
        .orderBy("label", "prediction")
        .collect()
    )

    print(f"\n===== {model_name} METRICS =====")
    print(f"Accuracy:  {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall:    {recall}")
    print(f"F1-score:  {f1}")
    print(f"ROC-AUC:   {auc}")

    print("\nConfusion Matrix:")
    for row in confusion:
        print(row)

    return {
        "model_name": model_name,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(auc),
        "checked_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


# ======================================================
# 1. Logistic Regression
# ======================================================
assembler_lr = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features_raw"
)

scaler = StandardScaler(
    inputCol="features_raw",
    outputCol="features",
    withStd=True,
    withMean=True
)

lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=50
)

lr_pipeline = Pipeline(stages=[assembler_lr, scaler, lr])

lr_model = lr_pipeline.fit(train_df)
lr_predictions = lr_model.transform(test_df)

lr_metrics = evaluate_model("Logistic Regression", lr_predictions)


# ======================================================
# 2. Random Forest
# ======================================================
assembler_rf = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features"
)

rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="label",
    numTrees=100,
    maxDepth=8,
    seed=42
)

rf_pipeline = Pipeline(stages=[assembler_rf, rf])

rf_model = rf_pipeline.fit(train_df)
rf_predictions = rf_model.transform(test_df)

rf_metrics = evaluate_model("Random Forest", rf_predictions)


# ======================================================
# 3. Lưu metrics vào MongoDB
# ======================================================
metrics_df = spark.createDataFrame([lr_metrics, rf_metrics])

print("\n===== MODEL METRICS =====")
metrics_df.show(truncate=False)

(
    metrics_df.write.format("mongodb")
    .mode("overwrite")
    .option("database", DATABASE)
    .option("collection", METRICS_COLLECTION)
    .save()
)

print("\nSaved model metrics to MongoDB collection: model_metrics")


# ======================================================
# 4. Chọn model tốt hơn theo F1-score
# ======================================================
best_model_name = (
    "Random Forest"
    if rf_metrics["f1_score"] >= lr_metrics["f1_score"]
    else "Logistic Regression"
)

best_predictions = rf_predictions if best_model_name == "Random Forest" else lr_predictions

print(f"\nBest model based on F1-score: {best_model_name}")


# ======================================================
# 5. Lưu prediction results vào MongoDB
# ======================================================
prediction_output = (
    best_predictions
    .select(
        "id",
        "age_years",
        "gender",
        "height",
        "weight",
        "ap_hi",
        "ap_lo",
        "cholesterol",
        "gluc",
        "smoke",
        "alco",
        "active",
        "bmi",
        F.col("label").alias("actual_cardio"),
        F.col("prediction").alias("predicted_cardio"),
        F.col("probability").cast("string").alias("probability"),
    )
    .withColumn("model_name", F.lit(best_model_name))
    .withColumn("predicted_at", F.lit(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
)

(
    prediction_output.write.format("mongodb")
    .mode("overwrite")
    .option("database", DATABASE)
    .option("collection", PREDICTION_COLLECTION)
    .save()
)

print("\nSaved prediction results to MongoDB collection: prediction_results")

spark.stop()