# Heart Disease Data Quality Assessment and Prediction using PySpark and MongoDB

## 1. Project Overview

This project is a Big Data course project focusing on **data quality assessment** and **heart disease risk prediction** using **PySpark** and **MongoDB**.

The main objective is to build an end-to-end data pipeline that imports a cardiovascular disease dataset into MongoDB, reads the data using MongoDB Spark Connector, evaluates data quality with PySpark, cleans invalid records, trains machine learning models using Spark MLlib, and writes the results back to MongoDB.

## 2. Project Topic

**Đánh giá chất lượng dữ liệu và dự đoán nguy cơ bệnh tim bằng PySpark và MongoDB**

English title:

**Data Quality Assessment and Heart Disease Risk Prediction using PySpark and MongoDB**

## 3. Main Requirements

The project follows the Big Data processing workflow:

```text
CSV Dataset
    ↓
MongoDB
    ↓
MongoDB Spark Connector
    ↓
PySpark
    ↓
Data Quality Assessment
    ↓
Data Cleaning
    ↓
Machine Learning with Spark MLlib
    ↓
MongoDB
```

Data quality is evaluated based on four dimensions:

```text
Completeness
Validity
Uniqueness
Consistency
```

## 4. Technologies Used

| Technology              | Purpose                                  |
| ----------------------- | ---------------------------------------- |
| Ubuntu WSL              | Development environment                  |
| Python 3.12             | Programming language                     |
| PySpark 4.1.2           | Big Data processing and machine learning |
| MongoDB 8.0             | NoSQL database                           |
| MongoDB Spark Connector | Connection between MongoDB and Spark     |
| pandas                  | CSV import and result export             |
| pymongo                 | MongoDB operations                       |
| matplotlib              | Visualization                            |
| Spark MLlib             | Machine learning models                  |

## 5. Dataset

Dataset used:

```text
Cardiovascular Disease Dataset
File: cardio_train.csv
Records: 70,000
```

Main attributes:

| Column      | Meaning                            |
| ----------- | ---------------------------------- |
| age         | Age in days                        |
| gender      | Gender                             |
| height      | Height in cm                       |
| weight      | Weight in kg                       |
| ap_hi       | Systolic blood pressure            |
| ap_lo       | Diastolic blood pressure           |
| cholesterol | Cholesterol level                  |
| gluc        | Glucose level                      |
| smoke       | Smoking status                     |
| alco        | Alcohol intake                     |
| active      | Physical activity                  |
| cardio      | Target label: heart disease or not |

The dataset file is not included in this repository. Place the dataset at:

```text
data/raw/cardio_train.csv
```

## 6. Project Structure

```text
bigdata_project/
│
├── data/
│   ├── raw/
│   │   └── cardio_train.csv
│   └── processed/
│
├── scripts/
│   ├── import_mongodb.py
│   ├── mongodb_connector.py
│   ├── data_quality.py
│   ├── cleaning.py
│   ├── prediction.py
│   ├── collection_quality_score.py
│   ├── export_results.py
│   ├── generate_figures.py
│   └── generate_collection_quality_figure.py
│
├── results/
│   ├── quality_report.csv
│   ├── cleaning_summary.csv
│   ├── model_metrics.csv
│   ├── collection_quality_scores.csv
│   └── figures/
│
├── reports/
│
├── requirements.txt
├── README.md
└── .gitignore
```

## 7. MongoDB Database Design

Database name:

```text
heart_disease_db
```

Collections:

| Collection                | Description                                     |
| ------------------------- | ----------------------------------------------- |
| raw_patients              | Original dataset imported from CSV              |
| quality_report            | Data quality score of raw data                  |
| clean_patients            | Cleaned patient records                         |
| cleaning_summary          | Summary of the cleaning process                 |
| model_metrics             | Evaluation metrics of ML models                 |
| prediction_results        | Prediction output from the best model           |
| collection_quality_scores | Data quality scores for each MongoDB collection |

## 8. How to Run the Project

### Step 1: Activate virtual environment

```bash
cd ~/bigdata_project
source .venv/bin/activate
```

### Step 2: Start MongoDB

```bash
sudo systemctl start mongod
systemctl status mongod
```

### Step 3: Import CSV data into MongoDB

```bash
python scripts/import_mongodb.py
```

Expected result:

```text
Imported 70000 records into MongoDB collection raw_patients
```

### Step 4: Test MongoDB Spark Connector

```bash
spark-submit \
--packages org.mongodb.spark:mongo-spark-connector_2.13:10.5.0 \
scripts/mongodb_connector.py
```

Expected result:

```text
TOTAL RECORDS
70000
```

### Step 5: Run Data Quality Assessment

```bash
spark-submit \
--packages org.mongodb.spark:mongo-spark-connector_2.13:10.5.0 \
scripts/data_quality.py
```

### Step 6: Clean the data

```bash
spark-submit \
--packages org.mongodb.spark:mongo-spark-connector_2.13:10.5.0 \
scripts/cleaning.py
```

### Step 7: Train prediction models

```bash
spark-submit \
--packages org.mongodb.spark:mongo-spark-connector_2.13:10.5.0 \
scripts/prediction.py
```

### Step 8: Score data quality for each MongoDB collection

```bash
spark-submit \
--packages org.mongodb.spark:mongo-spark-connector_2.13:10.5.0 \
scripts/collection_quality_score.py
```

### Step 9: Export results to CSV

```bash
python scripts/export_results.py
```

### Step 10: Generate figures

```bash
python scripts/generate_figures.py
python scripts/generate_collection_quality_figure.py
```

## 9. Experimental Results

### 9.1 Data Cleaning Result

| Metric          |  Value |
| --------------- | -----: |
| Raw records     | 70,000 |
| Clean records   | 68,628 |
| Removed records |  1,372 |
| Retention rate  | 98.04% |

### 9.2 Collection-Level Data Quality Scores

| Collection     | Completeness | Validity | Uniqueness | Consistency | Overall |
| -------------- | -----------: | -------: | ---------: | ----------: | ------: |
| raw_patients   |      100.00% |   98.17% |    100.00% |      98.18% |  99.09% |
| clean_patients |      100.00% |  100.00% |    100.00% |     100.00% | 100.00% |

### 9.3 Machine Learning Results

| Model               | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| ------------------- | -------: | --------: | -----: | -------: | ------: |
| Logistic Regression |   0.7279 |    0.7321 | 0.7279 |   0.7266 |  0.7938 |
| Random Forest       |   0.7346 |    0.7389 | 0.7346 |   0.7332 |  0.8006 |

The best model is:

```text
Random Forest
```

Random Forest achieved the highest Accuracy, F1-score, and ROC-AUC among the tested models.

## 10. Key Findings

The original collection `raw_patients` already had high completeness and uniqueness, but some records violated validity and consistency rules. After applying data cleaning rules, the cleaned collection `clean_patients` achieved 100% data quality score across all four dimensions.

The experiment shows that data quality assessment is an important step before machine learning because invalid or inconsistent healthcare data can affect model reliability.

## 11. Project Contributions

This project demonstrates:

* How to import CSV data into MongoDB.
* How to connect MongoDB with PySpark using MongoDB Spark Connector.
* How to evaluate data quality with distributed Spark processing.
* How to clean invalid and inconsistent healthcare records.
* How to train machine learning models using Spark MLlib.
* How to write processed results back to MongoDB.
* How to generate reports and figures for analysis.

## 12. Limitations

* The dataset is not from a real hospital system.
* The project runs in local mode instead of a real Spark cluster.
* The prediction task uses basic machine learning models.
* Real-time streaming is not implemented.

## 13. Future Work

Future improvements may include:

* Deploying Spark on a real cluster.
* Adding Kafka for real-time healthcare data streaming.
* Building a dashboard for MongoDB results.
* Testing more machine learning algorithms.
* Applying the system to larger medical datasets.

## 14. Author

Student: Nguyễn Hà Thu
Student ID: 2474802010376
University: Van Lang University
Faculty: Faculty of Information Technology
Course project: Big Data
