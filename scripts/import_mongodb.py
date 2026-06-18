import pandas as pd
from pymongo import MongoClient

# Đọc dataset
df = pd.read_csv("data/raw/cardio_train.csv", sep=";")

print(f"Số dòng: {len(df)}")
print(f"Số cột: {len(df.columns)}")

# Kết nối MongoDB
client = MongoClient("mongodb://localhost:27017")

db = client["heart_disease_db"]
collection = db["raw_patients"]

# Xóa dữ liệu cũ
collection.delete_many({})

# Chuyển sang dictionary
records = df.to_dict("records")

# Import
collection.insert_many(records)

print(f"Đã import {len(records)} bản ghi")