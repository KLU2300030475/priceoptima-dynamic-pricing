import os
import pandas as pd
from datetime import datetime

RAW_FOLDER = "data/raw"
PROCESSED_FOLDER = "data/processed"

os.makedirs(PROCESSED_FOLDER, exist_ok=True)

for file in os.listdir(RAW_FOLDER):
    if file.endswith(".csv"):
        print(f"\nProcessing file: {file}")

        file_path = os.path.join(RAW_FOLDER, file)
        df = pd.read_csv(file_path)

        print("Original rows:", len(df))
        print("Columns:", df.columns.tolist())

        # Cleaning
        df = df.drop_duplicates()
        df = df.dropna()

        print("Rows after cleaning:", len(df))

        # Save cleaned version with timestamp
        today = datetime.now().strftime("%Y_%m_%d")
        new_file_name = f"processed_{today}_{file}"

        processed_path = os.path.join(PROCESSED_FOLDER, new_file_name)
        df.to_csv(processed_path, index=False)

        print(f"Saved cleaned file: {new_file_name}")

print("\n Daily ingestion completed successfully")