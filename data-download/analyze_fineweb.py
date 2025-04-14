import os
import pandas as pd
import pyarrow.parquet as pq
import random
from pathlib import Path

# Get the absolute path to the project root directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Path to the fineweb dataset
fineweb_dir = os.path.join(project_root, "reports", "corpus", "fineweb")

# List all parquet files
parquet_files = list(Path(fineweb_dir).glob("*.parquet"))
print(f"Found {len(parquet_files)} Parquet files in {fineweb_dir}")

# Select a random file to analyze
sample_file = random.choice(parquet_files)
print(f"\nAnalyzing file: {sample_file.name}")

# Read the Parquet file schema
parquet_file = pq.ParquetFile(sample_file)
schema = parquet_file.schema
print("\nSchema:")
print(schema)

# Get the number of rows
num_rows = parquet_file.metadata.num_rows
print(f"\nTotal rows in file: {num_rows:,}")

# Read a sample of rows (first 5)
print("\nSample rows (first 5):")
df_sample = pd.read_parquet(sample_file).head(5)
print(df_sample)

# Display column names
print("\nColumns:")
for col in df_sample.columns:
    print(f"- {col}")

# Basic statistics for numeric columns
print("\nBasic statistics for numeric columns:")
print(df_sample.describe(include=['number']))

# Sample text content if available
text_columns = [col for col in df_sample.columns if 'text' in col.lower() or 'content' in col.lower()]
if text_columns:
    print("\nSample text content:")
    for col in text_columns:
        print(f"\nFirst entry from '{col}':")
        if not df_sample[col].empty:
            text = df_sample[col].iloc[0]
            # Truncate long text
            if len(text) > 500:
                text = text[:500] + "..."
            print(text)
        else:
            print("(empty)")

# If you want to analyze multiple files, uncomment this section
"""
print("\nAnalyzing multiple files...")
for i, file_path in enumerate(parquet_files[:3]):  # Analyze first 3 files
    print(f"\nFile {i+1}: {file_path.name}")
    df = pd.read_parquet(file_path).head(10)
    print(f"Columns: {', '.join(df.columns)}")
    print(f"Number of rows: {len(df):,}")
""" 