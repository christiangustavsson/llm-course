import os
import pandas as pd
from pathlib import Path
import numpy as np

# Get the absolute path to the project root directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Path to the fineweb dataset
fineweb_dir = os.path.join(project_root, "datasets", "fineweb", "sample", "10BT")

# List all parquet files
parquet_files = list(Path(fineweb_dir).glob("*.parquet"))
print(f"Found {len(parquet_files)} Parquet files in {fineweb_dir}")

# Initialize counters
total_rows = 0
total_tokens = 0
min_tokens = float('inf')
max_tokens = 0
token_counts = []  # To calculate mean later

# Process each file
for i, file_path in enumerate(parquet_files, 1):
    print(f"\nProcessing file {i}/{len(parquet_files)}: {file_path.name}")
    
    # Read only the token_count column for efficiency
    df = pd.read_parquet(file_path, columns=['token_count'])
    
    # Update statistics
    file_rows = len(df)
    file_tokens = df['token_count'].sum()
    file_min = df['token_count'].min()
    file_max = df['token_count'].max()
    file_mean = df['token_count'].mean()
    
    total_rows += file_rows
    total_tokens += file_tokens
    min_tokens = min(min_tokens, file_min)
    max_tokens = max(max_tokens, file_max)
    
    print(f"  Rows: {file_rows:,}")
    print(f"  Tokens: {file_tokens:,}")
    print(f"  Min tokens: {file_min:,}")
    print(f"  Max tokens: {file_max:,}")
    print(f"  Mean tokens: {file_mean:.2f}")

# Calculate and print final statistics
print("\nFinal Statistics:")
print(f"Total number of documents: {total_rows:,}")
print(f"Total number of tokens: {total_tokens:,}")
print(f"Minimum token count: {min_tokens:,}")
print(f"Maximum token count: {max_tokens:,}")
print(f"Average tokens per document: {total_tokens/total_rows:.2f}") 