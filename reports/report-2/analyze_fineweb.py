import os
import pandas as pd
import pyarrow.parquet as pq
import random
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# Get the absolute path to the project root directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))  # Go up two levels to reach project root

# Path to the fineweb dataset
fineweb_dir = os.path.join(project_root, "datasets", "fineweb", "sample", "10BT")

# List all parquet files
parquet_files = list(Path(fineweb_dir).glob("*.parquet"))
print(f"Found {len(parquet_files)} Parquet files in {fineweb_dir}")

# Initialize counters for overall statistics
total_rows = 0
total_tokens = 0
min_tokens = float('inf')
max_tokens = 0
language_counts = Counter()
token_lengths = []

# Process each file
for i, file_path in enumerate(parquet_files, 1):
    print(f"\nProcessing file {i}/{len(parquet_files)}: {file_path.name}")
    
    # Read the file
    df = pd.read_parquet(file_path)
    
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
    
    # Collect language statistics
    if 'language' in df.columns:
        language_counts.update(df['language'].value_counts().to_dict())
    
    # Collect token length distribution
    token_lengths.extend(df['token_count'].tolist())
    
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
if total_rows > 0:
    print(f"Average tokens per document: {total_tokens/total_rows:.2f}")

# Print language distribution
print("\nLanguage Distribution:")
for lang, count in language_counts.most_common(10):
    percentage = (count / total_rows) * 100
    print(f"  {lang}: {count:,} ({percentage:.2f}%)")

# Create output directory for plots
plots_dir = os.path.join(script_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

# Plot token length distribution
if token_lengths:
    plt.figure(figsize=(12, 6))
    plt.hist(token_lengths, bins=100, alpha=0.7, color='blue')
    plt.title('Token Length Distribution')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Frequency')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, 'token_length_distribution.png'))
    plt.close()

    # Plot token length distribution (zoomed in on most common lengths)
    plt.figure(figsize=(12, 6))
    plt.hist(token_lengths, bins=100, alpha=0.7, color='blue', range=(0, 2000))
    plt.title('Token Length Distribution (0-2000 tokens)')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, 'token_length_distribution_zoomed.png'))
    plt.close()

# Sample a few documents for qualitative analysis
if parquet_files:
    print("\nSample Documents:")
    sample_file = random.choice(parquet_files)
    df_sample = pd.read_parquet(sample_file).head(5)

    for i, row in df_sample.iterrows():
        print(f"\nDocument {i+1}:")
        print(f"  URL: {row['url']}")
        print(f"  Language: {row['language']} (score: {row['language_score']:.2f})")
        print(f"  Tokens: {row['token_count']}")
        print(f"  Text preview: {row['text'][:200]}...") 