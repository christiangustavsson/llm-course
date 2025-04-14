import os
import tiktoken
import pyarrow.parquet as pq
import pyarrow as pa
from tqdm import tqdm
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import time

# Set paths
input_dir = os.path.join("reports", "corpus", "fineweb", "sample", "10BT")
output_dir = os.path.join("reports", "corpus", "tokenized")
os.makedirs(output_dir, exist_ok=True)

# Initialize tokenizer in the main process
print("Initializing GPT-2 tokenizer...")
enc = tiktoken.get_encoding("gpt2")
print("Tokenizer initialized successfully")

def tokenize_text(text):
    """Tokenize a single text using GPT-2 tokenizer"""
    return enc.encode(text, disallowed_special=())

def process_file(file_info):
    """Process a single parquet file"""
    input_file, output_file, worker_id = file_info
    
    # Initialize tokenizer for this process
    local_enc = tiktoken.get_encoding("gpt2")
    
    print(f"\nWorker {worker_id}: Processing {os.path.basename(input_file)}")
    
    # Read the parquet file
    table = pq.read_table(input_file)
    df = table.to_pandas()
    total_texts = len(df)
    print(f"Worker {worker_id}: Found {total_texts:,} texts to process")
    
    # Find text column
    text_columns = [col for col in df.columns if "text" in col.lower()]
    if not text_columns:
        raise ValueError("No text column found in the parquet file")
    text_column = text_columns[0]
    print(f"Worker {worker_id}: Using text column: {text_column}")
    
    # Tokenize all texts
    print(f"Worker {worker_id}: Tokenizing texts...")
    tokens = []
    token_counts = []
    
    for text in tqdm(df[text_column], desc=f"Worker {worker_id}", total=total_texts):
        tokenized = local_enc.encode(text, disallowed_special=())
        tokens.append(tokenized)
        token_counts.append(len(tokenized))
    
    # Create output table
    output_table = pa.Table.from_pydict({
        'tokens': tokens,
        'token_count': token_counts
    })
    
    # Save to parquet
    pq.write_table(output_table, output_file)
    print(f"Worker {worker_id}: Saved tokenized data to: {output_file}")
    
    return True

def main():
    # List all parquet files
    parquet_files = [f for f in os.listdir(input_dir) if f.endswith('.parquet')]
    total_files = len(parquet_files)
    print(f"Found {total_files} files to process")
    
    # Determine number of workers (use 75% of available CPU cores)
    num_workers = 2 # Set by me
    print(f"Using {num_workers} worker processes")
    
    # Prepare file info for each worker
    file_info_list = []
    for i, file in enumerate(parquet_files):
        input_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, f"tokenized_{file}")
        file_info_list.append((input_path, output_path, i % num_workers))
    
    # Process files in parallel
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Use list to force execution and show progress
        list(tqdm(executor.map(process_file, file_info_list), 
                 total=total_files, 
                 desc="Overall Progress"))
    
    end_time = time.time()
    print(f"\nTotal processing time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main() 