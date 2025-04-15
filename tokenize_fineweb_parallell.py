import os
import tiktoken
import pyarrow.parquet as pq
import pyarrow as pa
from tqdm import tqdm
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import time
import argparse
import tempfile
import shutil

# Set paths
input_dir = os.path.join("corpus", "fineweb", "sample", "10BT")
output_dir = os.path.join("corpus", "fineweb", "tokenized")
os.makedirs(output_dir, exist_ok=True)

# Initialize tokenizer in the main process
print("Initializing GPT-2 tokenizer...")
enc = tiktoken.get_encoding("gpt2")
print("Tokenizer initialized successfully")

def process_chunk(chunk, local_enc):
    """Process a chunk of texts efficiently"""
    # Pre-allocate arrays for better memory efficiency
    tokens = []
    token_counts = np.zeros(len(chunk), dtype=np.int32)
    
    for i, text in enumerate(chunk):
        tokenized = local_enc.encode(text, disallowed_special=())
        tokens.append(tokenized)
        token_counts[i] = len(tokenized)
    
    return tokens, token_counts

def process_file(file_info):
    """Process a single parquet file in chunks with incremental writing"""
    input_file, output_file, worker_id = file_info
    
    try:
        # Initialize tokenizer for this process
        local_enc = tiktoken.get_encoding("gpt2")
        
        print(f"\nWorker {worker_id}: Processing {os.path.basename(input_file)}")
        
        # Open the parquet file as a stream
        parquet_file = pq.ParquetFile(input_file)
        total_rows = parquet_file.metadata.num_rows
        print(f"Worker {worker_id}: Found {total_rows:,} texts to process")
        
        # Find text column
        schema = parquet_file.schema
        text_columns = [col for col in schema.names if "text" in col.lower()]
        if not text_columns:
            raise ValueError("No text column found in the parquet file")
        text_column = text_columns[0]
        print(f"Worker {worker_id}: Using text column: {text_column}")
        
        # Process in smaller chunks and write incrementally
        chunk_size = 500  # Smaller chunk size for better memory management
        max_chunks_in_memory = 10  # Number of chunks to accumulate before writing
        
        # Create temporary directory for incremental files
        temp_dir = tempfile.mkdtemp()
        temp_files = []
        
        print(f"Worker {worker_id}: Tokenizing texts in chunks of {chunk_size}...")
        
        current_chunks = []
        current_counts = []
        chunk_counter = 0
        
        for batch in tqdm(parquet_file.iter_batches(batch_size=chunk_size), 
                         desc=f"Worker {worker_id}", 
                         total=total_rows//chunk_size + 1):
            # Convert batch to pandas and get text column
            df = batch.to_pandas()
            texts = df[text_column].tolist()
            
            # Process the chunk
            tokens, token_counts = process_chunk(texts, local_enc)
            current_chunks.extend(tokens)
            current_counts.extend(token_counts)
            
            # Clear memory
            del df, texts, tokens, token_counts
            
            # Write to disk if we've accumulated enough chunks
            if len(current_chunks) >= chunk_size * max_chunks_in_memory:
                temp_file = os.path.join(temp_dir, f"chunk_{chunk_counter}.parquet")
                output_table = pa.Table.from_pydict({
                    'tokens': current_chunks,
                    'token_count': current_counts
                })
                pq.write_table(output_table, temp_file)
                temp_files.append(temp_file)
                
                # Clear memory
                current_chunks = []
                current_counts = []
                chunk_counter += 1
        
        # Write any remaining chunks
        if current_chunks:
            temp_file = os.path.join(temp_dir, f"chunk_{chunk_counter}.parquet")
            output_table = pa.Table.from_pydict({
                'tokens': current_chunks,
                'token_count': current_counts
            })
            pq.write_table(output_table, temp_file)
            temp_files.append(temp_file)
        
        # Merge all temporary files
        print(f"Worker {worker_id}: Merging {len(temp_files)} chunks...")
        tables = [pq.read_table(f) for f in temp_files]
        final_table = pa.concat_tables(tables)
        pq.write_table(final_table, output_file)
        
        # Clean up temporary files
        shutil.rmtree(temp_dir)
        
        print(f"Worker {worker_id}: Saved tokenized data to: {output_file}")
        
        return True
    except Exception as e:
        print(f"Worker {worker_id}: Error processing {input_file}: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Tokenize FineWeb dataset in parallel')
    parser.add_argument('--workers', type=int, help='Number of worker processes to use (default: 75% of available CPU cores)')
    parser.add_argument('--chunk-size', type=int, default=500, help='Number of texts to process in each chunk (default: 500)')
    args = parser.parse_args()
    
    # List all parquet files
    parquet_files = [f for f in os.listdir(input_dir) if f.endswith('.parquet')]
    total_files = len(parquet_files)
    print(f"Found {total_files} files to process")
    
    # Determine number of workers
    if args.workers:
        num_workers = args.workers
    else:
        # Use 75% of available CPU cores, but at least 2 and at most 32
        num_workers = max(2, min(32, int(mp.cpu_count() * 0.75)))
    print(f"Using {num_workers} worker processes (out of {mp.cpu_count()} available CPU cores)")
    print(f"Processing in chunks of {args.chunk_size} texts")
    
    # Prepare file info for each worker
    file_info_list = []
    for i, file in enumerate(parquet_files):
        input_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, f"tokenized_{file}")
        file_info_list.append((input_path, output_path, i % num_workers))
    
    # Process files in parallel
    start_time = time.time()
    successful_files = 0
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(process_file, file_info_list), 
                          total=total_files, 
                          desc="Overall Progress"))
        successful_files = sum(results)
    
    end_time = time.time()
    print(f"\nProcessing completed:")
    print(f"Successfully processed files: {successful_files}/{total_files}")
    print(f"Total processing time: {end_time - start_time:.2f} seconds")
    print(f"Average time per file: {(end_time - start_time)/total_files:.2f} seconds")

if __name__ == "__main__":
    main() 