import os
import tiktoken
import pyarrow.parquet as pq
import pyarrow as pa
from tqdm import tqdm
import numpy as np

# Set paths
input_dir = os.path.join("corpus", "fineweb", "sample", "10BT")
output_dir = os.path.join("corpus", "fineweb", "tokenized")
os.makedirs(output_dir, exist_ok=True)

# Initialize tokenizer
print("Initializing GPT-2 tokenizer...")
enc = tiktoken.get_encoding("gpt2")
print("Tokenizer initialized successfully")

def tokenize_text(text):
    """Tokenize a single text using GPT-2 tokenizer"""
    # Allow all special tokens to be encoded as normal text
    return enc.encode(text, disallowed_special=())

def process_file(input_file, output_file, pbar=None):
    """Process a single parquet file"""
    if pbar:
        pbar.set_description(f"Processing {os.path.basename(input_file)}")
    
    # Read the parquet file
    table = pq.read_table(input_file)
    df = table.to_pandas()
    total_texts = len(df)
    print(f"\nFound {total_texts:,} texts to process")
    
    # Find text column
    text_columns = [col for col in df.columns if "text" in col.lower()]
    if not text_columns:
        raise ValueError("No text column found in the parquet file")
    text_column = text_columns[0]
    print(f"Using text column: {text_column}")
    
    # Tokenize all texts
    print("Tokenizing texts...")
    tokens = []
    token_counts = []
    
    for text in tqdm(df[text_column], desc="Tokenizing", total=total_texts):
        tokenized = tokenize_text(text)
        tokens.append(tokenized)
        token_counts.append(len(tokenized))
    
    # Create output table
    output_table = pa.Table.from_pydict({
        'tokens': tokens,
        'token_count': token_counts
    })
    
    # Save to parquet
    pq.write_table(output_table, output_file)
    print(f"Saved tokenized data to: {output_file}")
    
    if pbar:
        pbar.update(1)

def main():
    # List all parquet files
    parquet_files = [f for f in os.listdir(input_dir) if f.endswith('.parquet')]
    total_files = len(parquet_files)
    print(f"Found {total_files} files to process")
    
    # Process each file with overall progress bar
    with tqdm(total=total_files, desc="Overall Progress") as pbar:
        for file in parquet_files:
            input_path = os.path.join(input_dir, file)
            output_path = os.path.join(output_dir, f"tokenized_{file}")
            process_file(input_path, output_path, pbar)

if __name__ == "__main__":
    main()