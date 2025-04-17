import os
import torch
from torch.utils.data import Dataset, DataLoader
import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path
import gc

class TokenizedParquetDataset(Dataset):
    def __init__(self, parquet_dir, max_length, stride, cache_size=2, chunk_size=50, max_tokens_per_row=2000):
        """
        parquet_dir: Directory containing tokenized parquet files
        max_length: Maximum sequence length
        stride: Stride length for overlapping sequences
        cache_size: Number of files to keep in memory
        chunk_size: Number of rows to process at a time from each parquet file
        max_tokens_per_row: Maximum number of tokens to process from each row
        """
        self.max_length = max_length
        self.stride = stride
        self.cache_size = cache_size
        self.chunk_size = chunk_size
        self.max_tokens_per_row = max_tokens_per_row
        
        # List all parquet files
        self.parquet_files = sorted(list(Path(parquet_dir).glob("*.parquet")))
        print(f"Found {len(self.parquet_files)} tokenized parquet files")
        
        # Estimate total sequences by sampling first file
        print("Estimating dataset size...")
        sample_file = self.parquet_files[0]
        pf = pq.ParquetFile(sample_file)
        
        # Sample first chunk to estimate sequences per chunk
        batch = next(pf.iter_batches(batch_size=self.chunk_size))
        df = batch.to_pandas()
        sequences_per_chunk = sum(
            min((len(tokens[:self.max_tokens_per_row]) - max_length) // stride + 1, 
                (self.max_tokens_per_row - max_length) // stride + 1)
            for tokens in df['tokens'] if len(tokens) >= max_length
        )
        
        # Estimate total sequences
        total_chunks = sum(1 for _ in pf.iter_batches(batch_size=self.chunk_size))
        sequences_per_file = sequences_per_chunk * total_chunks
        self.total_sequences = sequences_per_file * len(self.parquet_files)
        
        print(f"Estimated total sequences: {self.total_sequences:,}")
        
        # Initialize file tracking
        self.current_file_index = 0
        self.current_chunks = []
        self.chunks_position = 0
        self.file_cache = {}
    
    def _load_next_chunks(self):
        """Load next chunks from current file"""
        if self.current_file_index >= len(self.parquet_files):
            return False
            
        file_path = self.parquet_files[self.current_file_index]
        print(f"\nLoading chunks from {os.path.basename(file_path)}")
        
        try:
            # Read parquet file in chunks
            pf = pq.ParquetFile(file_path)
            
            # Get a batch of rows
            batch = next(pf.iter_batches(batch_size=self.chunk_size))
            df = batch.to_pandas()
            
            # Process the chunks
            self.current_chunks = []
            for tokens in df['tokens']:
                # Limit tokens and create sequences
                tokens = tokens[:self.max_tokens_per_row]
                if len(tokens) >= self.max_length:
                    for i in range(0, len(tokens) - self.max_length + 1, self.stride):
                        input_chunk = tokens[i:i + self.max_length]
                        target_chunk = tokens[i + 1:i + self.max_length + 1]
                        self.current_chunks.append((input_chunk, target_chunk))
            
            self.chunks_position = 0
            if not self.current_chunks:  # If no valid sequences were found
                self.current_file_index += 1
                return self._load_next_chunks()
            return True
            
        except Exception as e:
            print(f"Error loading file {file_path}: {str(e)}")
            self.current_file_index += 1
            return self._load_next_chunks()
    
    def __len__(self):
        return self.total_sequences
    
    def __getitem__(self, idx):
        if not self.current_chunks or self.chunks_position >= len(self.current_chunks):
            if not self._load_next_chunks():
                self.current_file_index = 0  # Reset to start
                if not self._load_next_chunks():
                    raise StopIteration
        
        sequence = self.current_chunks[self.chunks_position]
        self.chunks_position += 1
        return torch.tensor(sequence[0]), torch.tensor(sequence[1])

def create_tokenized_dataloader(parquet_dir, batch_size, max_length, stride,
                              shuffle=True, drop_last=True, num_workers=0):
    """Create a DataLoader from tokenized parquet files"""
    print("Creating dataset...")
    dataset = TokenizedParquetDataset(
        parquet_dir, 
        max_length, 
        stride,
        chunk_size=50,  # Process only 50 rows at a time
        max_tokens_per_row=2000  # Limit tokens per row
    )
    
    # Create train/val split
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    
    print(f"\nSplitting into train ({train_size:,} sequences) and validation ({val_size:,} sequences)")
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        drop_last=drop_last, 
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        drop_last=False, 
        num_workers=num_workers
    )
    
    return train_loader, val_loader 