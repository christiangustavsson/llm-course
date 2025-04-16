import matplotlib.pyplot as plt
import os
import torch
import urllib.request
import tiktoken
import time
import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import gc
from includes.config import *

# Import from local files
from includes.model import GPTModel, generate_text_simple


class TokenizedParquetDataset(Dataset):
    def __init__(self, parquet_dir, max_length, stride, cache_size=2):
        """
        parquet_dir: Directory containing tokenized parquet files
        max_length: Maximum sequence length
        stride: Stride length for overlapping sequences
        cache_size: Number of files to keep in memory
        """
        self.max_length = max_length
        self.stride = stride
        self.cache_size = cache_size
        
        # List all parquet files
        self.parquet_files = sorted(list(Path(parquet_dir).glob("*.parquet")))
        print(f"Found {len(self.parquet_files)} tokenized parquet files")
        
        # Calculate total number of sequences
        self.total_sequences = 0
        self.file_offsets = [0]  # Start with 0
        
        print("Calculating dataset size...")
        for file_path in tqdm(self.parquet_files):
            # Read only the token_count column for efficiency
            df = pd.read_parquet(file_path, columns=['token_count'])
            sequences_in_file = sum((count - max_length) // stride + 1 
                                  for count in df['token_count'] if count >= max_length)
            self.total_sequences += sequences_in_file
            self.file_offsets.append(self.total_sequences)
            
            # Clear memory
            del df
            gc.collect()
        
        print(f"Total sequences available: {self.total_sequences:,}")
        
        # Cache for multiple files
        self.file_cache = {}
        self.cache_order = []
    
    def _load_file(self, file_idx):
        """Load a file into cache, removing oldest if cache is full"""
        if file_idx in self.file_cache:
            # Move to end of cache order
            self.cache_order.remove(file_idx)
            self.cache_order.append(file_idx)
            return self.file_cache[file_idx]
        
        # Remove oldest file if cache is full
        if len(self.file_cache) >= self.cache_size:
            oldest_idx = self.cache_order.pop(0)
            del self.file_cache[oldest_idx]
            gc.collect()
        
        # Load new file
        df = pd.read_parquet(self.parquet_files[file_idx])
        self.file_cache[file_idx] = df
        self.cache_order.append(file_idx)
        return df
    
    def __len__(self):
        return self.total_sequences
    
    def __getitem__(self, idx):
        # Find which file contains this index
        file_idx = next(i for i, offset in enumerate(self.file_offsets[1:], 1) 
                       if offset > idx) - 1
        local_idx = idx - self.file_offsets[file_idx]
        
        # Load the file (from cache if possible)
        df = self._load_file(file_idx)
        
        # Find the sequence in the current file
        current_pos = 0
        for i, tokens in enumerate(df['tokens']):
            num_sequences = (len(tokens) - self.max_length) // self.stride + 1
            if current_pos + num_sequences > local_idx:
                # Found the right sequence
                seq_idx = local_idx - current_pos
                start_idx = seq_idx * self.stride
                input_chunk = tokens[start_idx:start_idx + self.max_length]
                target_chunk = tokens[start_idx + 1:start_idx + self.max_length + 1]
                return torch.tensor(input_chunk), torch.tensor(target_chunk)
            current_pos += num_sequences
        
        raise IndexError(f"Index {idx} is out of bounds")

def create_tokenized_dataloader(parquet_dir, batch_size, max_length, stride,
                              shuffle=True, drop_last=True, num_workers=0):
    """Create a DataLoader from tokenized parquet files"""
    print("Creating dataset...")
    dataset = TokenizedParquetDataset(parquet_dir, max_length, stride)
    
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


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1

    # Set gradient clipping
    max_grad_norm = 1.0

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Progress bar for this epoch
        pbar = tqdm(train_loader, desc=f"Training")
        
        for input_batch, target_batch in pbar:
            try:
                optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
                loss = calc_loss_batch(input_batch, target_batch, model, device)
                loss.backward()  # Calculate loss gradients
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                optimizer.step()  # Update model weights using loss gradients
                tokens_seen += input_batch.numel()
                global_step += 1

                # Update progress bar
                pbar.set_postfix({"loss": f"{loss.item():.3f}"})

                # Optional evaluation step
                if global_step % eval_freq == 0:
                    train_loss, val_loss = evaluate_model(
                        model, train_loader, val_loader, device, eval_iter)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)
                    print(f"\nStep {global_step:06d}: Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("\nWARNING: Out of memory. Skipping this batch.")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e

        # Print a sample text after each epoch
        print("\nGenerating sample text...")
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots()

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    # plt.show()


def main(gpt_config, settings):
    torch.manual_seed(123)
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                        "cuda" if torch.cuda.is_available() else 
                        "cpu")
    
    print("Using device: ", device)

    ##############################
    # Initialize model
    ##############################
    print("Initializing model...")
    model = GPTModel(gpt_config)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )

    ##############################
    # Set up dataloaders
    ##############################
    print("\nSetting up data loaders...")
    tokenized_dir = os.path.join("corpus", "fineweb", "tokenized")
    
    # Determine optimal number of workers based on CPU cores
    num_workers = min(4, os.cpu_count() or 1)
    
    # Create train and validation dataloaders with parallel loading
    train_loader, val_loader = create_tokenized_dataloader(
        tokenized_dir,
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=num_workers
    )

    ##############################
    # Train model
    ##############################
    print("\nStarting training...")
    tokenizer = tiktoken.get_encoding("gpt2")

    # Start timer
    start_time = time.time()

    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=settings["num_epochs"], eval_freq=5, eval_iter=1,
        start_context="Every effort moves you", tokenizer=tokenizer
    )

    # End timer
    end_time = time.time()
    print(f"\nTraining took {end_time - start_time:.2f} seconds on {device}.")

    return train_losses, val_losses, tokens_seen, model


if __name__ == "__main__":
    os.system("clear")

    # GPT Config settings for memory-constrained environment
    cfg = GPT_CONFIG_small.copy()  # Make a copy so we don't modify the original
    cfg["context_length"] = 256  # Reduced from 1024 to save memory
    
    # Try to determine optimal batch size based on available memory
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory
        # Start with a conservative estimate
        batch_size = 4
    else:
        # For CPU/MPS, use a smaller batch size
        batch_size = 2
    
    OTHER_SETTINGS = {
        "learning_rate": 1e-4,
        "num_epochs": 3,         # Reduced for initial testing
        "batch_size": batch_size,
        "weight_decay": 0.1
    }

    print("\nModel Configuration:")
    print("-" * 40)
    for k, v in cfg.items():
        print(f"{k}: {v}")
    print("\nTraining Settings:")
    print("-" * 40)
    for k, v in OTHER_SETTINGS.items():
        print(f"{k}: {v}")
    print("-" * 40)

    ###########################
    # Initiate training
    ###########################

    try:
        train_losses, val_losses, tokens_seen, model = main(cfg, OTHER_SETTINGS)

        ###########################
        # After training
        ###########################

        # Plot results
        epochs_tensor = torch.linspace(0, OTHER_SETTINGS["num_epochs"], len(train_losses))
        plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
        plt.savefig("loss.pdf")

        # Save model
        print("\nSaving model...")
        torch.save(model.state_dict(), "model.pth")
        print("Model saved successfully!")

    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\nERROR: Out of memory error occurred.")
            print("Suggestions:")
            print("1. Reduce batch_size (currently", OTHER_SETTINGS["batch_size"], ")")
            print("2. Reduce context_length (currently", cfg["context_length"], ")")
            print("3. Use fewer layers (currently", cfg["n_layers"], "layers )")
            print("\nFull error:", str(e))
        else:
            raise e