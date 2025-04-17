# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

import matplotlib.pyplot as plt
import os
import gc
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import torch.optim.lr_scheduler as lr_scheduler
import urllib.request
import tiktoken
import psutil
import time
import glob
from torch.utils.data import Dataset, DataLoader
import random

# Set CUDA memory allocation configuration
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'

import torch
import urllib.request
import tiktoken
import torch.optim.lr_scheduler as lr_scheduler


# Import from local files
from previous_chapters import GPTModel, create_dataloader_v1, generate_text_simple


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


def get_gpu_memory_usage():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2  # Convert to MB
    return 0

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()

def train_model_simple(model, train_loader, val_loader, optimizer, scheduler, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1
    grad_accum_steps = 4  # Accumulate gradients over 4 steps
    memory_clear_freq = 100  # Clear memory every 100 steps

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        optimizer.zero_grad()  # Reset gradients at the start of each epoch

        for i, (input_batch, target_batch) in enumerate(train_loader):
            # Clear memory periodically
            if global_step % memory_clear_freq == 0:
                clear_gpu_memory()
                print(f"GPU Memory after clearing: {get_gpu_memory_usage():.2f} MB")

            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss = loss / grad_accum_steps  # Scale loss by accumulation steps
            loss.backward()  # Calculate loss gradients
            
            if (i + 1) % grad_accum_steps == 0:
                optimizer.step()  # Update model weights using loss gradients
                scheduler.step()  # Update learning rate
                optimizer.zero_grad()  # Reset gradients
            
            tokens_seen += input_batch.numel()
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}, "
                      f"LR: {scheduler.get_last_lr()[0]:.2e}, "
                      f"GPU Mem: {get_gpu_memory_usage():.2f} MB")

        # Clear memory after each epoch
        clear_gpu_memory()

        # Print a sample text after each epoch
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


class StreamingParquetDataset(Dataset):
    def __init__(self, parquet_dir, context_length, batch_size=4):
        self.parquet_dir = parquet_dir
        self.context_length = context_length
        self.batch_size = batch_size
        
        # Get list of all parquet files
        self.parquet_files = sorted(glob.glob(f"{parquet_dir}/*.parquet"))
        print(f"Found {len(self.parquet_files)} parquet files")
        
        # Initialize current file and data
        self.current_file_idx = 0
        self.current_df = None
        self.current_row_idx = 0
        
        # Load first file
        self._load_next_file()
    
    def _load_next_file(self):
        """Load the next parquet file in sequence"""
        if self.current_file_idx >= len(self.parquet_files):
            # If we've gone through all files, start over
            self.current_file_idx = 0
        
        # Load the next file
        file_path = self.parquet_files[self.current_file_idx]
        print(f"Loading file: {file_path}")
        self.current_df = pd.read_parquet(file_path)
        self.current_row_idx = 0
        self.current_file_idx += 1
    
    def __len__(self):
        # Return a large number since we're streaming
        return 1000000000  # Arbitrary large number
    
    def __getitem__(self, idx):
        # Get a batch of sequences
        sequences = []
        for _ in range(self.batch_size):
            # Check if we need to load next file
            if self.current_df is None or self.current_row_idx >= len(self.current_df):
                self._load_next_file()
            
            # Get next sequence
            sequence = torch.tensor(self.current_df.iloc[self.current_row_idx]['tokens'], dtype=torch.long)
            self.current_row_idx += 1
            
            # Take first context_length tokens or pad if needed
            if len(sequence) > self.context_length:
                sequence = sequence[:self.context_length]
            else:
                sequence = torch.nn.functional.pad(
                    sequence, 
                    (0, self.context_length - len(sequence)),
                    mode='constant',
                    value=0
                )
            
            sequences.append(sequence)
        
        # Stack sequences into a batch
        input_batch = torch.stack(sequences)
        target_batch = torch.roll(input_batch, -1, dims=1)
        target_batch[:, -1] = 0  # Set last token to padding token
        
        return input_batch, target_batch

def main(gpt_config, settings):
    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Clear GPU memory at start
    clear_gpu_memory()
    print(f"Initial GPU Memory: {get_gpu_memory_usage():.2f} MB")

    ##############################
    # Initialize model
    ##############################

    model = GPTModel(gpt_config)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )

    ##############################
    # Set up dataloaders
    ##############################

    # Create a single dataset instance
    parquet_dir = "corpus/fineweb/tokenized"
    print(f"Setting up streaming dataset from {parquet_dir}")
    
    dataset = StreamingParquetDataset(
        parquet_dir=parquet_dir,
        context_length=gpt_config["context_length"],
        batch_size=settings["batch_size"]
    )
    
    # Create dataloaders using the same dataset
    train_loader = DataLoader(
        dataset,
        batch_size=None,  # Dataset already returns batches
        num_workers=0
    )
    
    val_loader = DataLoader(
        dataset,
        batch_size=None,  # Dataset already returns batches
        num_workers=0
    )

    # Initialize learning rate scheduler
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=settings["num_epochs"] * len(train_loader),
        eta_min=settings["learning_rate"] * 0.1
    )

    ##############################
    # Train model
    ##############################

    tokenizer = tiktoken.get_encoding("gpt2")

    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, scheduler, device,
        num_epochs=settings["num_epochs"], eval_freq=5, eval_iter=1,
        start_context="Every effort moves you", tokenizer=tokenizer
    )

    return train_losses, val_losses, tokens_seen, model


if __name__ == "__main__":

    GPT_CONFIG_124M = {
        "vocab_size": 50257,    # Vocabulary size
        "context_length": 256,  # Reduced context length
        "emb_dim": 768,         # embedding dimension
        "n_heads": 12,           # number of attention heads
        "n_layers": 12,          # number of layers
        "drop_rate": 0.1,       # Dropout rate
        "qkv_bias": False       # Query-key-value bias
    }

    OTHER_SETTINGS = {
        "learning_rate": 5e-4,
        "num_epochs": 30,
        "batch_size": 16,        
        "weight_decay": 0.1
    }

    ###########################
    # Initiate training
    ###########################

    train_losses, val_losses, tokens_seen, model = main(GPT_CONFIG_124M, OTHER_SETTINGS)

    ###########################
    # After training
    ###########################

    # Save model
    torch.save(model.state_dict(), "model.pth")

    # Plot results
    epochs_tensor = torch.linspace(0, OTHER_SETTINGS["num_epochs"], len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    plt.savefig("loss.pdf")

    # Load model
    model = GPTModel(GPT_CONFIG_124M)
    model.load_state_dict(torch.load("model.pth", weights_only=True))