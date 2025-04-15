"""
Analyze perplexity of raw vs. cleaned text files.

This script calculates and compares the perplexity of raw and cleaned text files
using a pre-trained language model.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import numpy as np

def load_text(file_path):
    """Load text from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def calculate_perplexity(text, model, tokenizer, device, max_length=512):
    """
    Calculate perplexity of text using a language model.
    
    Args:
        text: The text to analyze
        model: The language model
        tokenizer: The tokenizer
        device: The device to run the model on
        max_length: Maximum sequence length
        
    Returns:
        Perplexity score
    """
    # Tokenize the text
    encodings = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
    input_ids = encodings.input_ids.to(device)
    
    # Calculate negative log likelihood
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    
    # Calculate perplexity
    perplexity = torch.exp(loss).item()
    return perplexity

def analyze_file(file_path, model, tokenizer, device):
    """
    Analyze a single file and return its perplexity.
    
    Args:
        file_path: Path to the text file
        model: The language model
        tokenizer: The tokenizer
        device: The device to run the model on
        
    Returns:
        Dictionary containing file name and perplexity
    """
    print(f"\nAnalyzing {os.path.basename(file_path)}...")
    text = load_text(file_path)
    perplexity = calculate_perplexity(text, model, tokenizer, device)
    
    return {
        'file': os.path.basename(file_path),
        'perplexity': perplexity
    }

def main():
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model_name = "gpt2"  # Using GPT-2 as it's relatively lightweight
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    print("Model loaded successfully")
    
    # Get script directory and data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "corpus", "toy-example")
    
    # List of files to analyze
    files = [
        "raw_website_data.txt",
        "clean_website_data.txt",
        "raw_pdf_data.txt",
        "clean_pdf_data.txt"
    ]
    
    # Analyze each file
    results = []
    for file in files:
        file_path = os.path.join(data_dir, file)
        if os.path.exists(file_path):
            result = analyze_file(file_path, model, tokenizer, device)
            results.append(result)
        else:
            print(f"File not found: {file_path}")
    
    # Print results
    print("\nPerplexity Analysis Results:")
    print("-" * 50)
    print(f"{'File':<30} {'Perplexity':<15}")
    print("-" * 50)
    
    for result in results:
        print(f"{result['file']:<30} {result['perplexity']:<15.2f}")
    
    # Compare raw vs clean versions
    print("\nComparison of Raw vs Clean Versions:")
    print("-" * 50)
    
    # Website data comparison
    website_raw = next(r for r in results if r['file'] == "raw_website_data.txt")
    website_clean = next(r for r in results if r['file'] == "clean_website_data.txt")
    website_improvement = (website_raw['perplexity'] - website_clean['perplexity']) / website_raw['perplexity'] * 100
    
    print(f"Website Data:")
    print(f"  Raw perplexity: {website_raw['perplexity']:.2f}")
    print(f"  Clean perplexity: {website_clean['perplexity']:.2f}")
    print(f"  Improvement: {website_improvement:.2f}%")
    
    # PDF data comparison
    pdf_raw = next(r for r in results if r['file'] == "raw_pdf_data.txt")
    pdf_clean = next(r for r in results if r['file'] == "clean_pdf_data.txt")
    pdf_improvement = (pdf_raw['perplexity'] - pdf_clean['perplexity']) / pdf_raw['perplexity'] * 100
    
    print(f"\nPDF Data:")
    print(f"  Raw perplexity: {pdf_raw['perplexity']:.2f}")
    print(f"  Clean perplexity: {pdf_clean['perplexity']:.2f}")
    print(f"  Improvement: {pdf_improvement:.2f}%")

if __name__ == "__main__":
    main() 