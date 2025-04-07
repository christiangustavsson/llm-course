"""
Course: Understanding and Building LLMs
Report 1: Developing a simple data pre-processing pipeline

This script loads tokenized data from a Parquet file and displays basic 
information about the tokenized sequences.

Christian Gustavsson, christian.gustavsson@liu.se
"""

import os
import pandas as pd

def load_tokenized_data():
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parquet_path = os.path.join(script_dir, "tokenized_data.parquet")
    
    # Load the Parquet file
    df = pd.read_parquet(parquet_path)
    
    # Get web and PDF data separately
    web_data = df[df['source'] == 'web']
    pdf_data = df[df['source'] == 'pdf']
    
    # Print basic information
    print("\nTokenized Data Summary:")
    print("-" * 40)
    print(f"Total number of sequences: {len(df)}")
    print(f"Web sequences: {len(web_data)}")
    print(f"PDF sequences: {len(pdf_data)}")
    
    # Print sequence information
    print("\nSequence Information:")
    print("-" * 40)
    print("Web data:")
    print(f"Sequence length: {web_data['sequence_length'].iloc[0]}")
    print(f"Vocabulary size: {web_data['vocab_size'].iloc[0]}")
    print(f"Tokenizer used: {web_data['tokenizer'].iloc[0]}")
    
    print("\nPDF data:")
    print(f"Sequence length: {pdf_data['sequence_length'].iloc[0]}")
    print(f"Vocabulary size: {pdf_data['vocab_size'].iloc[0]}")
    print(f"Tokenizer used: {pdf_data['tokenizer'].iloc[0]}")
    
    return df

if __name__ == "__main__":
    df = load_tokenized_data() 


    