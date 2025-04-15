"""

Course: Understanding and Building LLMs
Developing a simple data pre-processing pipeline

Main script that orchestrates the data collection and cleaning pipeline for a toy example.
This script runs the scraping and cleaning processes in sequence.

Christian Gustavsson, christian.gustavsson@liu.se 

"""

import os
import sys
import pandas as pd
from includes.scrape import scrape_url, scrape_pdf
from includes.clean import clean_web_data, clean_pdf_data
from includes.tokenizer import tokenize, textloader
from includes.get_tokens import load_tokenized_data

def run_pipeline():
    """
    Run the complete data processing pipeline:
    1. Scrape data from website and PDF
    2. Clean the scraped data
    """
    print("Starting data processing pipeline...")
    
    # Get script directory and output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "corpus", "toy-example")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Step 1: Scrape data
        print("\nStep 1: Scraping data...")
        
        # Scrape website data
        print("Scraping website data...")
        url = "https://en.wikipedia.org/wiki/Computer_security"
        website_data = scrape_url(url)
        
        # Save raw website data
        raw_web_path = os.path.join(output_dir, "raw_website_data.txt")
        with open(raw_web_path, "w", encoding='utf-8') as f:
            f.write(website_data)
        print(f"Saved raw website data to: {raw_web_path}")
        
        # Scrape PDF data
        print("Scraping PDF data...")
        # Use absolute path for PDF file
        pdf_path = os.path.join(output_dir, "nationell-strategi-for-cybersakerhet-2025-2029.pdf")
        
        # Check if PDF file exists
        if not os.path.exists(pdf_path):
            print(f"Warning: PDF file not found at {pdf_path}")
            print("Please ensure the PDF file is in the same directory as the script.")
            print("Skipping PDF processing...")
            
            # Create a dummy PDF data for testing
            pdf_data = "This is a placeholder for PDF data. The actual PDF file was not found."
        else:
            pdf_data = scrape_pdf(pdf_path)
        
        # Save raw PDF data
        raw_pdf_path = os.path.join(output_dir, "raw_pdf_data.txt")
        with open(raw_pdf_path, "w", encoding='utf-8') as f:
            f.write(pdf_data)
        print(f"Saved raw PDF data to: {raw_pdf_path}")
        
        # Step 2: Clean data
        print("\nStep 2: Cleaning data...")
        
        # Clean website data
        print("Cleaning website data...")
        cleaned_web = clean_web_data(website_data)
        clean_web_path = os.path.join(output_dir, "clean_website_data.txt")
        with open(clean_web_path, "w", encoding='utf-8') as f:
            f.write(cleaned_web)
        print(f"Saved cleaned website data to: {clean_web_path}")
        
        # Clean PDF data
        print("Cleaning PDF data...")
        cleaned_pdf = clean_pdf_data(pdf_data)
        clean_pdf_path = os.path.join(output_dir, "clean_pdf_data.txt")
        with open(clean_pdf_path, "w", encoding='utf-8') as f:
            f.write(cleaned_pdf)
        print(f"Saved cleaned PDF data to: {clean_pdf_path}")

        # Tokenizing the clean web data         
        tokenized_web = tokenize(cleaned_web)

        # Tokenizing the clean pdf data
        tokenized_pdf = tokenize(cleaned_pdf)

        # Save tokenized data as Parquet
        df = pd.DataFrame({
            'source': ['web', 'pdf'],
            'token_ids': [tokenized_web, tokenized_pdf],
            'sequence_length': [len(tokenized_web), len(tokenized_pdf)],
            'vocab_size': [50257, 50257],  # GPT-2 vocabulary size
            'tokenizer': ['gpt2', 'gpt2']
        })
        
        # Save to Parquet in the output directory
        parquet_path = os.path.join(output_dir, "tokenized_data.parquet")
        df.to_parquet(parquet_path, index=False)
        print(f"\nSaved tokenized data to: {parquet_path}")

        print("\nPipeline completed successfully!")

        df = load_tokenized_data() 

    except Exception as e:
        print(f"\nError in pipeline: {str(e)}")
        sys.exit(1)



if __name__ == "__main__":
    run_pipeline()



