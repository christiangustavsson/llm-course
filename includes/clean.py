"""

Course: Understanding and Building LLMs
Report 1: Developing a simple data pre-processing pipeline

Implementation of a simple data cleaning pipeline for data
collected from a website and a pdf-file.

Christian Gustavsson, christian.gustavsson@liu.se

"""

import requests
import os
import re
import unicodedata
from bs4 import BeautifulSoup
import html

def normalize_text(text):
    """
    Common normalization function for both web and PDF text.
    Preserves Swedish characters and other special characters.
    """
    # Convert to lowercase for consistency
    text = text.lower()
    
    # Use NFC normalization instead of NFKD to preserve composed characters
    text = unicodedata.normalize('NFC', text)
    
    # Remove truly non-printable characters while preserving special chars
    text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C' or char in '\n\t')
    
    # Normalize whitespace without affecting special characters
    text = re.sub(r'[ \t]+', ' ', text)
    
    return text

def clean_web_data(data):
    """
    Clean text extracted from web pages with HTML content.
    """
    # Parse HTML using BeautifulSoup
    soup = BeautifulSoup(data, 'html.parser')
    
    # Remove script and style elements
    for script in soup(["script", "style", "meta", "link", "noscript"]):
        script.decompose()
        
    # Remove comments
    for comment in soup.find_all(string=lambda text: isinstance(text, str) and '<!--' in text):
        comment.extract()
    
    # Get text content
    text = soup.get_text(separator=' ')
    
    # Unescape HTML entities
    text = html.unescape(text)
    
    # Apply common normalization
    text = normalize_text(text)
    
    # Remove common web artifacts
    text = re.sub(r'cookie policy|privacy policy|terms of service', '', text, flags=re.IGNORECASE)
    text = re.sub(r'subscribe to our newsletter', '', text, flags=re.IGNORECASE)
    text = re.sub(r'share on \w+', '', text)  # Remove social media sharing text
    
    # Handle common navigation text
    text = re.sub(r'menu|home|search|next|previous', '', text)
    
    # Remove URLs and email addresses
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '', text)
    
    return text.strip()

def clean_pdf_data(data):
    """
    Clean text extracted from PDF files with enhanced processing.
    Preserves Swedish characters and prevents incorrect word splitting.
    """
    # Apply common normalization first
    data = normalize_text(data)
    
    # Remove common PDF artifacts
    data = re.sub(r'page \d+ of \d+', '', data)  # Remove page numbers
    data = re.sub(r'page \d+', '', data)         # Remove simple page numbers
    data = re.sub(r'\f', '', data)               # Remove form feed characters
    
    # Handle reference numbers more carefully to avoid splitting words
    data = re.sub(r'(?<!\w)\[[\d\s,]+\](?!\w)', '', data)  # Remove reference numbers
    
    # Handle line breaks and hyphens more carefully
    data = re.sub(r'(?<=\w)-\n(?=\w)', '', data)  # Remove hyphenation only between words
    data = re.sub(r'(?<!\n)\n(?!\n)', ' ', data)  # Single newlines to spaces
    data = re.sub(r'\n{2,}', '\n\n', data)        # Multiple newlines to double newlines
    
    # Clean up whitespace while preserving special characters
    data = re.sub(r'[ \t]+', ' ', data)           # Normalize spaces and tabs
    data = re.sub(r'^\s+|\s+$', '', data, flags=re.MULTILINE)  # Trim lines
    
    # Remove common headers/footers
    data = re.sub(r'(?i)confidential|draft|all rights reserved', '', data)
    
    # Remove URLs and email addresses if needed
    data = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', data)
    data = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '', data)
    
    # Final cleanup - be careful with whitespace to preserve special characters
    data = re.sub(r'[ \t]+', ' ', data).strip()
    
    return data

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Loading and cleaning website data
    output_path = os.path.join(script_dir, "raw_website_data.txt")
    with open(output_path, "r") as f:
        website_data = f.read()
    cleaned_web = clean_web_data(website_data)
    
    # Save cleaned web data
    clean_web_path = os.path.join(script_dir, "clean_website_data.txt")
    with open(clean_web_path, "w") as f:
        f.write(cleaned_web)

    # Loading and cleaning PDF data
    output_path = os.path.join(script_dir, "raw_pdf_data.txt")
    with open(output_path, "r") as f:
        pdf_data = f.read()
    cleaned_pdf = clean_pdf_data(pdf_data)
    
    # Save cleaned PDF data
    clean_pdf_path = os.path.join(script_dir, "clean_pdf_data.txt")
    with open(clean_pdf_path, "w") as f:
        f.write(cleaned_pdf)

if __name__ == "__main__":
    main()