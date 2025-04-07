## Report 1: Developing a simple data pre-processing pipeline

To run the pipeline code:
1. `pip install -r requirements.txt`
2. Make sure `nationell-strategi-for-cybersakerhet-2025-2029.pdf` is downloaded.
3. `python main.py`

Additional code is also available (after running the pipeline):
- `analyze_patterns.py`
- `calculate_perplexity.py`

# Expected output:
Five files should be created during the pipeline. 
```
raw_pdf_data.txt
raw_website_data.txt
clean_pdf_data.txt
clean_website_data.txt
tokenized_data.parquet
```
In a real-case scenario, some clean-up would be performed along the way. 

# Terminal output:
When running the script, the terminal should output this:

```
Starting data processing pipeline...

Step 1: Scraping data...
Scraping website data...
Saved raw website data to: /Users/christian/gitrepos/llm-course/reports/report-1/raw_website_data.txt
Scraping PDF data...
Saved raw PDF data to: /Users/christian/gitrepos/llm-course/reports/report-1/raw_pdf_data.txt

Step 2: Cleaning data...
Cleaning website data...
Saved cleaned website data to: /Users/christian/gitrepos/llm-course/reports/report-1/clean_website_data.txt
Cleaning PDF data...
Saved cleaned PDF data to: /Users/christian/gitrepos/llm-course/reports/report-1/clean_pdf_data.txt

Saved tokenized data to: /Users/christian/gitrepos/llm-course/reports/report-1/tokenized_data.parquet

Pipeline completed successfully!

Tokenized Data Summary:
----------------------------------------
Total number of sequences: 2
Web sequences: 1
PDF sequences: 1

Sequence Information:
----------------------------------------
Web data:
Sequence length: 20651
Vocabulary size: 50257
Tokenizer used: gpt2

PDF data:
Sequence length: 30450
Vocabulary size: 50257
Tokenizer used: gpt2
```
