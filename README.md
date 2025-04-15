# Understanding and Building LLMs

Coursework on building LLMs from scratch, following these building blocks:
- Developing a simple data pre-processing pipeline
- Pre-train a GPT-style LLM
- Fine-tune the LLM
- Evaluate the LLM

## Preparations for running the code:

### Foundations
1. Clone the repository
2. `pip install -r requirements.txt`

### Download and tokenize FineWeb Corpus (10BT)
3. `python corpus_download.py`, note that this is approximately 31 GB of data.
- If interested, some deeper analysis could be extracted by running `python analyze_fineweb.py`.
5. `python tokenize_fineweb.py`, note that this will take some time. **OR**, for a powerful computer, `python tokenize_fineweb_parallell.py --workers X` is also available, where X is substituted for your choice. 

## 1. Simple Pre-processing pipeline
A simple pre-processing pipeline has been implemented, scraping data from a website and a PDF file. FineWeb is used to train the GPT-style model. However, this simple scraping and cleaning process is a good exercise.

1. Run `python preprocessing_pipeline.py`
- If interested, some deeper text analysis could be extracted by running `python analyze_patterns.py`.
- A perplexity comparison between raw and cleaned texts is given by running `python analyze_perplexity.py`.

### Expected file output:
Five files should be created during the pipeline. 
```
raw_pdf_data.txt
raw_website_data.txt
clean_pdf_data.txt
clean_website_data.txt
tokenized_data.parquet
```
In a real-case scenario, some clean-up would be performed along the way. 

### Expected terminal output:
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
