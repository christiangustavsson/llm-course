# Understanding and Building LLMs

Coursework on building LLMs from scratch, following these building blocks:
- Developing a simple data pre-processing pipeline
- Pre-train a GPT-style LLM
- Fine-tune the LLM
- Evaluate the LLM

## Preparations for running the code:

### Basic steps
1. Clone the repository
2. `pip install -r requirements.txt`

### Download and tokenize FineWeb Corpus (10BT)
The FineWeb dataset, by Penedo et al. (Penedo et al, 2024), is used for the training corpus. The dataset is preprocessed to remove HTML, scripts, and other artifacts, making it suitable for direct use in language modeling tasks. 

The choice of training corpus is an important problem for this project. It should be large enough to result in a reasonably good model. However, access to computational resources is limited, making it necessary to keep the size manageable. This work uses a smaller, sampled subset of the total data volume (10BT). 

**Some statistics for the FineWeb 10BT subset:**
- Total Documents: 14,868,862 documents across all files
- Total Tokens: 10,371,489,838 (about 10.4 billion tokens)
- Minimum Token Count: 24 tokens (shortest document)
-  Maximum Token Count: 381,395 tokens (longest document)
- Average Tokens per Document: 697.53 tokens

3. `python corpus_download.py`, note that this is approximately 31 GB of data.
- If interested, some deeper analysis could be extracted by running `python analyze_fineweb.py`.
4. `python tokenize_fineweb_parallell.py --workers 1 --chunk-size 100`, arguments can be increased depending on system performance. 

## 1. Simple Pre-processing pipeline
A simple pre-processing pipeline has been implemented, scraping data from a website and a PDF file as toy examples. A sampled version of FineWeb will be used to train the GPT-style model. However, this simple scraping and cleaning process is a good exercise.

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

## 2. Pre-training

## References:
Guilherme Penedo, Hynek Kydlíček, Loubna Ben allal, Anton Lozhkov, Margaret Mitchell, Colin Raffel, Leandro Von Werra, Thomas Wolf, The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale, en, arXiv:2406.17557 [cs], Oct. 2024. doi: 10.48550/arXiv.2406.
