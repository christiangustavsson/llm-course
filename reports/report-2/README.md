# Report 2: FineWeb Dataset Analysis

This directory contains scripts and analysis for working with the FineWeb dataset.

## Files

- `fineweb-download.py`: Script to download the FineWeb dataset from HuggingFace
- `analyze_fineweb.py`: Script to analyze the FineWeb dataset, including:
  - Basic statistics (document count, token count, etc.)
  - Language distribution
  - Token length distribution (with visualizations)
  - Sample documents for qualitative analysis

## Dataset Information

The FineWeb dataset is a large-scale web crawl dataset containing diverse web content. This analysis focuses on the 10BT sample, which contains approximately:

- 15 million documents
- 10.4 billion tokens
- Documents ranging from 24 to 381,395 tokens in length
- Average document length of ~700 tokens

## Usage

1. Download the dataset:
   ```
   python fineweb-download.py
   ```

2. Analyze the dataset:
   ```
   python analyze_fineweb.py
   ```

3. View the generated plots in the `plots` directory

## Notes

- The dataset is stored in the `datasets/fineweb` directory
- The analysis focuses on the 10BT sample, which is a subset of the full dataset
- The token counts are pre-calculated and stored in the dataset 