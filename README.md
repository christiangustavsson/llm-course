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
3. Make sure `nationell-strategi-for-cybersakerhet-2025-2029.pdf` is downloaded.

### Download and tokenize FineWeb Corpus (10BT)
4. 'python corpus-download.py', note that this is approximately 20 GB of data.
5. 'python tokenize_fineweb.py', note that this will take some time. For a powerful computer, 'python tokenize_fineweb.py' is also available.
