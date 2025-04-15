"""

Course: Understanding and Building LLMs
Report 1: Developing a simple data pre-processing pipeline

This script is a BPE tokenizer that reads a text file, tokenizes it and creates 
a vocabulary. 

Christian Gustavsson, christian.gustavsson@liu.se

"""


import os
import re

from importlib.metadata import version
import tiktoken # At this time, 0.9.0

def textloader(file_path):
    with open(file_path, 'r', encoding="utf-8") as f:
        raw = f.read()
    # print(f"Characters read from files: {len(raw)}")
    return raw


def tokenize(text:str) -> list[int]:
    tokenizer = tiktoken.get_encoding("gpt2")

    return tokenizer.encode(text)


def main():
    os.system("clear")

    # Toy example using the-verdict.txt
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    file_path = os.path.join(root_dir, 'the-verdict.txt')
    
    raw = textloader(file_path)
    text = tokenize(raw)

    print("First 100 tokens:", text[:100])


if __name__ == "__main__":
    main()