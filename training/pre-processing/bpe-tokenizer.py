# This script is a BPE tokenizer that reads a text file, tokenizes it and creates a vocabulary.
# The vocabulary is a dictionary with the words as keys and the integers as values.
# The script is a good starting point for pre-processing text data.

import os
import re

from importlib.metadata import version
import tiktoken # At this time, 0.9.0

def textloader(file_path):
    with open(file_path, 'r', encoding="utf-8") as f:
        raw = f.read()
    print(f"Characters read from files: {len(raw)}")
    return raw

def main():
    os.system("clear")

    tokenizer = tiktoken.get_encoding("gpt2")

    raw = textloader('the-verdict.txt')
    text = raw[50:]
    encoded = tokenizer.encode(text)

    context_size = 4
    x = encoded[:context_size]
    y = encoded[1:context_size+1]

    for i in range(1,context_size+1):
        context = encoded[:i]
        desired = encoded[i]
        print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))


    # print(f"x:  {x}")
    # print(f"y:       {y}")

if __name__ == "__main__":
    main()