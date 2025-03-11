# This script is a basic tokenizer that reads a text file, tokenizes it and creates a vocabulary.
# The vocabulary is a dictionary with the words as keys and the integers as values.
# The script is a good starting point for pre-processing text data.

import os
import re

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [
        item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        preprocessed = [item if item in self.str_to_int
                        else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text

def load_text(file_path):
    with open(file_path, 'r', encoding="utf-8") as f:
        raw = f.read()
    print(f"Characters read from files: {len(raw)}")
    return raw

def vocabulary(raw):
    tokens = re.split(r'([,.:;?_!"()\']|--|\s)', raw)
    tokens = [item.strip() for item in tokens if item.strip()] # Remove white spaces: Might be good, might not.
    all_tokens = sorted(set(tokens))
    all_tokens.extend(["<|endoftext|>", "<|unk|>", "[PAD]", "[BOS]", "[EOS]"])
    vocab_size = len(all_tokens)
    print(f"Vocabulary size: {vocab_size}")
    vocab = {token:integer for integer, token in enumerate(all_tokens)}

    return vocab

def main():
    os.system("clear")

    raw = "Hello, world. Is this-- a test?"
    raw = load_text('pre-processing/the-verdict.txt')

    vocab = vocabulary(raw)

    tokenizer = SimpleTokenizerV2(vocab)

    # text1 = """"It's the last he painted, you know," Mrs. Gisburn said with pardonable pride."""
    text2 = "Hello, would you like to play a game?" # WarGames
    text3 = "In the sunlit terraces of the palace."

    text = text = " <|endoftext|> ".join((text2, text3))
    print(text)

    ids = tokenizer.encode(text)
    print(f"Tokenized text string: {ids}")
    print(f"Reconverted back to text: {tokenizer.decode(ids)}")

if __name__ == "__main__":
    main()