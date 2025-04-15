""""
PhD Course: Building and Understanding LLMs
Christian Gustavsson

This script is part of the pre-processing pipeline for the project. It contains the 
code for creating a dataloader for the GPT-2 model. For the pre-training, roughly a 
quarter of OpenWebText is used. Citations for the datasets can be found in the lab 
report. Store .db files in datasets/openwebtext
"""

import os
import tiktoken
import sqlite3
import torch
from torch.utils.data import Dataset, DataLoader

### Using best possible device for Torch workload
device = torch.device("mps" if torch.backends.mps.is_available() else 
                      "cuda" if torch.cuda.is_available() else 
                      "cpu")

### Placement of the database file
#db_path = "datasets/openwebtext/smaller.db" # For testing purposes, use smaller.db, 744 entries
db_path = "datasets/openwebtext/data.db" # Full >10GB database, 2,019,435 entries

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    
def create_dataloader_v1(txt, batch_size, max_length, stride,
                        shuffle=False, drop_last=True, num_workers=0):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader

def open_db(db_path):
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print(f"Connected to database: {db_path}")

        # Get table name (assuming there is only one table)
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        table_name = cursor.fetchone()
        
        if table_name:
            table_name = table_name[0]
            print(f"Found table: {table_name}")
            cursor.execute(f"SELECT text FROM {table_name};")

            print(table_name)

            return conn, cursor, table_name

        else:
            print("No tables found in the database.")
            return None, None, None

    except sqlite3.Error as e:
        print("Error connecting to database:", e)
        return None, None, None

def fetch_records(cursor, table_name, no_records=1):
    rows = cursor.fetchmany(no_records) # Fetch first x records

    return rows

def close_db(conn):
    conn.close()
    print("Connection closed.")

def main():
    print("Connecting to database.")    
    conn, cursor, table_name = open_db(db_path)

    vocab_size = 50257
    output_dim = 256
    context_length = 1024

    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

    batch_size = 16
    max_length = 8
    stride = 2

    entries = 0

    # while True:
    #     rows = fetch_records(cursor, table_name, no_records=10000)
    #     if rows.__len__() == 0: # Break when no more records
    #         break
        
    #     for row in rows:
    #         raw_text = row[0]

    #         dataloader = create_dataloader_v1(
    #             raw_text,
    #             batch_size=batch_size,
    #             max_length=max_length,
    #             stride=stride
    #             )
    #         for batch in dataloader:
    #             x, y = batch

    #             token_embeddings = token_embedding_layer(x)
    #             pos_embeddings = pos_embedding_layer(torch.arange(max_length))

    #             input_embeddings = token_embeddings + pos_embeddings

    #             print("Input embeddings shape:", input_embeddings.shape)

    #             entries += 1

    #             break

    # print(f"Entries processed: {entries}")

    close_db(conn)

if __name__ == "__main__":
    os.system("clear") # clear for Linux machines, cls for Windows
    main()