import tiktoken
import torch
import sqlite3  # Example for SQLite, replace with other DB drivers if needed
from torch.utils.data import Dataset, DataLoader

### Using best possible device for Torch workload
device = torch.device("cuda" if torch.cuda.is_available() else 
                      "mps" if torch.backends.mps.is_available() else 
                      "cpu")
print(f"Using device: {device}")

### Placement of the database file
#db_path = "datasets/openwebtext/smaller.db"  # Testing: 744 entries
db_path = "datasets/openwebtext/data.db"  # Full database: >10GB, 2,019,435 entries

class GPTDatasetFromDB(Dataset):
    def __init__(self, db_path, query, tokenizer, max_length, stride):
        """
        db_path: Path to the SQLite database (or connection string for other DBs)
        query: SQL query to fetch text data
        tokenizer: The tokenizer for encoding text
        max_length: Sequence length
        stride: Overlap between sequences
        """
        self.input_ids = []
        self.target_ids = []
        self.tokenizer = tokenizer

        # Connect to database and fetch data
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute(query)

        # Process each row of the fetched dataset
        for row in self.cursor.fetchall():
            print("Processing row...")
            text = row[0]  # Assuming the text is in the first column
            self.process_text(text, max_length, stride)

    def process_text(self, text, max_length, stride):
        """ Tokenizes text and chunks it into overlapping sequences. """
        token_ids = self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]

            # Move tensors to device (GPU/CPU)
            self.input_ids.append(torch.tensor(input_chunk, device=device))
            self.target_ids.append(torch.tensor(target_chunk, device=device))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

    def __del__(self):
        """ Close DB connection when the dataset object is deleted. """
        self.conn.close()


def create_dataloader_from_db(db_path, query, batch_size, max_length, stride,
                              shuffle=True, drop_last=True, num_workers=0):
    """ Create a DataLoader that reads from a database instead of a text file. """
    tokenizer = tiktoken.get_encoding("gpt2")

    dataset = GPTDatasetFromDB(db_path, query, tokenizer, max_length, stride)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                            drop_last=drop_last, num_workers=num_workers)

    return dataloader


# Example usage: Fetching text data from a database
query = "SELECT text FROM data"  # Modify as per your DB schema

batch_size = 4
max_length = 128
stride = 32

dataloader = create_dataloader_from_db(db_path, query, batch_size, max_length, stride, shuffle=False)

# Example iteration over dataloader
i = 0
for batch in dataloader:
    x, y = batch

    # Ensure tensors remain on the correct device
    x, y = x.to(device), y.to(device)

    print("Batch Inputs:\n", x.shape)
    print("\nBatch Targets:\n", y.shape)

    # Print first batch index
    i += 1
    print(f"Processed batch {i} on {device}")
    break  # Process only first batch for testing
