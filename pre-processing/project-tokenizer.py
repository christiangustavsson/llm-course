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
db_path = "datasets/openwebtext/smaller.db" # For testing purposes, use smaller.db
# db_path = "datasets/openwebtext/data.db" # Full >10GB database



def main():
    pass

if __name__ == "__main__":
    os.system("clear") # clear for Linux machines, cls for Windows
    main()