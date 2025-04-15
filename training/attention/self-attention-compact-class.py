"""
    Working along with video tutorial on attention mechanism
    https://www.youtube.com/watch?v=5vcj8kSwBCY

    Now, implementing a compact casual-self-attention class
"""

import os
import torch
import torch.nn as nn

class CasualAttention(nn.Module):
    
    def __init__(self, d_in, d_out, context_length, dropout = 0.0, qkv_bias=False):
        super().__init__()
        self.W_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_keys = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = torch.nn.Dropout(dropout)
        # Prepares a mask that can be moved to device, also saves time to generate once
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))


    def forward(self, inputs):
        batch_size, num_tokens, d_in = inputs.shape
        # inputs = batch, 2 x 6 x 3 for this example
        queries = self.W_query(inputs)
        keys = self.W_keys(inputs)
        value = self.W_value(inputs)

        attn_scores = queries @ keys.transpose(1, 2)                 # Changed transpose
        attn_scores.masked_fill_(                                   # New, _ operations are in-place
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf) # 'num_tokens' to account for cases where...
        attn_weights = torch.softmax(                               # Same
            attn_scores / keys.shape[-1]**0.5, dim = -1)
        attn_weights = self.dropout(attn_weights)                   # New, dropout

        context_vector = attn_weights @ value

        print(context_vector)

        return context_vector

def main():
    ### For simplicity, one word is a token and embeddning vector is small
    inputs = torch.tensor(   
        [[0.43, 0.15, 0.89], # Your         (x^1)
         [0.55, 0.87, 0.66], # journey      (x^2)
         [0.57, 0.85, 0.64], # starts       (x^3)  
         [0.22, 0.58, 0.33], # with         (x^4)
         [0.77, 0.25, 0.10], # one          (x^5)
         [0.05, 0.80, 0.55]] # step         (x^)
    )

    torch.manual_seed(789)

    # Creating a batch of 2 for testing
    batch = torch.stack([inputs, inputs], dim=0)  # (2,6,3)

    d_in = inputs.shape[1]              # Dimension of input vector
    d_out = 2                           # Dimension of output vector, choosen.
    context_length = batch.shape[1]     # Number of tokens in the input
    dropout = 0.0
    ca = CasualAttention(d_in, d_out, context_length, dropout)
    ca(batch)


if __name__ == "__main__":
    os.system("clear")
    main()
    print("End of program.\n")