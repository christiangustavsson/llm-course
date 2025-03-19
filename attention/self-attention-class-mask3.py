"""
    Working along with video tutorial on attention mechanism
    https://www.youtube.com/watch?v=5vcj8kSwBCY

    Now, a more complex self-attention mechanism is implemented, with
    trainable weigts. 

    Implements a dropout mask; Meant to reduce overfitting. Not used 
    that often anymore. Added to series for completeness.
"""
import os
import torch
import torch.nn as nn

class SelfAttention_v4(nn.Module):
    
    def __init__(self, d_in, d_out,qkv_bias=False):
        super().__init__()
        self.W_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_keys = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = torch.nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, inputs):
        queries = self.W_query(inputs)
        keys = self.W_keys(inputs)
        value = self.W_value(inputs)
        attn_score = queries @ keys.T
        d_k = keys.shape[1]
        context_length = attn_score.shape[0]

        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
        masked = attn_score.masked_fill(mask.bool(), -torch.inf)

        attn_weights = torch.softmax(masked / d_k**0.5, dim = -1)
        print(attn_weights)

        dropout_mask = torch.nn.Dropout(p=0.5)
        attn_weights = dropout_mask(attn_weights)
        # Note, he forgot to normalize the new attn_weights        

        context_vector = attn_weights @ value

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

    x_2 = inputs[1]
    d_in = inputs.shape[1]  # Dimension of input vector
    d_out = 2               # Dimension of output vector, choosen.

    torch.manual_seed(123)

    sa_v1_1 = SelfAttention_v4(d_in, d_out)
    result = sa_v1_1(inputs)
    print(result)

if __name__ == "__main__":
    os.system("clear")
    main()
    # print("End of program.\n")