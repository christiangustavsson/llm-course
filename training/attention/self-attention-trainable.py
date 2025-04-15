"""
    Working along with video tutorial on attention mechanism
    https://www.youtube.com/watch?v=5vcj8kSwBCY

    Now, a more complex self-attention mechanism is implemented, with
    trainable weigts. This initial implementation only works with respect to x^2
"""
import os
import torch
import torch.nn as nn

class Attention(nn.Module):
    pass

def softmax_naive(x): # Not numerically stable
    return torch.exp(x) / torch.exp(x).sum(dim=0)

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

    W_query = torch.nn.Parameter(torch.rand(d_in, d_out))
    W_keys = torch.nn.Parameter(torch.rand(d_in, d_out))
    W_value = torch.nn.Parameter(torch.rand(d_in, d_out))

    # print(W_query)
    # print(W_key)
    # print(W_value)

    query_2 = x_2 @ W_query
    # print(query_2)

    keys = inputs @ W_keys
    value = inputs @ W_value
    # print(keys)
    # print(value)

    keys_2 = keys[1] 
    attn_score_22 = torch.dot(query_2, keys_2)
    print(attn_score_22)

    attn_score_2 = query_2 @ keys.T
    print(attn_score_2)

    d_k = keys.shape[1]
    attn_weights_2 = torch.softmax(attn_score_2 / d_k**0.5, dim = -1)
    print(attn_weights_2)

    context_vector_2 = attn_weights_2 @ value
    print(context_vector_2)

    # Next, we want to generalize this work. 

if __name__ == "__main__":
    os.system("clear")
    main()
    # print("End of program.\n")