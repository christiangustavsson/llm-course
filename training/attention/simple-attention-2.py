"""
    Working along with video tutorial on attention mechanism
    https://www.youtube.com/watch?v=5vcj8kSwBCY

    Now, we do the work for the complete input vector, not just x^2    
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

    ### The attention weights will now be a matrix, 6x6. "Self-attention".

    ### Calculating the attention weights, generalized from before
    query = inputs[1]

    attn_weights = torch.empty(inputs.shape[0], inputs.shape[0])

    # The for-loop version of this calculation
    # for i, x_i in enumerate(inputs):
    #     for j, x_j in enumerate(inputs):
    #         attn_weights[i, j] = torch.dot(x_i, x_j)
    # print(attn_weights)
    # attn_weights = torch.softmax(attn_weights, dim=0)
    # print(attn_weights)

    attn_weights = inputs @ inputs.T # Matix multiplication version
    attn_weights = torch.softmax(attn_weights, dim=1)
    # print(attn_weights)

    ### Calculating the context vector
    context_vector = attn_weights @ inputs
    print(context_vector)

if __name__ == "__main__":
    os.system("clear")
    main()
    # print("End of program.\n")