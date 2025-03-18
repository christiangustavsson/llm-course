"""
    Working along with video tutorial on attention mechanism
    https://www.youtube.com/watch?v=5vcj8kSwBCY
    
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

    ### Attention weights, dot product of x^2 and x^i

    query = inputs[1] # journey      (x^2)
    attn_scores_2 = torch.empty(inputs.shape[0])
    for i, x_i in enumerate(inputs):
        attn_scores_2[i] = torch.dot(query, x_i)
    # print(attn_scores_2)

    attn_scores_2_tmp = attn_scores_2 / attn_scores_2.sum() # Simple normalization
    # print(attn_scores_2_tmp)

    # print(softmax_naive(attn_scores_2)) # Using function, not numerically stable

    # print(attn_scores_2.softmax(dim=0)) # Using PyTorch, numerically stable?
    attn_weights_2 = torch.softmax(attn_scores_2, dim=0) # Same as above


    ### Calculating the context vector
    query = inputs[1]                           # journey (x^2)
    context_vec_2 = torch.zeros(query.shape)    # This is z^2, context vector for x^2
    for i, x_i in enumerate(inputs):
        context_vec_2 += attn_weights_2[i] * x_i

    print(context_vec_2)


if __name__ == "__main__":
    os.system("clear")
    main()
    print("End of program.\n")