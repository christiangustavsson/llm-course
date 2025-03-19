"""
    Working along with video tutorial on attention mechanism
    https://www.youtube.com/watch?v=5vcj8kSwBCY

    Now, implementing a a more efficient compact casual-self-attention class with 
    multi-head attention. Last step in this chapter.

    Also trying to use PyTorch's built-in scaled dot-product attention function.
"""

import os
import time
import torch
import torch.nn as nn

### Using best possible device for Torch workload
device = torch.device("mps" if torch.backends.mps.is_available() else 
                      "cuda" if torch.cuda.is_available() else 
                      "cpu")

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout = 0.0, num_heads = 2, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "Number of heads must be divisible by output dimension"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # Dimension of each head

        self.W_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = torch.nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out) # Linear layer to combine (project) the output
        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        batch_size, num_tokens, d_in = x.shape

        keys = self.W_key(x)    # Shape (batch_size, num_tokens, d_out)
        values = self.W_value(x)
        queries = self.W_query(x)

        # We implicity split the matrix by adding a 'num_heads' dimension
        # Unroll last dim: (batch_size, num_tokens, d_out) -> (batch_size, num_tokens, num_heads, head_dim)

        keys = keys.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        values = values.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim)

        # Transpose (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        queries = queries.transpose(1, 2)

        # Calculate scaled dot-product attention (aka self-attention) with casual mask
        attn_scores = queries @ keys.transpose(2, 3) # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / (keys.shape[-1] ** 0.5), dim = -1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (batch_size, num_tokens, num_heads, head_dim)
        context_vector = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vector = context_vector.contiguous().view(batch_size, num_tokens, self.d_out)
        context_vector = self.out_proj(context_vector) # optional projection

        return context_vector

class MHAPyTorchScaledDotProduct(nn.Module):
    def __init__(self, d_in, d_out, num_heads, context_length, dropout=0.0, qkv_bias=False):
        super().__init__()

        assert d_out % num_heads == 0, "embed_dim is indivisible by num_heads"

        self.num_heads = num_heads
        self.context_length = context_length
        self.head_dim = d_out // num_heads
        self.d_out = d_out

        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.proj = nn.Linear(d_out, d_out)
        self.dropout = dropout

    def forward(self, x):
        batch_size, num_tokens, embed_dim = x.shape

        # (b, num_tokens, embed_dim) --> (b, num_tokens, 3 * embed_dim)
        qkv = self.qkv(x)

        # (b, num_tokens, 3 * embed_dim) --> (b, num_tokens, 3, num_heads, head_dim)
        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)

        # (b, num_tokens, 3, num_heads, head_dim) --> (3, b, num_heads, num_tokens, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # (3, b, num_heads, num_tokens, head_dim) -> 3 times (b, num_heads, num_tokens, head_dim)
        queries, keys, values = qkv

        use_dropout = 0. if not self.training else self.dropout

        context_vec = nn.functional.scaled_dot_product_attention(
            queries, keys, values, attn_mask=None, dropout_p=use_dropout, is_causal=True)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.d_out)

        context_vec = self.proj(context_vec)

        return context_vec

class MHAPyTorchSDPAWithoutFlash(nn.Module):
    def __init__(self, d_in, d_out, num_heads, context_length, dropout=0.0, qkv_bias=False):
        super().__init__()

        assert d_out % num_heads == 0, "embed_dim is indivisible by num_heads"

        self.num_heads = num_heads
        self.context_length = context_length
        self.head_dim = d_out // num_heads
        self.d_out = d_out

        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.proj = nn.Linear(d_out, d_out)
        self.dropout = dropout

        # Register buffer ensures it's always on the right device
        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1).bool()
        self.register_buffer("mask", mask)

    def forward(self, x):
        x = x.to(self.qkv.weight.device)  # Ensure input is on the right device

        batch_size, num_tokens, embed_dim = x.shape

        # (b, num_tokens, embed_dim) --> (b, num_tokens, 3 * embed_dim)
        qkv = self.qkv(x)

        # (b, num_tokens, 3 * embed_dim) --> (b, num_tokens, 3, num_heads, head_dim)
        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)

        # (b, num_tokens, 3, num_heads, head_dim) --> (3, b, num_heads, num_tokens, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # (3, b, num_heads, num_tokens, head_dim) -> 3 times (b, num_heads, num_tokens, head_dim)
        queries, keys, values = qkv

        use_dropout = 0. if not self.training else self.dropout

        # Dynamically adjust the mask size based on actual input length
        attn_mask = self.mask[:num_tokens, :num_tokens].to(x.device)

        context_vec = nn.functional.scaled_dot_product_attention(
            queries, keys, values, attn_mask=attn_mask, dropout_p=use_dropout, is_causal=False
        )

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.d_out)

        context_vec = self.proj(context_vec)

        return context_vec


def main():
    print("Running device: ", device)

    ### For simplicity, one word is a token and embeddning vector is small
    inputs = torch.tensor(   
        [[0.43, 0.15, 0.89], # Your         (x^1)
         [0.55, 0.87, 0.66], # journey      (x^2)
         [0.57, 0.85, 0.64], # starts       (x^3)  
         [0.22, 0.58, 0.33], # with         (x^4)
         [0.77, 0.25, 0.10], # one          (x^5)
         [0.05, 0.80, 0.55]] # step         (x^)
    )

    torch.manual_seed(123)

    # Creating a batch of 2 for testing
    batch = torch.stack([inputs, inputs], dim=0)  # (2,6,3)

    batch_size, context_length, d_in = batch.shape
    d_out = 4
    dropout = 0.0

    start_time = time.time() # Start timer

    mha = MultiHeadAttention(d_in, d_out, context_length, dropout = 0.0, num_heads = 2 )

    context_vecs = mha(batch).to(device)
    # End timer
    end_time = time.time() # End timer, Computation took 0.019063 seconds
    print(f"Computation took {end_time - start_time:.6f} seconds")

    print(context_vecs)
    print("context_vecs.shape: ", context_vecs.shape)

    start_time = time.time() # Start timer

    mha_pytorch_scaled = MHAPyTorchScaledDotProduct(
        d_in=d_in,
        d_out=d_out,
        context_length=context_length,
        dropout=0.0,
        num_heads=2,
        qkv_bias=False
    ).to(device)

    batch = batch.to(device)  # Move batch to same device as model
    out = mha_pytorch_scaled(batch)

    # End timer
    end_time = time.time() # End timer, Computation took 0.048227 seconds
    print(f"Computation took {end_time - start_time:.6f} seconds")

    print(out)
    print(out.shape)

    start_time = time.time() # Start timer

    mha_pytorch_sdpa_no_flash = MHAPyTorchSDPAWithoutFlash(
        d_in=d_in,
        d_out=d_out,
        context_length=context_length,
        dropout=0.0,
        num_heads=2,
        qkv_bias=False
    ).to(device)

    out = mha_pytorch_sdpa_no_flash(batch)

    # End timer
    end_time = time.time() # End timer, Computation took 0.002886 seconds
    print(f"Computation took {end_time - start_time:.6f} seconds")

    print(out.shape)  # Expected Output: torch.Size([2, 6, 64])


if __name__ == "__main__":
    os.system("clear")
    main()
    print("End of program.\n")