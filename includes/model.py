import os
import torch
import torch.nn as nn
import torch.optim as optim
import tiktoken
import math
from .attention import MultiHeadAttention
from .config import * 
from torch.nn import functional as F

### Using best possible device for Torch workload
device = torch.device("mps" if torch.backends.mps.is_available() else 
                      "cuda" if torch.cuda.is_available() else 
                      "cpu")



class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.blocks = nn.Sequential(
            *[Block(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = LayerNorm(cfg["emb_dim"])
        self.attn = MultiHeadAttention(cfg)
        self.ln_2 = LayerNorm(cfg["emb_dim"])
        self.ff = FeedForward(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.ff(self.ln_2(x))
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.emb_dim = cfg["emb_dim"]
        self.n_heads = cfg["n_heads"]
        self.head_dim = self.emb_dim // self.n_heads

        self.qkv_proj = nn.Linear(
            cfg["emb_dim"], 3 * cfg["emb_dim"], bias=False
        )
        self.out_proj = nn.Linear(
            cfg["emb_dim"], cfg["emb_dim"], bias=False
        )
        self.attn_drop = nn.Dropout(cfg["drop_rate"])
        self.resid_drop = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        batch_size, seq_len, emb_dim = x.shape

        qkv = self.qkv_proj(x)
        qkv = qkv.view(
            batch_size, seq_len, self.n_heads, 3 * self.head_dim
        )
        qkv = qkv.transpose(1, 2)
        q, k, v = qkv.chunk(3, dim=-1)

        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = attn / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, seq_len, emb_dim)
        out = self.out_proj(out)
        out = self.resid_drop(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            nn.GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
            nn.Dropout(cfg["drop_rate"])
        )

    def forward(self, x):
        return self.net(x)

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.gamma = nn.Parameter(torch.ones(emb_dim))
        self.beta = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out

def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
        # Crop current context if it exceeds the context window
        idx_cond = idx[:, -context_size:]
        # Get predictions
        with torch.no_grad():
            logits = model(idx_cond)
        # Focus on last time step
        logits = logits[:, -1, :]  # Becomes (B, C)
        # Get probabilities
        probs = F.softmax(logits, dim=-1)  # (B, C)
        # Sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
        # Append sampled index to running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
    return idx

class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        # Use a placeholder for TransformerBlock
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        
        # Use a placeholder for LayerNorm
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # A simple placeholder

    def forward(self, x):
        # This block does nothing and just returns its input.
        return x


class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        # The parameters here are just to mimic the LayerNorm interface.

    def forward(self, x):
        # This layer does nothing and just returns its input.
        return x
    
def print_gradients(model, x):
    # Forward pass
    output = model(x)
    target = torch.tensor([[0.]])

    # Calculate loss based on how close the target
    # and output are
    loss = nn.MSELoss()
    loss = loss(output, target)
    
    # Backward pass to calculate the gradients
    loss.backward()

    for name, param in model.named_parameters():
        if 'weight' in name:
            # Print the mean absolute gradient of the weights
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")

def main():
    # GPT Config settings, availabel in config.py
    cfg = GPT_CONFIG_small
    # cfg = GPT_CONFIG_medium
    # cfg = GPT_CONFIG_large
    # cfg = GPT_CONFIG_XL

    tokenizer = tiktoken.get_encoding("gpt2")

    batch = []

    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"

    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch = torch.stack(batch, dim=0)
    # print(batch)

    # torch.manual_seed(123)
    # model = DummyGPTModel(GPT_CONFIG_124M)

    # logits = model(batch)
    # print("Output shape:", logits.shape)
    # print(logits)

    # # create 2 training examples with 5 dimensions (features) each
    # torch.manual_seed(123)
    # batch_example = torch.randn(2, 5) 

    # layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
    # out = layer(batch_example)
    # print(out)

    # mean = out.mean(dim=-1, keepdim=True)
    # var = out.var(dim=-1, keepdim=True)

    # print("Mean:\n", mean)
    # print("Variance:\n", var)

    # out_norm = (out - mean) / torch.sqrt(var)
    # print("Normalized layer outputs:\n", out_norm)

    # mean = out_norm.mean(dim=-1, keepdim=True)
    # var = out_norm.var(dim=-1, keepdim=True)
    # print("Mean:\n", mean)
    # print("Variance:\n", var)

    # torch.set_printoptions(sci_mode=False)
    # print("Mean:\n", mean)
    # print("Variance:\n", var)

    # ln = LayerNorm(emb_dim=5)
    # out_ln = ln(batch_example)

    # mean = out_ln.mean(dim=-1, keepdim=True)
    # var = out_ln.var(dim=-1, unbiased=False, keepdim=True)

    # print("Mean:\n", mean)
    # print("Variance:\n", var)

    # ffn = FeedForward(GPT_CONFIG_124M)

    # # input shape: [batch_size, num_token, emb_size]
    # torch.manual_seed(123)
    # x = torch.rand(2, 3, 768) 
    # out = ffn(x)
    # print(out.shape)

    # layer_sizes = [3, 3, 3, 3, 3, 1]  

    # sample_input = torch.tensor([[1., 0., -1.]])

    # torch.manual_seed(123)
    # model_without_shortcut = ExampleDeepNeuralNetwork(
    #     layer_sizes, use_shortcut=False
    # )
    # print_gradients(model_without_shortcut, sample_input)

    # torch.manual_seed(123)
    # model_with_shortcut = ExampleDeepNeuralNetwork(
    #     layer_sizes, use_shortcut=True
    # )
    # print_gradients(model_with_shortcut, sample_input)

    # torch.manual_seed(123)

    # x = torch.rand(2, 4, 768)  # Shape: [batch_size, num_tokens, emb_dim]
    # block = TransformerBlock(GPT_CONFIG_124M)
    # output = block(x)

    # print("Input shape:", x.shape)
    # print("Output shape:", output.shape)

    torch.manual_seed(123)
    model = GPTModel(cfg)

    out = model(batch)
    print("Input batch:\n", batch)
    print("\nOutput shape:", out.shape)
    print(out)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")

    print("Token embedding layer shape:", model.tok_emb.weight.shape)
    print("Output layer shape:", model.out_head.weight.shape)

    total_params_gpt2 =  total_params - sum(p.numel() for p in model.out_head.parameters())
    print(f"Number of trainable parameters considering weight tying: {total_params_gpt2:,}")

    # Calculate the total size in bytes (assuming float32, 4 bytes per parameter)
    total_size_bytes = total_params * 4

    # Convert to megabytes
    total_size_mb = total_size_bytes / (1024 * 1024)

    print(f"Total size of the model: {total_size_mb:.2f} MB")

    start_context = "Hello, I am"

    encoded = tokenizer.encode(start_context)
    print("encoded:", encoded)

    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    model.eval() # disable dropout

    out = generate_text_simple(
        model=model,
        idx=encoded_tensor, 
        max_new_tokens=6, 
        context_size=cfg["context_length"]
    )

    print("Output:", out)
    print("Output length:", len(out[0]))

    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    print(decoded_text)

if __name__ == "__main__":
    os.system("clear")
    main()
    print("--- End of program. --- \n")