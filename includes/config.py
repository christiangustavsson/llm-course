GPT_CONFIG_small = {
    "vocab_size": 50257,  # Vocabulary size
    "emb_dim": 768,       # Embedding dimension
    "context_length": 1024,  # Maximum context length
    "n_layers": 12,       # Number of transformer layers
    "n_heads": 12,        # Number of attention heads
    "drop_rate": 0.1,     # Dropout rate
    "qkv_bias": False     # Whether to use bias in QKV projections
}

GPT_CONFIG_medium = {
    "vocab_size": 50257,
    "emb_dim": 1024,
    "context_length": 1024,
    "n_layers": 24,
    "n_heads": 16,
    "drop_rate": 0.1,
    "qkv_bias": False
}

GPT_CONFIG_large = {
    "vocab_size": 50257,
    "emb_dim": 1280,
    "context_length": 1024,
    "n_layers": 36,
    "n_heads": 20,
    "drop_rate": 0.1,
    "qkv_bias": False
}

GPT_CONFIG_XL = {
    "vocab_size": 50257,
    "emb_dim": 1600,
    "context_length": 1024,
    "n_layers": 48,
    "n_heads": 25,
    "drop_rate": 0.1,
    "qkv_bias": False
}