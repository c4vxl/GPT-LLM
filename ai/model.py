import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import dataclasses

# Define a data class to hold model arguments
@dataclasses.dataclass
class ModelArgs:
    block_size: int = 1024  # Size of each block
    vocab_size: int = 50257  # Size of the vocabulary
    n_layer: int = 12        # Number of transformer layers
    n_head: int = 12         # Number of attention heads
    n_embd: int = 768        # Dimension of the embedding
    bias: bool = False       # Whether to include bias terms
    device: str = "cuda" if torch.cuda.is_available() else "cpu"  # Device for computation

    # Method to set model arguments
    def set(self,
            block_size: int = block_size, 
            vocab_size: int = vocab_size, 
            n_layer: int = n_layer, 
            n_head: int = n_head, 
            n_embd: int = n_embd, 
            bias: bool = bias,
            device: str = device):
        """
        Modify model arguments

        Args:
            block_size (int): Size of each block.
            vocab_size (int): Size of the vocabulary.
            n_layer (int): Number of transformer layers.
            n_head (int): Number of attention heads.
            n_embd (int): Dimension of the embedding.
            bias (bool): Whether to include bias terms.
            device (str): Device for computation.

        Returns:
            ModelArgs: Updated ModelArgs object.
        """
        
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.bias = bias
        self.device = device
        return self

# Multi-Head Attention Module
class MHAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        assert args.n_embd % args.n_head == 0
        self.d_k = args.n_embd // args.n_head
        self.n_head = args.n_head

        # Linear transformations for query, key, value, and output
        self.q_linear = nn.Linear(args.n_embd, args.n_embd, bias=args.bias)
        self.k_linear = nn.Linear(args.n_embd, args.n_embd, bias=args.bias)
        self.v_linear = nn.Linear(args.n_embd, args.n_embd, bias=args.bias)
        self.out = nn.Linear(args.n_embd, args.n_embd, bias=args.bias)

    def forward(self, x):
        """
        Forward pass through the multi-head attention module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """

        batch_size = x.size(0)
        # Linear transformations followed by reshaping and transposing for query, key, and value
        q = self.q_linear(x).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)

        # Compute attention scores, apply softmax, and perform weighted sum
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)
        
        # Concatenate and apply linear transformation for output
        concat = output.transpose(1, 2).contiguous().view(batch_size, -1, self.n_head * self.d_k)
        return self.out(concat)

# Layer Normalization Module
class LayerNorm(nn.Module):
    def __init__(self, ndim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        # Initialize bias if bias is enabled
        self.bias = nn.Parameter(torch.zeros(ndim)) if ModelArgs.bias else None

    def forward(self, input):
        """
        Apply layer normalization.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor.
        """

        return F.layer_norm(input, input.shape[-1:], self.weight, self.bias, 1e-5)

# Feed Forward Module
class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # Define a feed-forward network with ReLU activation
        self.net = nn.Sequential(
            nn.Linear(args.n_embd, 4 * args.n_embd, bias=args.bias),
            nn.ReLU(),
            nn.Linear(4 * args.n_embd, args.n_embd, bias=args.bias),
        )

    def forward(self, x):
        """
        Pass input through the feed-forward network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.net(x)
    
# Transformer Block Module
class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # Initialize multi-head attention, layer normalization, and feed-forward layers
        self.attention = MHAttention(args)
        self.ln1 = LayerNorm(args.n_embd)
        self.ff = FeedForward(args)
        self.ln2 = LayerNorm(args.n_embd)

    def forward(self, x):
        """
        Forward pass through the transformer block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """

        # Forward pass through the transformer block
        attn_output = self.attention(x)
        x = self.ln1(x + attn_output)
        ff_output = self.ff(x)
        x = self.ln2(x + ff_output)
        return x

# Main Model Module
class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        
        # Define word and position embeddings, transformer blocks, and output layer
        self.wte = nn.Embedding(args.vocab_size, args.n_embd)
        self.wpe = nn.Embedding(args.block_size, args.n_embd)
        self.blocks = nn.ModuleList([TransformerBlock(args) for _ in range(args.n_layer)])
        self.ln_f = LayerNorm(args.n_embd)
        self.h = nn.Linear(args.n_embd, args.vocab_size, bias=args.bias)

    def forward(self, idx):
        """
        Forward pass through the model.

        Args:
            idx (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output logits.
        """

        # Forward pass through the model
        b, t = idx.size()
        assert t <= self.args.block_size, f"Cannot forward sequence of length {t}, block size is only {self.args.block_size}"
        assert (idx >= 0).all() and (idx < self.args.vocab_size).all(), "Invalid indices in input"

        token_embeddings = self.wte(idx)
        position_ids = torch.arange(t, dtype=torch.long, device=idx.device).unsqueeze(0)
        position_embeddings = self.wpe(position_ids)
        x = token_embeddings + position_embeddings

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.h(x)
        return logits

    def export_to_file(self, path: str = "model.pth"):
        # Save model to file
        torch.save(self.state_dict(), path)

    @staticmethod
    def load_from_file(path: str = "model.pth", args: ModelArgs = ModelArgs()):
        # Load model from file
        if not os.path.isfile(path): 
            return Model(args)

        state_dict = torch.load(path, map_location=torch.device(args.device))
        model = Model(args)

        state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}

        model.load_state_dict(state_dict, strict=False)
        return model