"""
This file contains the model for the transformer. The model is composed of the following:
- InputEmbedding
[TODO: Add more]

I will be doing this in a modular way, so that I can easily swap out different components of the model.
Also I will use pytorch to build the model, so that I can easily use the GPU to train the model.

author: Joshua Nathan Mugerwa (haha so formal right? suggested by github lol) concaption it is.
"""

import math
import torch
import torch.nn as nn

class InputEmbedding(nn.Module):
    """
    This class is the input embedding layer of the transformer. It is a simple embedding layer, but it also
    scales the embedding by the square root of the embedding dimension. This is done to prevent the gradients
    from exploding. This is done in the forward pass.
    """
    def __init__(self, embedding_dim: int, vocab_size: int):
        super(InputEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, x):
        """
        Forward pass of the input embedding layer. This is a simple embedding layer, but it also
        scales the embedding by the square root of the embedding dimension. This is done to prevent the gradients
        from exploding.
        """

        return self.embedding(x) * math.sqrt(self.embedding_dim)
    

class PositionalEncoding(nn.Module):
    """
    This class is the positional encoding layer of the transformer. 
    """
    def __init__(self, embedding_dim: int, max_len : int=5000, dropout: float=0.1,):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embedding_dim = embedding_dim
        self.max_len = max_len

        # Create the positional encoding matrix (max_len, embedding_dim)
        self.pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len).unsqueeze(1)

        # LaTeX: PE_{pos, 2i} = sin(pos / 10000^{2i / d_{model}}) for even i
        # LaTeX: PE_{pos, 2i+1} = cos(pos / 10000^{2i / d_{model}}) for odd i
        # This is the same as the formula in the paper.
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0) # Add a batch dimension (1, max_len, embedding_dim)
        self.register_buffer('pe', self.pe)
        
    def forward(self, x):
        """
        Forward pass of the positional encoding layer.
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)