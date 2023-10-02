import torch
import torch.nn as nn
from cross_attn.flash import FlashAttention





class MultiModalCrossAttn(nn.Module):
    """
    Cross attention module for multi-modal (text and image) attention.
    """
    def __init__(
        self, 
        dim: int, 
        heads: int, 
        dropout: float = 0.1, 
        causal: bool = True, 
        flash: bool = True
    ):
        super().__init__()

        self.to_query = nn.Linear(dim, dim * heads)
        self.to_key = nn.Linear(dim, dim * heads)
        self.to_value = nn.Linear(dim, dim * heads)
        

        self.attend = FlashAttention(
            # causal=causal,
            # dropout=dropout,
            # flash=flash
        )

    def forward(self, text_features, image_features):
        """
        Forward pass for cross attention.
        
        Args:
        - text_features (torch.Tensor): Text features for the cross attention.
        - image_features (torch.Tensor): Image features for the cross attention.

        Returns:
        - cross_attn_out (torch.Tensor): Output after applying cross attention.
        """
        
        queries = self.to_query(text_features[:-1])
        keys = self.to_key(image_features[:-1])
        values = self.to_value(image_features[:-1])

        cross_attn_out = self.attend(queries, keys, values)
        
        return cross_attn_out





