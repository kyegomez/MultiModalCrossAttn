import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiModalCrossAttention(nn.Module):
    def __init__(self, dim):
        super(MultiModalCrossAttention, self).__init__()

        # Projection layers for text and image
        self.Wq_text = nn.Linear(dim, dim)
        self.Wk_text = nn.Linear(dim, dim)
        self.Wv_text = nn.Linear(dim, dim)

        self.Wq_image = nn.Linear(dim, dim)
        self.Wk_image = nn.Linear(dim, dim)
        self.Wv_image = nn.Linear(dim, dim)

    def forward(self, Q_text, K_text, V_text, Q_image, K_image, V_image):
        # Project input representations
        Qcross_text = self.Wq_text(Q_text)
        Kcross_image = self.Wk_image(K_image)
        Vcross_image = self.Wv_image(V_image)

        Qcross_image = self.Wq_image(Q_image)
        Kcross_text = self.Wk_text(K_text)
        Vcross_text = self.Wv_text(V_text)

        # Compute cross-attention weights using scaled dot product
        dk = Q_text.size(-1) ** 0.5
        AttnWeights_text_to_image = F.softmax(
            torch.bmm(Qcross_text, Kcross_image.transpose(-2, -1)) / dk, dim=-1
        )
        AttnWeights_image_to_text = F.softmax(
            torch.bmm(Qcross_image, Kcross_text.transpose(-2, -1)) / dk, dim=-1
        )

        # Obtain the cross attention output representations
        Hcross_text = torch.bmm(AttnWeights_text_to_image, Vcross_image)
        Hcross_image = torch.bmm(AttnWeights_image_to_text, Vcross_text)

        return Hcross_text, Hcross_image


# import torch
# import torch.nn as nn
# from cross_attn.attend import Attention


# class MultiModalCrossAttn(nn.Module):
#     """
#     Cross attention module for multi-modal (text and image) attention.
#     """
#     def __init__(
#         self,
#         dim: int,
#         heads: int,
#         dropout: float = 0.1,
#         causal: bool = True,
#         flash: bool = True
#     ):
#         super().__init__()

#         self.to_query = nn.Linear(dim, dim * heads)
#         self.to_key = nn.Linear(dim, dim * heads)
#         self.to_value = nn.Linear(dim, dim * heads)

#         self.attend = Attention(dropout, causal, flash)

#     def forward(self, text_features, image_features):
#         """
#         Forward pass for cross attention.

#         Args:
#         - text_features (torch.Tensor): Text features for the cross attention.
#         - image_features (torch.Tensor): Image features for the cross attention.

#         Returns:
#         - cross_attn_out (torch.Tensor): Output after applying cross attention.
#         """

#         queries = self.to_query(text_features[:-1])
#         print(f"Queries embeds: {queries} and shape {queries.shape}")

#         keys = self.to_key(image_features[:-1])
#         print(f"Keys embeds: {keys} and shape {keys.shape}")

#         values = self.to_value(image_features[:-1])
#         print(f"Values embeds: {values} and shape {values.shape}")

#         cross_attn_out = self.attend(queries, keys, values, mask=None)
#         print(f"Cross attn out: {cross_attn_out} and shape {cross_attn_out.shape}")

#         return cross_attn_out


import torch
import torch.nn as nn
from cross_attn import Attention


class MultiModalCrossAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.0):
        super().__init__()

        self.dim_head = dim // heads

        # Projection layers for text and image
        self.text_to_image_projections = nn.ModuleList(
            [nn.Linear(dim, heads * self.dim_head) for _ in range(3)]
        )
        self.image_to_text_projections = nn.ModuleList(
            [nn.Linear(dim, heads * self.dim_head) for _ in range(3)]
        )

        # Attention mechanism
        self.text_to_image_attn = Attention(dropout=dropout)
        self.image_to_text_attn = Attention(dropout=dropout)

    def forward(self, Q_text, K_text, V_text, Q_image, K_image, V_image):
        Q_text_to_img, K_text_to_img, V_text_to_img = [
            proj(x)
            for proj, x in zip(
                self.text_to_image_projections, [Q_text, K_image, V_image]
            )
        ]
        Q_img_to_text, K_img_to_text, V_img_to_text = [
            proj(x)
            for proj, x in zip(
                self.image_to_text_projections, [Q_image, K_text, V_text]
            )
        ]

        Hcross_text = self.text_to_image_attn(
            Q_text_to_img, K_text_to_img, V_text_to_img
        )
        Hcross_image = self.image_to_text_attn(
            Q_img_to_text, K_img_to_text, V_img_to_text
        )

        return Hcross_text, Hcross_image


# Usage Example
dim = 512
model = MultiModalCrossAttention(dim=dim)

# Dummy data for text and image
B, L_text, L_image, D = 8, 100, 80, dim
Q_text, K_text, V_text = (
    torch.randn(B, L_text, D),
    torch.randn(B, L_text, D),
    torch.randn(B, L_text, D),
)
Q_image, K_image, V_image = (
    torch.randn(B, L_image, D),
    torch.randn(B, L_image, D),
    torch.randn(B, L_image, D),
)

Hcross_text, Hcross_image = model(Q_text, K_text, V_text, Q_image, K_image, V_image)
