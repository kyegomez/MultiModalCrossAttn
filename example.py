import torch
from cross_attn.main import MultiModalCrossAttention

# Test the MultiModalCrossAttention module
dim = 512  # For example
num_heads = 8

cross_attn = MultiModalCrossAttention(dim, num_heads)

Hllm_sample = torch.randn(32, dim, dim)  # Batch size = 32, Sequence length = 10
Himg_sample = torch.randn(32, dim, dim)

output = cross_attn(Hllm_sample, Himg_sample)
print(output)

print(output.shape)  # Expected: [32, 10, 512]
