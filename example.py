import torch
from cross_attn.main import MultiModalCrossAttn

dim = 512
dk = 512
batch_size = 16
seq_len = 1000


#random tensors for h1llm and himg
hllm_random = torch.randn(batch_size, seq_len, dim)
himg_random = torch.randn(batch_size, seq_len, dim)

cros_attn = MultiModalCrossAttn(dim, dk)

output = cros_attn(hllm_random, himg_random)
print(output)