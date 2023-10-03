[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# MultiModalCrossAttn
The open source implementation of the cross attention mechanism from the paper: "JOINTLY TRAINING LARGE AUTOREGRESSIVE MULTIMODAL MODELS"


[Paper Link](https://arxiv.org/pdf/2309.15564.pdf)

# Appreciation
* Lucidrains
* Agorians



# Install
`pip install cross-attn`

# Usage
```python
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

```



# License
MIT

# Citations
```
@misc{2309.15564,
Author = {Emanuele Aiello and Lili Yu and Yixin Nie and Armen Aghajanyan and Barlas Oguz},
Title = {Jointly Training Large Autoregressive Multimodal Models},
Year = {2023},
Eprint = {arXiv:2309.15564},
}
```