import torch
from cross_attn.main import MultiModalCrossAttn


# Parameters
BATCH_SIZE = 32  # number of samples in a batch
SEQ_LEN = 100  # sequence length of both text and image features
DIM = 512  # dimension of the features
HEADS = 8  # number of attention heads
DROPOUT = 0.1  # dropout probability

# Initialize the module
cross_attn_module = MultiModalCrossAttn(dim=DIM, heads=HEADS, dropout=DROPOUT)

# Generate random inputs
text_features = torch.randn(BATCH_SIZE, SEQ_LEN, DIM)
image_features = torch.randn(BATCH_SIZE, SEQ_LEN, DIM)

# Forward pass
output = cross_attn_module(text_features, image_features)

print(output.shape)  # Expected to be [BATCH_SIZE, SEQ_LEN - 1, DIM]
