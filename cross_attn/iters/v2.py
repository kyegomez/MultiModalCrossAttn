import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiModalCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiModalCrossAttention, self).__init__()
        
        self.num_heads = num_heads
        self.d_model = d_model
        self.dk = d_model // num_heads
        
        # Query, Key, Value projection layers for Timg -> Tllm
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        
        # Query, Key, Value projection layers for Tllm -> Timg (reverse)
        self.Wq_reverse = nn.Linear(d_model, d_model)
        self.Wk_reverse = nn.Linear(d_model, d_model)
        self.Wv_reverse = nn.Linear(d_model, d_model)

        # Output linear layer after attention computation
        self.linear_out = nn.Linear(2*d_model, d_model)

    def forward(self, Hllm, Himg):
        """
        Hllm: Hidden states from Tllm
        Himg: Hidden states from Timg
        """
        
        # Timg -> Tllm
        Qcross = self.Wq(Hllm)
        Kcross = self.Wk(Himg)
        Vcross = self.Wv(Himg)
        attn_weights = F.softmax(Qcross @ Kcross.transpose(-2, -1) / torch.sqrt(torch.tensor(self.dk).float()), dim=-1)
        Hcross = attn_weights @ Vcross
        
        # Tllm -> Timg (Symmetric process)
        Qcross_reverse = self.Wq_reverse(Himg)
        Kcross_reverse = self.Wk_reverse(Hllm)
        Vcross_reverse = self.Wv_reverse(Hllm)
        attn_weights_reverse = F.softmax(Qcross_reverse @ Kcross_reverse.transpose(-2, -1) / torch.sqrt(torch.tensor(self.dk).float()), dim=-1)
        Hcross_reverse = attn_weights_reverse @ Vcross_reverse
        
        # Concatenate the results
        output = torch.cat((Hcross, Hcross_reverse), dim=-1)
        
        # Pass through linear layer
        output = self.linear_out(output)
        
        return output

# Test the MultiModalCrossAttention module
d_model = 512  # For example
num_heads = 8
cross_attn = MultiModalCrossAttention(d_model, num_heads)
Hllm_sample = torch.randn(32, 10, d_model)  # Batch size = 32, Sequence length = 10
Himg_sample = torch.randn(32, 10, d_model)
output = cross_attn(Hllm_sample, Himg_sample)

print(output.shape)  # Expected: [32, 10, 512]