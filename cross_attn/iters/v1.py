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
        AttnWeights_text_to_image = F.softmax(torch.bmm(Qcross_text, Kcross_image.transpose(-2, -1)) / dk, dim=-1)
        AttnWeights_image_to_text = F.softmax(torch.bmm(Qcross_image, Kcross_text.transpose(-2, -1)) / dk, dim=-1)

        # Obtain the cross attention output representations
        Hcross_text = torch.bmm(AttnWeights_text_to_image, Vcross_image)
        Hcross_image = torch.bmm(AttnWeights_image_to_text, Vcross_text)

        return Hcross_text, Hcross_image
