import torch
import torch.nn as nn
import torch.nn.functional as F


class RecombinationAttention(nn.Module):
    """
    Implements: w_t = softmax( softmax(QK^T / sqrt(Dk)) * (Ft W_V) )
    where Ft is the input of shape [B, N, D].
    """

    def __init__(self, embed_dim, key_dim):
        super().__init__()
        self.W_Q = nn.Linear(embed_dim, key_dim, bias=False)
        self.W_K = nn.Linear(embed_dim, key_dim, bias=False)
        self.W_V = nn.Linear(embed_dim, 1, bias=False)
        self.scale = key_dim**0.5

    def forward(self, Ft):
        """
        Ft: Tensor of shape [B, N, D]
        returns: w_t of shape [B, N] — a probability distribution
        """

        Q = self.W_Q(Ft)
        K = self.W_K(Ft)

        attn_logits = torch.matmul(Q, K.transpose(1, 2)) / self.scale
        A = F.softmax(attn_logits, dim=-1)

        V = self.W_V(Ft).squeeze(-1)

        w_raw = torch.matmul(A, V.unsqueeze(-1)).squeeze(-1)

        w_t = F.softmax(w_raw, dim=-1)

        return w_t
