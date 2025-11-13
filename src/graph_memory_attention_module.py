
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphMemoryAttention(nn.Module):
    """
    Adaptive Attention over Graph Memory.
    Applies attention between current embeddings and stored temporal graph memory.
    """

    def __init__(self, embed_dim, heads=1):
        super(GraphMemoryAttention, self).__init__()
        self.embed_dim = embed_dim
        self.heads = heads
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=heads, batch_first=True)

    def forward(self, current_embedding, memory):
        """
        Args:
            current_embedding: Tensor of shape [N, D] – embedding at current time step
            memory: Tensor of shape [T_mem, N, D] – graph memory of previous time steps

        Returns:
            output: Tensor of shape [N, D] – attention-enhanced embedding
        """
        if memory.shape[0] == 0:
            return current_embedding  # no past memory available

        query = current_embedding.unsqueeze(1)     # [N, 1, D]
        key = memory.permute(1, 0, 2)              # [N, T, D]
        value = key                                # same as key

        # Apply attention: output → [N, 1, D]
        out, _ = self.attn(query, key, value)

        # Squeeze time dimension → [N, D]
        return out.squeeze(1)
