import math
import torch
import torch.nn as nn
from MultiheadAttention import sequence_mask
from config import device

class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.
    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    """

    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    @staticmethod
#     def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
#         """Build sinusoidal embeddings.
#         This matches the implementation in tensor2tensor, but differs slightly
#         from the description in Section 3.5 of "Attention Is All You Need".
#         """
#         half_dim = embedding_dim // 2
#         emb = math.log(10000) / (half_dim - 1)
#         emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
#         emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
#         emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
#         if embedding_dim % 2 == 1:
#             # zero pad
#             emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
#         if padding_idx is not None:
#             emb[padding_idx, :] = 0
#         return emb

    
    def get_embedding(input_length, embedding_dim):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        input_length = input_length.cpu() #bz
        batch_size = input_length.size(0) 
        mask = sequence_mask(input_length) #bz, sent_len_max
        sent_len_max = mask.size(1)
        
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb) #half_dim
        emb = torch.mm(torch.arange(sent_len_max, dtype=torch.float).unsqueeze(1), emb.unsqueeze(0)) 
        #print(emb.size())
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(sent_len_max, -1)
        #print(emb.size())
        
        emb = emb.unsqueeze(0).expand(batch_size, sent_len_max, embedding_dim)
        mask = mask.unsqueeze(-1).expand(batch_size, sent_len_max, embedding_dim)
        emb = emb.masked_fill(1-mask, float(0))
        return emb.to(device).detach()
    
    def forward(self, input_length):
        emb = SinusoidalPositionalEmbedding.get_embedding(input_length, self.embedding_dim)
        return emb


#     def forward(self, src_tokens, timestep=None):
#         """Input is expected to be of size [bsz x seqlen]."""
#         bsz, seq_len = src_tokens.size()
#         max_pos = self.padding_idx + 1 + seq_len
#         if self.weights is None or max_pos > self.weights.size(0):
#             # recompute/expand embeddings if needed
#             self.weights = SinusoidalPositionalEmbedding.get_embedding(
#                 max_pos,
#                 self.embedding_dim,
#                 self.padding_idx,
#             )
#         self.weights = self.weights.type_as(self._float_tensor)
#         return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()

    
