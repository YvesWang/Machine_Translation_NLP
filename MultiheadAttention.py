import torch
from torch import nn
import torch.nn.functional as F

class MultiheadAttention(nn.Module):
    """
    Multi-headed attention.
    """

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_zero_attn=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        if bias:
            self.Linear_Q = nn.Linear(embed_dim, embed_dim, bias = True)
            self.Linear_K = nn.Linear(embed_dim, embed_dim, bias = True)
            self.Linear_V = nn.Linear(embed_dim, embed_dim, bias = True)
        else:
            self.Linear_Q = nn.Linear(embed_dim, embed_dim, bias = False)
            self.Linear_K = nn.Linear(embed_dim, embed_dim, bias = False)
            self.Linear_V = nn.Linear(embed_dim, embed_dim, bias = False)

        self.attn_dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.add_zero_attn = add_zero_attn
        self.reset_parameters()

    def forward(self, query, key, value, key_padding_mask=None, incremental_state=None,
                need_weights=True, static_kv=False, attn_mask=None):
        
        batch_size , tgt_len, embed_dim = query.size()
        src_len = key.size(1)
        assert embed_dim == self.embed_dim
        assert key.size() == value.size()

        Q = self.Linear_Q(query.view(batch_size * tgt_len, embed_dim)).view(batch_size*self.num_heads,tgt_len,self.num_heads)
        K = self.Linear_K(query.view(batch_size * src_len, embed_dim)).view(batch_size*self.num_heads,src_len,self.num_heads)
        V = self.Linear_V(query.view(batch_size * src_len, embed_dim)).view(batch_size*self.num_heads,src_len,self.num_heads)

        attn_weights = torch.bmm(Q, K.transpose(1, 2))
        assert attn_weights.size() == (batch_size * self.num_heads, tgt_len, src_len)

        #### Mask here #########################
        if attn_mask is not None:
            attn_weights += attn_mask.unsqueeze(0)
        ############################################
        ######### Here we only mask src is padding ###################################
        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(batch_size, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.float().masked_fill(
                sequence_mask(key_padding_mask).unsqueeze(1).unsqueeze(2),
                float('-inf'),
            ).type_as(attn_weights)  # FP16 support: cast to float and back
            attn_weights = attn_weights.view(batch_size * self.num_heads, tgt_len, src_len)
        ############################################

        scores_normalized = F.softmax(attn_weights, dim=-1)
        scores_normalized = self.attn_dropout(scores_normalized)

        attn = torch.bmm(attn_weights, V)
        assert attn.size() == (batch_size * self.num_heads, tgt_len, self.head_dim)

        attn = attn.contiguous().view(batch_size * tgt_len, embed_dim)
        attn = self.out_proj(attn).view(batch_size, tgt_len, embed_dim)

        if need_weights:
            # average attention weights over heads
            attn_weights = attn_weights.view(batch_size, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.sum(dim=1) / self.num_heads
        else:
            attn_weights = None

        return attn, attn_weights

def sequence_mask(lengths):
    batch_size = lengths.numel()
    max_len = lengths.max()
    return (torch.arange(0, max_len).type_as(lengths).repeat(batch_size,1).lt(lengths.unsqueeze(1)))




        