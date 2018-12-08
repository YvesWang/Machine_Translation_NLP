import torch
from torch import nn
import torch.nn.functional as F
from config import device

class MultiheadAttention(nn.Module):
    """
    Multi-headed attention.
    """

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True):
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

    def forward(self, query, key, value, src_true_len=None, query_mask=False, need_weights=True):
        """
        Args:
            query (Tensor): input to the layer of shape `(batch, tgt_len, embed_dim)`
            key,value (Tensor): input to the layer of shape `(batch, src_len, embed_dim)`
            src_true_len (Tensor): true len for mask padding node
            query_mask: mask future information
            need_weight: return attention weight or not
        Returns:
            encoded output of shape `(batch, tgt_len, embed_dim)`
        """
        
        batch_size , tgt_len, embed_dim = query.size()
        src_len = key.size(1)
        assert embed_dim == self.embed_dim
        assert key.size() == value.size()

        Q = self.Linear_Q(query).view(batch_size,tgt_len,self.num_heads,self.head_dim).transpose(1,2).contiguous().view(batch_size*self.num_heads,tgt_len,self.head_dim) #  bsz*n tgt head
        K = self.Linear_K(key).view(batch_size,src_len,self.num_heads,self.head_dim).transpose(1,2).contiguous().view(batch_size*self.num_heads,src_len,self.head_dim)  #  bsz*n src head
        V = self.Linear_V(value).view(batch_size,src_len,self.num_heads,self.head_dim).transpose(1,2).contiguous().view(batch_size*self.num_heads,src_len,self.head_dim) #  bsz*n src head

        attn_weights = self.scaling * torch.bmm(Q, K.transpose(1, 2)) #  bsz*n tgt src
        assert attn_weights.size() == (batch_size * self.num_heads, tgt_len, src_len)

        #### Mask here #########################
        ############################################
        ######### Here we only mask src is padding ###################################
        if src_true_len is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(batch_size, self.num_heads, tgt_len, src_len)
            mask = sequence_mask(src_true_len).unsqueeze(1).unsqueeze(2).expand(batch_size, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.float().masked_fill(
                1-mask.to(device),float('-inf')) # FP16 support: cast to float and back
            attn_weights = attn_weights.view(batch_size * self.num_heads, tgt_len, src_len)
        ############################################
        ############## Here mask for inference ##################################
        if query_mask:
            attn_weights = attn_weights.view(batch_size, self.num_heads, tgt_len, src_len)
            assert tgt_len == src_len
            mask = sequence_mask(torch.arange(tgt_len) + 1).unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.float().masked_fill(
                1-mask.to(device),float('-inf')) # FP16 support: cast to float and back
            attn_weights = attn_weights.view(batch_size * self.num_heads, tgt_len, src_len)
        

        scores_normalized = F.softmax(attn_weights, dim=-1)
        scores_normalized = self.attn_dropout(scores_normalized)
        
        attn = torch.bmm(scores_normalized, V) # bsz*n tgt head
        assert attn.size() == (batch_size * self.num_heads, tgt_len, self.head_dim)

        attn = attn.contiguous().view(batch_size , self.num_heads, tgt_len, self.head_dim).transpose(1,2).contiguous().view(batch_size, tgt_len, embed_dim) # bsz, tgt, embed
        attn = self.out_proj(attn)
        
        if need_weights:
            # average attention weights over heads
            attn_weights = attn.view(batch_size, self.num_heads, tgt_len, src_len)
            attn_weights = attn.sum(dim=1) / self.num_heads
        else:
            attn_weights = None

        return attn, attn_weights

def sequence_mask(lengths):
    batch_size = lengths.numel()
    max_len = lengths.max()
    return (torch.arange(0, max_len).type_as(lengths).repeat(batch_size,1).lt(lengths.unsqueeze(1)))




        