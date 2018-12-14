import torch
from torch import nn
import torch.nn.functional as F
from MultiheadAttention import MultiheadAttention
import math
from PositionalEmbedding import SinusoidalPositionalEmbedding
from config import device

class Encoder(nn.Module):
    def __init__(self, args, num_layers, embedding_weight = None, vocab_size = None, use_position_emb = False):
        super(Encoder, self).__init__()
        self.dropout = nn.Dropout(args['dropout'])
        if embedding_weight is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding_weight, freeze = False)
        else:
            self.embedding = nn.Embedding(vocab_size,args['encoder_embed_dim'])
        ############# we need an embedding position matrix ##################
        ############# we need an embed scale matrix #########################
        if use_position_emb:
            self.embed_positions = SinusoidalPositionalEmbedding(args['encoder_embed_dim'])
        else:
            self.embed_positions = None
        #embed_dim = embedding_weight.shape
        #self.max_source_positions = args.max_source_positions
        self.embed_scale = math.sqrt(args['encoder_embed_dim'])
        # self.embed_positions = PositionalEmbedding(
        #     args.max_source_positions, embed_dim, self.padding_idx,
        #     left_pad=left_pad,
        #     learned=args.encoder_learned_pos,
        # ) if not args.no_token_positional_embeddings else None
        
        self.num_layers = num_layers
        self.compress_hidden_type = args['compress_hidden_type']

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(args)
            for i in range(args['encoder_layers'])
        ])
        
        self.normalize = args['output_normalize']
        if self.normalize:
            self.layer_norm = nn.LayerNorm(args['encoder_embed_dim'])
            

    
    def forward(self, src_tokens, src_lengths):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`

        Returns:
            tuple:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(batch, src_len, embed_dim)`
                
        """
        # embed tokens and positions
        x = self.embed_scale * self.embedding(src_tokens)
        if self.embed_positions is not None:
            x += self.embed_positions(src_lengths)
        x = self.dropout(x)
        
        batch_size, src_length, hidden_size = x.size()
        
        # encoder layers
        for layer in self.layers:
            x = layer(x, src_lengths)

        if self.normalize:
            x = self.layer_norm(x)
        
        hidden, cell = None,None
        if self.compress_hidden_type == 'sum':
            hidden = torch.sum(x, dim=1).unsqueeze(1).expand(batch_size, self.num_layers, hidden_size)
            cell = torch.sum(x, dim=1).unsqueeze(1).expand(batch_size, self.num_layers, hidden_size) 
            return x, hidden.transpose(0,1).contiguous(), cell.transpose(0,1).contiguous()
        elif self.compress_hidden_type == 'random':
            hidden = torch.randn(self.num_layers, batch_size, hidden_size, device=device)
            cell = torch.randn(self.num_layers, batch_size, hidden_size, device=device)
            return x, hidden, cell
        else:
            print('compress hidden type Error')
            return None,None,None

        #return {
        #    'encoder_out': x,  # B x T x C
        #    'encoder_padding_mask': src_lengths,  # B x T
        #}

class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super(TransformerEncoderLayer, self).__init__()
        self.embed_dim = args['encoder_embed_dim']
        self.self_attn = MultiheadAttention(
            self.embed_dim, 
            args['encoder_attention_heads'],
            dropout=args['attention_dropout']
        )
        self.dropout = args['dropout']
        self.normalize_before = args['encoder_normalize_before']
        self.fc1 = Linear(self.embed_dim, args['encoder_hidden_dim'])
        self.fc2 = Linear(args['encoder_hidden_dim'], self.embed_dim)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(self.embed_dim) for i in range(2)])
        self.encoder_dropout1 = nn.Dropout(self.dropout)
        self.encoder_dropout2 = nn.Dropout(self.dropout)
        self.encoder_dropout3 = nn.Dropout(self.dropout)

    def forward(self, x, src_lengths):
        """
        Args:
            x (Tensor): input to the layer of shape `(batch, src_len, embed_dim)`
        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        #batch_size , tgt_len, embed_dim = x.size()
        residual = x
        x = self.layer_norm(0, x, before=True)

        x, _ = self.self_attn(query=x, key=x, value=x, src_true_len=src_lengths, need_weights=False)
        x = self.encoder_dropout1(x)
        
        assert x.size() == residual.size()
        x = residual + x
        x = self.layer_norm(0, x, after=True)

        residual = x
        x = self.layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = self.encoder_dropout2(x)
        x = self.fc2(x)
        x = self.encoder_dropout3(x)
        x = residual + x
        x = self.layer_norm(1, x, after=True)
        return x


    def layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x


def LayerNorm(embedding_dim):
    m = nn.LayerNorm(embedding_dim)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m
