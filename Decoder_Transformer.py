import torch
from torch import nn
import torch.nn.functional as F
from MultiheadAttention import MultiheadAttention
import math

class Decoder(nn.Module):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs.
            Default: ``False``
    """

    def __init__(self, args, embedding_weight):
        super(Decoder, self).__init__()
        ############# we need an embedding position matrix ##################
        ############# we need an embed scale matrix #########################
        self.embedding = nn.Embedding.from_pretrained(embedding_weight, freeze = False)
        self.embed_positions = None
        # self.embed_positions = PositionalEmbedding(
        #     args.max_target_positions, embed_dim, padding_idx,
        #     left_pad=left_pad,
        #     learned=args.decoder_learned_pos,
        # ) if not args.no_token_positional_embeddings else None

        embed_dim = args.decoder_embed_dim
        self.dropout = nn.Dropout(args['dropout'])
        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim) 
        
        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerDecoderLayer(args)
            for _ in range(args['decoder_layers'])
        ])

        self.project_out_dim = Linear(embed_dim, args['output_vocab_size'], bias=False) 
        self.logsoftmax = nn.LogSoftmax(dim=2)
        
        self.normalize = args['decoder_normalize_before']
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)
        

    def forward(self, output_tokens, src_lengths, encoder_out=None):
        """
        Args:
            output_tokens (LongTensor): decoder input of shape
                `(batch, tgt_len)`, for input feeding/teacher forcing
            src_lengths (Tensor): source input for masking padding node
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention
        Returns:
            tuple:
                - the last decoder layer's output of shape `(batch, tgt_len,
                  vocab)`
                - the last decoder layer's attention weights of shape `(batch,
                  tgt_len, src_len)`
        """


        # embed tokens and positions
        x = self.embed_scale * self.embedding(output_tokens)    # bsz tgt emb
        if self.embed_positions is not None:
            x += self.embed_positions(output_tokens)
        x = self.dropout(x)

        inner_states = [x]
        attn_states = []
        # decoder layers
        for layer in self.layers:
            x, attn = layer(
                x,
                encoder_out,
                src_lengths= src_lengths
            )
            inner_states.append(x)
            attn_states.append(atten)

        if self.normalize:
            x = self.layer_norm(x)

        logits = self.project_out_dim(x) # bsz tgt vocab
        output = self.logsoftmax(logits)
        return output, {'attn': attn_states, 'inner_states': inner_states}
    



class TransformerDecoderLayer(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs.
            Default: ``False``
    """

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args['decoder_embed_dim']
        self.normalize_before = args['decoder_normalize_before']
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout = args['dropout']

        self.self_attn = MultiheadAttention(
            self.embed_dim, args['decoder_attention_heads'],
            dropout=args['attention_dropout'],
        )

        self.encoder_attn = MultiheadAttention(
            self.embed_dim, args['decoder_attention_heads'],
            dropout=args['attention_dropout'],
        )
        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)

        self.fc1 = Linear(self.embed_dim, args.decoder_ffn_embed_dim)
        self.fc2 = Linear(args.decoder_ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = LayerNorm(self.embed_dim)
        self.need_attn = args['decoder_need_atten_weight']
        self.decoder_dropout1 = nn.Dropout(self.dropout)
        self.decoder_dropout2 = nn.Dropout(self.dropout)
        self.decoder_dropout3 = nn.Dropout(self.dropout)
        self.decoder_dropout4 = nn.Dropout(self.dropout)


    def forward(self, x, encoder_out, src_lengths):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)

        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            query_mask=True,
            need_weights=False,
        )
        x = self.decoder_dropout1(x)
        x = residual + x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, before=True)
            
        x, attn = self.encoder_attn(
            query=x,
            key=encoder_out,
            value=encoder_out,
            src_true_len=src_lengths,
            need_weights= self.need_attn
        )
        x = self.decoder_dropout2(x)
        x = residual + x
        x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = F.relu(self.fc1(x))
        x = self.decoder_dropout3(x)
        x = self.fc2(x)
        x = self.decoder_dropout4(x)
        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)

        return x, attn


    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn

def LayerNorm(embedding_dim):
    m = nn.LayerNorm(embedding_dim)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m
