src_vocab_size = 20000
tgt_vocab_size = 10000

tgt_max_length = 60
max_src_len_dataloader, max_tgt_len_dataloader = 60, 60

args = dict(
    output_vocab_size = 10004,
    encoder_embed_dim = 256,
    decoder_embed_dim = 256,
    encoder_attention_heads = 8,
    decoder_attention_heads = 8,
    dropout = 0.1,                      #   Encoder Decoder: dropout
    attention_dropout = 0.1,            #   Attention: attention_dropout
    encoder_normalize_before = True, 
    decoder_normalize_before = True,
    output_normalize = True,
    encoder_hidden_dim = 1024,           #   hidden dimension between to fc layers
    decoder_hidden_dim = 1024,           #   hidden dimension between to fc layers
    encoder_layers = 4,
    decoder_layers = 6,
    decoder_need_atten_weight = False
    )




####################### encoder_normalize_before ###################################################
#    In the original paper each operation (multi-head attention or FFN) is
#    postprocessed with: `dropout -> add residual -> layernorm`. In the
#    tensor2tensor code they suggest that learning is more robust when
#    preprocessing each layer with layernorm and postprocessing with:
#    `dropout -> add residual`. We default to the approach in the paper, but the
#    tensor2tensor approach can be enabled by setting
#    *args.encoder_normalize_before* to ``True``.
#############################################################################

