args = dict(
    output_vocab_size = 47131,
    encoder_embed_dim = 300,
    encoder_attention_heads = 6,
    dropout = 0.1,                      #   Encoder Decoder: dropout
    attention_dropout = 0.1,            #   Attention: attention_dropout
    encoder_normalize_before = False, 
    output_normalize = True,
    encoder_hidden_dim = 1000,           #   hidden dimension between to fc layers
    encoder_layers = 2,
    compress_hidden_type = 'random'  # 'sum'
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

