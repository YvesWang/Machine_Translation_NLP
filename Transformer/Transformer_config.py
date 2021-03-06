src_vocab_size = 26109 #47127 #26109
tgt_vocab_size = 24418 #31553 #24418

tgt_max_length = 71 #72 #71
max_src_len_dataloader, max_tgt_len_dataloader = 72, 71 #67, 72#72, 71 

args = dict(
    output_vocab_size = 24422, #31557, #24422,
    encoder_embed_dim = 512,
    decoder_embed_dim = 512,
    encoder_attention_heads = 4,
    decoder_attention_heads = 4,
    dropout = 0.3,                      #   Encoder Decoder: dropout
    attention_dropout = 0.3,            #   Attention: attention_dropout
    encoder_normalize_before = False, 
    decoder_normalize_before = False,
    output_normalize = True,
    encoder_hidden_dim = 1024,           #   hidden dimension between to fc layers
    decoder_hidden_dim = 1024,           #   hidden dimension between to fc layers
    encoder_layers = 2,
    decoder_layers = 2,
    decoder_need_atten_weight = False
    )

model_path = 'trans_models/vi-en-512-4-dropout3_validation/'




####################### encoder_normalize_before ###################################################
#    In the original paper each operation (multi-head attention or FFN) is
#    postprocessed with: `dropout -> add residual -> layernorm`. In the
#    tensor2tensor code they suggest that learning is more robust when
#    preprocessing each layer with layernorm and postprocessing with:
#    `dropout -> add residual`. We default to the approach in the paper, but the
#    tensor2tensor approach can be enabled by setting
#    *args.encoder_normalize_before* to ``True``.
#############################################################################

