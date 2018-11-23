import torch

src_vocab_size = 50000
tgt_vocab_size = 50000

tgt_max_length = 50 

PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

address_book = dict(
    train_src = './iwsltzhen/iwslt-zh-en/train.tok.zh',
    train_tgt = './iwsltzhen/iwslt-zh-en/train.tok.en',
    val_src = './iwsltzhen/iwslt-zh-en/dev.tok.zh',
    val_tgt = './iwsltzhen/iwslt-zh-en/dev.tok.en',
    src_emb = '../embedding/wiki.zh.vec',
    tgt_emb = '../embedding/wiki.en.vec'
)
