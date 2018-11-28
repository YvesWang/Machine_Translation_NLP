import torch

src_vocab_size = 10000
tgt_vocab_size = 10000

tgt_max_length = 200

PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

address_book1 = dict(
    train_src = 'Machine_Translation_NLP/iwsltzhen/iwslt-zh-en/train_sortByEn.tok.zh',
    train_tgt = 'Machine_Translation_NLP/iwsltzhen/iwslt-zh-en/train_sortByEn.tok.en',
    val_src = 'Machine_Translation_NLP/iwsltzhen/iwslt-zh-en/dev.tok.zh',
    val_tgt = 'Machine_Translation_NLP/iwsltzhen/iwslt-zh-en/dev.tok.en',
    src_emb = 'embedding/wiki.zh.vec',
    tgt_emb = 'embedding/wiki.en.vec'
)

address_book1 = dict(
    train_src = 'Machine_Translation_NLP/iwsltzhen/iwslt-zh-en/train.tok.zh',
    train_tgt = 'Machine_Translation_NLP/iwsltzhen/iwslt-zh-en/train.tok.en',
    val_src = 'Machine_Translation_NLP/iwsltzhen/iwslt-zh-en/dev.tok.zh',
    val_tgt = 'Machine_Translation_NLP/iwsltzhen/iwslt-zh-en/dev.tok.en',
    src_emb = 'embedding/wiki.zh.vec',
    tgt_emb = 'embedding/wiki.en.vec'
)

address_book = dict(
    train_src = 'data/zh-en/train_sortByEn_10w.tok.zh',
    train_tgt = 'data/zh-en/train_sortByEn_10w.tok.en',
    val_src = 'Machine_Translation_NLP/iwsltzhen/iwslt-zh-en/dev.tok.zh',
    val_tgt = 'Machine_Translation_NLP/iwsltzhen/iwslt-zh-en/dev.tok.en',
    src_emb = 'embedding/wiki.zh.vec',
    tgt_emb = 'embedding/wiki.en.vec'
)

address_book1 = dict(
    train_src = 'Data/src_tokens',
    train_tgt = 'Data/tgt_tokens',
    val_src = 'Machine_Translation_NLP/iwsltzhen/iwslt-zh-en/dev.tok.zh',
    val_tgt = 'Machine_Translation_NLP/iwsltzhen/iwslt-zh-en/dev.tok.en',
    src_emb = 'embedding/wiki.zh.vec',
    tgt_emb = 'embedding/wiki.en.vec'
)
