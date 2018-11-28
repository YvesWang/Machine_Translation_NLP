import torch

src_vocab_size = 10000
tgt_vocab_size = 10000

tgt_max_length = 200

PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3

vocab_prefix = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']

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
    train_src = 'Data/news_token_zh_en.zh',
    train_tgt = 'Data/news_token_zh_en_nltk.en',
    val_src = 'Machine_Translation_NLP/iwsltzhen/iwslt-zh-en/dev.tok.zh',
    val_tgt = 'Machine_Translation_NLP/iwsltzhen/iwslt-zh-en/dev.tok.en',
    src_emb = 'embedding/wiki.zh.vec',
    tgt_emb = 'embedding/wiki.en.vec'
)

address_book = dict(
    train_src = 'Data/csrc_tokens',
    train_tgt = 'Data/ctgt_tokens',
    val_src = 'Machine_Translation_NLP/iwsltzhen/iwslt-zh-en/dev.tok.zh',
    val_tgt = 'Machine_Translation_NLP/iwsltzhen/iwslt-zh-en/dev.tok.en',
    src_emb = 'embedding/wiki.zh.vec',
    tgt_emb = 'embedding/wiki.en.vec'
)

address_book1 = dict(
    train_src = 'Data/src_tokens',
    train_tgt = 'Data/tgt_tokens',
    val_src = 'Data/src_tokens',
    val_tgt = 'Data/tgt_tokens',
    src_emb = 'embedding/wiki.zh.vec',
    tgt_emb = 'embedding/wiki.en.vec'
)

