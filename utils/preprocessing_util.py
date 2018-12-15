import numpy as np
from config import *
from collections import Counter

def preposs_toekn(tokens):
    return [token for token in tokens if token != '']

def load_emb_vectors(fasttest_home):
    max_num_load = 500000
    words_dict = {}
    with open(fasttest_home) as f:
        for num_row, line in enumerate(f):
            if num_row >= max_num_load:
                break
            s = line.split()
            words_dict[s[0]] = np.asarray(s[1:])
    return words_dict
# def read_embedding(fasttest_home = './wiki-news-300d-1M.vec', words_to_load = 10000):
#     words_ft = {}
#     idx2words_ft = {}
    
#     words_ft['$PAD$'] = PAD_token
#     idx2words_ft[PAD_token] = '$PAD$'
#     words_ft['$SOS$'] = SOS_token
#     idx2words_ft[SOS_token] = '$SOS$'
#     words_ft['$EOS$'] = EOS_token
#     idx2words_ft[EOS_token] = '$EOS$'
#     words_ft['$UNK$'] = UNK_token
#     idx2words_ft[UNK_token] = '$UNK$'
    
#     with open(fasttest_home) as f:
#         loaded_embeddings_ft = np.zeros((words_to_load, 300)) 
#         ordered_words_ft = []
#         # f.readline()
#         for i, line in enumerate(f):
#             i = i+4
#             if i >= words_to_load: 
#                 break
#             s = line.split()
#             try:
#                 loaded_embeddings_ft[i, :] = np.asarray(s[1:])
#             except:
#                 print('')
                
#             words_ft[s[0]] = i
#             idx2words_ft[i] = s[0]
#             ordered_words_ft.append(s[0])
    
#     return words_ft,idx2words_ft,loaded_embeddings_ft.astype(np.float32)

class Lang:
    def __init__(self, name, max_vocab_size, emb_pretrained_add):
        self.name = name
        self.word2index = None #{"$PAD$": PAD_token, "$SOS$": SOS_token, "$EOS$": EOS_token, "$UNK$": UNK_token}
        #self.word2count = None #{"$PAD$": 0, "$SOS$" : 0, "$EOS$": 0, "$UNK$": 0}
        self.index2word = None #{PAD_token: "$PAD$", SOS_token: "$SOS$", EOS_token: "$EOS$", UNK_token: "$UNK$"}
        self.max_vocab_size = max_vocab_size  # Count SOS and EOS
        self.vocab_size = None
        self.emb_pretrained_add = emb_pretrained_add
        self.embedding_matrix = None

    def build_vocab(self, train_data):
        all_tokens = []
        for sent in train_data:
            all_tokens.extend(sent)
        token_counter = Counter(all_tokens)
        print('The number of unique tokens totally in train data: ', len(token_counter))
        vocab, count = zip(*token_counter.most_common(self.max_vocab_size))
        self.index2word = vocab_prefix + list(vocab)
        word2index = dict(zip(vocab, range(len(vocab_prefix),len(vocab_prefix)+len(vocab)))) 
        for idx, token in enumerate(vocab_prefix):
            word2index[token] = idx
        self.word2index = word2index
        return None 

    def build_emb_weight(self):
        words_emb_dict = load_emb_vectors(self.emb_pretrained_add)
        vocab_size = len(self.index2word)
        self.vocab_size = vocab_size
        emb_weight = np.zeros([vocab_size, 300])
        for i in range(len(vocab_prefix), vocab_size):
            emb = words_emb_dict.get(self.index2word[i], None)
            if emb is not None:
                try:
                    emb_weight[i] = emb
                except:
                    pass
                    #print(len(emb), self.index2word[i], emb)
        self.embedding_matrix = emb_weight
        return None

    # def addSentence(self, sentence):
    #     for word in sentence.split(' '):
    #         self.addWord(word)

    # def addWord(self, word):
    #     if word not in self.word2index:
    #         self.word2index[word] = self.n_words
    #         self.word2count[word] = 1
    #         self.index2word[self.n_words] = word
    #         self.n_words += 1
    #     else:
    #         self.word2count[word] += 1
    
    # def load_embedding(self, address, words_to_load):
    #     self.word2index, self.index2word, self.embedding_matrix = read_embedding(address, words_to_load)
        
def text2index(data, word2index):
    indexdata = []
    for line in data:
        indexdata.append([word2index[c] if c in word2index.keys() else UNK_token for c in line])
        #indexdata[-1].append(EOS_token)
    print('finish')
    return indexdata

def construct_Lang(name, max_vocab_size, emb_pretrained_add, train_data):
    lang = Lang(name, max_vocab_size, emb_pretrained_add)
    lang.build_vocab(train_data)
    lang.build_emb_weight()
    return lang

# def preparelang(name, data):
#     lang = Lang(name)
#     for line in data:
#         for word in line:
#             lang.addWord(word)
#     return lang
