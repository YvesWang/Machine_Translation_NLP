import numpy as np
from config import *


def preposs_toekn(tokens):
    return [token for token in tokens if token != '']

def read_embedding(fasttest_home = './wiki-news-300d-1M.vec', words_to_load = 50000):
    words_ft = {}
    idx2words_ft = {}
    
    words_ft['$PAD$'] = PAD_token
    idx2words_ft[PAD_token] = '$PAD$'
    words_ft['$SOS$'] = SOS_token
    idx2words_ft[SOS_token] = '$SOS$'
    words_ft['$EOS$'] = EOS_token
    idx2words_ft[EOS_token] = '$EOS$'
    words_ft['$UNK$'] = UNK_token
    idx2words_ft[UNK_token] = '$UNK$'
    
    with open(fasttest_home) as f:
        loaded_embeddings_ft = np.zeros((words_to_load, 300)) 
        ordered_words_ft = []
        f.readline()
        for i, line in enumerate(f):
            i = i+4
            if i >= words_to_load: 
                break
            s = line.split()
            try:
                loaded_embeddings_ft[i, :] = np.asarray(s[1:])
            except:
                print('')
                
            words_ft[s[0]] = i
            idx2words_ft[i] = s[0]
            ordered_words_ft.append(s[0])
    
    return words_ft,idx2words_ft,loaded_embeddings_ft

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {"PAD" : PAD_token,"$SOS$" : SOS_token, "$EOS$" : EOS_token, "$UNK$" : UNK_token}
        self.word2count = {"PAD" : 0, "$SOS$" : 0, "$EOS$" : 0, "$UNK$" : 0}
        self.index2word = {PAD_token: "PAD", SOS_token: "$SOS$", EOS_token: "$EOS$", UNK_token: "$UNK$"}
        self.n_words = 3  # Count SOS and EOS
        self.embedding_matrix = None

#     def addSentence(self, sentence):
#         for word in sentence.split(' '):
#             self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
    
    def load_embedding(self,address,words_to_load):
        self.word2index, self.index2word,self.embedding_matrix = read_embedding(address,words_to_load)
        
def text2index(data,word2index):
    indexdata = []
    for line in data:
        indexdata.append([word2index[c] if c in word2index.keys() else UNK_token  for c in line])
        indexdata[-1].append(EOS_token)
    print('finish')
    return indexdata


def preparelang(name,data):
    lang = Lang(name)
    for line in data:
        for word in line:
            lang.addWord(word)
    return lang