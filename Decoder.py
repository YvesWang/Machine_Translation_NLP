import torch.nn as nn
import torch
import torch.nn.functional as F

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, vocab_size, embedding_weight, device):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.device = device
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, src_input, hidden):
        output = self.embedding(src_input)
        #print(output.size())
        output, hidden = self.gru(output, hidden)
        logits = self.out(output[:,0,:])
        output = self.softmax(logits)
        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)
    

class DecoderAttention(nn.Module):
    def __init__(self, hidden_size, vocab_size, embedding_weight, device):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.device = device
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, src_input, hidden):
        output = self.embedding(src_input)
        #print(output.size())
        output, hidden = self.gru(output, hidden)
        logits = self.out(output[:,0,:])
        output = self.softmax(logits)
        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)
    
    def score(self):
        #Multi-layer Perceptron
        a(q, k) = w2 tanh(W1[q; k])
    
    