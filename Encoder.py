import torch.nn as nn
import torch
import torch.nn.functional as F

class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_direction, embedding_weight, device):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_direction = num_direction
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.device = device
        if num_direction == 1:
            self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)
        elif num_direction == 2:
            self.gru = nn.GRU(embed_size, hidden_size, batch_first=True, bidirectional = True)
        else:
            print('number of direction out of bound')

    def forward(self, x, hidden, lengths):
        embed = self.embedding(x)
        embed = torch.nn.utils.rnn.pack_padded_sequence(embed, lengths.numpy(), batch_first=True)
        rnn_out, hidden = self.gru(embed, hidden)
        rnn_out, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True)
        return rnn_out, hidden

    def initHidden(self, batch_size):
        hidden = torch.randn(self.num_direction, batch_size, self.hidden_size, device= self.device)
        return hidden