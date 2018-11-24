import torch.nn as nn
import torch
import torch.nn.functional as F

class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_direction, embedding_weight, device):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_direction = num_direction
        self.embedding = nn.Embedding.from_pretrained(embedding_weight, freeze = False)
        self.device = device
        self.direction = num_direction
        if num_direction == 1:
            self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)
        elif num_direction == 2:
            self.gru = nn.GRU(embed_size, hidden_size, batch_first=True, bidirectional = True)
        else:
            print('number of direction out of bound')

    def forward(self, x, hidden, lengths):
        embed = self.embedding(x)
        batch_size = embed.size(0)
        embed = torch.nn.utils.rnn.pack_padded_sequence(embed, lengths.cpu().numpy(), batch_first=True)
        rnn_out, hidden = self.gru(embed, hidden)
        rnn_out, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True)
        hidden = hidden.transpose(0,1).contiguous().view(1, batch_size, self.hidden_size*self.num_direction)
        return rnn_out, hidden

    def initHidden(self, batch_size):
        hidden = torch.randn(self.num_direction, batch_size, self.hidden_size, self.device)
        return hidden
