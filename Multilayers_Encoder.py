import torch.nn as nn
import torch
import torch.nn.functional as F
from config import device, embedding_freeze

class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, num_direction, deal_bi, rnn_type = 'GRU', embedding_weight = None, dropout_rate = 0.1):
        super(EncoderRNN, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.num_direction = num_direction
        if embedding_weight is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding_weight, freeze = embedding_freeze)
        else:
            self.embedding = nn.Embedding(vocab_size,embed_size)
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout_rate)
        self.deal_bi = deal_bi
        self.rnn_type = rnn_type
#         if num_direction == 1:
#             self.gru = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True)
#         elif num_direction == 2:
#             self.gru = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
#             if deal_bi == 'linear':
#                 self.linear_compress = nn.Linear(2*self.hidden_size, self.hidden_size, bias=False)
        if num_direction == 1:
            if rnn_type == 'GRU':
                self.gru = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True, dropout = dropout_rate)
            elif rnn_type == 'LSTM':
                self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout = dropout_rate)
            else:
                print('RNN TYPE ERROR')
        elif num_direction == 2:
            if rnn_type == 'GRU':
                self.gru = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True, dropout = dropout_rate, bidirectional=True)
            elif rnn_type == 'LSTM':
                self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout = dropout_rate, bidirectional=True)
            else:
                print('RNN TYPE ERROR')
            if deal_bi == 'linear':
                self.linear_compress = nn.Linear(2*self.hidden_size, self.hidden_size, bias=False) 
                self.linear_compress_cell = nn.Linear(2*self.hidden_size, self.hidden_size, bias=False) 
                #self.linear_hidden_compress = nn.Linear(2*self.hidden_size, self.hidden_size, bias=False)
        else:
            print('number of direction out of bound')

    def forward(self, x, hidden, lengths, cell = None):
        embed = self.embedding(x) #(bz, src_len, emb_size)
        embed = self.dropout(embed) 
        batch_size = embed.size(0)
        embed = torch.nn.utils.rnn.pack_padded_sequence(embed, lengths.cpu().numpy(), batch_first=True)
        if self.rnn_type == 'GRU':
            rnn_out, hidden = self.gru(embed, hidden)
        else:
            rnn_out, (hidden, cell) = self.lstm(embed, (hidden, cell))
        #rnn_out, hidden = self.gru(embed, hidden)
        rnn_out, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True) # (bz, src_len, num_directions * hidden_size)
        if self.num_direction == 2:
            hidden = hidden.view(self.num_layers, self.num_direction, batch_size, self.hidden_size)
            if cell is not None:
                cell = cell.view(self.num_layers, self.num_direction, batch_size, self.hidden_size)
            if self.deal_bi == 'linear':
                hidden = self.linear_compress(hidden.transpose(1,2).contiguous().view(self.num_layers, batch_size, self.num_direction*self.hidden_size))
                rnn_out = self.linear_compress(rnn_out)
                if cell is not None:
                    cell = self.linear_compress_cell(cell.transpose(1,2).contiguous().view(self.num_layers, batch_size, self.num_direction*self.hidden_size))
            elif self.deal_bi == 'sum':
                hidden = torch.sum(hidden, dim=1)
                if cell is not None:
                    cell = torch.sum(cell, dim=1)
                src_len_batch = rnn_out.size(1)
                rnn_out = torch.sum(rnn_out.view(batch_size, src_len_batch, self.num_direction, self.hidden_size), dim=2)
            else:
                print('deal_bi Error')
        elif self.num_direction == 1:
            pass
        else:
            pass
        #hidden = hidden.view(self.num_layers, self.num_direction, batch_size, self.hidden_size).index_select(0,torch.tensor([(self.num_layers-1)]).to(self.device)).squeeze(0)
        #hidden = hidden.transpose(0,1).contiguous().view(1, batch_size, self.hidden_size*self.num_direction)
        return rnn_out, hidden, cell # (bz, src_len, hidden_size) (num_layers, bz, hidden_size)

    def initHidden(self, batch_size):
        cell = None
        if self.rnn_type == 'GRU':
            hidden = torch.randn(self.num_direction*self.num_layers, batch_size, self.hidden_size, device=self.device)
        else:
            hidden = torch.randn(self.num_direction*self.num_layers, batch_size, self.hidden_size, device=self.device)
            cell = torch.randn(self.num_direction*self.num_layers, batch_size, self.hidden_size, device=self.device)
        return hidden, cell
    
    def get_rnn_type(self):
        return self.rnn_type
        

