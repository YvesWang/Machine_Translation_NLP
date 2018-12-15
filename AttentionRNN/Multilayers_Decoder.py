import torch.nn as nn
import torch
import torch.nn.functional as F
from config import embedding_freeze

class DecoderRNN(nn.Module):
    def __init__(self, emb_size, hidden_size, vocab_size, num_layers, rnn_type = 'GRU', embedding_weight = None, dropout_rate = 0.1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout_rate)
        if embedding_weight is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding_weight, freeze = embedding_freeze)
        else:
            self.embedding = nn.Embedding(vocab_size,emb_size)
        self.rnn_type = rnn_type
        if rnn_type == 'GRU':
            self.gru = nn.GRU(emb_size, hidden_size, num_layers, batch_first=True, dropout = dropout_rate)
        elif rnn_type == 'LSTM':
            self.lstm = nn.LSTM(emb_size, hidden_size, num_layers, batch_first=True, dropout = dropout_rate)
        else:
            print('RNN TYPE ERROR')
        self.out = nn.Linear(hidden_size, vocab_size)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, tgt_input, hidden, true_len = None, encoder_outputs = None, cell = None):
        output = self.embedding(tgt_input)
        #print(output.size())
        if self.rnn_type == 'GRU':
            output, hidden = self.gru(output, hidden)
        else:
            output, (hidden, cell) = self.lstm(output,(hidden, cell))
        logits = self.out(output.squeeze(1))
        output = self.logsoftmax(logits)
        return output, hidden, None, cell

    # def initHidden(self, encoder_hidden):
    #     batch_size = encoder_hidden.size(1)
    #     return encoder_hidden.expand(self.num_layers, batch_size, self.hidden_size).contiguous()
    

class DecoderAtten(nn.Module):
    def __init__(self, emb_size, hidden_size, vocab_size, num_layers, rnn_type = 'GRU', embedding_weight = None, atten_type = 'dot_prod', dropout_rate = 0.1):
        super(DecoderAtten, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout_rate)
        self.num_layers = num_layers
        if embedding_weight is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding_weight, freeze = False)
        else:
            self.embedding = nn.Embedding(vocab_size, emb_size)
        self.rnn_type = rnn_type
        if rnn_type == 'GRU':
            self.gru = nn.GRU(emb_size, hidden_size, num_layers, batch_first=True)
        elif rnn_type == 'LSTM':
            self.lstm = nn.LSTM(emb_size, hidden_size, num_layers, batch_first=True)
        else:
            print('RNN TYPE ERROR')
        self.atten = AttentionLayer(hidden_size, atten_type= atten_type)
        
        self.linear = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, tgt_input, hidden, true_len, encoder_outputs, cell = None):
        output = self.embedding(tgt_input)
        output = self.dropout(output)
        #print(output.size())
        if self.rnn_type == 'GRU':
            output, hidden = self.gru(output, hidden)
        else:
            output, (hidden, cell) = self.lstm(output, (hidden, cell))
        #output, hidden = self.gru(output, hidden)
        ### add attention
        atten_output, atten_weight = self.atten(output, encoder_outputs, true_len)
        out1 = torch.cat((output,atten_output),-1)
        out2 = self.linear(out1.squeeze(1))
        out2 = F.relu(out2)
        logits = self.out(out2)
        output = self.logsoftmax(logits)
        return output, hidden, atten_weight, cell
    
#    def initHidden(self, encoder_hidden):
#        batch_size = encoder_hidden.size(1)
#        return encoder_hidden.expand(self.num_layers, batch_size, self.hidden_size).contiguous()


class AttentionLayer(nn.Module):
    def __init__(self, hidden_size, atten_type):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.mode = atten_type
        if atten_type == 'dot_prod':
            print('dot_prod')
        elif atten_type == 'general':
            print('general')
            self.general_linear = nn.Linear(hidden_size, hidden_size, bias = False)
        elif atten_type == 'concat':
            print('concat')
            self.content_linear = nn.Linear(hidden_size * 2, hidden_size, bias = True)
            self.score_linear = nn.Linear(hidden_size , 1, bias = False)
        else:
            print('mode out of bound')
        #elif atten_type == 
        #self.atten = nn.Linear(key_size, 1)

    def forward(self, query, memory_bank, true_len):
        #batch_size, src_len, hidden_size = memory_bank.size()
        #query_len = query.size(1)
        scores = self.atten_score(query, memory_bank)
        
        mask_matrix = sequence_mask(true_len).unsqueeze(1)
        scores.masked_fill_(1-mask_matrix, float('-inf'))
        
        scores_normalized = F.softmax(scores, dim=-1)
        #scores_normalized = F.softmax(scores.view(batch_size * query_len, seq_len), dim=-1).view(batch_size, query_len, seq_len)
        context = torch.bmm(scores_normalized, memory_bank)
        
        return context, scores_normalized
    
    def atten_score(self, query, memory_bank):
        """
        query: batch * tgt_length * hidden_size
        memory_bank: batch * src_length * hidden_size
        return: batch * tgt_length * src_length
        """

        batch_size, src_len, hidden_size = memory_bank.size()
        query_len = query.size(1)
        if self.mode == 'dot_prod':
            out = torch.bmm(query, memory_bank.transpose(1, 2))
        elif self.mode == 'general':
            temp = self.general_linear(query.view(batch_size * query_len, hidden_size))
            out = torch.bmm(temp.view(batch_size,query_len,hidden_size),memory_bank.transpose(1, 2))
        elif self.mode == 'concat':
            query_temp = query.unsqueeze(2).expand(batch_size,query_len,src_len,hidden_size)
            memory_temp = memory_bank.unsqueeze(1).expand(batch_size,query_len,src_len,hidden_size)
            content_out = self.content_linear(torch.cat((query_temp,memory_temp),-1).view(batch_size * query_len * src_len, hidden_size*2))
            content_out = torch.tanh(content_out)
            out = self.score_linear(content_out)
            out = out.view(batch_size , query_len , src_len).squeeze(-1)
        else:
            print('mode out of bound')
        return out

def sequence_mask(lengths):
    batch_size = lengths.numel()
    max_len = lengths.max()
    return (torch.arange(0, max_len).type_as(lengths).repeat(batch_size,1).lt(lengths.unsqueeze(1)))
