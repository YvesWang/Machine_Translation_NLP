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
        return output, hidden, None
    

class DecoderAtten(nn.Module):
    def __init__(self, hidden_size, vocab_size, embedding_weight, device):
        super(DecoderAtten, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.atten = AttentionLayer(hidden_size,atten_type = 'dot_prod')
        
        self.linear = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.encoder_output = None

    def forward(self, src_input, hidden):
        output = self.embedding(src_input)
        #print(output.size())
        output, hidden = self.gru(output, hidden)
        ### add attention
        atten_output, atten_weight = self.atten(output, self.encoder_output)
        out1 = torch.cat((output,atten_output),-1)
        out2 = self.linear(out1[:,0,:])
        logits = self.out(out2)
        output = self.logsoftmax(logits)
        return output, hidden, atten_weight
    
    def readEncoderOutput(self,encoder_output):
        self.encoder_output = encoder_output


class AttentionLayer(nn.Module):
    def __init__(self, hidden_size, atten_type):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.mode = atten_type
        if atten_type == 'dot_prod':
            print('dot_prod')
        else:
            print('mode out of bound')
        #elif atten_type == 
        #self.atten = nn.Linear(key_size, 1)

    def forward(self, query, memory_bank):
        batch, seq_len, hidden_size = memory_bank.size()
        #query_len = query.size(1)
        
        scores = self.atten_score(query, memory_bank)
        scores_normalized = F.softmax(scores, dim=2)
        
        context = torch.bmm(scores_normalized, memory_bank)
        
        return context, scores
    
    def atten_score(self, query, memory_bank):
        """
        query is: b t_q n
        memory_bank is b t_k n
        return batch * target length * sequence length
        """

        batch, seq_len, hidden_size = list(memory_bank.size())
        #query_len = query.size(1)
        if self.mode == 'dot_prod':
            out = torch.bmm(query, memory_bank.transpose(1, 2))
        else:
            print('mode out of bound')

        return out
    
    