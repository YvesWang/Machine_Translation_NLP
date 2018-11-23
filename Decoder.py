import torch.nn as nn
import torch
import torch.nn.functional as F

class DecoderRNN(nn.Module):
    def __init__(self, emb_size, hidden_size, vocab_size, num_encoder_direction,  embedding_weight, device):
        super(DecoderRNN, self).__init__()
        hidden_size = hidden_size * num_encoder_direction
        self.hidden_size = hidden_size
        if embedding_weight is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding_weight, freeze = False)
        else:
            self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(emb_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.device = device
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, src_input, hidden, true_len = None, encoder_outputs = None):
        output = self.embedding(src_input)
        #print(output.size())
        output, hidden = self.gru(output, hidden)
        logits = self.out(output[:,0,:])
        output = self.softmax(logits)
        return output, hidden, None
    

class DecoderAtten(nn.Module):
    def __init__(self, emb_size, hidden_size, vocab_size, num_encoder_direction,  embedding_weight, device):
        super(DecoderAtten, self).__init__()
        hidden_size = hidden_size * num_encoder_direction
        self.hidden_size = hidden_size
        if embedding_weight is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding_weight, freeze = False)
        else:
            self.embedding = nn.Embedding(vocab_size, emb_size)
        self.gru = nn.GRU(emb_size, hidden_size, batch_first=True)
        self.atten = AttentionLayer(hidden_size,atten_type = 'dot_prod')
        
        self.linear = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, src_input, hidden, true_len, encoder_outputs):
        output = self.embedding(src_input)
        #print(output.size())
        output, hidden = self.gru(output, hidden)
        ### add attention
        atten_output, atten_weight = self.atten(output, encoder_outputs, true_len)
        out1 = torch.cat((output,atten_output),-1)
        out2 = self.linear(out1.squeeze(1))
        out2 = F.tanh(out2)
        logits = self.out(out2)
        output = self.logsoftmax(logits)
        return output, hidden, atten_weight
    
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

    def forward(self, query, memory_bank, true_len):
        batch_size, seq_len, hidden_size = memory_bank.size()
        query_len = query.size(1)
        #print('11111111111111111111')
        scores = self.atten_score(query, memory_bank)
        
        #print('111222222222222211111111')
        mask_matrix = sequence_mask(true_len).unsqueeze(1)
        scores.masked_fill_(1-mask_matrix, float('-inf'))
        
        #print('111111333333333333331111')
        #scores_normalized = F.softmax(scores, dim=2)
        scores_normalized = F.softmax(scores.view(batch_size * query_len, seq_len), dim=-1).view(batch_size, query_len, seq_len)
        context = torch.bmm(scores_normalized, memory_bank)
        
        #print('1111111444444444444441111')
        return context, scores
    
    def atten_score(self, query, memory_bank):
        """
        query is: batch * target length * hidden size
        memory_bank is batch * sequence length * hidden size
        return batch * target length * sequence length
        """

        batch, seq_len, hidden_size = memory_bank.size()
        #query_len = query.size(1)
        if self.mode == 'dot_prod':
            out = torch.bmm(query, memory_bank.transpose(1, 2))
        else:
            print('mode out of bound')
        return out

def sequence_mask(lengths):
    batch_size = lengths.numel()
    max_len = lengths.max()
    return (torch.arange(0, max_len).type_as(lengths).repeat(batch_size,1).lt(lengths.unsqueeze(1)))
