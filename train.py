import numpy as np
import time
import os.path
import torch.nn as nn
import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from Data_utils import VocabDataset, vocab_collate_func
from preprocessing_util import preposs_toekn, read_embedding, Lang, text2index, preparelang
from Encoder import EncoderRNN
from Decoder import DecoderRNN, DecoderAtten
from config import *
import random
from evaluation import evaluate


####################Define Global Variable#########################

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)


####################Define Global Variable#########################


def train(input_tensor, input_lengths, target_tensor, target_lengths,
          encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, 
          teacher_forcing_ratio, attention):
    '''
    finish train for a batch
    '''
    batch_size = input_tensor.size()[0]
    encoder_hidden = encoder.initHidden(batch_size)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    loss = 0

    encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden, input_lengths)
    decoder_input = torch.tensor([[SOS_token]*batch_size], device=device).transpose(0,1)
    decoder_hidden = encoder_hidden
    #print('encoddddddddddder finishhhhhhhhhhhhhhh')
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        target_lengths = target_lengths.cpu().numpy()
        sent_not_end_index = list(range(batch_size))
        decoding_token_index = 0
        while len(sent_not_end_index) > 0:
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, input_lengths, encoder_outputs)
            sent_not_end_index = torch.LongTensor(sent_not_end_index).to(device)
            loss += criterion(decoder_output.index_select(0,sent_not_end_index), 
                              target_tensor[:,decoding_token_index].index_select(
                                  0,sent_not_end_index))
            decoder_input = target_tensor[:,decoding_token_index].unsqueeze(1)  # Teacher forcing
            decoding_token_index += 1
            end_or_not = target_lengths > decoding_token_index
            sent_not_end_index = list(np.where(end_or_not)[0])
            

    else:
        # Without teacher forcing: use its own predictions as the next input
        target_lengths = target_lengths.cpu().numpy()
        sent_not_end_index = list(range(batch_size))
        decoding_token_index = 0
        while len(sent_not_end_index) > 0:
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden,input_lengths, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.detach()  # detach from history as input
            #print(type(sent_not_end_index[0]))
            sent_not_end_index = torch.LongTensor(sent_not_end_index).to(device)
            loss += criterion(decoder_output.index_select(0,sent_not_end_index), 
                              target_tensor[:,decoding_token_index].index_select(
                                  0,sent_not_end_index))
            decoding_token_index += 1
            end_or_not = target_lengths > decoding_token_index
            #(target_lengths > decoding_token_index)*(decoder_input.squeeze().numpy() != EOS_token)
            sent_not_end_index = list(np.where(end_or_not)[0])
            
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_lengths.mean()


def trainIters(train_loader, val_loader, encoder, decoder, num_epochs, 
               learning_rate,teacher_forcing_ratio, attention, srcLang, tgtLang):
    start = time.time()
    plot_losses = []
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    criterion = nn.NLLLoss()
    for epoch in range(num_epochs): 
        n_iter = 0
        plot_losses = []
        for input_tensor, input_lengths, target_tensor, target_lengths in train_loader:
            n_iter += 1
            print('start_step: ', n_iter)
            loss = train(input_tensor, input_lengths, target_tensor, target_lengths, 
                         encoder, decoder, encoder_optimizer, decoder_optimizer, 
                         criterion, teacher_forcing_ratio, attention)
            plot_losses.append(loss)
            print('*********',loss)
            if n_iter%100 == 0:
                val_bleu, val_loss = evaluate(val_loader, encoder, decoder, criterion, tgt_max_length,srcLang.index2word ,tgtLang.index2word)
                print('epoch: [{}], step: [{}/{}], val_bleu: {}, val_loss: {}'.format(epoch, n_iter, len(train_loader), val_bleu, val_loss))
        val_bleu, val_loss = evaluate(val_loader, encoder, decoder, criterion, tgt_max_length,srcLang.index2word ,tgtLang.index2word)
        print('epoch: [{}], val_bleu: {}, val_loss: {}'.format(epoch, val_bleu, val_loss))
    return None
    

def start_train(transtype, paras):
    teacher_forcing_ratio = paras['teacher_forcing_ratio']
    emb_size = paras['emb_size']
    hidden_size = paras['hidden_size']
    num_direction = paras['num_direction']
    learning_rate=paras['learning_rate']
    num_epochs=paras['num_epochs']
    batch_size = paras['batch_size']
    attention = paras['attention']
    
    
    train_src_add = address_book['train_src']
    train_tgt_add = address_book['train_tgt']
    val_src_add = address_book['val_src']
    val_tgt_add = address_book['val_tgt']

    train_src = []
    with open(train_src_add) as f:
        for line in f:
            train_src.append(preposs_toekn(line[:-1].strip().split(' ')))

    train_tgt = []
    with open(train_tgt_add) as f:
        for line in f:
            train_tgt.append(preposs_toekn(line[:-1].strip().split(' ')))
        
    val_src = []
    with open(val_src_add) as f:
        for line in f:
            val_src.append(preposs_toekn(line[:-1].strip().split(' ')))

    val_tgt = []
    with open(val_tgt_add) as f:
        for line in f:
            val_tgt.append(preposs_toekn(line[:-1].strip().split(' ')))

    srcLang = Lang('src')
    srcLang.load_embedding(address_book['src_emb'],src_vocab_size)
    tgtLang = Lang('tgt')
    tgtLang.load_embedding(address_book['tgt_emb'],tgt_vocab_size)
    train_input_index = text2index(train_src,srcLang.word2index)
    train_output_index = text2index(train_tgt,tgtLang.word2index)
    val_input_index = text2index(val_src,srcLang.word2index)
    val_output_index = text2index(val_tgt,tgtLang.word2index)
    
    

    train_dataset = VocabDataset(train_input_index,train_output_index)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               collate_fn=vocab_collate_func,
                                               shuffle=False)

    val_dataset = VocabDataset(val_input_index,val_output_index)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                            batch_size=1,
                                            collate_fn=vocab_collate_func,
                                            shuffle=False)

    # test_dataset = VocabDataset(test_data)
    # test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
    #                                            batch_size=BATCH_SIZE,
    #                                            collate_fn=vocab_collate_func,
    #                                            shuffle=False)

    embedding_src_weight = torch.from_numpy(srcLang.embedding_matrix).to(device)
    embedding_tgt_weight = torch.from_numpy(tgtLang.embedding_matrix).to(device)
    print(embedding_src_weight.size(),embedding_tgt_weight.size())
    if attention:
        encoder = EncoderRNN(src_vocab_size, emb_size,hidden_size,num_direction, embedding_weight = embedding_src_weight, device = device)
        decoder = DecoderAtten(emb_size, hidden_size, tgt_vocab_size, num_direction, embedding_weight = embedding_tgt_weight, device = device)
    else:      
        encoder = EncoderRNN(src_vocab_size, emb_size,hidden_size,num_direction, embedding_weight = embedding_src_weight, device = device)
        decoder = DecoderRNN(emb_size, hidden_size, tgt_vocab_size, num_direction, embedding_weight = embedding_tgt_weight, device = device)
    
    encoder, decoder = encoder.to(device), decoder.to(device)
    print(encoder)
    print(decoder)
    trainIters(train_loader, val_loader, encoder, decoder, num_epochs, learning_rate, teacher_forcing_ratio, attention, srcLang, tgtLang)
    

if __name__ == "__main__":
    transtype = 'zh2en'
    paras = dict( 
        teacher_forcing_ratio = 0,
        emb_size = 300,
        hidden_size = 100,
        num_direction = 2,
        learning_rate=0.01,
        num_epochs=10,
        batch_size = 20, 
        attention = False
    )
    start_train(transtype, paras)







        
