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


####################Define Global Variable#########################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


####################Define Global Variable#########################


def train(input_tensor, input_lengths, target_tensor, target_lengths,
          encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, 
          teacher_forcing_ratio, attention):
    
    batch_size = input_tensor.size()[0]
    encoder_hidden = encoder.initHidden(batch_size)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    loss = 0

    encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden, input_lengths)
    
    if attention:
        decoder.readEncoderOutput(encoder_outputs)
        
    decoder_input = torch.tensor([[SOS_token]*batch_size], device=device).transpose(0,1)
    decoder_hidden = encoder_hidden
    
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        target_lengths = target_lengths.cpu().numpy()
        sent_not_end_index = list(range(batch_size))
        decoding_token_index = 0
        while len(sent_not_end_index) > 0:
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
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
                decoder_input, decoder_hidden)
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

    return loss.item() / target_lengths


def trainIters(train_loader, encoder, decoder, num_epochs, 
               learning_rate,teacher_forcing_ratio, attention):
    start = time.time()
    plot_losses = []
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    criterion = nn.NLLLoss()
    n_iter = 0
    
    
    for epoch in range(num_epochs): 
        plot_losses = []
        for input_tensor, input_lengths, target_tensor, target_lengths in train_loader:
            n_iter += 1
            loss = train(input_tensor, input_lengths, target_tensor, target_lengths, 
                         encoder, decoder, encoder_optimizer, decoder_optimizer, 
                         criterion, teacher_forcing_ratio, attention)
            plot_losses.append(loss)

    #         print_loss_total += loss
    #         plot_loss_total += loss

    #         if n_iter % print_every == 0:
    #             print_loss_avg = print_loss_total / print_every
    #             print_loss_total = 0
    #             print('(%d %d%%) %.4f' % (n_iter, n_iter / n_iter * 100, print_loss_avg))

    #         if n_iter % plot_every == 0:
    #             plot_loss_avg = plot_loss_total / plot_every
    #             plot_losses.append(plot_loss_avg)
    #             plot_loss_total = 0

    showPlot(plot_losses)


def start_train(transtype, paras):
    teacher_forcing_ratio = paras['teacher_forcing_ratio']
    emb_size = paras['emb_size']
    hidden_size = paras['hidden_size']
    num_direction = paras['num_direction']
    learning_rate=paras['learning_rate']
    num_epochs=paras['num_epochs']
    batch_size = paras['batch_size']
    attention = paras['attention']
    
    
    if transtype == 'en2zh':
        train_en_add = './iwsltzhen/iwslt-zh-en/train.tok.en'
        train_zh_add = './iwsltzhen/iwslt-zh-en/train.tok.zh'
        val_en_add = './iwsltzhen/iwslt-zh-en/dev.tok.en'
        val_zh_add = './iwsltzhen/iwslt-zh-en/dev.tok.zh'

        train_en = []
        with open(train_en_add) as f:
            for line in f:
                train_en.append(preposs_toekn(line[:-1].strip().split(' ')))

        train_zh = []
        with open(train_zh_add) as f:
            for line in f:
                train_zh.append(preposs_toekn(line[:-1].strip().split(' ')))
        enLang = Lang('en')
        enLang.load_embedding('/scratch/tw1682/embedding/wiki.en.vec',src_vocab_size)
        zhLang = Lang('zh')
        zhLang.load_embedding('/scratch/tw1682/embedding/wiki.zh.vec',tgt_vocab_size)
        train_input_index = text2index(train_en,enLang.word2index)
        train_output_index = text2index(train_zh,zhLang.word2index)
    
    elif transtype == 'zh2en':
        train_en_add = './iwsltzhen/iwslt-zh-en/train.tok.en'
        train_zh_add = './iwsltzhen/iwslt-zh-en/train.tok.zh'
        val_en_add = './iwsltzhen/iwslt-zh-en/dev.tok.en'
        val_zh_add = './iwsltzhen/iwslt-zh-en/dev.tok.zh'

        train_en = []
        with open(train_en_add) as f:
            for line in f:
                train_en.append(preposs_toekn(line[:-1].strip().split(' ')))

        train_zh = []
        with open(train_zh_add) as f:
            for line in f:
                train_zh.append(preposs_toekn(line[:-1].strip().split(' ')))
                
        enLang = Lang('en')
        enLang.load_embedding('/scratch/tw1682/embedding/wiki.en.vec',src_vocab_size)
        zhLang = Lang('zh')
        zhLang.load_embedding('/scratch/tw1682/embedding/wiki.zh.vec',tgt_vocab_size)
        train_input_index = text2index(train_zh,zhLang.word2index)
        train_output_index = text2index(train_en,enLang.word2index)
    
    else:
        print('translation type error, we support zh2en and en2zh')
    

    train_dataset = VocabDataset(train_input_index,train_output_index)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               collate_fn=vocab_collate_func,
                                               shuffle=False)

    # val_dataset = VocabDataset(val_data)
    # val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
    #                                            batch_size=BATCH_SIZE,
    #                                            collate_fn=vocab_collate_func,
    #                                            shuffle=True)

    # test_dataset = VocabDataset(test_data)
    # test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
    #                                            batch_size=BATCH_SIZE,
    #                                            collate_fn=vocab_collate_func,
    #                                            shuffle=False)
    
    if attention:
        if transtype == 'en2zh':
            encoder = EncoderRNN(src_vocab_size, embed_size, hidden_size,vocab_size, embed_size, hidden_sizenum_direction = 1,embedding_weight = enLang.embedding_matrix, device = device)
            decoder = DecoderAtten(hidden_size, tgt_vocab_size, embedding_weight = zhLang.embedding_matrix, device = device)
        elif transtype == 'zh2en':
            encoder = EncoderRNN(src_vocab_size, emb_size,hidden_size,num_direction = 1,embedding_weight = zhLang.embedding_matrix, device = device)
            decoder = DecoderAtten(hidden_size, tgt_vocab_size, embedding_weight = enLang.embedding_matrix, device = device)
        else:
            print('You should not see this')
    else:      
        if transtype == 'en2zh':
            encoder = EncoderRNN(src_vocab_size, embed_size, hidden_size,vocab_size, embed_size, hidden_sizenum_direction = 1,embedding_weight = enLang.embedding_matrix, device = device)
            decoder = DecoderRNN(hidden_size, tgt_vocab_size, embedding_weight = zhLang.embedding_matrix, device = device)
        elif transtype == 'zh2en':
            encoder = EncoderRNN(src_vocab_size, emb_size,hidden_size,num_direction = 1,embedding_weight = zhLang.embedding_matrix, device = device)
            decoder = DecoderRNN(hidden_size, tgt_vocab_size, embedding_weight = enLang.embedding_matrix, device = device)
        else:
            print('You should not see this')
    
    trainIters(train_loader, encoder, decoder, num_epochs, learning_rate, teacher_forcing_ratio, attention)
    

if __name__ == "__main__":
    transtype = 'zh2en'
    paras = dict( 
        teacher_forcing_ratio = 0,
        emb_size = 300,
        hidden_size = 100,
        num_direction = 1,
        learning_rate=0.01,
        num_epochs=100,
        batch_size = 10, 
        attention = True
    )
    start_train(transtype,paras)







        
