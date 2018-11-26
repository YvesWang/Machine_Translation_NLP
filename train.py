import numpy as np
import time
import os
import torch.nn as nn
import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from Data_utils import VocabDataset, vocab_collate_func
from preprocessing_util import preposs_toekn, read_embedding, Lang, text2index, preparelang
from Multilayers_Encoder import EncoderRNN
from Multilayers_Decoder import DecoderRNN, DecoderAtten
from config import *
import random
from evaluation import evaluate_batch


####################Define Global Variable#########################

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)

####################Define Global Variable#########################


def train(input_tensor, input_lengths, target_tensor, target_lengths,
          encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, 
          teacher_forcing_ratio):
    '''
    finish train for a batch
    '''
    batch_size = input_tensor.size(0)
    encoder_hidden = encoder.initHidden(batch_size)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    loss = 0

    encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden, input_lengths)
    decoder_input = torch.tensor([[SOS_token]*batch_size], device=device).transpose(0,1)
    decoder_hidden = decoder.initHidden(encoder_hidden)
    #print(decoder_hidden.size())
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
                              target_tensor[:,decoding_token_index].index_select(0,sent_not_end_index))
            decoder_input = target_tensor[:,decoding_token_index].unsqueeze(1)  # Teacher forcing
            decoding_token_index += 1
            end_or_not = target_lengths > decoding_token_index
            sent_not_end_index = list(np.where(end_or_not)[0])
            

    else:
        ### debug 
        # Without teacher forcing: use its own predictions as the next input
        target_lengths_numpy = target_lengths.cpu().numpy()
        sent_not_end_index = list(range(batch_size))
        decoding_token_index = 0
        while len(sent_not_end_index) > 0:
            decoder_output, decoder_hidden, decoder_attention_weights = decoder(
                decoder_input, decoder_hidden, input_lengths, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.detach()  # detach from history as input
            #print(type(sent_not_end_index[0]))
            sent_not_end_index = torch.LongTensor(sent_not_end_index).to(device)
            loss += criterion(decoder_output.index_select(0,sent_not_end_index), 
                              target_tensor[:,decoding_token_index].index_select(0,sent_not_end_index))
            decoding_token_index += 1
            end_or_not = target_lengths_numpy > decoding_token_index
            #(target_lengths_numpy > decoding_token_index)*(decoder_input.squeeze().numpy() != EOS_token)
            sent_not_end_index = list(np.where(end_or_not)[0])
    
    # average loss        
    loss = torch.div(loss, target_lengths.type_as(loss).mean())
    loss.backward()

    ### TODO
    # clip for gradient exploding 
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item()  #/target_lengths.mean()


def trainIters(train_loader, val_loader, encoder, decoder, num_epochs, 
               learning_rate, teacher_forcing_ratio, srcLang, tgtLang, model_save_info):

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

    if model_save_info['model_path_for_resume'] is not None:
        check_point_state = torch.load(model_save_info['model_path_for_resume'])
        encoder.load_state_dict(check_point_state['encoder_state_dict'])
        encoder_optimizer.load_state_dict(check_point_state['encoder_optimizer_state_dict'])
        decoder.load_state_dict(check_point_state['decoder_state_dict'])
        decoder_optimizer.load_state_dict(check_point_state['decoder_optimizer_state_dict'])

    criterion = nn.NLLLoss()
    max_val_bleu = 0

    for epoch in range(num_epochs): 
        n_iter = -1
        start_time = time.time()
        for input_tensor, input_lengths, target_tensor, target_lengths in train_loader:
            n_iter += 1
            #print('start_step: ', n_iter)
            loss = train(input_tensor, input_lengths, target_tensor, target_lengths, 
                         encoder, decoder, encoder_optimizer, decoder_optimizer, 
                         criterion, teacher_forcing_ratio)
            if n_iter % 60 == 0:
                print('Loss:', loss)
                #eva_start = time.time()
                #val_bleu_sacre, val_bleu_nltk, val_loss = evaluate_batch(val_loader, encoder, decoder, criterion, tgt_max_length, tgtLang.index2word)
                #print((time.time()-eva_start)/60)
                #print('epoch: [{}/{}], step: [{}/{}], train_loss:{}, val_bleu_sacre: {}, val_bleu_nltk: {}, val_loss: {}'.format(
                 #   epoch, num_epochs, n_iter, len(train_loader), loss, val_bleu_sacre[0], val_bleu_nltk, val_loss))
               # for p in decoder.parameters():
               #     print('Decoder grad mean:')
               #     print(p.grad.data.abs().mean().item(), end=' ')
               #     print('---------')
               # for p in encoder.parameters():
               #     print('Encoder grad mean:')
               #     print(p.grad.data.abs().mean().item(), end=' ')
               #     print('----------')
        val_bleu_sacre, val_bleu_nltk, val_loss = evaluate_batch(val_loader, encoder, decoder, criterion, tgt_max_length, tgtLang.index2word)
        print('epoch: [{}/{}] (Running time {:.6f} min), val_bleu_sacre: {}, val_bleu_nltk: {}, val_loss: {}'.format(epoch, num_epochs, (time.time()-start_time)/60, val_bleu_sacre, val_bleu_nltk, val_loss))
        if max_val_bleu < val_bleu_sacre.score:
            max_val_bleu = val_bleu_sacre.score
            ### TODO save best model
        if epoch % model_save_info['epochs_per_save_model'] == 0:
            check_point_state = {
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'encoder_optimizer_state_dict': encoder_optimizer.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'decoder_optimizer_state_dict': decoder_optimizer.state_dict()
                }
            torch.save(check_point_state, '{}epoch_{}.pth'.format(model_save_info['model_path'], epoch))

    return None
    

def start_train(transtype, paras):
    teacher_forcing_ratio = paras['teacher_forcing_ratio']
    emb_size = paras['emb_size']
    hidden_size = paras['hidden_size']
    num_layers = paras['num_layers']
    num_direction = paras['num_direction']
    learning_rate = paras['learning_rate']
    num_epochs = paras['num_epochs']
    batch_size = paras['batch_size']
    attention_type = paras['attention_type']
    model_save_info = paras['model_save_info']

    train_src_add = address_book['train_src']
    train_tgt_add = address_book['train_tgt']
    val_src_add = address_book['val_src']
    val_tgt_add = address_book['val_tgt']
	
    # make dir for saving models
    if not os.path.exists(model_save_info['model_path']):
        os.makedirs(model_save_info['model_path'])

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

    print('The number of train samples: ', len(train_src))
    print('The number of val samples: ', len(val_src))
    srcLang = Lang('src')
    srcLang.load_embedding(address_book['src_emb'], src_vocab_size)
    tgtLang = Lang('tgt')
    tgtLang.load_embedding(address_book['tgt_emb'], tgt_vocab_size)
    train_input_index = text2index(train_src, srcLang.word2index)
    train_output_index = text2index(train_tgt, tgtLang.word2index)
    val_input_index = text2index(val_src, srcLang.word2index)
    val_output_index = text2index(val_tgt, tgtLang.word2index)
    
    train_dataset = VocabDataset(train_input_index,train_output_index)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               collate_fn=vocab_collate_func,
                                               shuffle=False)

    val_dataset = VocabDataset(val_input_index,val_output_index)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                            batch_size=batch_size,
                                            collate_fn=vocab_collate_func,
                                            shuffle=False)

    # test_dataset = VocabDataset(test_data)
    # test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
    #                                            batch_size=BATCH_SIZE,
    #                                            collate_fn=vocab_collate_func,
    #                                            shuffle=False)

    embedding_src_weight = torch.from_numpy(srcLang.embedding_matrix).to(device)
    embedding_tgt_weight = torch.from_numpy(tgtLang.embedding_matrix).to(device)
    print(embedding_src_weight.size(), embedding_tgt_weight.size())
    if attention_type:
        encoder = EncoderRNN(src_vocab_size, emb_size, hidden_size, num_layers, num_direction, embedding_weight = embedding_src_weight, device = device)
        decoder = DecoderAtten(emb_size, hidden_size, tgt_vocab_size, num_layers, num_direction, embedding_weight = embedding_tgt_weight, atten_type = attention_type, device = device)
    else:      
        encoder = EncoderRNN(src_vocab_size, emb_size,hidden_size, num_layers, num_direction, embedding_weight = embedding_src_weight, device = device)
        decoder = DecoderRNN(emb_size, hidden_size, tgt_vocab_size, num_layers, num_direction, embedding_weight = embedding_tgt_weight, device = device)
    
    encoder, decoder = encoder.to(device), decoder.to(device)
    print('Encoder:')
    print(encoder)
    print('Decoder:')
    print(decoder)
    trainIters(train_loader, val_loader, encoder, decoder, num_epochs, learning_rate, teacher_forcing_ratio, srcLang, tgtLang, model_save_info)
    

if __name__ == "__main__":
    transtype = 'zh2en'
    paras = dict( 
        teacher_forcing_ratio = 0,
        emb_size = 300,
        hidden_size = 100,
        num_layers = 2,
        num_direction = 2,
        learning_rate = 1e-4,
        num_epochs = 60,
        batch_size = 100, 
        attention_type = None, # None, dot_prod, general, concat

        model_save_info = dict(
            model_path = 'nmt_models/model_test/para221e4/',
            epochs_per_save_model = 10,
            model_path_for_resume = None #'nmt_models/epoch_0.pth'
            )
    )
    print('paras: ', paras)
    start_train(transtype, paras)



