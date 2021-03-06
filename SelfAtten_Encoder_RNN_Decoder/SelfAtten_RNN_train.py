import numpy as np
import time
import os
import torch.nn as nn
import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from Data_utils import VocabDataset, vocab_collate_func
from preprocessing_util import preposs_toekn, Lang, text2index, construct_Lang
from selfAtten_RNN_Encoder import Encoder
from Multilayers_Decoder import DecoderAtten
from selfatten_RNN_evaluation import evaluate_batch
from config import device, PAD_token, SOS_token, EOS_token, UNK_token, embedding_freeze, vocab_prefix
import random
import pickle 
from SelfAtten_RNN_config import args
from torch.optim.lr_scheduler import LambdaLR

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
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    loss = 0
    encoder_outputs, encoder_hidden, encoder_cell = encoder(input_tensor, input_lengths)

    decoder_input = torch.tensor([[SOS_token]*batch_size], device=device).transpose(0,1)
    decoder_hidden,decoder_cell = encoder_hidden, encoder_cell  #decoder.initHidden(encoder_hidden)
    #print(decoder_hidden.size())
    #print('encoddddddddddder finishhhhhhhhhhhhhhh')
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        #### Teacher forcing: Feed the target as the next input
        # target_lengths = target_lengths.cpu().numpy()
        # sent_not_end_index = list(range(batch_size))
        # decoding_token_index = 0
        # while len(sent_not_end_index) > 0:
        #     decoder_output, decoder_hidden, decoder_attention = decoder(
        #         decoder_input, decoder_hidden, input_lengths, encoder_outputs)
        #     sent_not_end_index = torch.LongTensor(sent_not_end_index).to(device)
        #     loss += criterion(decoder_output.index_select(0,sent_not_end_index), 
        #                       target_tensor[:,decoding_token_index].index_select(0,sent_not_end_index))
        #     decoder_input = target_tensor[:,decoding_token_index].unsqueeze(1)  # Teacher forcing
        #     decoding_token_index += 1
        #     end_or_not = target_lengths > decoding_token_index
        #     sent_not_end_index = list(np.where(end_or_not)[0])
        ### simple version; 
        decoding_token_index = 0
        tgt_max_len_batch = target_lengths.cpu().max().item()
        assert(tgt_max_len_batch==target_tensor.size(1))
        while decoding_token_index < tgt_max_len_batch:
            decoder_output, decoder_hidden, decoder_attention, decoder_cell = decoder(
                decoder_input, decoder_hidden, input_lengths, encoder_outputs, decoder_cell)
            loss += criterion(decoder_output, target_tensor[:,decoding_token_index])
            decoder_input = target_tensor[:,decoding_token_index].unsqueeze(1)  # Teacher forcing
            decoding_token_index += 1

    else:
        ### debug 
        # # Without teacher forcing: use its own predictions as the next input
        # target_lengths_numpy = target_lengths.cpu().numpy()
        # sent_not_end_index = list(range(batch_size))
        # decoding_token_index = 0
        # while len(sent_not_end_index) > 0:
        #     decoder_output, decoder_hidden, decoder_attention_weights = decoder(
        #         decoder_input, decoder_hidden, input_lengths, encoder_outputs)
        #     topv, topi = decoder_output.topk(1)
        #     decoder_input = topi.detach()  # detach from history as input
        #     #print(type(sent_not_end_index[0]))
        #     sent_not_end_index = torch.LongTensor(sent_not_end_index).to(device)
        #     loss += criterion(decoder_output.index_select(0,sent_not_end_index), 
        #                       target_tensor[:,decoding_token_index].index_select(0,sent_not_end_index))
        #     decoding_token_index += 1
        #     end_or_not = target_lengths_numpy > decoding_token_index
        #     #(target_lengths_numpy > decoding_token_index)*(decoder_input.squeeze().numpy() != EOS_token)
        #     sent_not_end_index = list(np.where(end_or_not)[0])
        ### simple version
        decoding_token_index = 0
        tgt_max_len_batch = target_lengths.cpu().max().item()
        assert(tgt_max_len_batch==target_tensor.size(1))
        while decoding_token_index < tgt_max_len_batch:
            decoder_output, decoder_hidden, decoder_attention_weights, decoder_cell = decoder(
                decoder_input, decoder_hidden, input_lengths, encoder_outputs, decoder_cell)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.detach()  # detach from history as input
            loss += criterion(decoder_output, target_tensor[:,decoding_token_index])
            decoding_token_index += 1

    
    # average loss        
    #target_lengths.type_as(loss).mean()
    loss.backward()

    ### TODO
    # clip for gradient exploding 
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item()/tgt_max_len_batch  #torch.div(loss, target_lengths.type_as(loss).mean()).item()  #/target_lengths.mean()


def trainIters(train_loader, val_loader, encoder, decoder, num_epochs, 
               learning_rate, teacher_forcing_ratio, srcLang, tgtLang, model_save_info, tgt_max_len, beam_size):

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-09)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-09)

    if model_save_info['model_path_for_resume'] is not None:
        check_point_state = torch.load(model_save_info['model_path_for_resume'])
        encoder.load_state_dict(check_point_state['encoder_state_dict'])
        encoder_optimizer.load_state_dict(check_point_state['encoder_optimizer_state_dict'])
        decoder.load_state_dict(check_point_state['decoder_state_dict'])
        decoder_optimizer.load_state_dict(check_point_state['decoder_optimizer_state_dict'])
    
    lr_decay = False
    def lr_foo(step,warmup_steps = 4000):
        if step < 500:
            return 0.001/(step+1)
        else:
            return 1.0/np.sqrt(args['encoder_embed_dim'])* min(1.0/np.sqrt(step), step*warmup_steps**(-1.5))
    
    criterion = nn.NLLLoss() #nn.NLLLoss(ignore_index=PAD_token)
    max_val_bleu = 0
    
    if lr_decay:
        lambda_T = lambda step: lr_foo(step)
        scheduler_encoder = LambdaLR(encoder_optimizer, lr_lambda=lambda_T)
        scheduler_decoder = LambdaLR(decoder_optimizer, lr_lambda=lambda_T)
    
    for epoch in range(num_epochs): 
        n_iter = -1
        start_time = time.time()
        for input_tensor, input_lengths, target_tensor, target_lengths in train_loader:
            n_iter += 1
            
            if lr_decay:
                scheduler_encoder.step()
                scheduler_decoder.step()
            #print('start_step: ', n_iter)
            loss = train(input_tensor, input_lengths, target_tensor, target_lengths, 
                         encoder, decoder, encoder_optimizer, decoder_optimizer, 
                         criterion, teacher_forcing_ratio)
            if n_iter % 200 == 0:
                print('Loss:', loss)
                #eva_start = time.time()
                #val_bleu_sacre, val_bleu_nltk, val_loss = evaluate_batch(val_loader, encoder, decoder, criterion, tgt_max_len, tgtLang.index2word, srcLang.index2word)
                #print((time.time()-eva_start)/60)
                #print('epoch: [{}/{}], step: [{}/{}], train_loss:{}, val_bleu_sacre: {}, val_bleu_nltk: {}, val_loss: {}'.format(
               #     epoch, num_epochs, n_iter, len(train_loader), loss, val_bleu_sacre[0], val_bleu_nltk, val_loss))
               # print('Decoder parameters grad:')
               # for p in decoder.named_parameters():
               #     print(p[0], ': ',  p[1].grad.data.abs().mean().item(), p[1].grad.data.abs().max().item(), p[1].data.abs().mean().item(), p[1].data.abs().max().item(), end=' ')
               # print('\n')
               # print('Encoder Parameters grad:')
               # for p in encoder.named_parameters():
               #     print(p[0], ': ',  p[1].grad.data.abs().mean().item(), p[1].grad.data.abs().max().item(), p[1].data.abs().mean().item(), p[1].data.abs().max().item(), end=' ')
               # print('\n')
        val_bleu_sacre, val_bleu_nltk, val_loss = evaluate_batch(val_loader, encoder, decoder, criterion, tgt_max_len, tgtLang.index2word, srcLang.index2word)
        print('epoch: [{}/{}] (Running time {:.3f} min), val_bleu_sacre: {}, val_bleu_nltk: {}, val_loss: {}'.format(epoch, num_epochs, (time.time()-start_time)/60, val_bleu_sacre, val_bleu_nltk, val_loss))
        #val_bleu_sacre_beam, _, _ = evaluate_beam_batch(beam_size, val_loader, encoder, decoder, criterion, tgt_max_len, tgtLang.index2word)
        #print('epoch: [{}/{}] (Running time {:.3f} min), val_bleu_sacre_beam: {}'.format(epoch, num_epochs, (time.time()-start_time)/60, val_bleu_sacre_beam))
        if max_val_bleu < val_bleu_sacre.score:
            max_val_bleu = val_bleu_sacre.score
            ### TODO save best model
        if (epoch+1) % model_save_info['epochs_per_save_model'] == 0:
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
    src_max_vocab_size = paras['src_max_vocab_size']
    tgt_max_vocab_size = paras['tgt_max_vocab_size']
    tgt_max_len = paras['tgt_max_len']
    max_src_len_dataloader = paras['max_src_len_dataloader']
    max_tgt_len_dataloader = paras['max_tgt_len_dataloader']

    teacher_forcing_ratio = paras['teacher_forcing_ratio']
    emb_size = paras['emb_size']
    hidden_size = paras['hidden_size']
    num_layers = paras['num_layers']
    num_direction = paras['num_direction']
    deal_bi = paras['deal_bi']
    learning_rate = paras['learning_rate']
    num_epochs = paras['num_epochs']
    batch_size = paras['batch_size']
    rnn_type = paras['rnn_type']
    attention_type = paras['attention_type']
    beam_size = paras['beam_size']
    model_save_info = paras['model_save_info']
    dropout_rate = paras['dropout_rate']

    address_book=dict(
        train_src = 'Machine_Translation_NLP/iwsltzhen/iwslt-{}-{}/train.tok.{}'.format(transtype[0], transtype[1], transtype[0]),
        train_tgt = 'Machine_Translation_NLP/iwsltzhen/iwslt-{}-{}/train.tok.{}'.format(transtype[0], transtype[1], transtype[1]),
        val_src = 'Machine_Translation_NLP/iwsltzhen/iwslt-{}-{}/dev.tok.{}'.format(transtype[0], transtype[1], transtype[0]),
        val_tgt = 'Machine_Translation_NLP/iwsltzhen/iwslt-{}-{}/dev.tok.{}'.format(transtype[0], transtype[1], transtype[1]),
        src_emb = 'embedding/wiki.{}.vec'.format(transtype[0]),
        tgt_emb = 'embedding/wiki.{}.vec'.format(transtype[1])
        )
    #print(address_book)
    train_src_add = address_book['train_src']
    train_tgt_add = address_book['train_tgt']
    val_src_add = address_book['val_src']
    val_tgt_add = address_book['val_tgt']
    # make dir for saving models
    if not os.path.exists(model_save_info['model_path']):
        os.makedirs(model_save_info['model_path'])
    ### save model hyperparameters
    with open(model_save_info['model_path']+'model_params.pkl', 'wb') as f:
        model_hyparams = paras
        model_hyparams['address_book'] = address_book
        pickle.dump(model_hyparams, f)
    print(model_hyparams)

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
    srcLang = construct_Lang('src', src_max_vocab_size, address_book['src_emb'], train_src)
    tgtLang = construct_Lang('tgt', tgt_max_vocab_size, address_book['tgt_emb'], train_tgt)
    train_input_index = text2index(train_src, srcLang.word2index) #add EOS token here 
    train_output_index = text2index(train_tgt, tgtLang.word2index)
    val_input_index = text2index(val_src, srcLang.word2index)
    val_output_index = text2index(val_tgt, tgtLang.word2index)
    ### save srcLang and tgtLang

    train_dataset = VocabDataset(train_input_index,train_output_index, max_src_len_dataloader, max_tgt_len_dataloader)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               collate_fn=vocab_collate_func,
                                               shuffle=True)

    val_dataset = VocabDataset(val_input_index,val_output_index, None, None)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                            batch_size=batch_size,
                                            collate_fn=vocab_collate_func,
                                            shuffle=False)

    # test_dataset = VocabDataset(test_data)
    # test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
    #                                            batch_size=BATCH_SIZE,
    #                                            collate_fn=vocab_collate_func,
    #                                            shuffle=False)

    embedding_src_weight = torch.from_numpy(srcLang.embedding_matrix).type(torch.FloatTensor).to(device)
    embedding_tgt_weight = torch.from_numpy(tgtLang.embedding_matrix).type(torch.FloatTensor).to(device)
    print(embedding_src_weight.size(), embedding_tgt_weight.size())
    if attention_type:
        encoder = Encoder(args, num_layers, vocab_size = srcLang.vocab_size, use_position_emb = False)
        decoder = DecoderAtten(emb_size, hidden_size, tgtLang.vocab_size, num_layers, rnn_type = rnn_type, atten_type = attention_type, dropout_rate = dropout_rate)
    else:      
        encoder = Encoder(args, num_layers, vocab_size = srcLang.vocab_size, use_position_emb = False)
        decoder = DecoderRNN(emb_size, hidden_size, tgtLang.vocab_size, num_layers, rnn_type = rnn_type, dropout_rate = dropout_rate)

    
    encoder, decoder = encoder.to(device), decoder.to(device)
    print('Encoder:')
    print(encoder)
    print('Decoder:')
    print(decoder)
    trainIters(train_loader, val_loader, encoder, decoder, num_epochs, learning_rate, teacher_forcing_ratio, srcLang, tgtLang, model_save_info, tgt_max_len, beam_size)
    

if __name__ == "__main__":
    transtype = ('vi', 'en')
    paras = dict( 
        src_max_vocab_size = 26109, # 47127, #26109,
        tgt_max_vocab_size = 24418, #31553, #24418,
        tgt_max_len = 128,
        max_src_len_dataloader = 72, #67, #72, 
        max_tgt_len_dataloader = 71, #72, #71, 

        emb_size = 300,
        hidden_size = 300,
        num_layers = 2,
        num_direction = 1,
        deal_bi = 'linear', #{'linear', 'sum'}
        rnn_type = 'LSTM', # LSTM
        attention_type = 'concat', #'dot_prod', general, concat
        teacher_forcing_ratio = 1,

        learning_rate = 3e-4,
        num_epochs = 50,
        batch_size = 128, 
        beam_size = 1,
        dropout_rate = 0.1,

        model_save_info = dict(
            model_path = 'nmt_models/vi-en-selfattention300-lrdecay_encoderlayer21_random_dr1_lige/',
            epochs_per_save_model = 2,
            model_path_for_resume = None #'nmt_models/epoch_0.pth'
            )
        )
    #print('paras: ', paras)
    start_train(transtype, paras)



