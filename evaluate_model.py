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
from Encoder import EncoderRNN
from Decoder import DecoderRNN, DecoderAtten
from config import *
import random
from evaluation import evaluate, evaluate_batch

def start_evaluate(transtype, paras):
    teacher_forcing_ratio = paras['teacher_forcing_ratio']
    emb_size = paras['emb_size']
    hidden_size = paras['hidden_size']
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
    
    # train_dataset = VocabDataset(train_input_index,train_output_index)
    # train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
    #                                            batch_size=batch_size,
    #                                            collate_fn=vocab_collate_func,
    #                                            shuffle=False)

    # val_dataset = VocabDataset(val_input_index,val_output_index)
    # val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
    #                                         batch_size=1,
    #                                         collate_fn=vocab_collate_func,
    #                                         shuffle=False)

    # test_dataset = VocabDataset(test_data)
    # test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
    #                                            batch_size=BATCH_SIZE,
    #                                            collate_fn=vocab_collate_func,
    #                                            shuffle=False)

    embedding_src_weight = torch.from_numpy(srcLang.embedding_matrix).to(device)
    embedding_tgt_weight = torch.from_numpy(tgtLang.embedding_matrix).to(device)
    print(embedding_src_weight.size(), embedding_tgt_weight.size())
    if attention_type:
        encoder = EncoderRNN(src_vocab_size, emb_size, hidden_size, num_direction, embedding_weight = embedding_src_weight, device = device)
        decoder = DecoderAtten(emb_size, hidden_size, tgt_vocab_size, num_direction, embedding_weight = embedding_tgt_weight, atten_type = attention_type, device = device)
    else:      
        encoder = EncoderRNN(src_vocab_size, emb_size,hidden_size,num_direction, embedding_weight = embedding_src_weight, device = device)
        decoder = DecoderRNN(emb_size, hidden_size, tgt_vocab_size, num_direction, embedding_weight = embedding_tgt_weight, device = device)
    
    encoder, decoder = encoder.to(device), decoder.to(device)
    print('Encoder:')
    print(encoder)
    print('Decoder:')
    print(decoder)
    
    # load model 
    if model_save_info['model_path_for_resume'] is not None:
        check_point_state = torch.load(model_save_info['model_path_for_resume'])
        encoder.load_state_dict(check_point_state['encoder_state_dict'])
        decoder.load_state_dict(check_point_state['decoder_state_dict'])

    # input data and process data

    sacre_bleu_score, nltk_bleu_score, loss, tgt_sents_sacre, tgt_pred_sents_sacre, atten_weights = evaluate_single(data, encoder, decoder, criterion, tgt_max_length, tgtLang.index2word)
    

if __name__ == "__main__":
    transtype = 'zh2en'
    paras = dict( 
        teacher_forcing_ratio = 0,
        emb_size = 300,
        hidden_size = 100,
        num_direction = 2,
        learning_rate = 1e-4,
        num_epochs = 60,
        batch_size = 32, 
        attention_type = None, # None, dot_prod, general, concat

        model_save_info = dict(
            model_path = 'nmt_models/model_test/',
            epochs_per_save_model = 10,
            model_path_for_resume = None #'nmt_models/epoch_0.pth'
            )
    )
    print('paras: ', paras)
    start_train(transtype, paras)
