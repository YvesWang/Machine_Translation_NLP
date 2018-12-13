from config import *
import numpy as np
import time
import os
import torch.nn as nn
import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader
from Data_utils import VocabDataset, vocab_collate_func
from preprocessing_util import preposs_toekn, Lang, text2index, construct_Lang
from Encoder_Transformer import Encoder
from Decoder_Transformer import Decoder
from Transformer_config import *
from torch.optim.lr_scheduler import LambdaLR
import pickle 

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

print('The number of train samples: ', len(train_src))
print('The number of val samples: ', len(val_src))
# srcLang = Lang('src')
# srcLang.load_embedding(address_book['src_emb'], src_vocab_size)
# tgtLang = Lang('tgt')
# tgtLang.load_embedding(address_book['tgt_emb'], tgt_vocab_size)
srcLang = construct_Lang('src', src_vocab_size, address_book['src_emb'], train_src)
tgtLang = construct_Lang('tgt', tgt_vocab_size, address_book['tgt_emb'], train_tgt)
train_input_index = text2index(train_src, srcLang.word2index) #add EOS token here 
train_output_index = text2index(train_tgt, tgtLang.word2index)
val_input_index = text2index(val_src, srcLang.word2index)
val_output_index = text2index(val_tgt, tgtLang.word2index)
### save srcLang and tgtLang

#trainIters(train_loader, val_loader, encoder, decoder, num_epochs, learning_rate, srcLang, tgtLang)

batch_size = 96
train_dataset = VocabDataset(train_input_index,train_output_index, max_src_len_dataloader, max_tgt_len_dataloader)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           collate_fn=vocab_collate_func,
                                           shuffle=True)

val_dataset = VocabDataset(val_input_index,val_output_index, max_src_len_dataloader,max_tgt_len_dataloader)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                        batch_size=batch_size,
                                        collate_fn=vocab_collate_func,
                                        shuffle=False)

# test_dataset = VocabDataset(test_data)
# test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
#                                            batch_size=BATCH_SIZE,
#                                            collate_fn=vocab_collate_func,
#                                            shuffle=False)

#embedding_src_weight = torch.from_numpy(srcLang.embedding_matrix).type(torch.FloatTensor).to(device)
#embedding_tgt_weight = torch.from_numpy(tgtLang.embedding_matrix).type(torch.FloatTensor).to(device)
#print(embedding_src_weight.size(), embedding_tgt_weight.size())

# encoder = Encoder(args, embedding_src_weight)
# decoder = Decoder(args, tgtLang.vocab_size, embedding_tgt_weight)

encoder = Encoder(args, vocab_size = srcLang.vocab_size, use_position_emb = True)
decoder = Decoder(args, tgtLang.vocab_size, use_position_emb = True)

encoder, decoder = encoder.to(device), decoder.to(device)
print('Encoder:')
print(encoder)
print('Decoder:')
print(decoder)

def train(input_tensor, input_lengths, target_tensor, target_lengths,
          encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    '''
    finish train for a batch
    '''
    batch_size = input_tensor.size(0)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    tgt_max_len_batch = target_lengths.cpu().max().item()
    
    #tf.concat((tf.ones_like(self.y[:, :1])*2, self.y[:, :-1]), -1) # 2:<S>
    #print(target_tensor.size())
    target_tensor_in = torch.cat((torch.ones(batch_size,1).type_as(target_tensor)*SOS_token, target_tensor[:,:-1]),1)
    
    encoder_outputs, src_lengths = encoder(input_tensor, input_lengths)
    #print(src_lengths.size(),target_lengths.size())
    decoder_outputs, _ = decoder(target_tensor_in, src_lengths, target_lengths, encoder_outputs)  # bsz tgt vocab
    #print(decoder_outputs.view(batch_size*tgt_max_len_batch, -1).size(), target_tensor.view(batch_size*tgt_max_len_batch).size())
    loss = criterion(decoder_outputs.view(batch_size*tgt_max_len_batch, -1), target_tensor.view(batch_size*tgt_max_len_batch))
    #loss = 0
    #for i_batch in range(batch_size):
    #    loss += criterion(decoder_outputs[i_batch],target_tensor[i_batch])
    # average loss        
    #target_lengths.type_as(loss).mean()
    loss.backward()

    ### TODO
    # clip for gradient exploding 
    #print(encoder_optimizer,decoder_optimizer)
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item()#/tgt_max_len_batch  #torch.div(loss, target_lengths.type_as(loss).mean()).item()  #/target_lengths.mean()


def trainIters(train_loader, val_loader, encoder, decoder, num_epochs, learning_rate, srcLang, tgtLang,lr_decay = True):

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-09)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-09)
    
    #encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate, momentum = 0.7)
    #decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate, momentum = 0.7)
    
    
    criterion = nn.NLLLoss() #nn.NLLLoss(ignore_index=PAD_token)
    
#     def lr_foo(step,warmup_steps = 1000):
#         if step < 100:
#             return 0.01/(step+1)
#         else:
#             return min(1.0/np.sqrt(step), step*warmup_steps**(-1.5))
    
#     def lr_foo(step):
#         if step < 2000:
#             return 1e-3
#         elif step < 6000:
#             return 1e-4
#         elif step < 10000:
#             return 1e-5
#         else:
#             return 1e-6
    def lr_foo(step,warmup_steps = 4000):
        if step < 1000:
            return 0.001/(step+1)
        else:
            return 1.0/np.sqrt(args['encoder_embed_dim'])* min(1.0/np.sqrt(step), step*warmup_steps**(-1.5))
    
#    def lr_foo(step,warmup_steps = 4000):
#        return 1.0/np.sqrt(args['encoder_embed_dim'])* min(1.0/np.sqrt(step), step*warmup_steps**(-1.5))
#     def lr_foo(step,warmup_steps = 4000):
#         if step < 1000:
#             return 0.001/(step+1)
#         else:
#             return min(1.0/np.sqrt(step), step*warmup_steps**(-1.5))
    #lr_decay = False    
    if lr_decay:
        lambda_T = lambda step: lr_foo(step)   
        scheduler_encoder = LambdaLR(encoder_optimizer, lr_lambda=lambda_T)
        scheduler_decoder = LambdaLR(decoder_optimizer, lr_lambda=lambda_T)
        
    
    check_point_state = {
                'epoch': 0,
                'encoder_state_dict': encoder.state_dict(),
                'encoder_optimizer_state_dict': encoder_optimizer.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'decoder_optimizer_state_dict': decoder_optimizer.state_dict()
                }
    torch.save(check_point_state, '{}epoch_{}.pth'.format(model_path, 0))
    
    max_val_bleu = 4
    for epoch in range(num_epochs): 
        n_iter = -1
        start_time = time.time()
        for input_tensor, input_lengths, target_tensor, target_lengths in train_loader:
            n_iter += 1
            #print('start_step: ', n_iter)
            if lr_decay:
                scheduler_encoder.step()
                scheduler_decoder.step()
            
            loss = train(input_tensor, input_lengths, target_tensor, target_lengths, 
                         encoder, decoder, encoder_optimizer, decoder_optimizer, 
                         criterion)
            if n_iter % 100 == 0:
                print('Loss:', loss)
                
#                 print('Decoder parameters grad:')
#                 for p in decoder.named_parameters():
#                     print(p[0])
#                     print(p[0], ': ',  p[1].grad.data.abs().mean().item(), p[1].grad.data.abs().max().item(), p[1].data.abs().mean().item(), p[1].data.abs().max().item(), end=' ')
#                 print('\n')
#                 print('Encoder Parameters grad:')
#                 for p in encoder.named_parameters():
#                     print(p[0])
#                     print(p[0], ': ',  p[1].grad.data.abs().mean().item(), p[1].grad.data.abs().max().item(), p[1].data.abs().mean().item(), p[1].data.abs().max().item(), end=' ')
#                 print('\n')
                
                #val_bleu_sacre, val_bleu_nltk, val_loss = evaluate_batch(val_loader, encoder, decoder, criterion, tgt_max_length, tgtLang.index2word, srcLang.index2word)
                #print('epoch: [{}/{}], step: [{}/{}], train_loss:{}, val_bleu_sacre: {}, val_bleu_nltk: {}, val_loss: {}'.format(
                #    epoch, num_epochs, n_iter, len(train_loader), loss, val_bleu_sacre[0], val_bleu_nltk, val_loss))
              
        val_bleu_sacre = evaluate_batch(val_loader, encoder, decoder, tgt_max_length, tgtLang.index2word, srcLang.index2word)
        
        if max_val_bleu < val_bleu_sacre.score:
            max_val_bleu = val_bleu_sacre.score
            ### TODO save best model
            check_point_state = {
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'encoder_optimizer_state_dict': encoder_optimizer.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'decoder_optimizer_state_dict': decoder_optimizer.state_dict()
                }
            torch.save(check_point_state, '{}epoch_{}.pth'.format(model_path, epoch))
            
        #print('epoch: [{}/{}] (Running time {:.3f} min), val_bleu_sacre: {}, val_bleu_nltk: {}, val_loss: {}'.format(epoch, num_epochs, (time.time()-start_time)/60, val_bleu_sacre, val_bleu_nltk, val_loss))
        #val_bleu_sacre_beam, _, _ = evaluate_beam_batch(beam_size, val_loader, encoder, decoder, criterion, tgt_max_length, tgtLang.index2word)
        #print('epoch: [{}/{}] (Running time {:.3f} min), val_bleu_sacre_beam: {}'.format(epoch, num_epochs, (time.time()-start_time)/60, val_bleu_sacre_beam))

    return None

#def evaluation(val_loader, encoder, decoder, criterion, tgt_max_length, tgtLang.index2word, srcLang.index2word):
import sacrebleu
def fun_index2token(index_list, idx2words):
    token_list = []
    for index in index_list:
        if index == EOS_token:
            break
        else:
            token_list.append(idx2words[index])
    return token_list

#def evaluation(val_loader, encoder, decoder, criterion, tgt_max_length, tgtLang.index2word, srcLang.index2word):
import sacrebleu
def fun_index2token(index_list, idx2words):
    token_list = []
    for index in index_list:
        if index == EOS_token:
            break
        else:
            token_list.append(idx2words[index])
    return token_list

def evaluate_batch(loader, encoder, decoder, tgt_max_length, tgt_idx2words, src_idx2words):
    tgt_sents_sacre = []
    tgt_pred_sents_sacre = []
    src_sents = []

    for input_tensor, input_lengths, target_tensor, target_lengths in val_loader:
        batch_size = input_tensor.size(0)
        encoder_outputs, src_lengths = encoder(input_tensor, input_lengths)    
        ############# PADDDDDDDD #########################
        target_input = (torch.ones([batch_size, tgt_max_length],dtype = torch.long) * PAD_token).to(device)
        target_max_lengths = (torch.ones([batch_size], dtype=torch.long) * tgt_max_length).to(device)
        temp_input = target_input        
        for auto_idx in range(tgt_max_length):
            temp_input_de = torch.cat((torch.ones(batch_size,1).type_as(temp_input)*SOS_token, temp_input[:,:-1]),1)
            temp_out, _ = decoder(temp_input_de, src_lengths, encoder_out = encoder_outputs, tgt_max_lengths = target_max_lengths)
            topv, topi = temp_out.topk(1,dim = -1)
            temp_input[:,auto_idx] = topi.squeeze(-1)[:,auto_idx]
            #print(temp_out[0][0])

        target_tensor_numpy = target_tensor.cpu().numpy()
        input_tensor_numpy = input_tensor.cpu().numpy()
        idx_token_pred = temp_input.cpu().detach().numpy()
        
        for i_batch in range(batch_size):
            tgt_sent_tokens = fun_index2token(target_tensor_numpy[i_batch].tolist(), tgt_idx2words) #:target_lengths_numpy[i_batch]
            #tgt_sents_nltk.append([tgt_sent_tokens])
            tgt_sents_sacre.append(' '.join(tgt_sent_tokens))
            tgt_pred_sent_tokens = fun_index2token(idx_token_pred[i_batch].tolist(), tgt_idx2words)
            #tgt_pred_sents_nltk.append(tgt_pred_sent_tokens)
            tgt_pred_sents_sacre.append(' '.join(tgt_pred_sent_tokens))
            src_sent_tokens = fun_index2token(input_tensor_numpy[i_batch].tolist(), src_idx2words)
            src_sents.append(' '.join(src_sent_tokens))
    
    sacre_bleu_score = sacrebleu.corpus_bleu(tgt_pred_sents_sacre, [tgt_sents_sacre], smooth='exp', smooth_floor=0.0, force=False, lowercase=False,
        tokenize='none', use_effective_order=True)
    print('src:', src_sents[0])
    print('Ref: ', tgt_sents_sacre[0])
    print('pred: ', tgt_pred_sents_sacre[0])
    print('*****************************')
    print('BLUE: ', sacre_bleu_score)
    print('*****************************')
    
    return sacre_bleu_score


if not os.path.exists(model_path):
    os.makedirs(model_path)

with open(model_path+'model_params.pkl', 'wb') as f:
    model_hyparams = args
    model_hyparams['address_book'] = address_book
    model_hyparams['src_vocab_size'] = src_vocab_size 
    model_hyparams['tgt_vocab_size'] = tgt_vocab_size 
    model_hyparams['tgt_max_length'] = tgt_max_length 
    model_hyparams['max_src_len_dataloader'] = max_src_len_dataloader
    model_hyparams['max_tgt_len_dataloader'] = max_tgt_len_dataloader 
    model_hyparams['model_path'] = model_path
    pickle.dump(model_hyparams, f)
    print(model_hyparams)

num_epochs, learning_rate = 200, 1
trainIters(train_loader, val_loader, encoder, decoder, num_epochs, learning_rate, srcLang, tgtLang)
