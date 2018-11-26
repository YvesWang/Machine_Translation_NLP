# config:
import torch
import numpy as np
from nltk.translate import bleu_score 
from config import SOS_token, EOS_token
import sacrebleu

def fun_index2token(index_list, idx2words):
    token_list = []
    for index in index_list:
        if index == EOS_token:
            break
        else:
            token_list.append(idx2words[index])
    return token_list

def evaluate(loader, encoder, decoder, criterion, tgt_max_length, tgt_idx2words):
    """
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    loss_all = []
    tgt_sents_nltk = []
    tgt_sents_sacre = []
    tgt_pred_sents_nltk = []
    tgt_pred_sents_sacre = []
    with torch.no_grad():
        for input_tensor, input_lengths, target_tensor, target_lengths in loader:
            batch_size = input_tensor.size(0)
            encoder_hidden = encoder.initHidden(batch_size)

            encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden, input_lengths)
            decoder_input = torch.tensor([[SOS_token]*batch_size], device=device).transpose(0,1)
            decoder_hidden = encoder_hidden
            decoding_token_index = 0
            loss = 0 
            target_lengths = target_lengths.cpu().squeeze().item()
            tgt_pred_sentence = []
            while decoding_token_index < tgt_max_length:
                decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, input_lengths, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.detach()  # detach from history as input
                idx_token_pred = decoder_input.squeeze().item()
                tgt_pred_sentence.append(idx_token_pred)
                if decoding_token_index < target_lengths:
                    loss += criterion(decoder_output, target_tensor[:,decoding_token_index])
                decoding_token_index += 1
                if idx_token_pred == EOS_token:
                    break
            tgt_sent_tokens = fun_index2token(target_tensor[:,:target_lengths].cpu().squeeze().tolist(), tgt_idx2words)
            tgt_sents_nltk.append([tgt_sent_tokens])
            tgt_sents_sacre.append(' '.join(tgt_sent_tokens))
            tgt_pred_sent_tokens = fun_index2token(tgt_pred_sentence, tgt_idx2words)
            tgt_pred_sents_nltk.append(tgt_pred_sent_tokens)
            tgt_pred_sents_sacre.append(' '.join(tgt_pred_sent_tokens))
            # if decoding_token_index == 0:
            #     print('dddddddddd',src_sents[-1],tgt_sents[-1])
            # if target_lengths == 0:
            #     print('fffffffffff',src_sents[-1],tgt_sents[-1])
            loss_all.append(loss.item()/min(decoding_token_index, target_lengths))
    nltk_bleu_score = bleu_score.corpus_bleu(tgt_sents_nltk, tgt_pred_sents_nltk)
    sacre_bleu_score = sacrebleu.corpus_bleu(tgt_pred_sents_sacre, [tgt_sents_sacre], smooth='exp', smooth_floor=0.0, force=False, lowercase=False,
        tokenize='none', use_effective_order=True)
    loss = np.mean(loss_all)

    return sacre_bleu_score[0], nltk_bleu_score*100, loss

def evaluate_batch(loader, encoder, decoder, criterion, tgt_max_length, tgt_idx2words):
    """
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    loss_all = []
    #tgt_sents_nltk = []
    tgt_sents_sacre = []
    #tgt_pred_sents_nltk = []
    tgt_pred_sents_sacre = []
    with torch.no_grad():
        for input_tensor, input_lengths, target_tensor, target_lengths in loader:
            batch_size = input_tensor.size(0)
            encoder_hidden = encoder.initHidden(batch_size)

            encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden, input_lengths)
            decoder_input = torch.tensor([[SOS_token]*batch_size], device=device).transpose(0,1)
            decoder_hidden = encoder_hidden

            decoding_token_index = 0
            loss = 0 
            target_lengths_numpy = target_lengths.cpu().numpy()
            #sent_not_end_index = list(range(batch_size))
            tgt_pred_sentence = ### 
            idx_token_pred = np.zeros(batch_size, tgt_max_length)
            while decoding_token_index < tgt_max_length:
                decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, input_lengths, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.detach()  # detach from history as input
                idx_token_pred_step = decoder_input.cpu().squeeze(1).numpy()
                idx_token_pred[:, decoding_token_index] = idx_token_pred_step
                loss += criterion(decoder_output, target_tensor[:,decoding_token_index])
                decoding_token_index += 1
                end_or_not = idx_token_pred_step != EOS_token
                sent_not_end_index = list(np.where(end_or_not)[0])
                if len(sent_not_end_index) == 0:
                    break

            target_tensor_numpy = target_tensor.cpu().numpy()
            for i_batch in range(batch_size):
                tgt_sent_tokens = fun_index2token(target_tensor_numpy[i_batch].tolist(), tgt_idx2words) #:target_lengths_numpy[i_batch]
                #tgt_sents_nltk.append([tgt_sent_tokens])
                tgt_sents_sacre.append(' '.join(tgt_sent_tokens))
                tgt_pred_sent_tokens = fun_index2token(idx_token_pred[i_batch].tolist(), tgt_idx2words)
                #tgt_pred_sents_nltk.append(tgt_pred_sent_tokens)
                tgt_pred_sents_sacre.append(' '.join(tgt_pred_sent_tokens))
            # if decoding_token_index == 0:
            #     print('dddddddddd',src_sents[-1],tgt_sents[-1])
            # if target_lengths == 0:
            #     print('fffffffffff',src_sents[-1],tgt_sents[-1])
            loss_all.append(loss.item()/decoding_token_index)
    #nltk_bleu_score = bleu_score.corpus_bleu(tgt_sents_nltk, tgt_pred_sents_nltk)
    sacre_bleu_score = sacrebleu.corpus_bleu(tgt_pred_sents_sacre, [tgt_sents_sacre], smooth='exp', smooth_floor=0.0, force=False, lowercase=False,
        tokenize='none', use_effective_order=True)
    loss = np.mean(loss_all)
    return sacre_bleu_score, None, loss
