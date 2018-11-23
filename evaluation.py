# config:
import torch
import numpy as np
from nltk.translate.bleu_score import corpus_bleu
from config import SOS_token, EOS_token

def fun_index2token(index_list, idx2words):
    return [idx2words[index] for index in index_list]

def evaluate(loader, encoder, decoder, criterion, tgt_max_length, src_idx2words, tgt_idx2words):
    """
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    loss_all = []
    src_sents = []
    tgt_sents = []
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
            tgt_sentence = []
            while decoding_token_index < tgt_max_length:
                decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, input_lengths, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.detach()  # detach from history as input
                idx_token_ml = decoder_input.squeeze().item()
                tgt_sentence.append(idx_token_ml)
                if decoding_token_index < target_lengths:
                    loss += criterion(decoder_output, target_tensor[:,decoding_token_index])
                decoding_token_index += 1
                if idx_token_ml == EOS_token:
                    break
            input_lengths = input_lengths.cpu().squeeze().item()
            input_tokens = fun_index2token(input_tensor[:,:input_lengths].cpu().squeeze().tolist(),src_idx2words)
            src_sents.append([input_tokens])
            tgt_tokens = fun_index2token(tgt_sentence,tgt_idx2words)
            tgt_sents.append(tgt_tokens)
            if decoding_token_index == 0:
                print('dddddddddd',src_sents[-1],tgt_sents[-1])
            if target_lengths == 0:
                print('fffffffffff',src_sents[-1],tgt_sents[-1])
            loss_all.append(loss.item()/min(decoding_token_index,target_lengths))
    bleu_score = corpus_bleu(src_sents, tgt_sents)
    loss = np.mean(loss_all)

    return bleu_score, loss
