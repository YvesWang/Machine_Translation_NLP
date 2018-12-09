# config:
import torch
import numpy as np
from nltk.translate import bleu_score 
from config import SOS_token, EOS_token, PAD_token
import sacrebleu
import beam

def fun_index2token(index_list, idx2words):
    token_list = []
    for index in index_list:
        if index == EOS_token:
            break
        else:
            token_list.append(idx2words[index])
    return token_list

def evaluate_batch(loader, encoder, decoder, criterion, tgt_max_length, tgt_idx2words, src_idx2words):
    """
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    loss_all = []
    #tgt_sents_nltk = []
    tgt_sents_sacre = []
    #tgt_pred_sents_nltk = []
    tgt_pred_sents_sacre = []
    src_sents = []
    with torch.no_grad():
        for input_tensor, input_lengths, target_tensor, target_lengths in loader:
            batch_size = input_tensor.size(0)
            encoder_hidden, encoder_cell = encoder.initHidden(batch_size)
            encoder_outputs, encoder_hidden, encoder_cell = encoder(input_tensor, encoder_hidden, input_lengths, encoder_cell)
            decoder_input = torch.tensor([[SOS_token]*batch_size], device=device).transpose(0,1)
            decoder_hidden,decoder_cell = encoder_hidden,encoder_cell

            decoding_token_index = 0
            loss = 0 
            target_lengths_numpy = target_lengths.cpu().numpy()
            #sent_not_end_index = list(range(batch_size))
            idx_token_pred = np.zeros((batch_size, tgt_max_length), dtype=np.int)
            while decoding_token_index < tgt_max_length:
                decoder_output, decoder_hidden, _, decoder_cell = decoder(decoder_input, decoder_hidden, input_lengths, encoder_outputs,decoder_cell)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.detach()  # detach from history as input
                idx_token_pred_step = decoder_input.cpu().squeeze(1).numpy()
                idx_token_pred[:, decoding_token_index] = idx_token_pred_step
                if decoding_token_index < target_lengths_numpy.max():
                    loss += criterion(decoder_output, target_tensor[:,decoding_token_index])
                decoding_token_index += 1
                end_or_not = idx_token_pred_step != EOS_token
                sent_not_end_index = list(np.where(end_or_not)[0])
                if len(sent_not_end_index) == 0:
                    break

            target_tensor_numpy = target_tensor.cpu().numpy()
            input_tensor_numpy = input_tensor.cpu().numpy()
            for i_batch in range(batch_size):
                tgt_sent_tokens = fun_index2token(target_tensor_numpy[i_batch].tolist(), tgt_idx2words) #:target_lengths_numpy[i_batch]
                #tgt_sents_nltk.append([tgt_sent_tokens])
                tgt_sents_sacre.append(' '.join(tgt_sent_tokens))
                tgt_pred_sent_tokens = fun_index2token(idx_token_pred[i_batch].tolist(), tgt_idx2words)
                #tgt_pred_sents_nltk.append(tgt_pred_sent_tokens)
                tgt_pred_sents_sacre.append(' '.join(tgt_pred_sent_tokens))
                src_sent_tokens = fun_index2token(input_tensor_numpy[i_batch].tolist(), src_idx2words)
                src_sents.append(' '.join(src_sent_tokens))
            # if decoding_token_index == 0:
            #     print('dddddddddd',src_sents[-1],tgt_sents[-1])
            # if target_lengths == 0:
            #     print('fffffffffff',src_sents[-1],tgt_sents[-1])
            loss_all.append(loss.item()/decoding_token_index)
    #nltk_bleu_score = bleu_score.corpus_bleu(tgt_sents_nltk, tgt_pred_sents_nltk)
    sacre_bleu_score = sacrebleu.corpus_bleu(tgt_pred_sents_sacre, [tgt_sents_sacre], smooth='exp', smooth_floor=0.0, force=False, lowercase=False,
        tokenize='none', use_effective_order=True)
    loss = np.mean(loss_all)
    if True:
        random_sample = np.random.randint(len(tgt_pred_sents_sacre))
        print('src:', src_sents[random_sample])
        print('Ref: ', tgt_sents_sacre[random_sample])
        print('pred: ', tgt_pred_sents_sacre[random_sample])
    return sacre_bleu_score, None, loss

def evaluate_beam_batch(beam_size, loader, encoder, decoder, criterion, tgt_max_length, tgt_idx2words):
    """
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    have_cell = True
    if encoder.get_rnn_type == 'GRU':
        have_cell = False
    #have_cell = True
    #loss_all = []
    #tgt_sents_nltk = []
    tgt_sents_sacre = []
    #tgt_pred_sents_nltk = []
    tgt_pred_sents_sacre = []
    with torch.no_grad():
        for input_tensor, input_lengths, target_tensor, target_lengths in loader:
            batch_size = input_tensor.size(0) #int 
            encoder_hidden,encoder_cell = encoder.initHidden(batch_size)
            encoder_outputs, encoder_hidden,encoder_cell = encoder(input_tensor, encoder_hidden, input_lengths, encoder_cell)

            beamers = [beam.beam(beam_size, min_length=0, n_best=1) for i in range(batch_size)]
            encoder_max_len, decoder_hidden_dim = encoder_outputs.size(1), encoder_outputs.size(2) #int
            assert(encoder_max_len==input_lengths.max())
            num_layers = encoder_hidden.size(0)
            encoder_hiddden_beam = encoder_hidden.unsqueeze(2).expand(num_layers, batch_size, beam_size, decoder_hidden_dim).contiguous().view(num_layers, batch_size*beam_size, decoder_hidden_dim)
            decoder_hidden = encoder_hiddden_beam
            if have_cell:
                encoder_cell_beam = encoder_cell.unsqueeze(2).expand(num_layers, batch_size, beam_size, decoder_hidden_dim).contiguous().view(num_layers, batch_size*beam_size, decoder_hidden_dim)
                decoder_cell = encoder_cell_beam
            input_lengths_beam = input_lengths.unsqueeze(1).expand(batch_size, beam_size).contiguous().view(batch_size*beam_size)
            encoder_outputs_beam = encoder_outputs.unsqueeze(1).expand(batch_size, beam_size, encoder_max_len, decoder_hidden_dim).contiguous().view(batch_size*beam_size, 
                encoder_max_len, decoder_hidden_dim)

            #loss = 0
            for decoding_token_index in range(tgt_max_length):
                decoder_input = torch.stack([beamer.next_ts[-1] for beamer in beamers], dim=0).unsqueeze(-1).view(batch_size*beam_size, 1).to(device)
                decoder_output, decoder_hidden, _, decoder_cell = decoder(decoder_input, decoder_hidden, input_lengths_beam, encoder_outputs_beam, decoder_cell)
                vocab_size = decoder_output.size(1)
                decoder_output_beam, decoder_hidden_beam = decoder_output.view(batch_size, beam_size, vocab_size), decoder_hidden.view(1, batch_size, beam_size, decoder_hidden_dim)
                if have_cell:
                    decoder_cell_beam = decoder_cell.view(1, batch_size, beam_size, decoder_hidden_dim)
                decoder_input_list = []
                decoder_hidden_list = []
                decoder_cell_list = []
                flag_stop = True
                for i_batch in range(batch_size):
                    beamer = beamers[i_batch]
                    if beamer.stopByEOS == False:
                        beamer.advance(decoder_output_beam[i_batch])
                        decoder_hidden_list.append(decoder_hidden_beam[:, i_batch, :, :].index_select(dim=1,index=beamer.prev_ps[-1]))
                        if have_cell:
                            decoder_cell_list.append(decoder_cell_beam[:, i_batch, :, :].index_select(dim=1,index=beamer.prev_ps[-1]))
                        decoder_input_list.append(beamer.next_ts[-1])
                        flag_stop = False
                    else:
                        decoder_hidden_list.append(decoder_hidden_beam[:,i_batch,:,:])
                        if have_cell:
                            decoder_cell_list.append(decoder_cell_beam[:,i_batch,:,:])
                        decoder_input_list.append(torch.LongTensor(beam_size).fill_(PAD_token).to(device))
                if flag_stop:
                    break
                decoder_input = torch.stack(decoder_input_list, 0).view(batch_size*beam_size, 1)
                decoder_hidden = torch.stack(decoder_hidden_list, 1).view(1, batch_size*beam_size, decoder_hidden_dim)
                if have_cell:
                    decoder_cell = torch.stack(decoder_cell_list, 1).view(1, batch_size*beam_size, decoder_hidden_dim)

            target_tensor_numpy = target_tensor.cpu().numpy()
            for i_batch in range(batch_size):
                beamer = beamers[i_batch]
                paths_sort = sorted(beamer.finish_paths, key=lambda x: x[0], reverse=True)
                if len(paths_sort) == 0:
                    best_path = (beamer.scores[0], len(beamer.prev_ps), 0)
                else:
                    best_path = paths_sort[0]
                score_best_path, tokens_best_path = beamer.get_pred_sentence(best_path)
                # ground true
                tgt_sent_tokens = fun_index2token(target_tensor_numpy[i_batch].tolist(), tgt_idx2words)
                tgt_sents_sacre.append(' '.join(tgt_sent_tokens))
                # prediction
                tgt_pred_sent_tokens = fun_index2token(tokens_best_path, tgt_idx2words)
                tgt_pred_sents_sacre.append(' '.join(tgt_pred_sent_tokens))

    #nltk_bleu_score = bleu_score.corpus_bleu(tgt_sents_nltk, tgt_pred_sents_nltk)
    sacre_bleu_score = sacrebleu.corpus_bleu(tgt_pred_sents_sacre, [tgt_sents_sacre], smooth='exp', smooth_floor=0.0, force=False, lowercase=False,
        tokenize='none', use_effective_order=True)
    if True:
        random_sample = 300 #np.random.randint(len(tgt_pred_sents_sacre))
        print('Ref: ', tgt_sents_sacre[random_sample])
        print('pred: ', tgt_pred_sents_sacre[random_sample])
    return sacre_bleu_score, None, None



def evaluate_single(data, encoder, decoder, criterion, tgt_max_length, tgt_idx2words, src_idx2words):
    """
    """
    input_tensor, input_lengths, target_tensor, target_lengths = data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    loss_all = []
    #tgt_sents_nltk = []
    src_sents = []
    tgt_sents_sacre = []
    #tgt_pred_sents_nltk = []
    tgt_pred_sents_sacre = []
    with torch.no_grad():
        batch_size = input_tensor.size(0)
        encoder_hidden = encoder.initHidden(batch_size)

        encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden, input_lengths)
        decoder_input = torch.tensor([[SOS_token]*batch_size], device=device).transpose(0,1)
        decoder_hidden = encoder_hidden

        decoding_token_index = 0
        loss = 0 
        target_lengths_numpy = target_lengths.cpu().numpy()
        #sent_not_end_index = list(range(batch_size))
        idx_token_pred = np.zeros((batch_size, tgt_max_length))
        while decoding_token_index < tgt_max_length:
            decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, input_lengths, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.detach()  # detach from history as input
            idx_token_pred_step = decoder_input.cpu().squeeze(1).numpy()
            idx_token_pred[:, decoding_token_index] = idx_token_pred_step
            if decoding_token_index < target_lengths_numpy.max():
                loss += criterion(decoder_output, target_tensor[:,decoding_token_index])
            decoding_token_index += 1
            end_or_not = idx_token_pred_step != EOS_token
            sent_not_end_index = list(np.where(end_or_not)[0])
            if len(sent_not_end_index) == 0:
                break

        target_tensor_numpy = target_tensor.cpu().numpy()
        input_tensor_numpy = input_tensor.cpu().numpy()
        for i_batch in range(batch_size):
            tgt_sent_tokens = fun_index2token(target_tensor_numpy[i_batch].tolist(), tgt_idx2words) #:target_lengths_numpy[i_batch]
            #tgt_sents_nltk.append([tgt_sent_tokens])
            tgt_sents_sacre.append(' '.join(tgt_sent_tokens))
            tgt_pred_sent_tokens = fun_index2token(idx_token_pred[i_batch].tolist(), tgt_idx2words)
            #tgt_pred_sents_nltk.append(tgt_pred_sent_tokens)
            tgt_pred_sents_sacre.append(' '.join(tgt_pred_sent_tokens))
            src_sent_tokens = fun_index2token(input_tensor_numpy[i_batch].tolist(), src_idx2words)
            src_sents.append(' '.join(src_sent_tokens))
        # if decoding_token_index == 0:
        #     print('dddddddddd',src_sents[-1],tgt_sents[-1])
        # if target_lengths == 0:
        #     print('fffffffffff',src_sents[-1],tgt_sents[-1])
        loss_all.append(loss.item()/decoding_token_index)
    #nltk_bleu_score = bleu_score.corpus_bleu(tgt_sents_nltk, tgt_pred_sents_nltk)
    sacre_bleu_score = sacrebleu.corpus_bleu(tgt_pred_sents_sacre, [tgt_sents_sacre], smooth='exp', smooth_floor=0.0, force=False, lowercase=False,
        tokenize='none', use_effective_order=True)
    loss = np.mean(loss_all)
    return sacre_bleu_score, None, loss, tgt_sents_sacre, tgt_pred_sents_sacre, src_sents, None # atten


# def evaluate(loader, encoder, decoder, criterion, tgt_max_length, tgt_idx2words):
#     """
#     """
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
#     loss_all = []
#     tgt_sents_nltk = []
#     tgt_sents_sacre = []
#     tgt_pred_sents_nltk = []
#     tgt_pred_sents_sacre = []
#     with torch.no_grad():
#         for input_tensor, input_lengths, target_tensor, target_lengths in loader:
#             batch_size = input_tensor.size(0)
#             encoder_hidden = encoder.initHidden(batch_size)

#             encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden, input_lengths)
#             decoder_input = torch.tensor([[SOS_token]*batch_size], device=device).transpose(0,1)
#             decoder_hidden = encoder_hidden
#             decoding_token_index = 0
#             loss = 0 
#             target_lengths = target_lengths.cpu().squeeze().item()
#             tgt_pred_sentence = []
#             while decoding_token_index < tgt_max_length:
#                 decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, input_lengths, encoder_outputs)
#                 topv, topi = decoder_output.topk(1)
#                 decoder_input = topi.detach()  # detach from history as input
#                 idx_token_pred = decoder_input.squeeze().item()
#                 tgt_pred_sentence.append(idx_token_pred)
#                 if decoding_token_index < target_lengths:
#                     loss += criterion(decoder_output, target_tensor[:,decoding_token_index])
#                 decoding_token_index += 1
#                 if idx_token_pred == EOS_token:
#                     break
#             tgt_sent_tokens = fun_index2token(target_tensor[:,:target_lengths].cpu().squeeze().tolist(), tgt_idx2words)
#             tgt_sents_nltk.append([tgt_sent_tokens])
#             tgt_sents_sacre.append(' '.join(tgt_sent_tokens))
#             tgt_pred_sent_tokens = fun_index2token(tgt_pred_sentence, tgt_idx2words)
#             tgt_pred_sents_nltk.append(tgt_pred_sent_tokens)
#             tgt_pred_sents_sacre.append(' '.join(tgt_pred_sent_tokens))
#             # if decoding_token_index == 0:
#             #     print('dddddddddd',src_sents[-1],tgt_sents[-1])
#             # if target_lengths == 0:
#             #     print('fffffffffff',src_sents[-1],tgt_sents[-1])
#             loss_all.append(loss.item()/min(decoding_token_index, target_lengths))
#     nltk_bleu_score = bleu_score.corpus_bleu(tgt_sents_nltk, tgt_pred_sents_nltk)
#     sacre_bleu_score = sacrebleu.corpus_bleu(tgt_pred_sents_sacre, [tgt_sents_sacre], smooth='exp', smooth_floor=0.0, force=False, lowercase=False,
#         tokenize='none', use_effective_order=True)
#     loss = np.mean(loss_all)
#     return sacre_bleu_score[0], nltk_bleu_score*100, loss
