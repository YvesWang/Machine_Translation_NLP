# config:
import torch
from nltk.translate.bleu_score import corpus_bleu

def index2token():
    return None

def evaluate(loader, encoder, decoder, criterion, tgt_max_length):
    """
    """    
    loss_all = []
    src_sents = []
    tgt_sents = []
    with torch.no_grad():
        for input_tensor, input_lengths, target_tensor, target_lengths in loader:
            input_lengths = input_lengths.cpu().squeeze().item()
            src_sents.append(input_tensor[:,:input_lengths].cpu().tolist())

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
                decoder_output, decoder_hidden, _ = decoder(
                    decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.detach()  # detach from history as input
                idx_token_ml = decoder_input.squeeze()
                tgt_sentence.append(idx_token_ml)
                if decoding_token_index < target_lengths:
                    loss += criterion(decoder_output, target_tensor[:,decoding_token_index])
                if idx_token_ml == EOS_token:
                    break
                decoding_token_index += 1
            tgt_sents.append(tgt_sentence)
            loss_all.append(loss.item())
    bleu_score = corpus_bleu(src_sents, tgt_sents)
    loss = np.mean(loss_all)

    return bleu_score, loss