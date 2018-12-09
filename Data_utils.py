import numpy as np
import time
import os.path

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from config import device, PAD_token, EOS_token

import random


class VocabDataset(Dataset):
    """
    Class that represents a train/validation/test dataset that's readable for PyTorch
    Note that this class inherits torch.utils.data.Dataset
    """

    def __init__(self, train_input, train_ouput, src_clip = None, tgt_clip = None):
        """
        @param data_list: list of character
        @param target_list: list of targets

        """
        self.src_clip = src_clip
        self.tgt_clip = tgt_clip
        self.data_list, self.target_list = train_input, train_ouput
        assert (len(self.data_list) == len(self.target_list))
        #self.word2index = word2index

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, key):
        """
        Triggered when you call dataset[i]
        """
        
        train = self.data_list[key]
        label = self.target_list[key]
        if self.src_clip is not None:
            train = train[:self.src_clip]
        train_length = len(train)

        if self.tgt_clip is not None:
            label = label[:self.tgt_clip]
        label_length = len(label)
        
        return train, train_length, label, label_length


def vocab_collate_func(batch):
    """
    Customized function for DataLoader that dynamically pads the batch so that all
    data have the same length
    """
    data_list = []
    label_list = []
    train_length_list = []
    label_length_list = []
    
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for datum in batch:
        train_length_list.append(datum[1]+1)
        label_length_list.append(datum[3]+1)
    
    batch_max_input_length = np.max(train_length_list)
    batch_max_output_length = np.max(label_length_list)
    # padding
    for datum in batch:
        padded_vec = np.pad(np.array(datum[0]+[EOS_token]),
                                pad_width=((PAD_token, batch_max_input_length-datum[1]-1)),
                                mode="constant", constant_values=0)
        data_list.append(padded_vec)
        
        padded_vec = np.pad(np.array(datum[2]+[EOS_token]),
                                pad_width=((PAD_token, batch_max_output_length-datum[3]-1)),
                                mode="constant", constant_values=0)
        label_list.append(padded_vec)
        
    ind_dec_order = np.argsort(train_length_list)[::-1]
    data_list = np.array(data_list)[ind_dec_order]
    train_length_list = np.array(train_length_list)[ind_dec_order]
    label_list = np.array(label_list)[ind_dec_order]
    label_length_list = np.array(label_length_list)[ind_dec_order]
    
    #print(type(np.array(data_list)),type(np.array(label_list)))
    
    return [torch.from_numpy(data_list).to(device), 
            torch.LongTensor(train_length_list).to(device), 
            torch.from_numpy(label_list).to(device), 
            torch.LongTensor(label_length_list).to(device)]


    




