# +
# Import libraries
import torch
import os
import sys
import gc

import random
from random import randint

import numpy as np
from torch.utils.data import Dataset, TensorDataset, DataLoader, ConcatDataset


device = "cuda" if torch.cuda.is_available() else "cpu"

class CMBDataset(Dataset):
    def __init__(self, CMB_loc, etadir_loc, number_of_sampls, device="cuda"):
        super().__init__()
        
        self.total = number_of_sampls
        self.device = device
        
        self.CMB_data = []
        self.eta_data = []
        
        self.CMB_files = [f for f in os.listdir(CMB_loc) if f.endswith('.npy')]
        self.eta_files = [f for f in os.listdir(etadir_loc) if f.endswith('.npy')]
        
        self.eta_labels = []
        self.g_labels = []
        self.labels = []
        
        self.load_file_paths(CMB_loc, etadir_loc)
    
    def load_file_paths(self, CMB_loc, etadir_loc):
        for file_name in self.CMB_files:
            self.CMB_data.append(CMB_loc + file_name)
            self.labels.append(1)
            self.eta_labels.append(1)
            self.g_labels.append(1)
            
        for file_name in self.eta_files:
            self.eta_data.append(etadir_loc + file_name)
        
    
    def __getitem__(self, index):
        
        # pure CMB = 0,  Implanted PHS = 1
        indicator = np.random.choice(np.arange(2), p = [0.5, 0.5])
        
        etalist = np.arange(50, 170, 20).tolist()
        eta_selec = random.choice(etalist)
        
        #glist = np.arange(3,7,1).tolist()
        glist = np.arange(12,18,1).tolist()
        g_selec = random.choice(glist)
        
        # CMB sampling
        CMBindex = randint(0,int(len(self.CMB_data)-1))
        item = torch.tensor(np.load(self.CMB_data[CMBindex])[randint(0,499)].astype('float32'))
        
        eta_specific_list = [string for string in self.eta_data if "eta"+str(int(eta_selec)) in string]
        etaindex = randint(0,int(len(eta_specific_list)-1))
        purePHS =torch.tensor(np.load(eta_specific_list[etaindex])[randint(0,499)].astype('float32'))
        
        # Pure CMB
        if indicator==0:
            pass
            
        # CMB + PHS
        else:
            item = item + (torch.tensor(g_selec))*purePHS
        
        
        item = item.reshape(1,item.shape[0],item.shape[1])
        
        label = [np.float32(0 if indicator<1 else 1)]
        label = torch.tensor(label)
        return item.to(self.device), label.to(self.device)
    
    def __len__(self):
        return self.total
    
    
    
def prepare_data(batch_size=32, num_workers=0, train_sample_size=10000, test_sample_size=2000):

    train_dataset = CMBDataset(CMB_loc = '/Work/CMB_ML/param_CNN/Data_hp/',
                               etadir_loc = '/Work/CMB_ML/param_CNN/Data/',
                               number_of_sampls = train_sample_size)

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


    test_dataset = CMBDataset(CMB_loc = '/Work/CMB_ML/param_CNN/Data_hp/',
                              etadir_loc = '/Work/CMB_ML/param_CNN/Data/',
                              number_of_sampls = test_sample_size)

    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    classes = ('background', 'signal')
    return trainloader, testloader, classes
