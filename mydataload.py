import pickle
import numpy as np
import pandas as pd
import torch
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch.nn.functional as F
import sys, argparse, os
import json
import glob
from utils import *
from sklearn.model_selection import train_test_split
from aeon.datasets import load_classification

class ParkinsonDataset(Dataset):
    def __init__(self, args, split='train', seed=0):
        super().__init__()
        self.args = args
        self.split = split
        self.seq_len = 976  
        self.num_channels = 132  


        patients_dir = r'/tmp/Parkinson/original_dataset/patients'
        data_dir = r'/tmp/Parkinson/original_dataset/preprocessed/movement'


        all_data = []
        all_labels = []

        json_files = glob.glob(os.path.join(patients_dir, '*.json'))

        
  
        bin_files = glob.glob(os.path.join(data_dir, '*.bin'))

        
        
        for json_file in json_files:

            json_basename = os.path.splitext(os.path.basename(json_file))[0]  # patient_001
            patient_num = json_basename.split('_')[1]  # 001
            bin_file = os.path.join(data_dir, f'{patient_num}_ml.bin')

           

            with open(json_file, 'r') as f:
                patient_info = json.load(f)
            
            condition = patient_info.get('condition', '')

            if condition == "Parkinson's":
                label = 0
            elif condition == "Healthy":
                label = 1
            else:
                label = 2  # Other
            
            all_data.append(bin_file)
            all_labels.append(label)
        
        all_data = np.array(all_data)
        all_labels = np.array(all_labels)


        train_data, val_data, train_labels, val_labels = train_test_split(
            all_data, all_labels, 
            test_size=0.3, 
            random_state=seed, 
            stratify=all_labels
        )
        
        if split == 'train':
            self.data_files = train_data
            self.labels = train_labels
        else:  # test/val
            self.data_files = val_data
            self.labels = val_labels
        
 
        self.label = F.one_hot(torch.tensor(self.labels), num_classes=3).float()
        self.feat_in = self.num_channels  
        self.max_len = self.seq_len
        self.num_class = 3
    
    def __getitem__(self, idx):
        bin_file = self.data_files[idx]
        data = np.fromfile(bin_file, dtype=np.float32).reshape(self.num_channels, self.seq_len)
        feats = torch.from_numpy(data).permute(1, 0).float()  # L*d

        label = self.label[idx].float()
        return feats, label
    
    def __len__(self):
        return len(self.labels)
    
    def proterty(self):
        return self.max_len, self.num_class, self.feat_in


class loadorean(Dataset):
    def __init__(self, args, split='train', seed=0):
        super().__init__()
        self.args = args
        self.split = split
        
        if args.dataset == 'JapaneseVowels':
            self.seq_len = 29
        elif args.dataset == 'SpokenArabicDigits':
            self.seq_len = 93
        elif args.dataset == 'CharacterTrajectories':
            self.seq_len = 182
        elif args.dataset == 'InsectWingbeat':
            self.seq_len = 78
        if split in ['train']:

            if args.dataset == 'InsectWingbeat':
                Xtr, ytr, meta =load_classification(name='InsectWingbeat', split='train',extract_path='../timeclass/dataset/')
            else:
                Xtr, ytr, meta = load_classification(name=args.dataset,split='train')
            # print(Xtr.shape)
            word_to_idx = {}
            for i in range(len(meta['class_values'])):
                word_to_idx[meta['class_values'][i]]=i
                
            ytr = [word_to_idx[i] for i in ytr]
            self.label =  F.one_hot(torch.tensor(ytr)).float()    
            self.FeatList = Xtr

        elif split == 'test': 
            if args.dataset == 'InsectWingbeat':
                Xte, yte, meta =load_classification(name='InsectWingbeat', split='test',extract_path='../timeclass/dataset/')
            else:
                Xte, yte, meta = load_classification(name=args.dataset,split='test')
            word_to_idx = {}
            for i in range(len(meta['class_values'])):
                word_to_idx[meta['class_values'][i]]=i

            yte = [word_to_idx[i] for i in yte]
            self.label = F.one_hot(torch.tensor(yte)).float()
            self.FeatList = Xte
        
        self.feat_in = self.FeatList[0].shape[0]        
        self.max_len = self.seq_len
        self.num_class =  self.label.shape[-1]
    def __getitem__(self, idx):

        feats = torch.from_numpy(self.FeatList[idx]).permute(1,0).float() #L*d
        min_len =self.seq_len
        feats = F.pad(feats, pad=(0, 0, min_len-feats.shape[0], 0))

        label = self.label[idx].float()
        return feats, label
         
    def __len__(self):
         return len(self.label)
    
    def proterty(self):
        return self.max_len,self.num_class,self.feat_in