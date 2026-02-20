import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import List
from tqdm import tqdm

from .augmenter import Augmenter


class AugDataset(Dataset):
    def __init__(self, original_dataset, augmenter: Augmenter,
                 parkinson_multiplier: int = 2,
                 healthy_multiplier: int = 7,
                 other_multiplier: int = 5,
                 training_epochs: int = 50,
                 batch_size: int = 16,
                 lr: float = 1e-4,
                 force_regenerate: bool = False):

        self.augmenter = augmenter
        self.batch_size = batch_size
        self.lr = lr
        
   
        if hasattr(original_dataset, 'dataset'):

            self.original_samples = [original_dataset.dataset.samples[i] for i in original_dataset.indices]
            self.original_labels = [original_dataset.dataset.labels_list[i] for i in original_dataset.indices]
        else:
            self.original_samples = original_dataset.samples
            self.original_labels = original_dataset.labels_list
        
        if isinstance(self.original_samples[0], dict):
            original_signals = [s['movement'] for s in self.original_samples]
        else:
            original_signals = self.original_samples

        if not force_regenerate and augmenter.check_exists():
    
            self.signals, self.labels = augmenter.load()
        else:
            augmenter.fit(original_signals, self.original_labels, epochs=training_epochs,
                         batch_size=batch_size, lr=lr)
            self.signals, self.labels = augmenter.augment(
                original_signals, self.original_labels,
                parkinson_multiplier, healthy_multiplier, other_multiplier
            )
            augmenter.save(self.signals, self.labels, len(original_signals))

        
        
        self.labels_onehot = F.one_hot(
            torch.tensor(self.labels), num_classes=3
        ).float()



        

        self.max_len = self.signals[0].shape[1] if self.signals[0].dim() == 2 else self.signals[0].shape[0]
        self.num_class = 3
        self.feat_in = self.signals[0].shape[0] if self.signals[0].dim() == 2 else self.signals[0].shape[1]
    
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        signal = self.signals[idx]
        label = self.labels_onehot[idx]
        if signal.dim() == 2 and signal.shape[0] < signal.shape[1]:
            feats = signal.transpose(0, 1)
        else:
            feats = signal
        return feats, label
    
    def proterty(self):
        return self.max_len, self.num_class, self.feat_in
