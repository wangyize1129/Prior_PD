import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from DataAugment import Augmenter, AugDataset
from tqdm import tqdm
import numpy as np

class Adapter(Dataset):
    def __init__(self, original_dataset,
                 num_diffusion_steps: int = 100,
                 training_epochs: int = 50,
                 batch_size: int = 16,
                 lr: float = 1e-4,
                 hidden_dim: int = 128,
                 parkinson_multiplier: int = 2,
                 healthy_multiplier: int = 7,
                 other_multiplier: int = 5,
                 lambda1: float = 0.1,
                 lambda2: float = 0.1,
                 device: str = 'cuda',
                 seed: int = 42,
                 augment_dir: str = '/tmp/Parkinson/multiscale_augment'):

        self.original_dataset = original_dataset
        self.device = device
        self.augment_dir = augment_dir
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.max_len = original_dataset.max_len
        self.num_class = original_dataset.num_class
        self.feat_in = original_dataset.feat_in

        self.augmenter = Augmenter(
            num_diffusion_steps=num_diffusion_steps,
            hidden_dim=hidden_dim,
            device=device,
            lambda1=lambda1,
            lambda2=lambda2,
            save_dir=augment_dir
        )

        wrapped_dataset = self._wrap_dataset(original_dataset)
        
      
        self.augmented_dataset = AugDataset(
            wrapped_dataset,
            self.augmenter,
            parkinson_multiplier=parkinson_multiplier,
            healthy_multiplier=healthy_multiplier,
            other_multiplier=other_multiplier,
            training_epochs=training_epochs,
            batch_size=batch_size,
            lr=lr
        )

    def _wrap_dataset(self, original_dataset):
        class WrappedDataset:
            def __init__(self, original_dataset):
                self.samples = []
                self.labels_list = []
                for idx in tqdm(range(len(original_dataset))):
                    feats, label = original_dataset[idx]
                    movement = feats.transpose(0, 1)  # (d, L)
                    label_int = torch.argmax(label).item()
                    sample = {
                        'movement': movement,
                        'metadata': torch.zeros(1),
                        'label': label
                    }
                    self.samples.append(sample)
                    self.labels_list.append(label_int)
                self.indices = list(range(len(self.samples)))
                from collections import Counter
                dist = Counter(self.labels_list)


        wrapped = WrappedDataset(original_dataset)

        class SubsetWrapper:
            def __init__(self, dataset):
                self.dataset = dataset
                self.indices = dataset.indices
        
        return SubsetWrapper(wrapped)
    
    def __len__(self):
        return len(self.augmented_dataset)
    
    def __getitem__(self, idx):

        return self.augmented_dataset[idx]
    
    def proterty(self):
        return self.max_len, self.num_class, self.feat_in


def Application(trainset, args,
                parkinson_multiplier: int = 2,
                healthy_multiplier: int = 7,
                other_multiplier: int = 5,
                training_epochs: int = 50,
                num_diffusion_steps: int = 100,
                batch_size: int = 16,
                lr: float = 1e-4,
                hidden_dim: int = 128,
                lambda1: float = 0.1,
                lambda2: float = 0.1,
                augment_dir: str = '/tmp/Parkinson/multiscale_augment'):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    augmented_trainset = Adapter(
        trainset,
        num_diffusion_steps=num_diffusion_steps,
        training_epochs=training_epochs,
        batch_size=batch_size,
        lr=lr,
        hidden_dim=hidden_dim,
        parkinson_multiplier=parkinson_multiplier,
        healthy_multiplier=healthy_multiplier,
        other_multiplier=other_multiplier,
        lambda1=lambda1,
        lambda2=lambda2,
        device=device,
        seed=args.seed,
        augment_dir=augment_dir
    )
    
    return augmented_trainset
