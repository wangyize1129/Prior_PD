import torch
import numpy as np
import os
import json
from typing import List, Tuple
from tqdm import tqdm
from .diffusion_handler import Hander

class Augmenter:
    NUM_BANDS = 3  
    def __init__(self, num_diffusion_steps: int = 100,
                 device: str = 'cuda', fs: float = 100.0,
                 lambda1: float = 0.1, lambda2: float = 0.1,
                 hidden_dim: int = 128,
                 save_dir: str = '/tmp/Parkinson/multiscale_augment'):

        self.num_diffusion_steps = num_diffusion_steps
        self.device = device
        self.fs = fs
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.hidden_dim = hidden_dim
        self.save_dir = save_dir
        
        self.handler = None
        self.target_psd = None
    
    def fit(self, signals: List[torch.Tensor], labels: List[int],
            epochs: int = 50, batch_size: int = 16, lr: float = 1e-4):

        C, T = signals[0].shape
        
        self.handler = Hander(
            seq_len=T, input_dim=C,
            device=self.device, num_steps=self.num_diffusion_steps,
            fs=self.fs, hidden_dim=self.hidden_dim
        )

        parkinson_signals = [s for s, l in zip(signals, labels) if l == 0]
        if parkinson_signals:
            pk_stack = torch.stack(parkinson_signals).to(self.device)
            self.target_psd = self.handler.freq_loss.extract_target_psd(pk_stack)

        optimizer = torch.optim.AdamW(self.handler.model.parameters(), lr=lr)

        signal_tensor = torch.stack(signals).to(self.device)  # (N, C, T)
        label_tensor = torch.tensor(labels, dtype=torch.long, device=self.device)
        
        dataset = torch.utils.data.TensorDataset(signal_tensor, label_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
 
        for epoch in range(epochs):
            epoch_losses = {'diff_loss': 0, 'recon_loss': 0, 'freq_loss': 0, 'total_loss': 0}
            num_batches = 0

            for batch_signals, batch_labels in dataloader:
          
                batch_target_psd = None
                if self.target_psd is not None:
                    parkinson_mask = batch_labels == 0
                    if parkinson_mask.any():
           
                        n_parkinson = parkinson_mask.sum().item()
                        indices = torch.randint(0, len(self.target_psd), (n_parkinson,))
                        batch_target_psd = torch.zeros(len(batch_labels), self.target_psd.shape[-1], device=self.device)
                        batch_target_psd[parkinson_mask] = self.target_psd[indices]
                
                losses = self.handler.train_step(
                    batch_signals, batch_labels, optimizer,
                    self.lambda1, self.lambda2, batch_target_psd
                )
                
                for key in epoch_losses:
                    epoch_losses[key] += losses.get(key, 0)
                num_batches += 1
            
            avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}
            print(f"Epoch {epoch + 1}/{epochs}: "
                    f"Diff={avg_losses['diff_loss']:.4f}, "
                    f"Recon={avg_losses['recon_loss']:.4f}, "
                    f"Freq={avg_losses['freq_loss']:.4f}, "
                    f"Total={avg_losses['total_loss']:.4f}")
    
    def augment(self, signals: List[torch.Tensor], labels: List[int],
                parkinson_multiplier: int = 2,
                healthy_multiplier: int = 7,
                other_multiplier: int = 5) -> Tuple[List[torch.Tensor], List[int]]:

        augmented_signals = list(signals)
        augmented_labels = list(labels)
        
        multipliers = {0: parkinson_multiplier, 1: healthy_multiplier, 2: other_multiplier}
        class_names = {0: "Parkinson", 1: "Healthy", 2: "Other"}

        for class_label, multiplier in multipliers.items():
            class_indices = [i for i, l in enumerate(labels) if l == class_label]
            if len(class_indices) == 0 or multiplier <= 1:
                continue
            
            num_to_generate = len(class_indices) * (multiplier - 1)
            
            class_signals = [signals[i] for i in class_indices]
            
            with tqdm(total=num_to_generate) as pbar:
                generated = 0
                while generated < num_to_generate:
        
                    ref_idx = generated % len(class_signals)
                    ref_signal = class_signals[ref_idx].unsqueeze(0).to(self.device)



                    gen_labels = torch.tensor([class_label], device=self.device)
                    gen_signal = self.handler.generate_full_signal(
                        1, gen_labels, ref_signal, guidance_scale=0.3
                    )
                    augmented_signals.append(gen_signal.squeeze(0).cpu())
                    augmented_labels.append(class_label)
                    generated += 1
                    pbar.update(1)
        
        return augmented_signals, augmented_labels
    
    def save(self, augmented_signals: List[torch.Tensor], 
             augmented_labels: List[int],
             original_count: int):

        os.makedirs(self.save_dir, exist_ok=True)
        movement_dir = os.path.join(self.save_dir, 'movement')
        os.makedirs(movement_dir, exist_ok=True)
        
        augmented_info = []
        
        for i, (signal, label) in enumerate(zip(augmented_signals, augmented_labels)):
            filename = f"sample_{i:05d}.npy"
            filepath = os.path.join(movement_dir, filename)
            np.save(filepath, signal.numpy())
            augmented_info.append({
                'file': filename,
                'label': label,
                'is_original': i < original_count
            })

    
    
        metadata = {
            'original_count': original_count,
            'total_count': len(augmented_signals),
            'augmented_count': len(augmented_signals) - original_count,
            'samples': augmented_info,
            'parameters': {
                'num_bands': self.NUM_BANDS,
                'freq_bands': ['0.5-3Hz', '3-7Hz', '7-12Hz'],
                'num_diffusion_steps': self.num_diffusion_steps,
                'lambda1': self.lambda1,
                'lambda2': self.lambda2,
                'fs': self.fs
            }
        }
        
        with open(os.path.join(self.save_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)