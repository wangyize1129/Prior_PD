import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.signal import welch
from typing import Tuple


class FrequencyLoss(nn.Module):
    def __init__(self, fs: float = 100.0, freq_low: float = 3.0, freq_high: float = 7.0):
        super().__init__()
        self.fs = fs
        self.freq_low = freq_low
        self.freq_high = freq_high
    
    def compute_psd(self, signal: torch.Tensor) -> Tuple[torch.Tensor, np.ndarray]:
        signal_np = signal.detach().cpu().numpy()
        if signal_np.ndim == 2:
            # (batch, T)
            batch_size, T = signal_np.shape
            psds = []
            for b in range(batch_size):
                freqs, psd = welch(signal_np[b], fs=self.fs, nperseg=min(256, T))
                psds.append(psd)
            psds = np.stack(psds)
        elif signal_np.ndim == 3:
            # (batch, C, T)
            batch_size, C, T = signal_np.shape
            psds = []
            for b in range(batch_size):
                channel_psds = []
                for c in range(C):
                    freqs, psd = welch(signal_np[b, c], fs=self.fs, nperseg=min(256, T))
                    channel_psds.append(psd)
                psds.append(np.mean(channel_psds, axis=0))  # 平均各通道PSD
            psds = np.stack(psds)
        
        return torch.tensor(psds, dtype=torch.float32, device=signal.device), freqs
    
    def forward(self, generated_signal: torch.Tensor, target_psd: torch.Tensor) -> torch.Tensor:
        gen_psd, freqs = self.compute_psd(generated_signal)
        
    
        freq_mask = (freqs >= self.freq_low) & (freqs <= self.freq_high)
        gen_psd_band = gen_psd[:, freq_mask]
        
     
        if target_psd.shape[-1] != gen_psd_band.shape[-1]:
            target_psd = F.interpolate(
                target_psd.unsqueeze(1), 
                size=gen_psd_band.shape[-1], 
                mode='linear', 
                align_corners=False
            ).squeeze(1)
        
        loss = F.mse_loss(gen_psd_band, target_psd)
        return loss
    
    def extract_target_psd(self, real_signals: torch.Tensor) -> torch.Tensor:
        psd, freqs = self.compute_psd(real_signals)
        freq_mask = (freqs >= self.freq_low) & (freqs <= self.freq_high)
        return psd[:, freq_mask]


class ReconstructionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, scale_signals: list, original_signal: torch.Tensor) -> torch.Tensor:
        
        reconstructed = torch.stack(scale_signals, dim=0).sum(dim=0)
        
      
        loss = F.mse_loss(reconstructed, original_signal)
        return loss
