import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional

from .diffusion_model import DiffModule
from .fft_decomposer import FFTDecomposer
from .losses import FrequencyLoss, ReconstructionLoss


class Hander:
    NUM_BANDS = 3  
    
    def __init__(self, seq_len: int, input_dim: int, 
                 device: str = 'cuda', num_steps: int = 100,
                 num_classes: int = 3, fs: float = 100.0,
                 hidden_dim: int = 128):

        self.seq_len = seq_len
        self.input_dim = input_dim  
        self.device = device
        self.num_steps = num_steps
        self.num_classes = num_classes
        self.fs = fs
        self.hidden_dim = hidden_dim
        
        self.model = DiffModule(
            seq_len=seq_len, 
            input_dim=input_dim,  # C
            hidden_dim=hidden_dim, 
            num_classes=num_classes,
            num_bands=self.NUM_BANDS
        ).to(device)
        


        self.decomposer = FFTDecomposer(fs=fs)
        
        
        self.freq_loss = FrequencyLoss(fs=fs)  
        self.recon_loss = ReconstructionLoss()
        
        beta_start, beta_end = 0.0001, 0.02
        self.betas = torch.linspace(beta_start, beta_end, num_steps).to(device)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
    
    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, 
                 noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
 
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        return sqrt_alpha_t * x_0 + sqrt_one_minus_alpha_t * noise, noise
    
    
    def decompose_and_concat(self, signal: torch.Tensor) -> torch.Tensor:

        batch_size = signal.shape[0]
        
        concat_list = []
        for b in range(batch_size):
            sig_np = signal[b].cpu().numpy()  # (C, T)
            modes, _ = self.decomposer.decompose(sig_np)  # (3, C, T)
            #  (3, C, T) -> (3*C, T)
            concat = modes.reshape(-1, modes.shape[-1])  # (3*C, T)
            concat_list.append(torch.tensor(concat, dtype=torch.float32, device=self.device))
        
        # (batch, 3*C, T) -> (batch, T, 3*C)
        return torch.stack(concat_list, dim=0).transpose(1, 2)
    
    def split_bands(self, concat_signal: torch.Tensor) -> List[torch.Tensor]:

        batch, T, total_dim = concat_signal.shape
        C = total_dim // self.NUM_BANDS
        
        # (batch, T, 3*C) -> (batch, 3*C, T)
        concat_signal = concat_signal.transpose(1, 2)
        
    
        bands = []
        for k in range(self.NUM_BANDS):
            band = concat_signal[:, k*C:(k+1)*C, :]  # (batch, C, T)
            bands.append(band)
        
        return bands
    
    def train_step(self, signals: torch.Tensor, labels: torch.Tensor,
                   optimizer: torch.optim.Optimizer,
                   lambda1: float = 0.1, lambda2: float = 0.1,
                   target_psd: Optional[torch.Tensor] = None) -> dict:
        batch_size = signals.shape[0]
        self.model.train()
        
    
        concat_signal = self.decompose_and_concat(signals)
        
   


        t = torch.randint(0, self.num_steps, (batch_size,), device=self.device)
        noise = torch.randn_like(concat_signal)
        x_noisy, _ = self.q_sample(concat_signal, t, noise)
        

        t_input = t.float().unsqueeze(1) / self.num_steps
        noise_pred = self.model(x_noisy, t_input, labels)
        
        
        diff_loss = F.mse_loss(noise_pred, noise)
        
        
        generated_concat = self.generate(batch_size, labels)  # (batch, T, 3*C)
        generated_bands = self.split_bands(generated_concat)  # List of (batch, C, T)
        
       
        recon_loss = self.recon_loss(generated_bands, signals)
        
      
        freq_loss = torch.tensor(0.0, device=self.device)
        parkinson_mask = labels == 0
        if parkinson_mask.any():
            

            tremor_signal = generated_bands[1][parkinson_mask]
            if target_psd is not None:
                target_psd_masked = target_psd[parkinson_mask]
                freq_loss = self.freq_loss(tremor_signal, target_psd_masked)
        
        
        total_loss = diff_loss + lambda1 * freq_loss + lambda2 * recon_loss
        
    
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        optimizer.step()
        
        losses = {
            'diff_loss': diff_loss.item(),
            'recon_loss': recon_loss.item(),
            'freq_loss': freq_loss.item() if isinstance(freq_loss, torch.Tensor) else freq_loss,
            'total_loss': total_loss.item()
        }
        
        return losses
    
    def generate(self, num_samples: int, labels: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        
        with torch.no_grad():
            # (batch, T, 3*C)
            x_t = torch.randn(num_samples, self.seq_len, self.input_dim * self.NUM_BANDS).to(self.device)
            
         
            for t in reversed(range(self.num_steps)):
                t_batch = torch.full((num_samples,), t, device=self.device, dtype=torch.long)
                x_t = self.p_sample(x_t, t_batch, labels)
            
            return x_t
    
    def generate_full_signal(self, num_samples: int, labels: torch.Tensor,
                            reference_signal: Optional[torch.Tensor] = None,
                            guidance_scale: float = 0.1) -> torch.Tensor:
   
        generated_concat = self.generate(num_samples, labels)  # (batch, T, 3*C)
        
      
        generated_bands = self.split_bands(generated_concat)  # List of (batch, C, T)
        
      
        full_signal = torch.stack(generated_bands, dim=0).sum(dim=0)  # (batch, C, T)
        
        if reference_signal is not None and guidance_scale > 0:
            full_signal = (1 - guidance_scale) * full_signal + guidance_scale * reference_signal
        
        return full_signal