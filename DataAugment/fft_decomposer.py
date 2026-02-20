import numpy as np
from scipy.fft import rfft, irfft
from typing import List, Tuple
from tqdm import tqdm

class FFTDecomposer:
    FIXED_FREQ_BANDS = [
        (0.5, 3),    
        (3, 7),      
        (7, 12),     
    ]
    K = 3  
    
    def __init__(self, fs: float = 100.0):
        self.fs = fs
        self.freq_bands = self.FIXED_FREQ_BANDS.copy()
    
    def decompose(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        if signal.ndim == 1:
            return self._decompose_1d(signal)
        elif signal.ndim == 2:
            return self._decompose_2d(signal)
    
    def _decompose_1d(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
       
        T = len(signal)



        signal_fft = rfft(signal)
        freqs = np.fft.rfftfreq(T, 1/self.fs)
        
        modes = np.zeros((self.K, T))
        center_freqs = np.zeros(self.K)
        
        for k, (f_low, f_high) in enumerate(self.freq_bands):
         
            mask = (freqs >= f_low) & (freqs < f_high)
            filtered_fft = signal_fft * mask
            modes[k] = irfft(filtered_fft, n=T)
            center_freqs[k] = (f_low + f_high) / 2
        
        return modes, center_freqs
    
    def _decompose_2d(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        
        C, T = signal.shape
        signal_fft = rfft(signal, axis=1)  # (C, T//2+1)
        freqs = np.fft.rfftfreq(T, 1/self.fs)
        
        modes = np.zeros((self.K, C, T))
        center_freqs = np.zeros(self.K)
        
        for k, (f_low, f_high) in enumerate(self.freq_bands):
            mask = (freqs >= f_low) & (freqs < f_high)
            filtered_fft = signal_fft * mask[np.newaxis, :]
            modes[k] = irfft(filtered_fft, n=T, axis=1)
            center_freqs[k] = (f_low + f_high) / 2
        
        return modes, center_freqs
    
    def decompose_batch(self, signals: List[np.ndarray], 
                        show_progress: bool = False) -> List[Tuple[np.ndarray, np.ndarray]]:

        results = []
        iterator = tqdm(signals) if show_progress else signals
        for signal in iterator:
            results.append(self.decompose(signal))
        return results
    
    def reconstruct(self, modes: np.ndarray) -> np.ndarray:
        return np.sum(modes, axis=0)
