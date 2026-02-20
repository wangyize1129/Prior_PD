import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffModule(nn.Module):
    def __init__(self, seq_len: int, input_dim: int, hidden_dim: int = 128, 
                 num_layers: int = 4, num_classes: int = 3, num_bands: int = 3):

        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim  
        self.num_classes = num_classes
        self.num_bands = num_bands
        self.total_dim = input_dim * num_bands  # 3*C
        
  
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
   
        self.class_embed = nn.Embedding(num_classes, hidden_dim)
        
 
        self.band_embed = nn.Parameter(torch.randn(1, num_bands, hidden_dim) * 0.02)
        
       
        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(self.total_dim if i == 0 else hidden_dim, hidden_dim, 3, padding=1),
                nn.GroupNorm(8, hidden_dim),
                nn.SiLU(),
                nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
                nn.GroupNorm(8, hidden_dim),
                nn.SiLU()
            ) for i in range(num_layers)
        ])

  
        self.condition_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        self.neck = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU()
        )



        self.decoder = nn.ModuleList([
            nn.Sequential(
                
                nn.Conv1d(hidden_dim * 2 if i < num_layers - 1 else hidden_dim, hidden_dim, 3, padding=1),
                nn.GroupNorm(8, hidden_dim),
                nn.SiLU(),
                nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
                nn.GroupNorm(8, hidden_dim),
                nn.SiLU()
            ) for i in range(num_layers)
        ])

        self.final_conv = nn.Conv1d(hidden_dim, self.total_dim, 1)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # (batch, T, 3*C) -> (batch, 3*C, T)
        x = x.transpose(1, 2)



       
        t_emb = self.time_embed(t)  # (batch, hidden_dim)
        c_emb = self.class_embed(y)  # (batch, hidden_dim)
        cond = self.condition_proj(torch.cat([t_emb, c_emb], dim=-1))  # (batch, hidden_dim)
        cond = cond.unsqueeze(2)  # (batch, hidden_dim, 1)

      
        skip_connections = []
        h = x
        for layer in self.encoder:
            h = layer(h)
            skip_connections.append(h)
            if h.shape[2] > 4:  # 下采样
                h = F.avg_pool1d(h, 2)

        
        h = h + cond.expand(-1, -1, h.shape[2])

        
        h = self.neck(h)
        
    
        for i, layer in enumerate(self.decoder):
            if i < len(skip_connections) - 1:
               
                target_size = skip_connections[-(i + 2)].shape[2]
                h = F.interpolate(h, size=target_size, mode='linear', align_corners=False)
                h = torch.cat([h, skip_connections[-(i + 2)]], dim=1)
            h = layer(h)




        noise_pred = self.final_conv(h)
        return noise_pred.transpose(1, 2)  # (batch, T, 3*C)