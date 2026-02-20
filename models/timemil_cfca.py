import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.inceptiontime import InceptionTimeFeatureExtractor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=4, pool_type='mean', use_layernorm=True):
        super().__init__()
        self.dim = dim
        self.pool_type = pool_type
        self.use_layernorm = use_layernorm

        hidden_dim = max(dim // reduction, 8)
        self.channel_mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim)
        )

        if use_layernorm:
            self.norm = nn.LayerNorm(dim)

        self.scale = nn.Parameter(torch.ones(1) * 0.1)
        
    def forward(self, x):

        B, T, C = x.shape

        if self.pool_type == 'mean':
            pooled = x.mean(dim=1)  # (B, C)
        elif self.pool_type == 'max':
            pooled = x.max(dim=1)[0]  # (B, C)
        elif self.pool_type == 'mean_max':
            pooled = x.mean(dim=1) + x.max(dim=1)[0]  # (B, C)
        else:
            pooled = x.mean(dim=1)

        channel_weights = self.channel_mlp(pooled)  # (B, C)
        channel_weights = torch.sigmoid(channel_weights)  # (B, C)
        x_reweighted = x * (1 - self.scale + self.scale * channel_weights.unsqueeze(1))
        if self.use_layernorm:
            x_reweighted = self.norm(x_reweighted)
        
        return x_reweighted, channel_weights


class MILTimeAttention(nn.Module):

    def __init__(self, dim, hidden_dim=None, dropout=0.0):
        super().__init__()
        self.dim = dim
        hidden_dim = hidden_dim or dim // 2

        self.attention_V = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Sigmoid()
        )
        self.attention_w = nn.Linear(hidden_dim, 1)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, return_attention=False):

        B, T, C = x.shape
        # Gated Attention: α = softmax(w^T (tanh(Vx) ⊙ sigmoid(Ux)))
        A_V = self.attention_V(x)  # (B, T, hidden)
        A_U = self.attention_U(x)  # (B, T, hidden)
        A = self.attention_w(A_V * A_U)  # (B, T, 1)
        A = A.squeeze(-1)  # (B, T)

        attention_weights = F.softmax(A, dim=1)  # (B, T)
        attention_weights = self.dropout(attention_weights)

        z = torch.bmm(attention_weights.unsqueeze(1), x).squeeze(1)  # (B, C)
        
        if return_attention:
            return z, attention_weights
        return z

def mexican_hat_wavelet(size, scale, shift):
    x = torch.linspace(-(size[1]-1)//2, (size[1]-1)//2, size[1]).to(device)
    x = x.reshape(1, -1).repeat(size[0], 1)
    x = x - shift
    C = 2 / (3**0.5 * torch.pi**0.25)
    wavelet = C * (1 - (x/scale)**2) * torch.exp(-(x/scale)**2 / 2) / (torch.abs(scale)**0.5)
    return wavelet


class WaveletEncoding(nn.Module):
    def __init__(self, dim=512, max_len=256, hidden_len=512, dropout=0.0):
        super().__init__()
        self.proj_1 = nn.Linear(dim, dim)
        
    def forward(self, x, wave1, wave2, wave3):
        cls_token, feat_token = x[:, 0], x[:, 1:]
        x = feat_token.transpose(1, 2)
        
        D = x.shape[1]
        scale1, shift1 = wave1[0, :], wave1[1, :]
        wavelet_kernel1 = mexican_hat_wavelet(size=(D, 19), scale=scale1, shift=shift1)
        scale2, shift2 = wave2[0, :], wave2[1, :]
        wavelet_kernel2 = mexican_hat_wavelet(size=(D, 19), scale=scale2, shift=shift2)
        scale3, shift3 = wave3[0, :], wave3[1, :]
        wavelet_kernel3 = mexican_hat_wavelet(size=(D, 19), scale=scale3, shift=shift3)
        


        pos1 = F.conv1d(x, wavelet_kernel1.unsqueeze(1), groups=D, padding='same')
        pos2 = F.conv1d(x, wavelet_kernel2.unsqueeze(1), groups=D, padding='same')
        pos3 = F.conv1d(x, wavelet_kernel3.unsqueeze(1), groups=D, padding='same')
        x = x.transpose(1, 2)
        
        x = x + self.proj_1(pos1.transpose(1, 2) + pos2.transpose(1, 2) + pos3.transpose(1, 2))
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TimeMIL_cfca(nn.Module):
    def __init__(self, in_features, n_classes=2, mDim=64, max_seq_len=400, dropout=0.,
                 channel_reduction=4, channel_pool='mean', use_channel_layernorm=True):
        super().__init__()
        
        self.mDim = mDim
        self.n_classes = n_classes

        inception_out_channels = mDim // 4  # 32 when mDim=128
        self.feature_extractor = InceptionTimeFeatureExtractor(
            n_in_channels=in_features,
            out_channels=inception_out_channels
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, mDim))
        # Wave parameters for WPE1
        self.wave1 = nn.Parameter(torch.randn(2, mDim, 1))
        self.wave1.data[0] = torch.ones(mDim, 1) + torch.randn(mDim, 1) * 0.1
        self.wave2 = nn.Parameter(torch.zeros(2, mDim, 1))
        self.wave2.data[0] = torch.ones(mDim, 1) + torch.randn(mDim, 1) * 0.1
        self.wave3 = nn.Parameter(torch.zeros(2, mDim, 1))
        self.wave3.data[0] = torch.ones(mDim, 1) + torch.randn(mDim, 1) * 0.1
        
        
        
        # Wave parameters for WPE2
        self.wave1_ = nn.Parameter(torch.randn(2, mDim, 1))
        self.wave1_.data[0] = torch.ones(mDim, 1) + torch.randn(mDim, 1) * 0.1
        self.wave2_ = nn.Parameter(torch.zeros(2, mDim, 1))
        self.wave2_.data[0] = torch.ones(mDim, 1) + torch.randn(mDim, 1) * 0.1
        self.wave3_ = nn.Parameter(torch.zeros(2, mDim, 1))
        self.wave3_.data[0] = torch.ones(mDim, 1) + torch.randn(mDim, 1) * 0.1
        
        hidden_len = 2 * max_seq_len
        self.pos_layer = WaveletEncoding(mDim, max_seq_len, hidden_len)
        self.pos_layer2 = WaveletEncoding(mDim, max_seq_len, hidden_len)
        self.channel_attention = ChannelAttention(
            dim=mDim,
            reduction=channel_reduction,
            pool_type=channel_pool,
            use_layernorm=use_channel_layernorm
        )
        self.mil_attention = MILTimeAttention(
            dim=mDim,
            hidden_dim=mDim // 2,
            dropout=dropout
        )
        classifier_input_dim = mDim

        self.norm = nn.LayerNorm(classifier_input_dim)
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, classifier_input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_input_dim, n_classes)
        )

        self.alpha = nn.Parameter(torch.ones(1))
        initialize_weights(self)
        
    def forward(self, x, warmup=False, return_attention=False):

        B = x.shape[0]
        # x: (B, T, C_in) -> (B, C_in, T) for conv
        x = self.feature_extractor(x.transpose(1, 2))
        x = x.transpose(1, 2)  # (B, T', mDim)
        
        BN, seq_len, D = x.shape

        global_token = x.mean(dim=1)

        cls_tokens = self.cls_token.expand(BN, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (BN, T+1, mDim)
        # WPE1
        x = self.pos_layer(x, self.wave1, self.wave2, self.wave3)
        # WPE2
        x = self.pos_layer2(x, self.wave1_, self.wave2_, self.wave3_)
      
        cls_token_out = x[:, 0]  # (BN, mDim)
        seq_tokens = x[:, 1:]    # (BN, T, mDim)
      
        seq_reweighted, channel_weights = self.channel_attention(seq_tokens)
        # Attention-based pooling: z = Σ α_t * H̃_t
        if return_attention:
            bag_repr, time_weights = self.mil_attention(seq_reweighted, return_attention=True)
        else:
            bag_repr = self.mil_attention(seq_reweighted)
            time_weights = None

        z = 0.5 * cls_token_out + 0.5 * bag_repr  # (BN, mDim)
        if warmup:
            z = 0.1 * z + 0.9 * global_token


        z = self.norm(z)
        logits = self.classifier(z)
        
        if return_attention:
            return logits, channel_weights, time_weights
        return logits
    
    def get_attention_weights(self, x,):
        return self.forward(x, warmup=False, return_attention=True)


if __name__ == "__main__":

    B, T, C_in = 4, 976, 132
    x = torch.randn(B, T, C_in).to(device)
    
    model = TimeMIL_cfca(
        in_features=C_in,
        n_classes=3,
        mDim=128,
        max_seq_len=T,
        dropout=0.2
    ).to(device)

