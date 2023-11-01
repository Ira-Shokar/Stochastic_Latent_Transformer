# coding=utf-8
# Copyright (c) 2023 Ira Shokar
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch, math, nn

class Autoencoder(torch.nn.Module):
    def __init__(self, feat_dim=256, lat_dim=64, width=1):
        super().__init__()

        self.enc = torch.nn.Sequential(
            nn.TEPC_1D(1, width, feat_dim, feat_dim)
            nn.GELU(),
            nn.TEPC_1D(width, 1, feat_dim, lat_dim)
        )

        self.dec = torch.nn.Sequential(
            nn.TEPC_1D(1, width, lat_dim, feat_dim)
            nn.GELU(),
            nn.TEPC_1D(width, 1, feat_dim, feat_dim)
        )   

    def Encoder(self, x):
        x = x.unsqueeze(-2)
        x = self.enc(x)
        x = x.squeeze(-2)
        return x
    
    def Decoder(self, x):
        x = x.unsqueeze(-2)
        x = self.dec(x)
        x = x.squeeze(-2)
        return x

    def forward(self, x):
        x = self.Encoder(x)
        x = self.Decoder(x)
        return x


class Stochastic_Transformer(torch.nn.Module):
    def __init__(self, dim=256, seq_len=5, num_heads=16):
        super().__init__()

        self.dim = dim

        if dim <= 32: num_heads = dim//4

        position    = torch.arange(0, seq_len).unsqueeze(-1)
        div_term    = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        PE          = torch.zeros(seq_len, dim)
        PE[:, 0::2] = torch.sin(position * div_term)
        PE[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('PE', PE)

        self.att_block_0 = nn.Attention_Block(dim, num_heads, stochastic=True)
        self.att_block_1 = nn.Attention_Block(dim, num_heads)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, dim*2),
            torch.nn.GELU(),
            torch.nn.Linear(dim*2, dim)
        )

        k = lambda x: torch.fft.fftfreq(x)*x
        self.register_buffer('k', k(dim//2+1))

    def shift_phase(self, z):

        z   = torch.fft.rfft(z, norm='ortho')
        z   = z[..., :self.dim//2+1]
        phi = torch.angle(z[:, -1, 1])
        phi = torch.einsum('i,k,j->ijk', phi, self.k, torch.ones(z.size(1), device=z.device))
        z   = z * torch.exp(-1j*phi)
        z   = torch.fft.irfft(z, norm='ortho')
        return z, phi[:, -1]

    def unshift_phase(self, z, phi):
        z = torch.fft.rfft(z, norm='ortho')
        z = z * torch.exp(1j*phi)
        z = torch.fft.irfft(z, norm='ortho')
        return z
        
    def forward(self, z):

        z, phi = self.shift_phase(z)

        z    = z + self.PE
        z, a = self.att_block_0(z)
        z, _ = self.att_block_1(z)

        z = self.mlp(z[:, -1])

        z = self.unshift_phase(z, phi)

        return z, a