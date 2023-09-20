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

import torch 

class SpectralConv1d(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, in_dim:int,  out_dim: int):
        super(SpectralConv1d, self).__init__()
        """
        Initializes the 1D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        Args:
            in_channels (int): input channels to the FNO layer
            out_channels (int): output channels of the FNO layer
        """

        self.in_dim  = in_dim //2+1
        self.out_dim = out_dim//2+1

        self.modes = max(self.in_dim, self.out_dim)

        self.scale   = 1 / (in_channels+out_channels)**0.5
        self.weights = torch.nn.Parameter(torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat) * self.scale)
        self.bias    = torch.nn.Parameter(torch.zeros(out_channels, self.modes, dtype=torch.cfloat))

        k = lambda x: torch.fft.fftfreq(x)*x

        self.register_buffer('k_in',  k(self.in_dim))
        self.register_buffer('k_out', k(self.out_dim))

    def shift_phase(self, z):
        z     = torch.fft.rfft(z, norm='ortho')
        phi   = torch.angle(z[..., 1])
        phi_k = torch.einsum('...i,k->...ik', phi, self.k_in)
        z     = z * torch.exp(-1j*phi_k)
        return z, phi[..., 0].unsqueeze(-1)

    def unshift_phase(self, z, phi):
        phi_k = torch.einsum('...i,k->...ik', phi, self.k_out)
        z     = z * torch.exp(1j*phi_k)
        z     = torch.fft.irfft(z, norm='ortho')
        return z

    def forward(self, x):

        # Transform to physical space and align phase
        x_ft, phi = self.shift_phase(x)

        # Pad to match input fourier modes
        x_ft = torch.nn.functional.pad(x_ft, (0, self.modes-x_ft.size(-1)))

        # Multiply relevant Fourier modes
        out_ft =torch.einsum("...ix,iox->...ox", x_ft, self.weights) + self.bias

        # Truncate to match output fourier modes
        out_ft = out_ft[..., :self.out_dim]

        #Return to physical space and shift phase back
        x = self.unshift_phase(out_ft, phi)
        return x


class Multihead_Self_Attention(torch.nn.Module):

    def __init__(self, dim, num_heads):
        super().__init__()

        self.dim       = dim
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads

        self.q_proj = torch.nn.Linear(dim, dim)
        self.k_proj = torch.nn.Linear(dim, dim)
        self.v_proj = torch.nn.Linear(dim, dim)
        self.o_proj = torch.nn.Linear(dim, dim)

    def forward(self, x):

        # Store Batch Size
        b    = x.size(0)

        # Linear Projections and Split into Mulitple Heads
        q    = self.q_proj(x).reshape(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k    = self.k_proj(x).reshape(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v    = self.v_proj(x).reshape(b, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Self Attention and Head Concatination
        a    = torch.matmul(q, k.transpose(-1, -2)) / self.head_dim**0.5
        mask = torch.triu(-float('inf')*torch.ones(a.size()[1:], device=a.device), diagonal=1)
        a    = torch.nn.functional.softmax(a + mask, dim=-1)
        v    = torch.matmul(a, v).transpose(1, 2).reshape(b, -1, self.dim)

        # Linear Projection
        o    = self.o_proj(v)

        return o, a.mean(1)


class Attention_Block(torch.nn.Module):
    def __init__(self, dim, num_heads, first=False, last=False):
        super().__init__()

        self.first   = first
        self.last    = last

        self.mha     = Multihead_Self_Attention(dim, num_heads)

        self.ln_1    = torch.nn.LayerNorm(dim)
        self.ln_2    = torch.nn.LayerNorm(dim)

        self.fc_mlp1 = torch.nn.Linear(dim, dim)
        self.fc_mlp2 = torch.nn.Linear(dim, dim)

    def forward(self, x):

        o = self.ln_1(x)

        if self.first==True: o = torch.cat([torch.randn_like(o[:, 0]).unsqueeze(1), x], dim=1)

        o, a = self.mha(o)

        if self.first==True: o = o[:, 1:]
            
        x = x + o

        if self.last==True: x = x[:, -1]

        o = self.ln_2(x)
        o = self.fc_mlp1(o)
        o = torch.nn.functional.gelu(o)
        o = self.fc_mlp2(o)

        x = x + o

        return x, a