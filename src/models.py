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
    """
    Autoencoder Module for feature encoding and decoding.
    """

    def __init__(self, feat_dim: int = 256, lat_dim: int = 64, width: int = 1):
        """
        Initialise Autoencoder module.

        Parameters:
        - feat_dim (int): Feature dimension.
        - lat_dim  (int): Latent dimension.
        - width    (int): Number of convolution filters.
        """
        super().__init__()

        # Encoder MLP using TEPC_1D layers and GELU activation
        self._enc = torch.nn.Sequential(
            nn.TEPC_1D(1, width, feat_dim, feat_dim),
            torch.nn.GELU(),
            nn.TEPC_1D(width, 1, feat_dim, lat_dim)
        )

        # Decoder MLP using TEPC_1D layers and GELU activation
        self._dec = torch.nn.Sequential(
            nn.TEPC_1D(1, width, lat_dim, feat_dim),
            torch.nn.GELU(),
            nn.TEPC_1D(width, 1, feat_dim, feat_dim)
        )


    def Encoder(self, x):
        """
        Encoder forward pass.

        Parameters:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Encoded tensor.
        """
        x = x.unsqueeze(-2)
        x = self._enc(x)
        x = x.squeeze(-2)
        return x

    def Decoder(self, x):
        """
        Decoder forward pass.

        Parameters:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Decoded tensor.
        """
        x = x.unsqueeze(-2)
        x = self._dec(x)
        x = x.squeeze(-2)
        return x

    def forward(self, x):
        """
        Full Autoencoder forward pass.

        Parameters:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor.
        """
        x = self.Encoder(x)
        x = self.Decoder(x)
        return x


class Stochastic_Transformer(torch.nn.Module):
    """
    Stochastic Transformer Module.
    """

    def __init__(self, dim: int = 256, seq_len: int = 5, num_heads: int = 16):
        """
        Initialise Stochastic Transformer module.

        Parameters:
        - dim       (int): Dimension of input.
        - seq_len   (int): Length of input sequence.
        - num_heads (int): Number of attention heads.
        """
        super().__init__()

        # Adjust number of heads if dimension is small
        if dim <= 32: num_heads = dim // 4

        # Temporal Encoding
        position    = torch.arange(0, seq_len).unsqueeze(-1)
        div_term    = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        TE          = torch.zeros(seq_len, dim)
        TE[:, 0::2] = torch.sin(position * div_term)
        TE[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('TE', TE)

        # Attention Blocks, first of which is Stochastic
        self.att_block_0 = nn.Attention_Block(dim, num_heads, seq_len, stochastic=True)
        self.att_block_1 = nn.Attention_Block(dim, num_heads, seq_len)
        self.att_block_2 = nn.Attention_Block(dim, num_heads, seq_len)

        # MLP
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 2),
            torch.nn.GELU(),
            torch.nn.Linear(dim * 2, dim)
        )

        # Frequencies
        k = lambda x: torch.fft.fftfreq(x) * x
        self.register_buffer('k', k(dim // 2 + 1))

    def _shift_phase(self, z):
        """
        Shift the phase of the input tensor in the frequency domain.

        Parameters:
        - z (torch.Tensor): Input tensor.

        Returns:
        - Tuple[torch.Tensor, torch.Tensor]: Shifted tensor and phase information.
        """
        z   = torch.fft.rfft(z, norm='ortho')
        phi = torch.angle(z[:, -1, 1])
        phi = torch.einsum('i,k,j->ijk', phi, self.k, torch.ones(z.size(1), device=z.device))
        z   = z * torch.exp(-1j * phi)
        z   = torch.fft.irfft(z, norm='ortho')
        return z, phi[:, -1]

    def _unshift_phase(self, z, phi):
        """
        Unshift the phase of the tensor to revert the alignment.

        Parameters:
        - z (torch.Tensor): Input tensor.
        - phi (torch.Tensor): Phase information.

        Returns:
        - torch.Tensor: Unshifted tensor.
        """
        z = torch.fft.rfft(z, norm='ortho')
        z = z * torch.exp(1j * phi)
        z = torch.fft.irfft(z, norm='ortho')
        return z

    def forward(self, z):
        """
        Forward pass of the Stochastic Transformer module.

        Parameters:
        - z (torch.Tensor): Input tensor.

        Returns:
        - Tuple[torch.Tensor, torch.Tensor]: Output tensor and attention weights.
        """
        z, phi = self._shift_phase(z)

        z    = z + self.TE
        z, a = self.att_block_0(z)
        z, _ = self.att_block_1(z)
        z, _ = self.att_block_2(z)

        z = self.mlp(z[:, -1])

        z = self._unshift_phase(z, phi)

        return z, a
```
