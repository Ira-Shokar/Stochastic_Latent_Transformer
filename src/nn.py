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


class TEPC_1D(torch.nn.Module):
    """
    1D Translatiion Equivarient Pointwise Convolution (TEPC) Layer.
    """

    def __init__(self, in_channels: int, out_channels: int, in_dim: int, out_dim: int):
        """
        Initialise TEPC_1D module.

        Parameters:
        - in_channels  (int): Number of input channels.
        - out_channels (int): Number of output channels.
        - in_dim       (int): Input dimension.
        - out_dim      (int): Output dimension.
        """
        super().__init__()

        # Adjust input and output dimensions for complex Fourier transform
        self.in_dim  = in_dim  // 2 + 1
        self.out_dim = out_dim // 2 + 1

        # Determine the number of Fourier modes
        self.modes = max(self.in_dim, self.out_dim)

        # Scaling factor for initialization
        self.scale = 1 / (in_channels * out_channels) ** 0.5

        # Learnable parameters for TEPC_1D
        self.weights = torch.nn.Parameter(
            torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat) * self.scale
        )
        self.bias = torch.nn.Parameter(torch.zeros(out_channels, self.modes, dtype=torch.cfloat))

        # Frequencies for input and output dimensions
        k = lambda x: torch.fft.fftfreq(x) * x
        self.register_buffer("k_in" , k(self.in_dim))
        self.register_buffer("k_out", k(self.out_dim))

    def shift_phase(self, z):
        """
        Shift the phase of the input tensor to align frequencies.

        Parameters:
        - z (torch.Tensor): Input tensor.

        Returns:
        - Tuple[torch.Tensor, torch.Tensor]: Shifted tensor and phase information.
        """
        z = torch.fft.rfft(z)
        phi = torch.angle(z[..., 1])
        phi_k = torch.einsum("...i,k->...ik", phi, self.k_in)
        z = z * torch.exp(-1j * phi_k)
        return z, phi[..., 0].unsqueeze(-1)

    def unshift_phase(self, z, phi):
        """
        Unshift the phase of the tensor to revert the alignment.

        Parameters:
        - z   (torch.Tensor): Input tensor.
        - phi (torch.Tensor): Phase information.

        Returns:
        - torch.Tensor: Unshifted tensor.
        """
        phi_k = torch.einsum("...i,k->...ik", phi, self.k_out)
        z = z * torch.exp(1j * phi_k)
        z = torch.fft.irfft(z)
        return z

    def forward(self, x):
        """
        Forward pass of the TEPC_1D module.

        Parameters:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor.
        """
        # Transform to spectral space and align phase
        x_ft, phi = self.shift_phase(x)

        # Pad to match input Fourier modes
        x_ft = torch.nn.functional.pad(x_ft, (0, self.modes - x_ft.size(-1)))

        # Pointwise Convolution = Linear Layer in Spectral Space
        out_ft = torch.einsum("...ix,iox->...ox", x_ft, self.weights) + self.bias

        # Truncate to match output Fourier modes
        out_ft = out_ft[..., :self.out_dim]

        # Return to physical space and shift phase back
        x = self.unshift_phase(out_ft, phi)
        return x


class Multihead_Attention(torch.nn.Module):
    """
    Multihead Attention Module.
    """

    def __init__(self, dim: int, num_heads: int, seq_len: int, stochastic: bool = False):
        """
        Initialise Multihead_Attention module.

        Parameters:
        - dim        (int) : Dimension of input.
        - num_heads  (int) : Number of attention heads.
        - seq_len    (int) : Length of input sequence.
        - stochastic (bool): Whether to include stochastic block.
        """
        super().__init__()

        # Module parameters
        self.dim        = dim
        self.num_heads  = num_heads
        self.head_dim   = dim // num_heads
        self.stochastic = stochastic

        # Pre-computed inverse square root of head dimension
        self.inv_sqrt_head_dim = 1 / self.head_dim ** 0.5

        # Linear projections for queries, keys, values, and output
        self.q_proj = torch.nn.Linear(dim, dim)
        self.k_proj = torch.nn.Linear(dim, dim)
        self.v_proj = torch.nn.Linear(dim, dim)
        self.o_proj = torch.nn.Linear(dim, dim)

        # Softmax operation
        self.softmax = torch.nn.Softmax(dim=-1)

        # Define the attention mask shape based on stochastic block
        if self.stochastic: mask_shape = torch.ones((self.head_dim, seq_len, seq_len+1))
        else              : mask_shape = torch.ones((self.head_dim, seq_len, seq_len))

        # Attention mask
        self.register_buffer("mask", torch.triu(-float("inf") * mask_shape, diagonal=1))

    def forward(self, x):
        """
        Forward pass of the Multihead_Attention module.

        Parameters:
        - x (torch.Tensor): Input tensor.

        Returns:
        - Tuple[torch.Tensor, torch.Tensor]: Output tensor and attention weights.
        """
        # Store Batch Size
        b = x.size(0)

        # Add Forcing if Stochastic Block
        y = torch.cat([torch.randn_like(x[:, 0]).unsqueeze(1), x], dim=1) if self.stochastic else x

        # Linear Projections and Split into Multiple Heads
        q = self.q_proj(x).reshape(b, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(y).reshape(b, -1, self.num_heads, self.head_dim).permute(0, 2, 3, 1)
        v = self.v_proj(y).reshape(b, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Attention
        a = self.softmax(q @ k * self.inv_sqrt_head_dim)
        o = a @ v

        # Linear Projection and Recombine Heads
        o = self.o_proj(o.transpose(1, 2).reshape(b, -1, self.dim))

        return o, a.mean(1)


class Attention_Block(torch.nn.Module):
    """
    Attention Block Module.
    """

    def __init__(self, dim: int, num_heads: int, seq_len: int, stochastic: bool

 = False):
        """
        Initialise Attention_Block module.

        Parameters:
        - dim        (int) : Dimension of input.
        - num_heads  (int) : Number of attention heads.
        - seq_len    (int) : Length of input sequence.
        - stochastic (bool): Whether to include stochastic block.
        """
        super().__init__()

        # Multihead Attention submodule
        self.mha = Multihead_Attention(dim, num_heads, seq_len, stochastic)

        # Layer normalization for the first and second submodules
        self.ln_1 = torch.nn.LayerNorm(dim)
        self.ln_2 = torch.nn.LayerNorm(dim)

        # Multi-Layer Perceptron (MLP) submodule
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 2),
            torch.nn.GELU(),
            torch.nn.Linear(dim * 2, dim)
        )

    def forward(self, x):
        """
        Forward pass of the Attention_Block module.

        Parameters:
        - x (torch.Tensor): Input tensor.

        Returns:
        - Tuple[torch.Tensor, torch.Tensor]: Output tensor and attention weights.
        """
        # Layer normalization and Multihead Attention
        o    = self.ln_1(x)
        o, a = self.mha(o)
        x    = x + o

        # Layer normalization and MLP
        o = self.ln_2(x)
        o = self.mlp(o)
        x = x + o

        return x, a
