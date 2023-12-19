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

import numpy as np, torch, tqdm, io, random, time, utils, models

class Stochastic_Latent_Transformer:
    """
    Stochastic Latent Transformer class for training and evaluation.
    """

    def __init__(self, feat_dim=256, latent_dim=256, seq_len=1, ens_size=4, epochs=1,
                 learning_rate=1e-4, training_steps=1, val_steps=1, num_heads=16,
                 save_path=None, file_name=None, layers=2, width=2):
        """
        Initialize Stochastic_Latent_Transformer module.

        Parameters:
        - feat_dim       (int)  : Feature dimension.
        - latent_dim     (int)  : Latent dimension.
        - seq_len        (int)  : Length of input sequence.
        - ens_size       (int)  : Ensemble size.
        - epochs         (int)  : Number of training epochs.
        - learning_rate  (float): Learning rate for optimization.
        - training_steps (int)  : Number of training steps per epoch.
        - val_steps      (int)  : Number of validation steps per epoch.
        - num_heads      (int)  : Number of attention heads.
        - save_path      (str)  : Path to save the trained model.
        - file_name      (str)  : Name of the saved model file.
        - layers         (int)  : Number of layers in the transformer.
        - width          (int)  : Width parameter.
        """
        super().__init__()

        # Define Parameters
        self.save_path = save_path
        self.file_name = file_name

        self.total_epochs   = epochs
        self.training_steps = training_steps
        self.val_steps      = val_steps
        self.lr             = learning_rate

        self.feat_dim   = feat_dim
        self.latent_dim = latent_dim
        self.seq_len    = seq_len
        self.ens_size   = ens_size
        self.num_heads  = num_heads
        self.layers     = layers
        self.width      = width

        # Define Models
        self.AE = models.Autoencoder(self.feat_dim, self.latent_dim, self.width).to(utils.device)
        self.Trans = models.Stochastic_Transformer(self.latent_dim, self.seq_len).to(utils.device)

        # Define Optimiser
        self.optimiser = torch.optim.Adam([
            {'params': self.Trans.parameters()},
            {'params': self.AE.parameters(), 'lr': self.lr * 5}
        ], lr=self.lr)

        # Define LR scheduler
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimiser, gamma=0.9825)

        # Loss Tracker
        self.loss_dict = {'MSE': 0, 'mirror': 0, 'z_dist': 0, 'ES': 0, 'rep': 0, 'spec': 0}
        self.val_loss_dict = {'MSE': 0, 'mirror': 0, 'z_dist': 0, 'ES': 0, 'rep': 0, 'spec': 0}

    def fit(self, data, val_data):
        """
        Train the Stochastic Latent Transformer.

        Parameters:
        - data: Training data.
        - val_data: Validation data.
        """
        print(f'Using {utils.device} hardware')

        for self.epoch in range(self.total_epochs):
            with tqdm.trange(self.training_steps, ncols=140) as pbar:
                self.loss_dict = {x: 0 for x in self.loss_dict}
                self.val_loss_dict = {x: 0 for x in self.val_loss_dict}

                for self.step, train_batch in zip(pbar, data):
                    self.train(*train_batch)
                    self.track_losses(pbar)

                    if self.step == (self.training_steps - 1):
                        self.scheduler.step()

                        for self.val_step, val_batch in zip(range(self.val_steps), val_data):
                            self.validate(*val_batch)

                            if self.val_step == (self.val_steps - 1):
                                self.track_losses(pbar, val=True)

        utils.save_model(self.AE   , "AE"   , self.latent_dim, self.save_path)
        utils.save_model(self.Trans, "Trans", self.latent_dim, self.save_path)

    def forward(self, x):
        """
        Forward pass of the Stochastic Latent Transformer.

        Parameters:
        - x (torch.Tensor): Input tensor.

        Returns:
        - Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Output tensors.
        """
        x = self.AE.Encoder(x)
        z = torch.stack([self.Trans(x)[0] for i in range(self.ens_size)], dim=1)
        u = self.AE.Decoder(z)
        x = self.AE.Decoder(x)
        return u, z, x

    def CRPS(self, x, y):
        """
        Calculate Continuous Ranked Probability Score.

        Parameters:
        - x (torch.Tensor): Prediction tensor.
        - y (torch.Tensor): Ground truth tensor.

        Returns:
        - Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: CRPS and its components.
        """
        MSE     = torch.cdist(y, x).mean() 
        ens_var = torch.cdist(x, x).mean(0).sum() / (self.ens_size * (self.ens_size - 1))

        MSE     /= x.size(-1) ** 0.5
        ens_var /= x.size(-1) ** 0.5

        return 2 * MSE - ens_var, MSE, ens_var

    def AE_loss(self, x, y):
        """
        Calculate Autoencoder loss.

        Parameters:
        - x (torch.Tensor): Prediction tensor.
        - y (torch.Tensor): Ground truth tensor.

        Returns:
        - Tuple[torch.Tensor, torch.Tensor]: MAE and spectral loss.
        """
        MAE  = torch.cdist(y, x).mean() / x.size(-1) ** 0.5
        x    = torch.fft.rfft(x, norm='ortho').abs()
        y    = torch.fft.rfft(y, norm='ortho').abs()
        spec = torch.cdist(y, x).mean() / x.size(-1) ** 0.5
        return MAE, spec

    def train(self, x, y):
        """
        Training step of the Stochastic Latent Transformer.

        Parameters:
        - x (torch.Tensor): Input tensor.
        - y (torch.Tensor): Ground truth tensor.
        """
        # Use GPU if available
        x = x.to(utils.device)
        y = y.to(utils.device)

        self.AE.train()
        self.Trans.train()

        o, z, m      = self.forward(x)                   # Forward Pass through SLT
        ES, MSE, rep = self.CRPS(o, y)                   # CRPS, MSE and Ens Var Losses
        z_dist, _, _ = self.CRPS(z, self.AE.Encoder(y))  # CRPS in latent space
        mirror, spec = self.AE_loss(m[:, 0].unsqueeze(1), x[:, 0].unsqueeze(1))

        # Backprop
        (ES + spec + mirror + z_dist).backward()
        self.optimiser.step()
        self.optimiser.zero_grad

        # Update Metric Tracker
        self.loss_dict['ES']     += ES.item()
        self.loss_dict['MSE']    += MSE.item()
        self.loss_dict['rep']    += rep.item()
        self.loss_dict['spec']   += spec.item()
        self.loss_dict['mirror'] += mirror.item()
        self.loss_dict['z_dist'] += z_dist.item()

    @torch.no_grad()
    def validate(self, x, y):
        """
        Validation step of the Stochastic Latent Transformer.

        Parameters:
        - x (torch.Tensor): Input tensor.
        - y (torch.Tensor): Ground truth tensor.
        """
        # Use GPU if available
        x = x.to(utils.device)
        y = y.to(utils.device)

        self.AE.eval()
        self.Trans.eval()

        # Model Losses
        o, z, m      = self.forward(x)                   # Forward Pass through SLT
        ES, MSE, rep = self.CRPS(o, y)                   # CRPS, MSE and Ens Var Losses
        z_dist, _, _ = self.CRPS(z, self.AE.Encoder(y))  # CRPS in latent space
        mirror, spec = self.AE_loss(m[:, 0].unsqueeze(1), x[:, 0].unsqueeze(1))

        # Update Metric Tracker
        self.val_loss_dict['ES']     += ES.item()
        self.val_loss_dict['MSE']    += MSE.item()
        self.val_loss_dict['rep']    += rep.item()
        self.val_loss_dict['spec']   += spec.item()
        self.val_loss_dict['mirror'] += mirror.item()
        self.val_loss_dict['z_dist'] += z_dist.item()
    
    def track_losses(self, pbar, val=False):
        """
        Track and display losses during training or validation.

        Parameters:
        - pbar: tqdm progress bar.
        - val (bool): Flag indicating validation.
        """
        loss_dict = {x: self.loss_dict[x] / (self.step + 1) for x in self.loss_dict}
        if val == False:
            pbar.set_postfix({

                'epoch'     : f"{self.epoch}/{self.total_epochs}",
                'ES'        : f"{loss_dict['ES']:.2E}",
                'ES_val'    : f"---------",
                'mirror'    : f"{loss_dict['mirror']:.2E}",
                'mirror_val': f"---------",
            })

        else:
            val_loss_dict = {x: self.val_loss_dict[x] / (self.val_step + 1) for x in self.val_loss_dict}

            pbar.set_postfix({
                'epoch'     : f"{self.epoch}/{self.total_epochs}",
                'ES'        : f"{loss_dict['ES']:.2E}",
                'ES_val'    : f"{val_loss_dict['ES']:.2E}",
                'mirror'    : f"{loss_dict['mirror']:.2E}",
                'mirror_val': f"{val_loss_dict['mirror']:.2E}",
            })