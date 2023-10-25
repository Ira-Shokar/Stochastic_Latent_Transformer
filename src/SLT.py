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
    def __init__(self, feat_dim=256, latent_dim=256, seq_len=1, ens_size=4, epochs=1,
                 learning_rate=1e-4, training_steps=1, val_steps=1, num_heads=16, 
                 layers=2, width=2, save_path=None, file_name=None):
        super().__init__()

        # Define Parameters
        self.save_path      = save_path
        self.file_name      = file_name   

        self.total_epochs   = epochs
        self.training_steps = training_steps
        self.val_steps      = val_steps
        self.lr             = learning_rate

        self.feat_dim       = feat_dim 
        self.latent_dim     = latent_dim
        self.seq_len        = seq_len
        self.ens_size       = ens_size
        self.num_heads      = num_heads
        self.layers         = layers
        self.width          = width
        
        # Define Hardware
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define Models
        self.AE    = models.Autoencoder(self.feat_dim, self.latent_dim, self.width).to(self.device)
        self.Trans = models.Stochatic_Transformer(self.latent_dim, self.seq_len).to(self.device)

        # Define Optimiser
        self.optimiser = torch.optim.Adam(
            list(self.AE.parameters()) + list(self.Trans.parameters()), lr=self.lr)

        # Define LR scheduler
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR( self.optimiser, gamma=0.9825)

        # Loss Tracker
        self.loss_dict     = {'MSE':0, 'mirror': 0, 'z_dist':0, 'ES':0, 'rep':0}
        self.val_loss_dict = {'MSE':0, 'mirror': 0, 'z_dist':0, 'ES':0, 'rep':0}


    def fit(self, data, val_data):
        print(f'Using {self.device} hardware')

        for self.epoch in range(self.total_epochs):
            with tqdm.trange(self.training_steps, ncols=140) as pbar:
                self.loss_dict     = {x:0 for x in self.loss_dict}
                self.val_loss_dict = {x:0 for x in self.val_loss_dict}
               
                for self.step, train_batch in zip(pbar, data):
                    self.train(*train_batch)
                    self.track_losses(pbar)

                    if self.step==(self.training_steps-1):
                        self.scheduler.step()
                        
                        for self.val_step, val_batch in zip(range(self.val_steps), val_data):
                            self.validate(*val_batch)

                            if self.val_step==(self.val_steps-1):
                                if self.epoch%10==0: self.save_model()
                                self.track_losses(pbar, val=True)
                  
    
    def forward(self, x):
        z = torch.zeros(
            x.size(0), self.ens_size, 1, self.latent_dim, device=self.device)

        x = self.AE.Encoder(x).unsqueeze(1).expand(-1, self.ens_size, -1, -1)
        for i in range(self.ens_size):
            z[:, i] = self.Trans(x[:, i])[0]
        u = self.AE.Decoder(z)
        x = self.AE.Decoder(x[:, 0])
        return u, z, x


    def energy_score(self, x, y, p=2):
        """
        Inputs:
        x: ["batch", "ensemble_size", "data_size"], contains all forecasts for each batch element
        y: ["batch", 1 , "data_size"], contains all verifications for each batch element
        p: norm, "Euclidean distance" if 2 

        Output:
        energy score:  E[k(x, y)]^p - 1/2.E[k(x, x)]^p
        """

        if y.dim()==3: y = y.unsqueeze(1)

        while x.dim()>3:
            x = x.flatten(-2, -1)
            y = y.flatten(-2, -1)

        MSE      = torch.cdist(x, y, p).mean() / x.size(-1)**0.5 
        ens_var  = torch.cdist(x, x, p).mean() / x.size(-1)**0.5
        ens_var *= self.ens_size/(self.ens_size - 1) # to result in 1/[m(m-1)]

        return 2*MSE-ens_var, MSE, ens_var


    def train(self, x, y): 
        # Use GPU if available
        x = x.to(self.device)
        y = y.to(self.device)

        self.AE.train()
        self.Trans.train()

        o, z, m      = self.forward(x)                          # Forward Pass through AE & Trans
        mirror       = torch.nn.functional.mse_loss(m, x)       # Mirror Loss
        ES, MSE, rep = self.energy_score(o, y)                  # Energy Score, MSE and Rep Losses
        z_dist, _, _ = self.energy_score(z, self.AE.Encoder(y)) # Energy Score in latent space

        # Backprop
        (mirror + z_dist + ES).backward()
        self.optimiser.step()
        self.optimiser.zero_grad()
                
        # Update Metric Tracker
        self.loss_dict['ES']     += ES.item()
        self.loss_dict['MSE']    += MSE.item()
        self.loss_dict['rep']    += rep.item()
        self.loss_dict['mirror'] += mirror.item()
        self.loss_dict['z_dist'] += z_dist.item()


    @torch.no_grad()
    def validate(self, x, y):
        # Use GPU if available
        x = x.to(self.device)
        y = y.to(self.device)

        self.AE.eval()
        self.Trans.eval()

        # Model Losses
        o, z, m      = self.forward(x)                          # Forward Pass through AE & Trans
        mirror       = torch.nn.functional.mse_loss(m, x)       # Mirror Loss
        ES, MSE, rep = self.energy_score(o, y)                  # Energy Score, MSE and Rep Losses
        z_dist, _, _ = self.energy_score(z, self.AE.Encoder(y)) # Energy Score in latent space

        # Update Metric Tracker
        self.val_loss_dict['ES']     += ES.item()
        self.val_loss_dict['MSE']    += MSE.item()
        self.val_loss_dict['rep']    += rep.item()
        self.val_loss_dict['mirror'] += mirror.item()
        self.val_loss_dict['z_dist'] += z_dist.item()


    def save_model(self):
        path = lambda i, j, k: f"{i}/weights_{j}_{k}.pt" 
        torch.jit.script(self.AE   ).save(path(self.save_path, 'AE'   , f'{self.latent_dim}'))
        torch.jit.script(self.Trans).save(path(self.save_path, 'Trans', f'{self.latent_dim}'))


    def track_losses(self, pbar, val=False):
        
        loss_dict = {x:self.loss_dict[x]/(self.step+1) for x in self.loss_dict}
        if val==False:
            pbar.set_postfix({
                
                'epoch'      : f"{self.epoch}/{self.total_epochs}",
                'ES'         : f"{loss_dict['ES']     :.2E}",
                'ES_val'     : f"---------",
                'mirror'     : f"{loss_dict['mirror'] :.2E}" ,
                'mirror_val' : f"---------",
                })
        
        else:
            val_loss_dict = {x:self.val_loss_dict[x]/(self.val_step+1) for x in self.val_loss_dict}

            pbar.set_postfix({
                'epoch'      : f"{self.epoch}/{self.total_epochs}",
                'ES'         : f"{loss_dict['ES']         :.2E}" ,
                'ES_val'     : f"{val_loss_dict['ES']     :.2E}" ,
                'mirror'     : f"{loss_dict['mirror']     :.2E}" ,
                'mirror_val' : f"{val_loss_dict['mirror'] :.2E}" ,
                })

@torch.no_grad()
def hellinger_distance_3D(p, q):
    d  = (torch.sqrt(p.mean((1,2))) - torch.sqrt(q.mean((1,2)))) ** 2
    d += (torch.sqrt(p.mean((0,2))) - torch.sqrt(q.mean((0,2)))) ** 2
    d += (torch.sqrt(p.mean((0,1))) - torch.sqrt(q.mean((0,1)))) ** 2
    h = torch.sqrt(torch.sum(d) / (2*3))
    return h


@torch.no_grad()
def load_model(arch, latent_size):
    weights_path = f'{utils.path}data/torch_script_models/'
    file_name    = f'weights_{arch}_{latent_size}'
    file         = f'{weights_path}{file_name}.pt'
    return torch.jit.load(io.BytesIO(open(file, 'rb').read()), map_location='cpu')


@torch.no_grad()
def SLT_prediction_ensemble(AE, Trans, seq_len, latent_size, truth_ensemble, evolution_time,
                            seed=0, inference=True, print_time=False):

    hardware = "cuda" if torch.cuda.is_available() else "cpu"
    device   = torch.device(hardware)

    # TODO - fix deterministic seeding:
    if inference==True and hardware=="cpu":
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(seed)
        random.seed(seed)

    AE.eval().to(device)
    Trans.eval().to(device)

    truth_ensemble, _, _ = utils.unit_gaussain(truth_ensemble.to(device))
    ens_size = truth_ensemble.size(0)
    
    z = torch.zeros((ens_size, seq_len+evolution_time, latent_size)).to(device)
    a = torch.zeros((ens_size, seq_len+evolution_time, seq_len+1, seq_len+1)).to(device)
    
    time_1 = time.time()

    z[:, :seq_len] = AE.Encoder(truth_ensemble[:, :seq_len])
    for t in range(seq_len, evolution_time+seq_len):
        z[:, t], a[:, t] = Trans(z[:, t-seq_len:t])
    u = AE.Decoder(z)

    time_2 = time.time()

    if print_time: 
        print(f'Prediction time: {(time_2-time_1)/ens_size:.4f} seconds')

    truth_ensemble = utils.un_unit_gaussain(truth_ensemble)*100
    prediction     = utils.un_unit_gaussain(u)*100

    return truth_ensemble, prediction, a


