import numpy as np, torch, tqdm, utils, math #type: ignore
#from torch_poly_lr_decay import PolynomialLRDecay

ARCH       = 'AE_Transformer'
DATA_PATH  = f'{utils.which_os}Beta_Plane_Jets/data/training_data/arrays/'
SAVE_PATH  = f'{utils.which_os}Beta_Plane_Jets/data/outputs/{ARCH}/'

BATCH_SIZE    = 200;
EPOCHS        = 500;
LEARNING_RATE = 5e-4;
LATENT_DIM    = 128; 
SEQ_LENGTH    = 10;
SEQ_FORWARD   = 1;
ENS_SIZE      = 1;
RUN_NUM       = 1; #1 no z loss
LAYERS        = 2;
WIDTH         = 4;


class Beta_Plane_ML:
    def __init__(self, feat_dim=256, latent_dim=256, seq_len=1, ens_size=4, epochs=1,
                 learning_rate=1e-4, seq_forward=2, training_steps=1, val_steps=1, num_heads=16, 
                 save_path=None, file_name=None, layers=2, width=2):
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
        self.seq_forward    = seq_forward
        self.num_heads      = num_heads
        self.layers         = layers
        self.width          = width
        self.best           = 10
        self.H_distance     = 10
        
        # Define Hardware
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define Models
        self.AE    = Autoencoder(self.feat_dim, self.latent_dim, 4).to(self.device)
        self.Trans = LSTM_VAE(self.latent_dim, 4).to(self.device)
        self.Disc  = Discriminator(self.feat_dim, 256).to(self.device)

        # Define Optimiser
        self.optimiser = torch.optim.Adam([
                {'params': self.Trans.parameters()},
                {'params': self.AE.parameters(), 'lr': 2e-3}
            ], lr = self.lr,
            )

        self.optimiser_D = torch.optim.Adam([
                {'params': self.Disc.parameters()},
            ], lr = self.lr,
            )

        # Define LR scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer  = self.optimiser,
            T_max      = self.total_epochs
        )

        # Validation Ensemblex#
        self.truth_ensemble = utils.truth_ensemble(
            seq_len         = SEQ_LENGTH,
            ens_size        = 8,
            evolution_time  = 500,
            time_step_start = 100)

        self.truth_long = utils.truth_long(seq_len=SEQ_LENGTH)

        # Loss function
        self.KL  = lambda mu, sigma: -0.5 *(1 + sigma - mu.pow(2) - sigma.exp()).sum()/mu.size(0)
        self.MSE = torch.nn.MSELoss()
        self.BCE = torch.nn.BCELoss()

        # Loss Tracker
        self.loss_dict     = {'MSE':0, 'mirror': 0, 'z_dist':0, 'ES':0, 'rep':0, 'd_real':0, 'd_fake':0}
        self.val_loss_dict = {'MSE':0, 'mirror': 0, 'z_dist':0, 'ES':0, 'rep':0, 'd_real':0, 'd_fake':0}


    def save_model(self, add_path=''):
        path       = lambda i, j, k: f"{i}weights/weights_{j}_{k}.pt" 
        file_name  = f'{self.latent_dim}_{0}_{self.total_epochs}_'
        file_name += f'{RUN_NUM}{add_path}'
        torch.jit.script(self.AE   ).save(path(self.save_path, 'AE' , file_name)) #type: ignore
        torch.jit.script(self.Trans).save(path(self.save_path, 'RNN', file_name)) #type: ignore


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
                                self.save_model()
                                self.track_losses(pbar, val=True)
                  
    
    def forward(self, x):
        # Store Tensors
        b  =  x.size(0)
        u  = torch.zeros(b, self.ens_size, self.seq_forward, self.feat_dim  , device=self.device)
        z  = torch.zeros(b, self.ens_size, self.seq_forward, self.latent_dim, device=self.device)
        mu = torch.zeros(b, self.ens_size, self.seq_forward, self.latent_dim, device=self.device)
        lv = torch.zeros(b, self.ens_size, self.seq_forward, self.latent_dim, device=self.device)

        z_in  = self.AE.Encoder(x) # Transform to latent space
        for i in range(self.ens_size): 
            h = torch.zeros(1)
            for t in range(self.seq_forward):
                z_t, mu[:, i, t], lv[:, i, t] = self.Trans(z_in)
                z[:, i, t] = z_t
                z_in = torch.cat((z_in[:, 1:], z_t.unsqueeze(1)), dim=1)[:, -self.seq_len:]

            u[:, i] = self.AE.Decoder(z[:, i])  # Transform back to observation space

        return u[:, 0], mu[:, 0, 0], lv[:, 0, 0], z[:, 0]
    

    def train(self, x, y): 
        # Use GPU if available
        x = x.to(self.device)
        y = y.to(self.device)

        self.AE.train()
        self.Trans.train()

        for _ in range(1):

            with torch.no_grad(): o, mu, sigma, z = self.forward(x)  # Forward Pass through AE & Trans

            d_real = self.Disc(y).mean()
            d_fake = self.Disc(o).mean()

            l_real = self.BCE(d_real, torch.ones_like(d_real))
            l_fake = self.BCE(d_fake, torch.zeros_like(d_fake))

            (l_real + l_fake).backward()
            self.optimiser_D.step()
            self.optimiser_D.zero_grad()

            self.AE.train()
            self.Trans.train()
            self.Disc.eval()

        o, mu, sigma, z = self.forward(x)                 # Forward Pass through AE & Trans
        mirror          = self.MSE(self.AE(x), x)         # Mirror Loss
        MSE             = self.MSE(y, o)                  # MSE Loss
        KL              = self.KL(mu, sigma)              # KL Loss
        z_dist          = self.MSE(z, self.AE.Encoder(y))
        d_fake_2        = self.Disc(o).mean()
        gen_loss        = self.BCE(d_fake_2, torch.ones_like(d_fake_2))

        # Backprop
        (mirror + MSE + KL + z_dist + gen_loss * self.epoch/self.total_epochs).backward()
        self.optimiser.step()
        self.optimiser.zero_grad()
                
        # Update Metric Tracker
        self.loss_dict['MSE']    += MSE.item()
        self.loss_dict['rep']    += KL.item()
        self.loss_dict['mirror'] += mirror.item()
        self.loss_dict['z_dist'] += z_dist.item()
        self.loss_dict['d_real'] += d_real.item()
        self.loss_dict['d_fake'] += d_fake.item()


    @torch.no_grad()
    def validate(self, x, y):
        # Use GPU if available
        x = x.to(self.device)
        y = y.to(self.device)

        self.AE.eval()
        self.Trans.eval()

        # Model Losses
        o, mu, sigma, z = self.forward(x)                 # Forward Pass through AE & Trans
        mirror          = self.MSE(self.AE(x), x)         # Mirror Loss
        MSE             = self.MSE(y, o)                  # MSE Loss
        KL              = self.KL(mu, sigma)              # KL Loss
        z_dist          = self.MSE(z, self.AE.Encoder(y))
        d_real          = self.Disc(y).mean()
        d_fake          = self.Disc(o).mean()
        
        # Update Metric Tracker
        self.val_loss_dict['MSE']    += MSE.item()
        self.val_loss_dict['rep']    += KL.item()
        self.val_loss_dict['mirror'] += mirror.item()
        self.val_loss_dict['z_dist'] += z_dist.item()
        self.val_loss_dict['d_real'] += d_real.item()
        self.val_loss_dict['d_fake'] += d_fake.item()



    def track_losses(self, pbar, val=False):
        
        loss_dict = {x:self.loss_dict[x]/(self.step+1) for x in self.loss_dict}
        if val==False:
            pbar.set_postfix({
                
                'epoch'      : f"{self.epoch}/{self.total_epochs}",
                'd_real'     : f"{loss_dict['d_real']     :.2E}",
                'd_fake'     : f"{loss_dict['d_fake']     :.2E}",
                'mirror'     : f"{loss_dict['mirror'] :.2E}" ,
                'mirror_val' : f"---------",
                })
        

        else:
            val_loss_dict = {x:self.val_loss_dict[x]/(self.val_step+1) for x in self.val_loss_dict}

            pbar.set_postfix({
                'epoch'      : f"{self.epoch}/{self.total_epochs}",
                'd_real'     : f"{loss_dict['d_real']     :.2E}",
                'd_fake'     : f"{loss_dict['d_fake']     :.2E}",
                'mirror'     : f"{loss_dict['mirror']     :.2E}" ,
                'mirror_val' : f"{val_loss_dict['mirror'] :.2E}" ,
                })



class TEPC_1D(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, in_dim:int,  out_dim: int):
        super(SpectralConv1d, self).__init__()

        self.in_dim  = in_dim //2+1
        self.out_dim = out_dim//2+1

        self.modes = max(self.in_dim, self.out_dim)

        self.scale   = 1 / (in_channels+out_channels)**0.5
        self.bias    = torch.nn.Parameter(torch.zeros(out_channels, self.modes, dtype=torch.cfloat))
        self.weights = torch.nn.Parameter(torch.rand(
                       in_channels, out_channels, self.modes, dtype=torch.cfloat) * self.scale)

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


class Discriminator(torch.nn.Module):
    def __init__(self, feat_dim=256, width=64):
        super().__init__()

        self.conv0  = SpectralConv1d(1    , width, feat_dim, feat_dim)
        self.conv1  = SpectralConv1d(width, 1, feat_dim, feat_dim)
        self.linear = torch.nn.Linear(feat_dim, 1)

    def forward(self, x):
        x = x.unsqueeze(-2)
        x = self.conv0(x)
        x = torch.nn.functional.gelu(x)
        x = self.conv1(x)
        x = torch.nn.functional.gelu(x)
        x = x.squeeze(-2)
        x = self.linear(x)
        x = torch.sigmoid(x)
        return x

class LSTM_VAE(torch.nn.Module):
    def __init__(self, dim=256, layers=4):
        super().__init__()

        self.dim     = dim

        self.lstm    = torch.nn.LSTM(self.dim, self.dim*2, num_layers=layers, batch_first=True)

        self.mu      = torch.nn.Linear(self.dim*2, self.dim)
        self.sigma   = torch.nn.Linear(self.dim*2, self.dim)

        self.fc_mlp1 = torch.nn.Linear(dim, dim*2)
        self.fc_mlp2 = torch.nn.Linear(dim*2, dim)

        k = lambda x: torch.fft.fftfreq(x)*x
        self.register_buffer('k', k(dim//2+1))

    def shift_phase(self, z):

        z   = torch.fft.rfft(z, norm='ortho')
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

        z      = self.lstm(z)[0][:, -1]

        mu     = self.mu(z)
        lv     = self.sigma(z)
        std    = torch.exp(0.5*lv)

        z      = mu + std * torch.randn_like(mu)

        z     = self.fc_mlp1(z)
        z     = torch.nn.functional.gelu(z)
        z     = self.fc_mlp2(z)

        z     = self.unshift_phase(z, phi)

        return z, mu, std


def train(config=None):  

    # Load data
    print('\nLoading Training Data. \n');
    X = np.load(f"{DATA_PATH}X_train_{SEQ_LENGTH}.npy")
    Y = np.load(f"{DATA_PATH}Y_train_{SEQ_LENGTH}.npy")
    print('Training Data Loaded. Number of Data Points = {}\n'.format(len(X)));

    X = utils.unit_gaussain(X)[0]
    Y = utils.unit_gaussain(Y)[0]

    #val   = round(X.shape[0]*(1-0.02))
    val   = 200000
    Val_X = torch.tensor(X[val:], dtype=torch.float32)
    Val_Y = torch.tensor(Y[val:], dtype=torch.float32)
    X     = torch.tensor(X[:val], dtype=torch.float32)#[:DATA_SET_SIZE]
    Y     = torch.tensor(Y[:val], dtype=torch.float32)#[:DATA_SET_SIZE]
    X, Y  = utils.shuffle(X, Y) #type: ignore

    file_name = f'{LATENT_DIM}_{SEQ_LENGTH}_{EPOCHS}_{RUN_NUM}';

    TRAINING_STEPS = len(X)     // BATCH_SIZE
    VAL_STEPS      = len(Val_X) // BATCH_SIZE
    training_set   = utils.DataGenerator(X    , Y    , TRAINING_STEPS, BATCH_SIZE, SEQ_FORWARD)
    validation_set = utils.DataGenerator(Val_X, Val_Y, VAL_STEPS     , BATCH_SIZE, SEQ_FORWARD)
        
    model = Beta_Plane_ML(
        epochs         = EPOCHS         ,
        seq_len        = SEQ_LENGTH     ,
        ens_size       = ENS_SIZE       ,
        seq_forward    = SEQ_FORWARD    ,
        latent_dim     = LATENT_DIM     ,
        learning_rate  = LEARNING_RATE  ,
        training_steps = TRAINING_STEPS ,
        val_steps      = VAL_STEPS      ,
        save_path      = SAVE_PATH      ,
        layers         = LAYERS         ,
        width          = WIDTH          ,
    )

    model.fit(training_set, validation_set)    


if __name__ == '__main__':
    train()
    