import numpy as np, torch, tqdm, wandb, io, random, utils, models

class Stochastic_Latent_Transformer:
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
        
        # Define Hardware
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define Models
        self.AE    = models.Autoencoder(self.feat_dim, self.latent_dim, 2).to(self.device)
        self.Trans = models.Transformer(self.latent_dim, self.seq_len).to(self.device)

        # Define Optimiser
        self.optimiser = torch.optim.Adam(
            list(self.AE.parameters()) + list(self.Trans.parameters()), lr=self.lr)
        
        self.trans_optimiser = torch.optim.Adam(self.Trans.parameters(), lr=self.lr)

        # Define LR scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer  = self.optimiser,
            T_max      = self.total_epochs)
        
        # Validation Ensemblex
        self.truth_ensemble = utils.truth_ensemble(
            seq_len         = self.seq_len,
            ens_size        = 8,
            evolution_time  = 500,
            time_step_start = 100)

        # Loss function
        self.MSE = torch.nn.MSELoss()

        # Loss Tracker
        self.loss_dict     = {'MSE':0, 'mirror': 0, 'z_dist':0, 'ES':0, 'rep':0}
        self.val_loss_dict = {'MSE':0, 'mirror': 0, 'z_dist':0, 'ES':0, 'rep':0}


    def save_model(self, add_path=''):
        path       = lambda i, j, k: f"{i}weights/weights_{j}_{k}.pt" 
        file_name  = f'{self.latent_dim}_{0}_{self.total_epochs}_'
        file_name += f'{RUN_NUM}{add_path}'
        torch.jit.script(self.AE   ).save(path(self.save_path, 'AE'   , file_name))
        torch.jit.script(self.Trans).save(path(self.save_path, 'Trans', file_name))


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
                                if self.epoch%10==0 or self.epoch<10: self.generate_plots()
                                if self.epoch%5==0: self.save_model()
                                self.track_losses(pbar, val=True)
                  
    
    def forward(self, x):
        z = torch.zeros(x.size(0), self.ens_size, self.seq_forward, self.latent_dim, device=self.device)

        x = self.AE.Encoder(x).unsqueeze(1).expand(-1, self.ens_size, -1, -1)
        for i in range(self.ens_size):
            z_hist = x[:, i]
            for t in range(self.seq_forward):
                z_t    = self.Trans(z_hist)[0]
                z_hist = torch.cat((z_hist, z_t.unsqueeze(1)), dim=1)[:, 1:]
            z[:, i] = z_hist[:, -self.seq_forward:]
        u = self.AE.Decoder(z)
        x = self.AE.Decoder(x[:, 0])
        return u, z, x


    def energy_score(self, x, y, p=2):
        """
        Inputs:
        x: ["batch", "ensemble_size", "data_size"], contains all forecasts for each batch element
        y: ["batch", 1              , "data_size"], contains all verifications for each batch element
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
        mirror       = self.MSE(m, x)                           # Mirror Loss
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
        mirror       = self.MSE(m, x)
        ES, MSE, rep = self.energy_score(o, y)                  # Energy Score, MSE and Rep Losses
        z_dist, _, _ = self.energy_score(z, self.AE.Encoder(y)) # Energy Score in latent space

        # Update Metric Tracker
        self.val_loss_dict['ES']     += ES.item()
        self.val_loss_dict['MSE']    += MSE.item()
        self.val_loss_dict['rep']    += rep.item()
        self.val_loss_dict['mirror'] += mirror.item()
        self.val_loss_dict['z_dist'] += z_dist.item()


    def generate_plots(self):

        truth_ens, preds_ens, att = utils.prediction_ensemble(self.AE, self.Trans, self.seq_len, self.latent_dim, self.truth_ensemble, 500)
        truth_mat, preds_mat      = utils.calculate_grad_fields(truth_ens, preds_ens)
        H_t, H_p, edges           = utils.calculate_pdfs(truth_mat, preds_mat)
        _, mse_slt, rep_slt       = utils.CRPS(preds_ens[1:], truth_ens[0])
        _, mse_t, rep_t           = utils.CRPS(truth_ens[1:], truth_ens[0])

        try: self.H_distance      = utils.calculate_1D_pdfs(truth_mat[::2], preds_mat[1::2], nbins=150)[3]
        except: pass

        if self.epoch==0: self.H_t_distance = utils.calculate_1D_pdfs(truth_mat[::2], truth_mat[1::2], nbins=150)[3]

        self.img       = utils.plot_ensembles(truth_ens[:4], preds_ens[:4], self.seq_len, show=False)
        self.psd       = utils.plot_spectra(  truth_ens, preds_ens, show=False)
        self.pdf       = utils.plot_pdf(   H_t, H_p, edges, show=False)
        self.pdf_1d    = utils.plot_1d_pdf(H_t, H_p, edges, show=False)
        self.att_w     = utils.plot_attention(att[:4], show=False)
        self.att_w_100 = utils.plot_attention(att[:4, :100], show=False)
        self.img_10    = utils.plot_ensembles(truth_ens[:4, :self.seq_len+10] , preds_ens[:4, :self.seq_len+10] , self.seq_len, show=False)
        self.img_100   = utils.plot_ensembles(truth_ens[:4, :self.seq_len+100], preds_ens[:4, :self.seq_len+100], self.seq_len, show=False)
        self.cprs      = utils.plot_CPRS(mse_slt, rep_slt, mse_t, rep_t, self.seq_len, show=False)


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

            wandb.log({
                "epoch"          : self.epoch                    ,
                "ES"             : loss_dict['ES']               ,
                "MSE"            : loss_dict['MSE']              ,
                "rep"            : loss_dict['rep']              ,
                "z_dist"         : loss_dict['z_dist']           ,
                "mirror"         : loss_dict['mirror']           ,
                "ES_val"         : val_loss_dict['ES']           ,
                "MSE_val"        : val_loss_dict['MSE']          ,
                "rep_val"        : val_loss_dict['rep']          ,
                "z_dist_val"     : val_loss_dict['z_dist']       ,
                "mirror_val"     : val_loss_dict['mirror']       ,
                'Helliger'       : self.H_distance               ,
                'Helliger_t'     : self.H_t_distance             ,
                "img"            : wandb.Image(self.img)         ,
                "img_10"         : wandb.Image(self.img_10)      ,
                "img_100"        : wandb.Image(self.img_100)     ,
                "psd"            : wandb.Image(self.psd)         ,
                "hist"           : wandb.Image(self.pdf)         ,
                "1d_hist"        : wandb.Image(self.pdf_1d)      ,
                'att_weights'    : wandb.Image(self.att_w)       ,
                'att_weights_100': wandb.Image(self.att_w_100)   ,
                'CRPS'           : wandb.Image(self.cprs)        ,
                'lr'             : self.optimiser.param_groups[0]['lr']
            })


@torch.no_grad()
def load_model(arch, latent_size, s_weight, epochs, run_num):
    weights_path = f'{utils.path}data/torch_script_models/weights/'
    file_name    = f'weights_{arch}_{latent_size}_{s_weight}_{epochs}_{run_num}'
    file         = f'{weights_path}{file_name}.pt'
    return torch.jit.load(io.BytesIO(open(file, 'rb').read()), map_location='cpu')


@torch.no_grad()
def SLT_prediction_ensemble(AE, RNN, seq_len, latent_size, truth_ensemble, evolution_time, seed=0):

    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    random.seed(seed)

    hardware = "cuda" if torch.cuda.is_available() else "cpu"
    device   = torch.device(hardware)
    AE.eval()
    RNN.eval()

    truth_ensemble, _, _ = utils.unit_gaussain(truth_ensemble.to(device))
    ens_size = truth_ensemble.size(0)
    
    z = torch.zeros((ens_size, seq_len+evolution_time, latent_size)).to(device)
    a = torch.zeros((ens_size, seq_len+evolution_time, seq_len+1, seq_len+1)).to(device)

    z[:, :seq_len] = AE.Encoder(truth_ensemble[:, :seq_len])
    for t in range(seq_len, evolution_time+seq_len):
        z[:, t], a[:, t] = RNN(z[:, t-seq_len:t])
    u = AE.Decoder(z)

    truth_ensemble = utils.un_unit_gaussain(truth_ensemble)*100
    prediction     = utils.un_unit_gaussain(u)*100

    return truth_ensemble, prediction, a


