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

import numpy as np, matplotlib.pyplot as plt, torch, subprocess, os, time, sklearn, scipy
from matplotlib.colors import LogNorm
from PIL import Image

### WHICH HARDWARE  ###############################################################################
path     = os.path.dirname(os.path.realpath(__file__)) + '/../'
device   = torch.device("mps"  if torch.backends.mps.is_available() else \
                       ("cuda" if torch.cuda.is_available() else "cpu"))

### DATA PROCESSING ###############################################################################

def unit_gaussain(data, calculate=False):
    if calculate==False:
        data_mean = 4.179747095942411e-11
        data_std  = 0.03395488737425462
    else:
        data_mean, data_std = data.mean(), data.std()
    data_norm = (data - data_mean) / data_std

    return data_norm, data_mean, data_std

torch.no_grad()
def un_unit_gaussain(data, data_mean=None, data_std=None):
    if data_mean is None or data_std is None:
        data_mean = 4.179747095942411e-11
        data_std  = 0.03395488737425462
    data = data * data_std + data_mean
    return data

def shuffle(X, Y):
    return sklearn.utils.shuffle(X, Y)


### DATA LOADER ###################################################################################


def load_training_data(path, history_len):

    dimensions   = 256
    spin_up_time = 50
    data_out     = np.zeros((0, 2, history_len, dimensions))

    for file in os.listdir(path):
        if '.csv' in file and 'shift' not in file:
            data     = np.genfromtxt(path+file, delimiter=',', dtype=np.float32)[:,spin_up_time:].T

            data_len = np.shape(data)[0]-(history_len*2)
            data_arr = np.zeros((data_len, 2, history_len, dimensions))

            X         = data[:-history_len]
            Y         = data[history_len:]

            for i in range(data_len):
                data_arr[i, 0] = X[i:i+history_len]
                data_arr[i, 1] = Y[i:i+history_len]

            data_out = np.concatenate((data_out, data_arr),axis=0)

    X = data_out[:,0]
    Y = data_out[:,1]

    return X, Y


class DataGenerator(torch.utils.data.Dataset):
    def __init__(self, X, Y, steps=None, batch_size=32, seq_forward=10):
        self.X           = X
        self.Y           = Y
        self.seq_forward = seq_forward
        self.batch_size  = batch_size
        self.steps       = steps
        self.len_X       = len(X)
        self.indexes     = np.arange(self.len_X)

    def __len__(self): return self.steps

    def __getitem__(self, idx):
        X_batch = self.X[idx*self.batch_size: (idx+1)*self.batch_size]
        Y_batch = self.Y[idx*self.batch_size: (idx+1)*self.batch_size]
        Y_batch = Y_batch[:, :self.seq_forward]
        return (X_batch, Y_batch)


def truth_ensemble(seq_len=5, evolution_time=256, ens_size=8, time_step_start=100):
    truth_ens   = np.zeros((ens_size, evolution_time+seq_len, 256)).astype('float32')
    truth_path  = f'{path}data/test_data/'
    t           = time_step_start
    test_data_1 = np.genfromtxt(truth_path+f'0_{t-100}_{t}_umean.csv', delimiter=',')
    test_data_1 = test_data_1[:, -seq_len:]

    for i in range(10):
        try:
            test_data_2 = np.genfromtxt(truth_path+f'{i+1}_{t}_{t+500}_umean.csv',delimiter=',')
            truth_ens[i, :seq_len] = test_data_1.T
            truth_ens[i, seq_len:] = test_data_2[:, :evolution_time].T
        except: pass
    return torch.from_numpy(truth_ens)
    

### MODEL EVALUATION ##############################################################################


def calculate_grad_fields(truth_ens, preds_ens):

    preds_ens_size, evolution_time, y_size = preds_ens.size()

    # ensmeble, time, lat, (u, dy, dt)
    truth_mat = torch.zeros((preds_ens_size, evolution_time, y_size, 3))
    preds_mat = torch.zeros((preds_ens_size, evolution_time, y_size, 3))
    
    for j in range(preds_ens_size):
        truth_mat[j, :, :, 0] = truth_ens[j]
        preds_mat[j, :, :, 0] = preds_ens[j]

        for t in range(evolution_time):
            truth_mat[j, t, :, 1] = torch.gradient(truth_ens[j, t])[0]
            preds_mat[j, t, :, 1] = torch.gradient(preds_ens[j, t])[0]

        for y in range(y_size):
            truth_mat[j, :, y, 2] = torch.gradient(truth_ens[j, :, y])[0]
            preds_mat[j, :, y, 2] = torch.gradient(preds_ens[j, :, y])[0]

    return truth_mat, preds_mat 


def calculate_pdfs(truth_mat, preds_mat, nbins=100):

    u_t = truth_mat.flatten(0, 2)
    u_p = preds_mat.flatten(0, 2)

    p, edges = torch.histogramdd(u_t, nbins)

    edges_  = [edges[0].min(), edges[0].max(),
               edges[1].min(), edges[1].max(),
               edges[2].min(), edges[2].max()]
    range_  = [float(i.detach().cpu().numpy()) for i in edges_]

    q, _, = torch.histogramdd(u_p, nbins, range=range_)

    p /= p.sum()
    q /= q.sum()

    return p, q, edges


def calculate_1D_pdfs(truth_mat, preds_mat, nbins=100):

    u_t = truth_mat[0].flatten()
    u_p = preds_mat[0].flatten()

    range_min = torch.min(torch.cat((u_t, u_p))).item()
    range_max = torch.max(torch.cat((u_t, u_p))).item()

    p, edges = torch.histogram(u_t, nbins, range=(range_min, range_max))
    q, _,    = torch.histogram(u_p, nbins, range=(range_min, range_max))

    p /= p.sum()
    q /= q.sum()

    hellinger = (0.5*torch.sum((p**0.5 - q**0.5)**2))**0.5

    return p, q, edges, hellinger.item()


def CRPS(x, y):
    crps = []
    mse  = []
    rep  = []

    y = y.unsqueeze(0)

    x = x.unsqueeze(0)
    y = y.unsqueeze(0)

    ens_size = x.size(1)

    for t in range(x.size(2)):

        MSE      = torch.cdist(x[:, :, t], y[:, :, t], 2).mean() / x.size(-1)**0.5 
        ens_var  = torch.cdist(x[:, :, t], x[:, :, t], 2).mean() / x.size(-1)**0.5
        ens_var *= ens_size/(ens_size - 1) # to result in 1/[m(m-1)]

        crps.append(2*MSE-ens_var)
        mse.append(MSE.detach().cpu().abs().numpy())
        rep.append(ens_var.detach().cpu().abs().numpy())

    return crps, mse, rep


### PLOTS #########################################################################################

def plot_ensembles(truth_ens, preds_ens, seq_len=5, damping_time=False, show=True):

    num_y     = truth_ens.size()[0]//4
    fig, axs  = plt.subplots(num_y*2, 4, figsize=(16, (1+1*num_y)*4), constrained_layout=True)
    tick_pos  = [-1, -0.5, 0, 0.5, 1]
    labels    = [r'-$\pi$', r'-$\pi/2$', r'0', r'$\pi/2$',r'$\pi$']
    truth_ens = truth_ens.detach().cpu().numpy()
    preds_ens = preds_ens.detach().cpu().numpy()

    if damping_time==True:
        extent  = [0, (6.25/500)*truth_ens.shape[-1], -1, 1]
        x_label = r'Damping Time ($\mu $t)'
        v_line = seq_len*(6.25/500)
    else:
        extent  = [0, truth_ens.shape[1], -1, 1]
        x_label = 't'
        v_line  = seq_len

    def plot(i, truth, title):
        shift = 0
        if title=='Neural Network Emulation': shift+=num_y
        ax = axs[(i//4) + shift, (i)%4]
        im = ax.imshow(truth.T, aspect='auto', extent=extent, vmin=-8.263900130987167, vmax=8.826799690723419)
        ax.axvline(x=v_line, linestyle='--', color='w')
        ax.set_title(f'{title} {i+1}')
        ax.set_ylabel('y - Latitude')
        ax.set_xlabel(x_label)
        ax.set_yticks(tick_pos, labels)
        return im

    for i, truth in enumerate(truth_ens): imt = plot(i, truth, 'Numerical Integtation')
    for i, pred in enumerate(preds_ens) : imp = plot(i, pred , 'Neural Network Emulation')

    fig.colorbar(imp, ax=axs.ravel().tolist(), shrink=0.465, pad=0.015, label=r'U(y,t)')
        
    if show==True:
        fig.suptitle('Latitude-Time Plots showing Zonally-Averaged Zonal Wind for Ensembles of ' +
                'Numerical Integrations and Neural Network Emulations \n from Identical ' +
                'Initial Conditions (the dotted line) for the Stochastic Beta-Plane System',\
                fontsize='x-large')
        plt.show()
    else:
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        plt.close()

        im    = Image.open(img_buf)
        im_np = np.array(im)
        img_buf.close()
        return im_np
    
def plot_attention(att_weights, seq_len=5, damping_time=False, show=True):

    num_y       = att_weights.size(0)//4
    fig, axs    = plt.subplots(num_y, 4, figsize=(16, (num_y)*4), constrained_layout=True)
    att_weights = att_weights[:, :, -1, 0].detach().cpu()

    if damping_time==True:
        extent  = [seq_len, (6.25/500)*att_weights.size(1), -1, 1]
        x_label = r'Damping Time ($\mu $t)'
    else:
        extent  = [seq_len, att_weights.size(1), -1, 1]
        x_label = 't'
        v_line  = seq_len

    def plot(i, att, title):
        if att_weights.size(0)>4: ax = axs[(i//4), (i)%4]
        else                    : ax = axs[i]
        ax.plot(att)
        ax.set_ylim(0, att_weights.max()*1.1)
        ax.set_xlim(seq_len, att_weights.size(1))
        ax.set_title(f'{title} {i+1}')
        ax.set_xlabel(x_label)

    for i, att in enumerate(att_weights): plot(i, att, 'Numerical Integration')
        
    if show==True:
        plt.show()
    else:
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        plt.close()

        im    = Image.open(img_buf)
        im_np = np.array(im)
        img_buf.close()
        return im_np

def plot_pdf(H_t, H_p, edges,show=True):

    label = ['U', r'$U_y$', r'$U_t$']

    #if show==True: fig, axs = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)
    if show==True: fig, axs = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=True)
    else         : fig, axs = plt.subplots(1, 4, figsize=(18, 4) , constrained_layout=True)

    if show==True: ax = axs[0, 0] 
    else         : ax = axs[0]
    c = ax.pcolor(edges[0].detach().cpu().numpy(),
        edges[1].detach().cpu().numpy(),
        H_t.sum(2).detach().cpu().numpy(),
        norm=LogNorm(), cmap='inferno')
    ax.set_xlabel(label[0]); ax.set_ylabel(label[1])
    ax.set_title(f'Truth PDF({label[0]}, {label[1]})')
    try: fig.colorbar(c, ax=ax)
    except: pass

    if show==True: ax = axs[0, 1] 
    else         : ax = axs[1]
    c = ax.pcolor(edges[0].detach().cpu().numpy(),
        edges[1].detach().cpu().numpy(),
        H_p.sum(2).detach().cpu().numpy(),
        norm=LogNorm(), cmap='inferno')
    ax.set_xlabel(label[0]); ax.set_ylabel(label[1])
    ax.set_title(f'Prediction PDF({label[0]}, {label[1]})')
    try: fig.colorbar(c, ax=ax)
    except: pass

    if show==True: ax = axs[1, 0] 
    else         : ax = axs[2]
    c = ax.pcolor(edges[0].detach().cpu().numpy(),
        edges[2].detach().cpu().numpy(),
        H_t.sum(1).detach().cpu().numpy(),
        norm=LogNorm(), cmap='inferno') 
    ax.set_xlabel(label[0]); ax.set_ylabel(label[2])
    ax.set_title(f'Truth PDF({label[0]}, {label[2]})')
    try: fig.colorbar(c, ax=ax)
    except: pass

    if show==True: ax = axs[1, 1] 
    else         : ax = axs[3]
    c = ax.pcolor(edges[0].detach().cpu().numpy(),
        edges[2].detach().cpu().numpy(),
        H_p.sum(1).detach().cpu().numpy(),
        norm=LogNorm(), cmap='inferno')
    ax.set_xlabel(label[0]); ax.set_ylabel(label[2])
    ax.set_title(f'Prediction PDF({label[0]}, {label[2]})')
    try: fig.colorbar(c, ax=ax)
    except: pass

    if show==True:
        plt.show()
    else:
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        plt.close()

        im    = Image.open(img_buf)
        im_np = np.array(im)
        img_buf.close()
        return im_np

def plot_1d_pdf(H_t, H_p, edges, show=True):

    label = [r'$U$', r'$\partial_y U$', r'$\partial_t U$']

    fig, axs = plt.subplots(2, 3, figsize=(16, 8) , constrained_layout=True)

    index = [(1, 2), (0, 2), (0, 1)]
    lims  = [1e-2, 1e-3, 1e-3, 2e-1, 2e-1, 5e-1]

    for i in range(3):
        
        h_t = H_t.mean(index[i]).detach().cpu().numpy()
        h_p = H_p.mean(index[i]).detach().cpu().numpy()
        h_t = (h_t[2:-2]+h_t[0:-4]+h_t[4:]+h_t[1:-3]+h_t[3:-1])/5
        h_p = (h_p[2:-2]+h_p[0:-4]+h_p[4:]+h_p[1:-3]+h_p[3:-1])/5
        x = np.linspace(edges[i].min(), edges[i].max(), len(h_t))

        ax = axs[0, i]

        ax.plot(x, h_t, 'g', label='Truth')
        ax.plot(x, h_p, 'r', label='ML')
        ax.set_xlabel(label[i]); ax.set_ylabel('Density')
        ax.set_title(f'Truth PDF({label[i]})')
        ax.legend()

        ax = axs[1, i]

        ax.plot(x, h_t, 'g', label='Truth')
        ax.plot(x, h_p, 'r', label='ML')
        ax.set_xlabel(label[i]); ax.set_ylabel('Density (log)')
        ax.set_yscale('log')
        #ax.set_ylim(lims[i], lims[i+3])
        ax.set_title(f'Truth PDF({label[i]})')
        ax.legend()

    if show==True:
        plt.show()
    else:
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        plt.close()

        im    = Image.open(img_buf)
        im_np = np.array(im)
        img_buf.close()
        return im_np

def plot_spectra(truth_ens, preds_ens, show=True):

    psd_t = torch.abs(torch.fft.rfft(truth_ens, norm='ortho')).pow(2).flatten(0,-2)
    psd_f = torch.abs(torch.fft.rfft(preds_ens, norm='ortho')).pow(2).flatten(0,-2)

    mean_t = psd_t.mean(0).detach().cpu().numpy()
    mean_f = psd_f.mean(0).detach().cpu().numpy()
    u_75_t = np.percentile(psd_t.detach().cpu().numpy(), 75, axis=0)
    u_75_f = np.percentile(psd_f.detach().cpu().numpy(), 75, axis=0)
    u_25_t = np.percentile(psd_t.detach().cpu().numpy(), 25, axis=0)
    u_25_f = np.percentile(psd_f.detach().cpu().numpy(), 25, axis=0)

    #fig = plt.figure(figsize=(16, 8))
    fig = plt.figure(figsize=(10, 5))
    plt.plot(mean_t, label='Truth', c='green')
    plt.plot(mean_f, label='ML'   , c='red')
    plt.fill_between(range(len(mean_t)), u_25_t, u_75_t, alpha=0.2, color='green')
    plt.fill_between(range(len(mean_f)), u_25_f, u_75_f, alpha=0.2, color='red')
    plt.ylim(1e-10, 1)
    plt.xlim(1, 100)
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('k')
    plt.ylabel('ε(k)')
    plt.title('Kinetic Energy Spectrum, time averaged')
    plt.legend()

    if show==True:
        plt.show()
    else:
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        plt.close()

        im    = Image.open(img_buf)
        im_np = np.array(im)
        img_buf.close()
        return im_np
    

def plot_CPRS(mse_slt, rep_slt, mse_t, rep_t, seq_len=10, show=True):

    fig, axs = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

    ax = axs[0]
    ax.plot(mse_slt[seq_len+1:], label=f'SLT', color='red')
    ax.plot(mse_t  [seq_len+1:], label=f'Num', color='green')
    ax.set_yscale('log')
    ax.set_title('MAE', fontsize=5.2*2)
    ax.set_xlabel('t', fontsize=5.2*2)
    ax.set_xlim(0, 250)
    ax.legend()

    ax = axs[1]
    ax.plot(rep_slt[seq_len+1:], label=f'SLT', color='red')
    ax.plot(rep_t  [seq_len+1:], label=f'Num', color='green')
    ax.set_yscale('log')
    ax.set_title('Ensemble Variation', fontsize=5.2*2)
    ax.set_xlabel('t', fontsize=5.2*2)
    ax.set_xlim(0, 250)
    ax.legend(fontsize=5.2*2)

    if show==True:
        plt.show()
    else:
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        plt.close()

        im    = Image.open(img_buf)
        im_np = np.array(im)
        img_buf.close()
        return im_np
    

### Probabilities ####################################################################################

'''def load_num_ens(s, t, seq_len=5, roll=False):
    truth_path = which_os+'Beta_Plane_Jets/data/test_data/csv_files/new_dir/'
    arr = []
    for i in range(1000, 3000):
        try:
            test_data = np.genfromtxt(truth_path+f'{i}_{s+t}_{s+100}_umean.csv', delimiter=',')
            if roll is True: test_data = np.roll(test_data, -180, axis=0)
            arr.append(test_data[:, t-seq_len:])
        except:
            try:
                np.genfromtxt(truth_path+f'0_{t-100}_{t}_umean.csv', delimiter=',')
            except:
                pass
    return np.stack(arr)


def load_num_ens2(s, t, seq_len=5, roll=False):
    truth_path = which_os+'Beta_Plane_Jets/data/test_data/csv_files/new_dir/'
    arr = []
    start = s+t
    end   = (start//10)*10 + 100
    for i in range(0, 700):
        try:
            test_data = np.genfromtxt(truth_path+f'{i}_{start}_{end}_umean.csv', delimiter=',')
            if roll is True: test_data = np.roll(test_data, -180, axis=0)
            arr.append(test_data[:, start%10-seq_len:])
        except: pass
    return np.stack(arr)


@torch.no_grad()
def generate_ML_ens(AE_model, RNN_model, s, t, ens_size=6, seq_len=5, latent_dim=256, time_steps=100, roll=0, print_time=False):

    if AE_model is not None:
        AE_model.to(device)
        AE_model.eval()
    RNN_model.to(device)
    RNN_model.eval()

    truth_path = which_os+'Beta_Plane_Jets/data/test_data/csv_files/'
    if t<seq_len:
        test_data    = np.genfromtxt(truth_path+f'0_{s-100}_{s}_umean.csv', delimiter=',')[:, t-seq_len:]
        app          = np.genfromtxt(truth_path+f'0_{s}_{s+100}_umean.csv', delimiter=',')[:, 0:t]
        test_data    = np.concatenate((test_data, app), axis=1)
    else : test_data = np.genfromtxt(truth_path+f'0_{s}_{s+100}_umean.csv', delimiter=',')[:, t-seq_len:t]

    test_data = np.roll(test_data, roll, axis=0)

    data = np.repeat(np.expand_dims(test_data, 0), ens_size, axis=0)
    data = torch.from_numpy(data).transpose(1, 2).float().to(device)

    if AE_model is None: data, preds, att = prediction_ensemble_trans(RNN_model, seq_len, data, time_steps)
    else               : data, preds, att = prediction_ensemble(AE_model, RNN_model, seq_len, latent_dim, data, time_steps)
    
    return preds.transpose(1,2), att


@torch.no_grad()
def generate_ML_ens2(AE_model, RNN_model, s, t, ens_size=6, seq_len=5, latent_dim=256, time_steps=100, roll=0, print_time=False):

    if AE_model is not None:
        AE_model.to(device)
        AE_model.eval()
    RNN_model.to(device)
    RNN_model.eval()

    truth_path = which_os+'Beta_Plane_Jets/data/test_data/csv_files/'
    test_data  = np.genfromtxt(truth_path+'1000_1000_6000_umean.csv', delimiter=',')
    test_data  = np.roll(test_data[:, s+t-seq_len:s+t], roll, axis=0)

    data = np.repeat(np.expand_dims(test_data, 0), ens_size, axis=0)
    data = torch.from_numpy(data).transpose(1, 2).float().to(device)

    if AE_model is None: data, preds, att = prediction_ensemble_trans(RNN_model, seq_len, data, time_steps)
    else               : data, preds, att = prediction_ensemble(AE_model, RNN_model, seq_len, latent_dim, data, time_steps)
    
    return preds.transpose(1,2), att


@torch.no_grad()
def generate_ML_ens3(AE_model, RNN_model, s, t, ens_size=6, seq_len=5, latent_dim=256, time_steps=100, roll=0, print_time=False):

    if AE_model is not None:
        AE_model.to(device)
        AE_model.eval()
    RNN_model.to(device)
    RNN_model.eval()

    truth_path = which_os+'Beta_Plane_Jets/data/test_data/csv_files/'
    test_data  = np.genfromtxt(truth_path+'e_umean.csv', delimiter=',')
    test_data  = np.roll(test_data[:, s+t-seq_len:s+t], roll, axis=0)

    data = np.repeat(np.expand_dims(test_data, 0), ens_size, axis=0)
    data = torch.from_numpy(data).transpose(1, 2).float().to(device)

    if AE_model is None: data, preds, att = prediction_ensemble_trans(RNN_model, seq_len, data, time_steps)
    else               : data, preds, att = prediction_ensemble(AE_model, RNN_model, seq_len, latent_dim, data, time_steps)
    
    return preds.transpose(1,2), att


def plot_ens(data, ens_size=6, seq_len=5, title=None):
    f = plt.figure(figsize = (16,12))
    for i in range(ens_size**2):
        plt.subplot(ens_size, ens_size, i+1)
        im = plt.imshow(data[i], aspect='auto', origin='lower')
        plt.axvline(x=seq_len, linestyle='--', color='w')
    if title is not None: plt.suptitle(title)
    plt.show()


def time_dist_ens(data, target_jets=3, seq_len=5, ens_size=5, threshold=0, min=0, t_max=0, width=7):
    time_arr = []
    for i in range(ens_size):
        if i%100==0: print(i)
        try:
            time_log = []
            if t_max==0: t_max = data[0].shape[-1]
            for t in range(seq_len+min, t_max):
                u = data[i, :, t]
                peak_1 = np.argmax(u)
                u = np.roll(u, -peak_1+50)
                jets = len(scipy.signal.find_peaks(u, height=threshold, distance=10, width=width)[0])
                time_log.append(jets)
                if sum(time_log[-5:])/5==target_jets:
                    time_arr.append(t-5)
                    break
                if t==data[0].shape[-1]-1:
                    time_arr.append(t_max+10)
        except IndexError:
            pass
        
    return time_arr


def num_jets(arr, threshold=0, width=7):
    count = []
    for u in arr:
        if torch.sum(u)!= 0:
            peak_1 = torch.argmax(u).item()
            u = torch.roll(u, -peak_1+50)
            jets = len(scipy.signal.find_peaks(u, height=threshold, distance=10, width=width)[0])
        count.append(jets)
    
    time_lag = 5
    mat = np.zeros((8, 3))
    B   = np.zeros(time_lag)
    l   = []
    c   = -1
    for i in range(time_lag, len(count)):
        A = count[i-time_lag:i]
        if 0 not in A:
            l.append(round(np.sum(A)/time_lag))
            c+=1
            if 0 not in B:
                if   l[c] > l[c-1]: mat[l[c-1], 2] += 1
                elif l[c] < l[c-1]: mat[l[c-1], 0] += 1
                else              : mat[l[c-1], 1] += 1
        B = A

    mat = mat[:5]
    mat/=np.sum(mat)
            
    return l, mat'''