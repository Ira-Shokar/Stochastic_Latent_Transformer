# Copyright (c) 2023 Ira Shokar

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.#

import numpy as np, torch, wandb, utils, SLT

ARCH       = 'Stochastic_Latent_Transformer'
DATA_PATH  = f'{utils.path}data/training_data/';
SAVE_PATH  = f'{utils.path}data/torch_script_outputs/';

BATCH_SIZE    = 200;
EPOCHS        = 500;
LEARNING_RATE = 2e-3;
LATENT_DIM    = 64; 
SEQ_LENGTH    = 10;
SEQ_FORWARD   = 1;
ENS_SIZE      = 4;
RUN_NUM       = 100; 
LAYERS        = 2;
WIDTH         = 4

if __name__ == '__main__':

    # load data
    print('\nLoading Training Data. \n');
    X, Y = utils.load_training_data(DATA_PATH, SEQ_LENGTH)
    print('Training Data Loaded. Number of Data Points = {}\n'.format(len(X)));

    # normalise data to standard normal distribution
    X = utils.unit_gaussain(X)[0]
    Y = utils.unit_gaussain(Y)[0]

    # split training and validation data
    val   = 200000
    Val_X = torch.tensor(X[val:], dtype=torch.float32)
    Val_Y = torch.tensor(Y[val:], dtype=torch.float32)
    X     = torch.tensor(X[:val], dtype=torch.float32)
    Y     = torch.tensor(Y[:val], dtype=torch.float32)

    # shuffle training data
    X, Y  = utils.shuffle(X, Y) 

    # number of steps per epoch
    TRAINING_STEPS = len(X)     // BATCH_SIZE
    VAL_STEPS      = len(Val_X) // BATCH_SIZE

    # define torch dataloader
    training_set   = utils.DataGenerator(X    , Y    , TRAINING_STEPS, BATCH_SIZE, SEQ_FORWARD)
    validation_set = utils.DataGenerator(Val_X, Val_Y, VAL_STEPS     , BATCH_SIZE, SEQ_FORWARD)

    # OPTIONAL: define wandb training tracker
    file_name = f'{LATENT_DIM}_{SEQ_LENGTH}_{EPOCHS}_{RUN_NUM}';
    run = wandb.init(project=ARCH, name=f'{ARCH}{file_name}', dir=SAVE_PATH, save_code=True);
        
    # define wandb training tracker
    model = SLT.Stochastic_Latent_Transformer(
        epochs         = EPOCHS         ,
        ens_size       = ENS_SIZE       ,
        seq_forward    = SEQ_FORWARD    ,
        seq_len        = SEQ_LENGTH     ,
        latent_dim     = LATENT_DIM     ,
        learning_rate  = LEARNING_RATE  ,
        training_steps = TRAINING_STEPS ,
        val_steps      = VAL_STEPS      ,
        save_path      = SAVE_PATH      ,
        layers         = LAYERS         ,
        width          = WIDTH          ,
    )

    # train model
    model.fit(training_set, validation_set)   

    # OPTIONAL: finish wandb tracking
    run.finish()