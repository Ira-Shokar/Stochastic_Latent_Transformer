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

import numpy as np, torch, utils, SLT

BATCH_SIZE    = 200
EPOCHS        = 400
LEARNING_RATE = 5e-4
LATENT_DIM    = 64
SEQ_LENGTH    = 10
SEQ_FORWARD   = 1
ENS_SIZE      = 4
LAYERS        = 2
WIDTH         = 4

def train():
    # Set paths
    DATA_PATH = f'{utils.path}data/training_data/'
    SAVE_PATH = f'{utils.path}data/torch_script_outputs/'

    # Load data
    X, Y = utils.load_training_data(DATA_PATH, SEQ_LENGTH)

    # Normalize data to standard normal distribution
    X = utils.unit_gaussain(X)[0]
    Y = utils.unit_gaussain(Y)[0]

    # Split training and validation data
    val   = 200000
    Val_X = torch.tensor(X[val:], dtype=torch.float32)
    Val_Y = torch.tensor(Y[val:], dtype=torch.float32)
    X     = torch.tensor(X[:val], dtype=torch.float32)
    Y     = torch.tensor(Y[:val], dtype=torch.float32)

    # Shuffle training data
    X, Y = utils.shuffle(X, Y)

    # Number of steps per epoch
    TRAINING_STEPS = len(X)     // BATCH_SIZE
    VAL_STEPS      = len(Val_X) // BATCH_SIZE

    # Define torch dataloader
    training_set   = utils.DataGenerator(X    , Y    , TRAINING_STEPS, BATCH_SIZE)
    validation_set = utils.DataGenerator(Val_X, Val_Y, VAL_STEPS     , BATCH_SIZE)

    # Define wandb training tracker
    model = SLT.Stochastic_Latent_Transformer(
        epochs         = EPOCHS         ,
        learning_rate  = LEARNING_RATE  ,
        ens_size       = ENS_SIZE       ,
        seq_len        = SEQ_LENGTH     ,
        latent_dim     = LATENT_DIM     ,
        layers         = LAYERS         ,
        width          = WIDTH          ,
        training_steps = TRAINING_STEPS ,
        val_steps      = VAL_STEPS      ,
        save_path      = SAVE_PATH      ,
    )

    # Train model
    model.fit(training_set, validation_set)

if __name__ == '__main__': train()
