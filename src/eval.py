# coding=utf-8
# Copyright (c) 2023 Ira Shokar
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import utils, SLT
# import packages
import numpy as np, matplotlib.pyplot as plt, torch, scipy

seed = 0 # To reproduce results, change this to generate different ensembles

# Hyperparameters
BATCH_SIZE    = 200;
EPOCHS        = 200;
LEARNING_RATE = 2e-3;
LATENT_DIM    = 64; 
SEQ_LENGTH    = 10;
SEQ_FORWARD   = 1;
ENS_SIZE      = 4;
S_WEIGHT      = 0;
RUN_NUM       = 6; 
LAYERS        = 2;
SWEEP         = False;
EVOLUTION_TIME = 500


time_step_start = 100   
ensemble_size   = 8
truth_ens       = utils.truth_ensemble(SEQ_LENGTH, EVOLUTION_TIME, ensemble_size, time_step_start)

AE                        = SLT.load_model('AE'   , LATENT_DIM, 0, EPOCHS, RUN_NUM)
RNN                       = SLT.load_model('Trans', LATENT_DIM, 0, EPOCHS, RUN_NUM)
truth_ens, preds_ens, att = SLT.SLT_prediction_ensemble(
    AE, RNN, SEQ_LENGTH, LATENT_DIM, truth_ens, EVOLUTION_TIME,
     seed, inference=False, print_time=True
    )