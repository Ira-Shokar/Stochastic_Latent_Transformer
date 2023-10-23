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

import utils, SLT                                      # import modules
import numpy as np, matplotlib.pyplot as plt, torch    # import packages      

#seed = 0 # TODO - To reproduce results, change this to generate different ensembles

# Define Parameters
LATENT_DIM = 64 
SEQ_LENGTH = 10

AE    = SLT.load_model('AE'   , LATENT_DIM)
Trans = SLT.load_model('Trans', LATENT_DIM)

time_step_start = 100   
ensemble_size   = 8
evolution_time  = 500

truth_ens = utils.truth_ensemble(
    SEQ_LENGTH, evolution_time, ensemble_size, time_step_start
    )

truth_ens, preds_ens, att = SLT.SLT_prediction_ensemble(
    AE, Trans, SEQ_LENGTH, LATENT_DIM, truth_ens, evolution_time,
    seed, inference=False, print_time=True
    )

# Plot the ensemble
utils.plot_ensembles(truth_ens, preds_ens, SEQ_LENGTH, show=True)