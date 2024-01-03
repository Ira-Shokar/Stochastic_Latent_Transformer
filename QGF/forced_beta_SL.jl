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

# Based on the GeophysicalFlows example

# Import necessary packages
using GeophysicalFlows, CUDA, Random, Printf #, Metal
using Statistics: mean
using LinearAlgebra: ldiv!
import FourierFlows as FF

# Import helper functions
include("utils.jl")

# Define simulation parameters
            dev = GPU();                # Device (CPU/GPU)
              L = 2π                    # domain size
              n = 256                   # 2D resolution: n² grid points

              β = 0.9                   # planetary PV gradient
              μ = 4e-4                  # bottom drag

             nν = 8                     # hyperviscosity order
              ν = (n/3)^(-nν*2)         # hyperviscosity coefficent

             dt = 4e-2                  # timestep
          t_max = 500                   # integration time
  save_substeps = 2500                  # number of timesteps after which output is saved
         nsteps = t_max*save_substeps   # total number of timesteps

      ε = 1e-6          # forcing energy input rate
    k_f = 16.0 * 2π/L   # the forcing wavenumber, `k_f`, for a spectrum that is a ring in wavenumber space
k_width = 1.5  * 2π/L   # the width of the forcing spectrum, `δ_f`

# Seed random number generator based on the device
if dev==CPU(); Random.seed!(0); else; CUDA.seed!(0); end;

# Generate forcing spectrum
forcing_spectrum = forcingspectrum(ε, k_f, k_width, FF.TwoDGrid(dev; nx=n, Lx=L))

# Define the simulation problem
prob = SingleLayerQG.Problem(
    dev; nx=n, ny=n, Lx=L, Ly=L, β=β, μ=μ, ν=ν, nν=nν, dt,
    stepper="FilteredRK4", calcF=calcF!, stochastic=true, aliased_fraction=1/3
)

# Initialise vorticity (q) field from rest
SingleLayerQG.set_q!(prob, device_array(dev)(zeros(prob.grid.nx, prob.grid.ny)))

# Define diagnostics for energy and enstrophy
E = Diagnostic(SingleLayerQG.energy, prob; nsteps, freq=save_substeps)
Z = Diagnostic(SingleLayerQG.enstrophy, prob; nsteps, freq=save_substeps)
diags = [E, Z];  # A list of Diagnostics types passed to "stepforward!" will be updated every timestep.

# Define output filename and remove file if it already exists
filename = "singlelayerqg_forcedbeta.jld2"
if isfile(filename); rm(filename); end

# Create output object
output = Output(prob, filename, (:u, get_u), (:q, get_q), (:ψ, get_ψ))

# Save problem and initial output
saveproblem(output)
saveoutput(output)

# Start simulation
startwalltime = time()

while prob.clock.step <= nsteps
    # step forward simulation and update variables
    stepforward!(prob, diags, save_substeps)
    SingleLayerQG.updatevars!(prob)

    # Print and save output at specified intervals
    if prob.clock.step % save_substeps == 0
        log = @sprintf("step: %04d, t: %d, E: %.3e, Q: %.3e, walltime: %.2f min",
                       prob.clock.step, prob.clock.t, E.data[E.i], Z.data[Z.i], (time() - startwalltime) / 60)
        println(log)
        saveoutput(output)
    end
end

# Save final diagnostics
savediagnostic(E, "energy", output.path)
savediagnostic(Z, "enstrophy", output.path)
