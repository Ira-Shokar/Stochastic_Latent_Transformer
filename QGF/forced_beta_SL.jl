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

#using Pkg
#Pkg.activate(".")

using GeophysicalFlows, CUDA, Random, Printf#, Metal
using Statistics: mean
using LinearAlgebra: ldiv!
import FourierFlows as FF

dev = GPU();     # Device (CPU/GPU)

if dev==CPU(); Random.seed!(0); else; CUDA.seed!(0); end;

              n = 256                   # 2D resolution: n² grid points
             dt = 4e-2                  # timestep
          t_max = 1000                  # integration time
  save_substeps = 2500                  # number of timesteps after which output is saved
         nsteps = t_max*save_substeps   # total number of timesteps

              L = 2π                    # domain size
              β = 0.9                   # planetary PV gradient
              μ = 4e-4                  # bottom drag
             nν = 8                     # hyperviscosity order
              ν = (n/3)^(-nν*2)         # hyperviscosity coefficent

              ε = 1e-6                  # forcing energy input rate
 
    k_f = 16.0 * 2π/L   # the forcing wavenumber, `k_f`, for a spectrum that is a ring in wavenumber space
k_width = 1.5  * 2π/L   # the width of the forcing spectrum, `δ_f`

function forcingspectrum(grid::AbstractGrid)

  K = @. sqrt(grid.Krsq)            # a 2D array with the total wavenumber

  forcing_spectrum = @. exp(-(K - k_f)^2 / (2 * k_width^2))
  @CUDA.allowscalar forcing_spectrum[grid.Krsq .== 0] .= 0 # ensure forcing has zero domain-average

  ε0 = FF.parsevalsum(forcing_spectrum .* grid.invKrsq / 2, grid) / (grid.Lx * grid.Ly)
  @. forcing_spectrum *= ε/ε0;       # normalize forcing to inject energy at rate ε
  return forcing_spectrum
end

function calcF!(Fh, sol, t, clock, vars, params, grid)
  T = eltype(grid)
  @. Fh = sqrt(forcing_spectrum) * cis(2π * rand(T)) / sqrt(clock.dt)
  return nothing
end;

forcing_spectrum = forcingspectrum(FF.TwoDGrid(dev; nx=n, Lx=L))

prob = SingleLayerQG.Problem(
    dev; nx=n,ny=n, Lx=L, Ly = L, β=β, μ=μ, ν=ν, nν = nν, dt,
    stepper="FilteredRK4", calcF=calcF!, stochastic=true, aliased_fraction = 1/3);

sol, clock, vars, params, grid = prob.sol, prob.clock, prob.vars, prob.params, prob.grid
x,  y  = grid.x,  grid.y
Lx, Ly = grid.Lx, grid.Ly;

SingleLayerQG.set_q!(prob, device_array(dev)(zeros(grid.nx, grid.ny)));

E = Diagnostic(SingleLayerQG.energy, prob; nsteps, freq=save_substeps)
Z = Diagnostic(SingleLayerQG.enstrophy, prob; nsteps, freq=save_substeps)
diags = [E, Z]; # A list of Diagnostics types passed to "stepforward!" will  be updated every timestep.

filepath = "."
plotpath = "./plots_forcedbetaturb"
plotname = "snapshots"
filename = joinpath(filepath, "singlelayerqg_forcedbeta.jld2");

if isfile(filename); rm(filename); end
if !isdir(plotpath); mkdir(plotpath); end;

get_q(prob) = Array(prob.vars.q)
get_ψ(prob) = Array(prob.vars.ψ)

function get_u(prob)
  @. prob.vars.qh = prob.sol
  MultiLayerQG2.streamfunctionfrompv!(prob.vars.ψh, prob.vars.qh, prob.params, prob.grid)
  @. prob.vars.uh = -im * prob.grid.l * prob.vars.ψh
  ldiv!(prob.vars.u, prob.params.rfftplan, prob.vars.uh)
  return Array(prob.vars.u)
end
    
output = Output(prob, filename, (:u, get_u), (:q, get_q), (:ψ, get_ψ))

saveproblem(output)
 saveoutput(output)

startwalltime = time()

while clock.step <= nsteps

  stepforward!(prob, diags, save_substeps)
  SingleLayerQG.updatevars!(prob)

  if clock.step % save_substeps == 0
    log = @sprintf("step: %04d, t: %d, E: %.3e, Q: %.3e, walltime: %.2f min",
    clock.step, clock.t, E.data[E.i], Z.data[Z.i], (time()-startwalltime)/60)
    println(log)
    saveoutput(output)
  end
end

savediagnostic(E, "energy", output.path)
savediagnostic(Z, "enstrophy", output.path)
