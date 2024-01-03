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

# Import necessary packages
using GeophysicalFlows, CUDA, Random #, Metal
using LinearAlgebra: ldiv!

"""
    forcingspectrum(ε, k_f, k_width, grid::AbstractGrid)

Generate the forcing spectrum based on the given parameters.

# Arguments
- `ε`: Forcing energy input rate.
- `k_f`: The forcing wavenumber for a spectrum that is a ring in wavenumber space.
- `k_width`: The width of the forcing spectrum.
- `grid`: AbstractGrid representing the simulation grid.

# Returns
A 2D array representing the forcing spectrum.
"""
function forcingspectrum(ε, k_f, k_width, grid::AbstractGrid)
    K = @. sqrt(grid.Krsq)            # a 2D array with the total wavenumber
    forcing_spectrum = @. exp(-(K - k_f)^2 / (2 * k_width^2))
    @CUDA.allowscalar forcing_spectrum[grid.Krsq .== 0] .= 0  # ensure forcing has zero domain-average

    ε0 = FourierFlows.parsevalsum(forcing_spectrum .* grid.invKrsq / 2, grid) / (grid.Lx * grid.Ly)
    @. forcing_spectrum *= ε / ε0       # normalize forcing to inject energy at rate ε
    return forcing_spectrum
end

"""
    calcF!(Fh, sol, t, clock, vars, params, grid)

Calculate the forcing term for the simulation.

# Arguments
- `Fh`: Output array for the forcing term.
- `sol`: Solution array.
- `t`: Current time.
- `clock`: Clock object.
- `vars`: Variables object.
- `params`: Parameters object.
- `grid`: AbstractGrid representing the simulation grid.

# Returns
Nothing
"""
function calcF!(Fh, sol, t, clock, vars, params, grid)
    random_uniform = CUDA.functional() ? CUDA.rand : rand
    T = eltype(grid)
    @CUDA.allowscalar d = random_uniform(T)  # to play nicely with CUDA 10.2
    @. vars.Fh = sqrt(params.spectrum) * cis(2π * d) / sqrt(clock.dt)
    return nothing
end;

"""
    get_q(prob)

Extract vorticity (q) field from the simulation problem.

# Arguments
- `prob`: Simulation problem.

# Returns
A 2D array representing the vorticity (q) field.
"""
get_q(prob) = Array(prob.vars.q)

"""
    get_ψ(prob)

Extract streamfunction (ψ) field from the simulation problem.

# Arguments
- `prob`: Simulation problem.

# Returns
A 2D array representing the streamfunction (ψ) field.
"""
get_ψ(prob) = Array(prob.vars.ψ)

"""
    get_u(prob)
Extract velocity field (u) from the simulation

"""

function get_u(prob)
    @. prob.vars.qh = prob.sol
    MultiLayerQG.streamfunctionfrompv!(prob.vars.ψh, prob.vars.qh, prob.params, prob.grid)
    @. prob.vars.uh = -im * prob.grid.l * prob.vars.ψh
    ldiv!(prob.vars.u, prob.params.rfftplan, prob.vars.uh)
    return Array(prob.vars.u)
  end
