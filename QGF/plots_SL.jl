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

#using Pkg
#Pkg.activate(".")

using JLD2, Plots
using Statistics: mean

file = jldopen("singlelayerqg_forcedbeta.jld2")

iterations = parse.(Int, keys(file["snapshots/t"]))
u  = [file["snapshots/u/$i"] for i ∈ iterations]
q  = [file["snapshots/q/$i"] for i ∈ iterations]
ψ  = [file["snapshots/ψ/$i"] for i ∈ iterations]

close(file)

t = length(u)
n = 256
u_bar = zeros(t,n)
for j in 1:t
    u_bar[j, :] = mean(u[j][:, :, 1], dims=1)
end

heatmap(transpose(u_bar), c= :viridis)
savefig("umean_ML_1.png")

heatmap(transpose(u[end][:, :, 1]), c= :viridis)
savefig("u1_ML.png")

heatmap(transpose(q[end][:, :, 1]), c= :viridis)
savefig("q1_ML.png")

heatmap(transpose(ψ[end][:, :, 1]), c= :viridis)
savefig("psi1_M.png")
