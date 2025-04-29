using LaTeXStrings
using JLD2
using LinearAlgebra
using Statistics
using DifferentialEquations
using Random
using ArgParse
using Dates
BLAS.set_num_threads(1)

function sampleSpinZPlus(n)
    θ = fill(acos(1 / sqrt(3)), n)
    ϕ = 2π * rand(n)
    return θ, ϕ
end

function get_neighbors_2d(nAtoms)
    matrix_size = sqrt(nAtoms) |> Int
    @assert matrix_size^2 == nAtoms "nAtoms must be a perfect square (e.g., 4, 9, 16)"
    rows = [(div(i - 1, matrix_size) + 1) for i in 1:nAtoms]
    cols = [(mod(i - 1, matrix_size) + 1) for i in 1:nAtoms]
    neighbor_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    neighbors = Vector{Vector{Int}}(undef, nAtoms)
    for i in eachindex(neighbors)
        row, col = rows[i], cols[i]
        atom_neighbors = [
            (row + dr - 1) * matrix_size + (col + dc)
            for (dr, dc) in neighbor_offsets
            if 1 <= row + dr <= matrix_size && 1 <= col + dc <= matrix_size
        ]
        neighbors[i] = atom_neighbors
    end
    return neighbors
end

function get_neighbors_3d(nAtoms)
    cube_size = cbrt(nAtoms) |> Int
    @assert cube_size^3 == nAtoms "nAtoms must be a perfect cube (e.g., 8, 27, 64)"
    xs = [(div((i - 1), cube_size^2) + 1) for i in 1:nAtoms]
    ys = [(div(mod(i - 1, cube_size^2), cube_size) + 1) for i in 1:nAtoms]
    zs = [(mod(i - 1, cube_size) + 1) for i in 1:nAtoms]
    neighbor_offsets = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
    neighbors = Vector{Vector{Int}}(undef, nAtoms)
    for i in eachindex(neighbors)
        x, y, z = xs[i], ys[i], zs[i]
        atom_neighbors = [
            (x + dx - 1) * cube_size^2 + (y + dy - 1) * cube_size + (z + dz)
            for (dx, dy, dz) in neighbor_offsets
            if 1 <= x + dx <= cube_size && 1 <= y + dy <= cube_size && 1 <= z + dz <= cube_size
        ]
        neighbors[i] = atom_neighbors
    end

    return neighbors
end

function drift!(du, u, p, t)
    Ω, Δ, V, Γ, γ, nAtoms, neighbors, dϕ_drift_sum = p
    θ = u[1:nAtoms]
    ϕ = u[nAtoms+1:2*nAtoms]
    sqrt_3 = sqrt(3)
    fill!(dϕ_drift_sum, 0)
    if case == 1
        dϕ_drift_sum[2:end-1] .= 2 .+ sqrt_3 .* (cos.(θ[1:end-2]) .+ cos.(θ[3:end]))
        dϕ_drift_sum[1] = 1 + sqrt_3 * cos(θ[2])
        dϕ_drift_sum[end] = 1 + sqrt_3 * cos(θ[end-1])
    else
        for n in eachindex(neighbors)
            neighbor_indices = neighbors[n]
            dϕ_drift_sum[n] = sum(1 .+ sqrt_3 * cos.(θ[neighbor_indices]))
        end
    end
    cotθ = cot.(θ)
    cscθ = csc.(θ)
    dθ_drift = 2 .* Ω .* sin.(ϕ) .+ Γ .* (cotθ .+ cscθ ./ sqrt_3)
    dϕ_drift = 2 .* Ω .* cotθ .* cos.(ϕ) .- V .* dϕ_drift_sum .+ Δ
    du[1:nAtoms] .= dθ_drift
    du[nAtoms+1:2*nAtoms] .= dϕ_drift
end

function diffusion!(du, u, p, t)
    Ω, Δ, V, Γ, γ = p
    θ = u[1:nAtoms]
    sqrt_3 = sqrt(3)
    term1 = 1
    term2 = 2 .* cot.(θ) .^ 2
    term3 = 2 .* cot.(θ) .* csc.(θ) ./ sqrt_3
    diffusion = sqrt.(Γ .* (term1 .+ term2 .+ term3) .+ 4 .* γ)
    du[1:nAtoms] .= 0.0
    du[nAtoms+1:2*nAtoms] .= diffusion
end

function computeTWA(nAtoms, tf, nT, nTraj, Ω, Δ, V, Γ, γ, case)
    tspan = (0, tf)
    tSave = LinRange(0, tf, nT)
    u0 = Vector{Float64}(undef, 2 * nAtoms)
    dϕ_drift_sum = zeros(nAtoms)
    
    neighbors = case == 2 ? get_neighbors_2d(nAtoms) : case == 3 ? get_neighbors_3d(nAtoms) : nothing
    p = (Ω, Δ, V, Γ, γ, nAtoms, neighbors, dϕ_drift_sum)
    
    prob = SDEProblem(drift!, diffusion!, u0, tspan, p)
    ensemble_prob = EnsembleProblem(prob; prob_func=(prob, i, repeat) -> begin
        θ, ϕ = sampleSpinZPlus(nAtoms)
        u0[1:nAtoms] = θ
        u0[nAtoms+1:2*nAtoms] = ϕ
        remake(prob, u0=u0)
    end)
    
    sol = solve(ensemble_prob, SRIW1(), EnsembleThreads(); 
                saveat=tSave, 
                trajectories=nTraj, 
                maxiters=1e7, 
                abstol=1e-3,
                reltol=1e-3, 
                dtmax=0.0001)
    
    sol_array = zeros(2 * nAtoms, nT, nTraj)
    for i in 1:nTraj
        for (j, u) in enumerate(sol[i].u)
            sol_array[:, j, i] = u
        end
    end
            
    return tSave, sol_array    
end

Γ = 1
Δ = 2000 * Γ
V = Δ
nAtoms = 400
tf = 160
nT = 400
nTraj = 16
case = 2

Ω_values = round.(vcat(8.5:0.15:13, 14:0.5:30), digits=2)
#vcat(0:2:16, 17:0.25:32, 33:1:60) 
#13.5:0.5:30 #vcat(0:2:10, 10.5:0.025:13, 14:2:30)
γ_values = [20]

omega_gamma_pairs = vec([[Ω, γ] for Ω in Ω_values, γ in γ_values])

script_dir = @__DIR__
omega_gamma_idx = parse(Int, ARGS[1])
Ω, γ = omega_gamma_pairs[omega_gamma_idx + 1]

println("Computing for nAtoms = $nAtoms, γ = $γ, Ω = $Ω")
flush(stdout)

data_folder = joinpath(script_dir, "results_data", "atoms=$(nAtoms),Δ=$(Δ),γ=$(γ)")
if !isdir(data_folder)
    mkpath(data_folder)
    println("Created directory: $data_folder")
else
    println("Directory exists: $data_folder")
end
flush(stdout)

println("Starting TWA computation for γ = $γ...")
flush(stdout)
t, sol_array = computeTWA(nAtoms, tf, nT, nTraj, Ω, Δ, V, Γ, γ, case)
println("TWA computation finished for γ = $γ.")
flush(stdout)

sol_filename = "$(data_folder)/temp_sol_$(case)D,Ω=$(Ω),Δ=$(Δ),γ=$(γ).jld2"
jldsave(sol_filename; t=t, sol=sol_array)
println("Solution saved temporarily: $sol_filename")
flush(stdout)

julia_path = joinpath(homedir(), "julia-1.11.2", "bin", "julia")
compute_sz_script = joinpath(script_dir, "compute_sz.jl")
cmd = `$julia_path $compute_sz_script $sol_filename $nAtoms $nTraj`
println("Executing command: $cmd")
flush(stdout)

run(cmd; wait=true)
println("Sz computation completed for γ = $γ.")
flush(stdout)

println("Computation completed for Ω = $Ω, γ = $γ.")
flush(stdout)

sleep(2)
