using LaTeXStrings
using JLD2
using LinearAlgebra
using Statistics
using DifferentialEquations
using Random
using ArgParse
using Dates

# Function to sample initial spin states
function sampleSpinZPlus(n)
    θ = fill(acos(1 / sqrt(3)), n)
    ϕ = 2π * rand(n)
    return θ, ϕ
end

# Precompute neighbor indices for a square lattice
function get_neighbors_vectorized(nAtoms)
    matrix_size = sqrt(nAtoms) |> Int
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

# Drift function (optimized with preallocated memory)
function drift!(du, u, p, t)
    Ω, Δ, V, Γ, γ, nAtoms, neighbors, dϕ_drift_sum = p
    θ = u[1:nAtoms]
    ϕ = u[nAtoms+1:2*nAtoms]
    sqrt_3 = sqrt(3)

    fill!(dϕ_drift_sum, 0)  # Reuse preallocated array

    if case == 1
        dϕ_drift_sum[2:end-1] .= 2 .+ sqrt_3 .* (cos.(θ[1:end-2]) .+ cos.(θ[3:end]))
        dϕ_drift_sum[1] = 1 + sqrt_3 * cos(θ[2])
        dϕ_drift_sum[end] = 1 + sqrt_3 * cos(θ[end-1])
    elseif case == 2
        for n in eachindex(neighbors)
            neighbor_indices = neighbors[n]
            dϕ_drift_sum[n] = sum(1 .+ sqrt_3 * cos.(θ[neighbor_indices]))
        end
    end

    cotθ = cot.(θ)
    cscθ = csc.(θ)
    dθ_drift = -2 .* Ω .* sin.(ϕ) .+ Γ .* (cotθ .+ cscθ ./ sqrt_3)
    dϕ_drift = -2 .* Ω .* cotθ .* cos.(ϕ) .+ (V / 2) .* dϕ_drift_sum .- Δ

    du[1:nAtoms] .= dθ_drift
    du[nAtoms+1:2*nAtoms] .= dϕ_drift
end

# Diffusion function (no major changes needed)
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

# Prob function fix: Avoids new memory allocation
function prob_func(prob, i, repeat, u0)
    θ, ϕ = sampleSpinZPlus(nAtoms)
    u0[1:nAtoms] = θ
    u0[nAtoms+1:2*nAtoms] = ϕ
    return remake(prob, u0=u0)
end

# Optimized computeTWA function
function computeTWA(nAtoms, tf, nT, nTraj, Ω, Δ, V, Γ, γ)
    tspan = (0, tf)
    tSave = LinRange(0, tf, nT)
    
    # Preallocate memory
    u0 = Vector{Float64}(undef, 2 * nAtoms)
    dϕ_drift_sum = zeros(nAtoms)
    
    # Precompute neighbors
    neighbors = get_neighbors_vectorized(nAtoms)

    p = (Ω, Δ, V, Γ, γ, nAtoms, neighbors, dϕ_drift_sum)
    prob = SDEProblem(drift!, diffusion!, u0, tspan, p)
    
    ensemble_prob = EnsembleProblem(prob; prob_func=(prob, i, repeat) -> prob_func(prob, i, repeat, u0))

    sol = solve(ensemble_prob, SRIW1();
        saveat=tSave, trajectories=nTraj, maxiters=1e7,
        abstol=1e-3, reltol=1e-3, dtmax=0.0001)

    Szs = sum(sqrt(3) * cos.(sol[1:nAtoms, :, :]), dims=1)
    return tSave, Szs
end

# Parameters
Γ = 1
γ = 1e-3 * Γ
Δ = 2000 * Γ
V = Δ
nAtoms = 400
tf = 40
nT = 400
nTraj = 100
case = 2

if case == 1
    Ω_values = 0:1:40
else
    Ω_values = 0:1:20
end

script_dir = @__DIR__

task_id = parse(Int, ARGS[1])
Ω_idx = task_id + 1
Ω = Ω_values[Ω_idx]

# Create output folder safely
data_folder = joinpath(script_dir, "results_data/atoms=$(nAtoms),Δ=$(Δ),γ=$(γ)")
if !isdir(data_folder)
    mkpath(data_folder)
    println("Created directory: $data_folder")
    flush(stdout)
else
    println("Directory exists: $data_folder")
    flush(stdout)
end

# Run simulation
println("Computing for nAtoms = $nAtoms, γ = $γ, Ω = $Ω")
flush(stdout)

println("Starting TWA computation...")
flush(stdout)
t, Szs = computeTWA(nAtoms, tf, nT, nTraj, Ω, Δ, V, Γ, γ)
println("TWA computation finished.")
flush(stdout)

# Save results
filename = "$(data_folder)/sz_mean_steady_for_$(case)D,Ω=$(Ω),Δ=$(Δ),γ=$(γ).jld2"
try
    jldsave(filename; t=t, Szs=Szs)
    println("File saved successfully: $filename")
    flush(stdout)
catch e
    println("Error saving file: $e")
    flush(stdout)
end

