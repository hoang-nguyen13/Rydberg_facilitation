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
    θ = fill(acos(1 / sqrt(3)), n)  # Fill θ with the same value for simplicity
    ϕ = 2π * rand(n)  # Random ϕ values between 0 and 2π
    return θ, ϕ
end

# Precompute neighbor indices for a square lattice
# This is an optimization. We now compute all neighbors once and reuse them.
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
    # Unpack parameters
    Ω, Δ, V, Γ, γ, nAtoms, neighbors, dϕ_drift_sum = p
    θ = u[1:nAtoms]
    ϕ = u[nAtoms+1:2*nAtoms]
    sqrt_3 = sqrt(3)

    fill!(dϕ_drift_sum, 0)  # Reuse the preallocated array, avoid re-allocating memory

    if case == 1
        # Drift calculation for case 1 (assuming some boundary conditions)
        dϕ_drift_sum[2:end-1] .= 2 .+ sqrt_3 .* (cos.(θ[1:end-2]) .+ cos.(θ[3:end]))
        dϕ_drift_sum[1] = 1 + sqrt_3 * cos(θ[2])
        dϕ_drift_sum[end] = 1 + sqrt_3 * cos(θ[end-1])
    elseif case == 2
        # Drift calculation for case 2 (using neighbor interactions)
        for n in eachindex(neighbors)
            neighbor_indices = neighbors[n]
            dϕ_drift_sum[n] = sum(1 .+ sqrt_3 * cos.(θ[neighbor_indices]))
        end
    end

    # Calculate derivatives for θ and ϕ
    cotθ = cot.(θ)
    cscθ = csc.(θ)
    dθ_drift = -2 .* Ω .* sin.(ϕ) .+ Γ .* (cotθ .+ cscθ ./ sqrt_3)
    dϕ_drift = -2 .* Ω .* cotθ .* cos.(ϕ) .+ (V / 2) .* dϕ_drift_sum .- Δ

    # Assign the results to the output vector `du`
    du[1:nAtoms] .= dθ_drift
    du[nAtoms+1:2*nAtoms] .= dϕ_drift
end

# Diffusion function (no major changes needed, just using the same efficient approach)
function diffusion!(du, u, p, t)
    Ω, Δ, V, Γ, γ = p
    θ = u[1:nAtoms]
    sqrt_3 = sqrt(3)

    term1 = 1
    term2 = 2 .* cot.(θ) .^ 2
    term3 = 2 .* cot.(θ) .* csc.(θ) ./ sqrt_3
    diffusion = sqrt.(Γ .* (term1 .+ term2 .+ term3) .+ 4 .* γ)

    # Diffusion part only affects the ϕ values (second half of u)
    du[1:nAtoms] .= 0.0  # No diffusion in θ
    du[nAtoms+1:2*nAtoms] .= diffusion  # Apply diffusion to ϕ
end

# Prob function fix: Avoids new memory allocation each time it's called
# Reuse `u0` from the calling function to prevent allocating memory every time.
function prob_func(prob, i, repeat, u0)
    θ, ϕ = sampleSpinZPlus(nAtoms)
    u0[1:nAtoms] = θ  # Update θ values in the pre-allocated array
    u0[nAtoms+1:2*nAtoms] = ϕ  # Update ϕ values in the pre-allocated array
    return remake(prob, u0=u0)  # Return a new problem with the updated `u0`
end

# Define a function to calculate Sz at each time step
function calculate_Sz_incrementally(sol, nAtoms, tSave, nTraj)
    Szs = zeros(size(tSave))  # Preallocate Szs to store results
    for t_idx in 1:length(tSave)
        # For each time step, sum the cosines of the θ angles over all atoms and trajectories
        Szs[t_idx] = sum(sqrt(3) * cos.(sol[1:nAtoms, t_idx, :]))
        GC.gc()
    end
    return Szs
end


# Optimized computeTWA function
# Main fix here: Precompute `neighbors` outside of loops, and pass preallocated memory for better efficiency.
function computeTWA(nAtoms, tf, nT, nTraj, Ω, Δ, V, Γ, γ)
    tspan = (0, tf)
    tSave = LinRange(0, tf, nT)
    
    # Preallocate memory
    u0 = Vector{Float64}(undef, 2 * nAtoms)  # Initial state vector
    dϕ_drift_sum = zeros(nAtoms)  # Preallocate the array used in drift calculation
    
    # Precompute neighbors for case 2
    neighbors = get_neighbors_vectorized(nAtoms)

    # Pack parameters together (passing precomputed neighbors and drift sum array)
    p = (Ω, Δ, V, Γ, γ, nAtoms, neighbors, dϕ_drift_sum)
    prob = SDEProblem(drift!, diffusion!, u0, tspan, p)
    
    # Setup ensemble problem (reuse prob_func without memory allocation overhead)
    ensemble_prob = EnsembleProblem(prob; prob_func=(prob, i, repeat) -> prob_func(prob, i, repeat, u0))

    # Solve the system with an ensemble solver
    sol = solve(ensemble_prob, SRIW1();
    saveat=tSave, trajectories=nTraj, maxiters=1e7,
    abstol=1e-3, reltol=1e-3, dtmax=0.0001)

    # Calculate observable Szs (sum of cosines of the θ angles)
    Szs = calculate_Sz_incrementally(sol, nAtoms, tSave, nTraj)
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
nTraj = 50
case = 2

if case == 1
    Ω_values = 0:1:40  # For case 1, Ω can range from 0 to 40
else
    Ω_values = 0:1:20  # For case 2, Ω can range from 0 to 20
end

script_dir = @__DIR__

task_id = parse(Int, ARGS[1])
Ω_idx = task_id + 1  # Use the task_id to index Ω values
Ω = Ω_values[Ω_idx]  # Get the corresponding Ω value

# Safely create data folder (this avoids issues in case the folder already exists)
data_folder = joinpath(script_dir, "results_data/atoms=$(nAtoms),Δ=$(Δ),γ=$(γ)")
if !isdir(data_folder)
    mkpath(data_folder)  # mkpath ensures it creates any missing intermediate directories
    println("Created directory: $data_folder")
    flush(stdout)
else
    println("Directory exists: $data_folder")
    flush(stdout)
end

# Print and start computation
println("Computing for nAtoms = $nAtoms, γ = $γ, Ω = $Ω")
flush(stdout)

println("Starting TWA computation...")
flush(stdout)
t, Szs = computeTWA(nAtoms, tf, nT, nTraj, Ω, Δ, V, Γ, γ)
println("TWA computation finished.")
flush(stdout)

# Save results to a file (handle errors in file saving)
filename = "$(data_folder)/sz_mean_steady_for_$(case)D,Ω=$(Ω),Δ=$(Δ),γ=$(γ).jld2"
try
    jldsave(filename; t=t, Szs=Szs)
    println("File saved successfully: $filename")
    flush(stdout)
catch e
    println("Error saving file: $e")
    flush(stdout)
end

