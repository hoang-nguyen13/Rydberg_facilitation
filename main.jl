using LaTeXStrings
using JLD2
using LinearAlgebra
using Statistics
using DifferentialEquations
using Random
using ArgParse
using Dates

function sampleSpinZPlus(n)
    θ = fill(acos(1 / sqrt(3)), n)
    ϕ = 2π * rand(n)
    return θ, ϕ
end

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

function prob_func(prob, i, repeat, u0)
    θ, ϕ = sampleSpinZPlus(nAtoms)
    u0[1:nAtoms] = θ
    u0[nAtoms+1:2*nAtoms] = ϕ
    return remake(prob, u0=u0)
end

function computeTWA(nAtoms, tf, nT, nTraj, Ω, Δ, V, Γ, γ)
    tspan = (0, tf)
    tSave = LinRange(0, tf, nT)
    u0 = Vector{Float64}(undef, 2 * nAtoms)
    dϕ_drift_sum = zeros(nAtoms)
    neighbors = get_neighbors_vectorized(nAtoms)
    p = (Ω, Δ, V, Γ, γ, nAtoms, neighbors, dϕ_drift_sum)
    prob = SDEProblem(drift!, diffusion!, u0, tspan, p)
    ensemble_prob = EnsembleProblem(prob; prob_func=(prob, i, repeat) -> prob_func(prob, i, repeat, u0))
    sol = solve(ensemble_prob, SRIW1(); saveat=tSave, trajectories=nTraj, maxiters=1e6, abstol=1e-3, reltol=1e-3, dtmax=0.1)

    # Convert EnsembleSolution to 3D array: [2*nAtoms, nT, nTraj]
    sol_array = zeros(2 * nAtoms, nT, nTraj)
    for i in 1:nTraj
        # Each sol[i].u is a Vector{Vector{Float64}}, where each inner vector is 2*nAtoms long
        for (j, u) in enumerate(sol[i].u)
            sol_array[:, j, i] = u
        end
    end

    println("Converted sol to array with dimensions: $(size(sol_array))")
    return tSave, sol_array  # Return the 3D array instead of EnsembleSolution
end

# Parameters
Γ = 1
Δ = 2000 * Γ
V = Δ
nAtoms = 16
tf = 60
nT = 400
nTraj = 1
case = 2

if case == 1
    Ω_values = 0:1:40
else
    Ω_values = vcat(0:1:15, 15.25:0.15:22.45, 25:1:30)
end

γ_values = [1e-3]

script_dir = @__DIR__

task_id = parse(Int, ARGS[1])
Ω_idx = task_id + 1
Ω = Ω_values[Ω_idx]

julia_path = joinpath(homedir(), "julia-1.11.2", "bin", "julia")

# Loop over γ values
for γ in γ_values
    data_folder = joinpath(script_dir, "results_data/atoms=$(nAtoms),Δ=$(Δ),γ=$(γ)")
    if !isdir(data_folder)
        mkpath(data_folder)
        println("Created directory: $data_folder")
        flush(stdout)
    else
        println("Directory exists: $data_folder")
        flush(stdout)
    end

    println("Computing for nAtoms = $nAtoms, γ = $γ, Ω = $Ω")
    flush(stdout)

    println("Starting TWA computation for γ = $γ...")
    flush(stdout)
    t, sol_array = computeTWA(nAtoms, tf, nT, nTraj, Ω, Δ, V, Γ, γ)
    println("TWA computation finished for γ = $γ.")
    flush(stdout)

    println("sol_array dimensions: $(size(sol_array))")
    flush(stdout)

    # Save sol as the 3D array
    sol_filename = "$(data_folder)/temp_sol_$(case)D,Ω=$(Ω),Δ=$(Δ),γ=$(γ).jld2"
    try
        jldsave(sol_filename; t=t, sol=sol_array)
        println("Solution saved temporarily: $sol_filename")
        flush(stdout)
    catch e
        println("Error saving solution: $e")
        flush(stdout)
    end

    # Call compute_sz.jl with nAtoms and nTraj
    compute_sz_script = joinpath(script_dir, "compute_sz.jl")
    if isfile(compute_sz_script)
        cmd = `$julia_path $compute_sz_script $sol_filename $nAtoms $nTraj`
        println("Executing command: $cmd")
        flush(stdout)
        try
            run(cmd)
            println("Sz computation completed for γ = $γ.")
            flush(stdout)
        catch e
            println("Error running compute_sz.jl: $e")
            flush(stdout)
        end
    else
        println("Error: compute_sz.jl not found at $compute_sz_script")
        flush(stdout)
    end

    # Clean up
    println("Cleaning up for γ = $γ...")
    flush(stdout)
end

println("All γ values processed for Ω = $Ω.")
flush(stdout)
