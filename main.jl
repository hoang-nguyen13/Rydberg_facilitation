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
            dϕ_drift_sum[n] = sum(1 .- sqrt_3 * cos.(θ[neighbor_indices]))
        end
    end
    cotθ = cot.(θ)
    cscθ = csc.(θ)
    dθ_drift = 2 .* Ω .* sin.(ϕ) .+ Γ .* (cotθ .+ cscθ ./ sqrt_3)
    dϕ_drift = 2 .* Ω .* cotθ .* cos.(ϕ) .+ (V / 2) .* dϕ_drift_sum .- Δ
    du[1:nAtoms] .= dθ_drift
    du[nAtoms+1:2*nAtoms] .= dϕ_drift
end

function diffusion!(du, u, p, t)
    Ω, Δ, V, Γ, γ, nAtoms, neighbors, dϕ_drift_sum = p
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
    sol = solve(ensemble_prob, SRIW1(); saveat=tSave, trajectories=nTraj, maxiters=1e7, abstol=1e-3, reltol=1e-3, dtmax=0.0001)
    
    sol_array = zeros(2 * nAtoms, nT, nTraj)
    for i in 1:nTraj
        for (j, u) in enumerate(sol[i].u)
            sol_array[:, j, i] = u
        end
    end
    
    return tSave, sol_array
end

# Parameters
Γ = 1
Δ = 2000 * Γ
V = Δ
nAtoms = 400
tf = 70
nT = 400
nTraj = 5
case = 2

if case == 1
    Ω_values = 0:1:40
else
    Ω_values = vcat(0:1:15, 15.25:0.15:22.45, 25:1:30)
end

γ_values = [1e-3, 0.1, 10, 100]

script_dir = @__DIR__

task_id = parse(Int, ARGS[1])
Ω_idx = task_id + 1
Ω = Ω_values[Ω_idx]

julia_path = joinpath(homedir(), "julia-1.11.2", "bin", "julia")

checkpoint_file = joinpath(script_dir, "checkpoints/checkpoint_Ω=$(Ω).jld2")
if !isdir(dirname(checkpoint_file))
    mkpath(dirname(checkpoint_file))
end

completed_γ = Float64[]
if isfile(checkpoint_file)
    jldopen(checkpoint_file, "r") do file
        completed_γ = file["completed_γ"]
    end
    println("Loaded completed γ values for Ω = $Ω: $completed_γ")
    flush(stdout)
end

for γ in γ_values
    if γ in completed_γ
        println("Skipping γ = $γ for Ω = $Ω (already completed)")
        flush(stdout)
        continue
    end

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

    max_retries = 2
    attempt = 1
    success = false
    t = nothing
    sol_array = nothing
    sol_filename = nothing

    while attempt <= max_retries && !success
        println("Attempt $attempt of $max_retries for γ = $γ, Ω = $Ω...")
        flush(stdout)

        try
            println("Starting TWA computation for γ = $γ...")
            flush(stdout)
            t, sol_array = computeTWA(nAtoms, tf, nT, nTraj, Ω, Δ, V, Γ, γ)
            println("TWA computation finished for γ = $γ.")
            flush(stdout)

            sol_filename = "$(data_folder)/temp_sol_$(case)D,Ω=$(Ω),Δ=$(Δ),γ=$(γ).jld2"
            jldsave(sol_filename; t=t, sol=sol_array)
            println("Solution saved temporarily: $sol_filename")
            flush(stdout)

            compute_sz_script = joinpath(script_dir, "compute_sz.jl")
            if !isfile(compute_sz_script)
                println("Error: compute_sz.jl not found at $compute_sz_script")
                flush(stdout)
                error("compute_sz.jl missing")
            end
            cmd = `$julia_path $compute_sz_script $sol_filename $nAtoms $nTraj`
            println("Executing command: $cmd")
            flush(stdout)
            
            process = run(cmd; wait=true)
            if process.exitcode != 0
                error("compute_sz.jl failed with exit code $(process.exitcode)")
            end
            println("Sz computation completed for γ = $γ.")
            flush(stdout)

            push!(completed_γ, γ)
            jldsave(checkpoint_file; completed_γ)
            println("Checkpoint updated: γ = $γ completed for Ω = $Ω")
            flush(stdout)

            success = true

        catch e
            println("Error in attempt $attempt for γ = $γ, Ω = $Ω: $e")
            flush(stdout)
            attempt += 1
            if attempt <= max_retries
                println("Retrying after cleanup...")
                flush(stdout)
                t = nothing
                sol_array = nothing
                if isfile(sol_filename)
                    try
                        rm(sol_filename)
                        println("Removed potentially corrupted file: $sol_filename")
                    catch rm_err
                        println("Error removing file: $rm_err")
                    end
                end
                GC.gc(true)
                sleep(5)
            else
                println("Max retries reached for γ = $γ, Ω = $Ω. Skipping this γ.")
                flush(stdout)
                t = nothing
                sol_array = nothing
                if isfile(sol_filename)
                    try
                        rm(sol_filename)
                    catch rm_err
                        println("Error removing file: $rm_err")
                    end
                end
                GC.gc(true)
                sleep(1)
                break
            end
        end
    end

    println("Cleaning up for γ = $γ...")
    flush(stdout)
    t = nothing
    sol_array = nothing
    GC.gc()
    sleep(1)
    println("Cleanup complete for γ = $γ.")
    flush(stdout)
end

println("All γ values processed for Ω = $Ω.")
flush(stdout)
