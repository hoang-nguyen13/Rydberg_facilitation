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
    dθ_drift = 2 .* Ω .* sin.(ϕ) .+ Γ .* (cotθ .+ cscθ ./ sqrt_3)
    dϕ_drift = 2 .* Ω .* cotθ .* cos.(ϕ) .- (V / 2) .* dϕ_drift_sum .+ Δ
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
    sol = solve(ensemble_prob, SRIW1(); saveat=tSave, trajectories=nTraj, maxiters=1e7, abstol=1e-3, reltol=1e-3, dtmax=0.0001)
    
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
nTraj = 1
case = 2

if case == 1
    Ω_values = 0:1:40
else
    Ω_values = vcat(0:4:17, 17.5:0.05:19, 20:2:30, 40:10:60)
end

γ_values = [0.1]

script_dir = @__DIR__
task_id = parse(Int, ARGS[1])
Ω_idx = task_id + 1
Ω = Ω_values[Ω_idx]

checkpoint_file = joinpath(script_dir, "checkpoints", "checkpoint_Ω=$(Ω).jld2")
if !isdir(dirname(checkpoint_file))
    mkpath(dirname(checkpoint_file))
end

# Load completed γ values
completed_γ = isfile(checkpoint_file) ? jldopen(checkpoint_file, "r")["completed_γ"] : Float64[]
println("Loaded completed γ values for Ω = $Ω: $completed_γ")
flush(stdout)

# Find incomplete γ values
incomplete_γ = filter(γ -> γ ∉ completed_γ, γ_values)
println("Incomplete γ values to process for Ω = $Ω: $incomplete_γ")
flush(stdout)

if isempty(incomplete_γ)
    println("All γ values completed for Ω = $Ω. Exiting.")
    flush(stdout)
    exit(0)
end

julia_path = joinpath(homedir(), "julia-1.11.2", "bin", "julia")

# Process each incomplete γ with retry logic
for γ in incomplete_γ
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
                println("Retrying γ = $γ after 5 seconds...")
                flush(stdout)
                sleep(5)
                t = nothing
                sol_array = nothing
                GC.gc()
            else
                println("Max retries reached for γ = $γ, Ω = $Ω. Signaling retry to shell.")
                flush(stdout)
                exit(2)  # Special exit code to signal retry
            end
        finally
            println("Cleaning up for γ = $γ...")
            flush(stdout)
            t = nothing
            sol_array = nothing
            GC.gc()
            sleep(1)
            println("Cleanup complete for γ = $γ.")
            flush(stdout)
        end
    end
end

println("All incomplete γ values processed for Ω = $Ω.")
flush(stdout)
