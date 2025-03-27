using LaTeXStrings
using JLD2
using LinearAlgebra
using Statistics
using DifferentialEquations
using Random
using ArgParse
using ThreadsX
BLAS.set_num_threads(1)

function sampleSpinZPlus(n)
    θ = fill(acos(1 / sqrt(3)), n)
    ϕ = 2π * rand(n)                  
    return θ, ϕ
end

function sampleSpinZMinus(n)
    θ = fill(π - acos(1 / sqrt(3)), n)   
    ϕ = 2π * rand(n)                  
    return θ, ϕ
end

function prob_func(prob, i, repeat)
    u0 = Vector{Float64}(undef, 2 * nAtoms)
    θ, ϕ = sampleSpinZPlus(nAtoms)
    u0[1:nAtoms] = θ
    u0[nAtoms+1:2*nAtoms] = ϕ
    return remake(prob, u0=u0)
end

function get_neighbors_vectorized(nAtoms)
    matrix_size = sqrt(nAtoms) |> Int
    rows = [(div(i - 1, matrix_size) + 1) for i in 1:nAtoms]
    cols = [(mod(i - 1, matrix_size) + 1) for i in 1:nAtoms]
    neighbor_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    neighbors = Vector{Vector{Int}}(undef, nAtoms)
    for i in 1:nAtoms
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
    Ω, Δ, V, Γ, γ = p
    θ = u[1:nAtoms]
    ϕ = u[nAtoms+1:2*nAtoms]
    sqrt_3 = sqrt(3)
    dϕ_drift_sum = zeros(nAtoms)
    if case == 1
        dϕ_drift_sum[2:end-1] .= 2 .+ sqrt_3 .* (cos.(θ[1:end-2]) .+ cos.(θ[3:end]))
        dϕ_drift_sum[1] = 1 + sqrt_3 * cos(θ[2]) 
        dϕ_drift_sum[end] = 1 + sqrt_3 * cos(θ[end-1])
    elseif case == 2
        neighbors = get_neighbors_vectorized(nAtoms)
        for n in 1:nAtoms
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

# function diffusion!(du, u, p, t)
#     Ω, Δ, V, Γ, γ = p
#     θ = u[1:nAtoms]
#     sqrt_3 = sqrt(3)
#     term1 = 9 / 6
#     term2 = (4 * sqrt_3 / 6) .* cos.(θ)
#     term3 = (3 / 6) .* cos.(2 .* θ)
#     cscθ2 = csc.(θ) .^ 2
#     diffusion = sqrt.(Γ .* (term1 .+ term2 .+ term3) .* cscθ2 .+ 4 .* γ)
#     du[1:nAtoms] .= 0.0
#     du[nAtoms+1:2*nAtoms] .= diffusion
# end

function diffusion!(du, u, p, t)
    Ω, Δ, V, Γ, γ = p
    θ = u[1:nAtoms]
    sqrt_3 = sqrt(3)
    term1 = 1
    term2 = 2 .* cot.(θ) .^ 2
    term3 = 2 .* cot.(θ) .* csc.(θ) ./ sqrt_3
    # cscθ2 = csc.(θ) .^ 2
    diffusion = NaNMath.sqrt.(Γ .* (term1 .+ term2 .+ term3) .+ 4 .* γ)
    du[1:nAtoms] .= 0.0
    du[nAtoms+1:2*nAtoms] .= diffusion
end

function computeTWA(nAtoms, tf, nT, nTraj, dt, Ω, Δ, V, Γ, γ)
    tspan = (0, tf)
    tSave = LinRange(0, tf, nT)
    u0 = Vector{Float64}(undef, 2 * nAtoms)
    p = (Ω, Δ, V, Γ, γ, nAtoms)

    prob = SDEProblem(drift!, diffusion!, u0, tspan, p)
    ensemble_prob = EnsembleProblem(prob; prob_func=prob_func)
    
    sol = solve(ensemble_prob, SOSRI2(), EnsembleThreads();
        saveat=tSave, trajectories=nTraj, maxiters=1e+7,
        abstol=1e-3, reltol=1e-2)
    
    Szs = sum(sqrt(3) * cos.(sol[1:nAtoms, :, :]), dims=1)  # Only compute Szs
    return tSave, Szs
end

Γ = 1
γ_values = [1e-3 * Γ, 1e-2 * Γ, 1e-1 * Γ, 1 * Γ, 10 * Γ, 20 * Γ, 30 * Γ, 40 * Γ]
Δ = 2000 * Γ
V = Δ
nAtoms = 400
tf = 15
nT = 400
nTraj = 500
dt = 1e-2
case = 2

if case == 1
    Ω_values = 0:1:40
else
    Ω_values = 0:1:20
end

script_dir = @__DIR__

@time begin
    index = parse(Int, ARGS[1])  # SLURM task ID (0 to 20)
    Ω_idx = index + 1
    Ω = Ω_values[Ω_idx]
    
    # Calculate threads per γ to avoid oversubscription
    n_threads_total = Threads.nthreads()  # 16 from -t 16
    n_γ = length(γ_values)  # 6
    threads_per_γ = max(1, n_threads_total ÷ n_γ)  # e.g., 16 ÷ 6 ≈ 2 threads per γ
    
    println("Total threads: $n_threads_total, Threads per γ: $threads_per_γ")
    
    # Parallelize over γ with limited threads per computeTWA
    ThreadsX.foreach(γ_values) do γ
        data_folder = joinpath(script_dir, "results_data/atoms=$(nAtoms),Δ=$(Δ),γ=$(γ)")
        if !isdir(data_folder)
            mkpath(data_folder)
        end
        println("Computing for nAtoms = $nAtoms, γ = $γ, Ω = $Ω on thread $(Threads.threadid())")
        
        # Limit threads for EnsembleThreads within computeTWA
        t, Szs = let
            # Temporarily set JULIA_NUM_THREADS for this scope
            old_threads = get(ENV, "JULIA_NUM_THREADS", string(n_threads_total))
            ENV["JULIA_NUM_THREADS"] = string(threads_per_γ)
            result = computeTWA(nAtoms, tf, nT, nTraj, dt, Ω, Δ, V, Γ, γ)
            ENV["JULIA_NUM_THREADS"] = old_threads  # Restore original
            result
        end
        @save "$(data_folder)/sz_mean_steady_for_$(case)D,Ω=$(Ω),Δ=$(Δ),γ=$(γ).jld2" t Szs compress=true
    end
end
