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
    sqrt_3 = sqrt(3)
    
    # Zero out dϕ_drift_sum
    for i in 1:length(dϕ_drift_sum)
        dϕ_drift_sum[i] = 0.0
    end
    
    # Compute dϕ_drift_sum and drift terms
    if case == 1
        # dϕ_drift_sum for case 1 (linear chain)
        for i in 2:(nAtoms-1)
            dϕ_drift_sum[i] = 2.0 + sqrt_3 * (cos(u[i-1]) + cos(u[i+1]))
        end
        dϕ_drift_sum[1] = 1.0 + sqrt_3 * cos(u[2])
        dϕ_drift_sum[nAtoms] = 1.0 + sqrt_3 * cos(u[nAtoms-1])
        
        # Compute dθ_drift and dϕ_drift
        for i in 1:nAtoms
            θ_i = u[i]
            ϕ_i = u[nAtoms + i]
            cotθ_i = cot(θ_i)
            cscθ_i = csc(θ_i)
            
            # dθ_drift = 2 * Ω * sin(ϕ_i) + Γ * (cotθ_i + cscθ_i / sqrt_3)
            du[i] = 2.0 * Ω * sin(ϕ_i) + Γ * (cotθ_i + cscθ_i / sqrt_3)
            
            # dϕ_drift = 2 * Ω * cotθ_i * cos(ϕ_i) - V * dϕ_drift_sum[i] + Δ
            du[nAtoms + i] = 2.0 * Ω * cotθ_i * cos(ϕ_i) - V * dϕ_drift_sum[i] + Δ
        end
    elseif case == 2
        # Single loop for dϕ_drift_sum and drift terms
        for i in 1:nAtoms
            θ_i = u[i]
            ϕ_i = u[nAtoms + i]
            cotθ_i = cot(θ_i)
            cscθ_i = csc(θ_i)
            
            # Compute dϕ_drift_sum[i]
            sum_val = 0.0
            for idx in neighbors[i]
                sum_val += 1.0 + sqrt_3 * cos(u[idx])
            end
            dϕ_drift_sum[i] = sum_val
            
            # dθ_drift = 2 * Ω * sin(ϕ_i) + Γ * (cotθ_i + cscθ_i / sqrt_3)
            du[i] = 2.0 * Ω * sin(ϕ_i) + Γ * (cotθ_i + cscθ_i / sqrt_3)
            
            # dϕ_drift = 2 * Ω * cotθ_i * cos(ϕ_i) - V * dϕ_drift_sum[i] + Δ
            du[nAtoms + i] = 2.0 * Ω * cotθ_i * cos(ϕ_i) - V * dϕ_drift_sum[i] + Δ
        end
    end
end

function diffusion!(du, u, p, t)
    Ω, Δ, V, Γ, γ = p
    sqrt_3 = sqrt(3)
    
    # Compute diffusion for each atom
    for i in 1:nAtoms
        θ_i = u[i]
        cotθ_i = cot(θ_i)
        cscθ_i = csc(θ_i)
        
        term1 = 1.0
        term2 = 2.0 * cotθ_i * cotθ_i
        term3 = 2.0 * cotθ_i * cscθ_i / sqrt_3
        diffusion_i = sqrt(Γ * (term1 + term2 + term3) + 4.0 * γ)
        
        # Set du
        du[i] = 0.0
        du[nAtoms + i] = diffusion_i
    end
end

function computeTWA(nAtoms, tf, nT, nTraj, Ω, Δ, V, Γ, γ, case)
    tspan = (0, tf)
    tSave = LinRange(0, tf, nT)
    u0 = Vector{Float64}(undef, 2 * nAtoms)
    dϕ_drift_sum = zeros(nAtoms)
    
    neighbors = case == 2 ? get_neighbors_vectorized(nAtoms) : nothing
    p = (Ω, Δ, V, Γ, γ, nAtoms, neighbors, dϕ_drift_sum)
    
    prob = SDEProblem(drift!, diffusion!, u0, tspan, p)
    ensemble_prob = EnsembleProblem(prob; prob_func=(prob, i, repeat) -> begin
        θ, ϕ = sampleSpinZPlus(nAtoms)
        u0[1:nAtoms] = θ
        u0[nAtoms+1:2*nAtoms] = ϕ
        remake(prob, u0=u0)
    end)
    
    sol = solve(ensemble_prob, SRIW1(); 
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
nTraj = 1
case = 2

Ω_values = 0:1:30
γ_values = [0.1, 20, 50, 100]

# Create array of [Ω, γ] pairs
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

