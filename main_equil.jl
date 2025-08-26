using Distributed

# Add workers if running under SLURM
if haskey(ENV, "SLURM_NTASKS")
    nprocs_requested = parse(Int, ENV["SLURM_NTASKS"])
    if nprocs() < nprocs_requested
        addprocs(nprocs_requested - nprocs())
    end
end

using DifferentialEquations, LinearAlgebra, JLD2, Plots, Statistics, NaNMath


# Initial conditions for spins aligned near +x
function sampleSpinXPlus()
    θ = π/2 .+ 0.1 * randn()  # Small fluctuations around θ = π/2
    θ = clamp.(θ, 0, π)
    ϕ = 0.1 * randn()  # Small fluctuations around φ = 0
    return θ, ϕ
end

# Neighbor list for 1D chain
function get_neighbors_1d(nAtoms)
    neighbor_offsets = [(-1,), (1,)]
    neighbors = Vector{Vector{Int}}(undef, nAtoms)
    @inbounds for i in eachindex(neighbors)
        atom_neighbors = [mod1(i + d, nAtoms) for (d,) in neighbor_offsets]
        neighbors[i] = atom_neighbors
    end
    return neighbors
end

function get_or_compute_neighbors(nAtoms, case, neighbors_dir="neighbors_data")
    @assert case == 1 "Only 1D case implemented"
    neighbors_file = joinpath(neighbors_dir, "neighbors_1D_n$(nAtoms).jld2")
    mkpath(neighbors_dir)
    if isfile(neighbors_file)
        try
            return jldopen(neighbors_file, "r") do file
                file["neighbors"]
            end
        catch e
            println("Error reading neighbors file $neighbors_file: $e. Recomputing...")
        end
    end
    neighbors = get_neighbors_1d(nAtoms)
    try
        jldopen(neighbors_file, "w") do file
            file["neighbors"] = neighbors
        end
        println("Saved neighbors to $neighbors_file")
    catch e
        println("Error saving neighbors to $neighbors_file: $e")
    end
    return neighbors
end

# Drift term with time-dependent J(t)
function drift!(du, u, p, t)
    h, J_max, t_f, nAtoms, neighbors = p
    J = (J_max * t) / t_f  # Linear sweep: J(t) = (J_max * t) / t_f
    θ = @view u[1:nAtoms]
    ϕ = @view u[(nAtoms+1):(2*nAtoms)]

    @inbounds for i in 1:nAtoms
        cotθ_i = cot(θ[i])
        dϕ_drift_sum = 0.0
        neighbors_indices = neighbors[i]
        for idx in neighbors_indices
            dϕ_drift_sum += sqrt(3) * cos(θ[idx])
        end
        du[i] = -2.0 * h * sin(ϕ[i])
        du[nAtoms+i] = -2.0 * h * cotθ_i * cos(ϕ[i]) + 2.0 * J * dϕ_drift_sum
    end
    nothing
end

# Diffusion term (zero for closed system)
function diffusion!(du, u, p, t)
    h, J_max, t_f, nAtoms, neighbors = p
    θ = @view u[1:nAtoms]
    ϕ = @view u[(nAtoms+1):(2*nAtoms)]
    @inbounds for i in 1:nAtoms
        du[i] = 0.0
        du[nAtoms+i] = 0.0
    end
    nothing
end

# Initialize trajectories
function prob_func(prob, i, repeat)
    nAtoms = prob.p[end-1]
    u0 = Vector{Float64}(undef, 2 * nAtoms)
    @inbounds for n = 1:nAtoms
        θn, ϕn = sampleSpinXPlus()
        u0[n] = θn
        u0[nAtoms+n] = ϕn
    end
    remake(prob, u0=u0)
end

# Compute correlation sum S = N <σ_i^z σ_{i+1}^z>
function compute_correlation_sz(sol, nAtoms)
    θ = sol[1:nAtoms, :, :]  # Shape: (nAtoms, nT, nTraj)
    corr = zeros(nAtoms, size(sol, 2))  # Correlation for each time step
    @inbounds for i in 1:nAtoms
        j = mod1(i + 1, nAtoms)  # Next neighbor with periodic boundary
        for t in 1:size(sol, 2), traj in 1:size(sol, 3)
            corr[i, t] += (sqrt(3) * cos(θ[i, t, traj])) * (sqrt(3) * cos(θ[j, t, traj]))
        end
    end
    corr ./= size(sol, 3)  # Average over trajectories
    return nAtoms * dropdims(mean(corr, dims=1), dims=1)  # S = N * mean(<σ_i^z σ_{i+1}^z>)
end

# Main simulation function
function computeTWA(nAtoms, tf, nT, nTraj, h, J_max, case)
    tspan = (0.0, tf)
    tSave = LinRange(0, tf, nT)
    u0 = Vector{Float64}(undef, 2 * nAtoms)
    neighbors = get_or_compute_neighbors(nAtoms, case)
    p = (h, J_max, tf, nAtoms, neighbors)
    prob = SDEProblem(drift!, diffusion!, u0, tspan, p)
    ensemble_prob = EnsembleProblem(prob; prob_func=prob_func)

    sol = solve(ensemble_prob, EM(), EnsembleThreads();
        saveat=tSave,
        trajectories=nTraj,
        maxiters=5e9,
        abstol=1e-6,
        reltol=1e-6,
        dt=0.0001
    )
    return tSave, sol
end

# Parameters
h = 1.0
J_max = 10.0
nAtoms = 10
case = 1
tf = 1000.0
nT = 400
nTraj = 1000  # Increased for better convergence

results_dir = joinpath(@__DIR__, "results_equil")
mkpath(results_dir)  # create folder if it doesn't exist
output_file = joinpath(results_dir, "equilibration_data.jld2")

# Run adiabatic sweep
@time begin
    t, sol = computeTWA(nAtoms, tf, nT, nTraj, h, J_max, case)
    S_values = compute_correlation_sz(sol, nAtoms)  # Shape: (nT,)
    J_values = (J_max * t) / tf  # J(t) = (J_max * t) / tf
    @save output_file J_values S_values
end
