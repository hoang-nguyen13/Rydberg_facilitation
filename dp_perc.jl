using JLD2
using DifferentialEquations
using Profile
using Base.Threads
using DiffEqCallbacks
using LinearAlgebra
using Statistics

BLAS.set_num_threads(1)

function sampleSpinZPlus()
    θ = acos(1/sqrt(3))
    ϕ = 2π * rand()
    return θ, ϕ
end

function sampleSpinZMinus()
    θ = acos(-1/sqrt(3))
    ϕ = 2π * rand()
    return θ, ϕ
end

function get_neighbors_1d(nAtoms)
    neighbor_offsets = [(-1,), (1,)]
    neighbors = Vector{Vector{Int}}(undef, nAtoms)
    @inbounds for i in eachindex(neighbors)
        atom_neighbors = [
            mod1(i + d, nAtoms)
            for (d,) in neighbor_offsets
            if 1 <= i + d <= nAtoms
        ]
        neighbors[i] = atom_neighbors
    end
    return neighbors
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

function get_or_compute_neighbors(nAtoms, case, neighbors_dir="neighbors_data")
    @assert case in [1, 2, 3] "case must be 1 (1D), 2 (2D), or 3 (3D)"
    neighbors_file = joinpath(neighbors_dir, "neighbors_$(case)D_n$(nAtoms).jld2")

    mkpath(neighbors_dir)

    lock(file_lock) do
        if isfile(neighbors_file)
            try
                return jldopen(neighbors_file, "r") do file
                    file["neighbors"]
                end
            catch e
                println("Error reading neighbors file $neighbors_file: $e. Recomputing...")
            end
        end

        neighbors = case == 1 ? get_neighbors_1d(nAtoms) :
                    case == 2 ? get_neighbors_2d(nAtoms) :
                    get_neighbors_3d(nAtoms)

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
end

function drift!(du, u, p, t)
    Ω, Δ, V, Γ, γ, nAtoms, neighbors = p
    sqrt_3 = sqrt(3)
    θ = @view u[1:nAtoms]
    ϕ = @view u[(nAtoms+1):(2*nAtoms)]

    @inbounds for i in 1:nAtoms
        cotθ_i = cot(θ[i])
        cscθ_i = csc(θ[i])
        dϕ_drift_sum = 0.0
        neighbor_indices = neighbors[i]
        for idx in neighbor_indices
            dϕ_drift_sum += 1.0 + sqrt_3 * cos(θ[idx])
        end
        du[i] = 2.0 * Ω * sin(ϕ[i]) + Γ * (cotθ_i + cscθ_i / sqrt_3)
        du[nAtoms+i] = 2.0 * Ω * cotθ_i * cos(ϕ[i]) - V * dϕ_drift_sum + Δ
    end
    nothing
end

function diffusion!(du, u, p, t)
    Ω, Δ, V, Γ, γ, nAtoms, neighbors = p
    sqrt_3 = sqrt(3)
    θ = @view u[1:nAtoms]
    ϕ = @view u[(nAtoms+1):(2*nAtoms)]

    @inbounds for i in 1:nAtoms
        cotθ_i = cot(θ[i])
        cscθ_i = csc(θ[i])
        du[i] = 0.0
        du[nAtoms+i] = sqrt(Γ * (1.0 + 2.0 * cotθ_i^2 + 2.0 * cotθ_i * cscθ_i / sqrt_3) + 4.0 * γ)
    end
    nothing
end

function prob_func(prob, i, repeat)
    println("Starting trajectory $i of $nTraj on thread: ", threadid())
    flush(stdout)
    nAtoms = prob.p[end-1]
    #spin_up_idx = div(nAtoms + matrix_size + 1, 2) # Initialize excited spin
    #matrix_size = cbrt(nAtoms) |> Int
    spin_up_idx = div(nAtoms, 2) + 1
    u0 = Vector{Float64}(undef, 2 * nAtoms)
    @inbounds for n = 1:nAtoms
        if n == spin_up_idx
            θn, ϕn = sampleSpinZPlus()
        else
            θn, ϕn = sampleSpinZMinus()
        end
        u0[n] = θn
        u0[nAtoms + n] = ϕn
    end
    remake(prob, u0=u0)
end

const file_lock = ReentrantLock()
const traj_counter = Atomic{Int}(0)

function output_func(sol, i)
    nAtoms = length(sol.u[1]) ÷ 2
    sqrt_3 = sqrt(3)
    sz_atoms = zeros(Float64, nAtoms, length(sol.t)) # nAtoms × nT matrix
    @inbounds for (j, t) in enumerate(sol.t)
        θ = @view sol(t)[1:nAtoms]
        for k in 1:nAtoms
            sz_atoms[k, j] = sqrt_3 * cos(θ[k]) # <s_z> for atom k at time t
        end
    end
    traj_idx = 0
    lock(file_lock) do
        jldopen(sol_filename, "a+") do file
            if !haskey(file, "sz")
                JLD2.Group(file, "sz")
            end
            sz_group = file["sz"]
            existing_traj = isempty(keys(sz_group)) ? 0 : maximum(parse(Int, match(r"traj_(\d+)", k).captures[1]) for k in keys(sz_group))
            traj_idx = existing_traj + i
            file["sz/traj_$traj_idx"] = sz_atoms # Save natoms × nT matrix
            if !haskey(file, "tSave")
                file["tSave"] = sol.t
            else
                if file["tSave"] != sol.t
                    @warn "tSave in file does not match current solution time points for trajectory $traj_idx"
                end
            end
        end
    end
    completed = atomic_add!(traj_counter, 1) + 1
    println("Completed trajectory $completed of $nTraj (total in file: $traj_idx)")
    flush(stdout)
    return (nothing, false)
end

function compute_and_save_sz_mean(src_file, dest_file)
    t_save = nothing
    lock(file_lock) do
        jldopen(src_file, "r") do file
            if !haskey(file, "sz")
                error("No sz group found in $src_file")
            end
            sz_group = file["sz"]
            traj_keys = sort([k for k in keys(sz_group) if occursin(r"traj_\d+", k)], by=k->parse(Int, match(r"traj_(\d+)", k).captures[1]))
            if isempty(traj_keys)
                error("No trajectories found in $src_file")
            end
            # Preload all trajectories to avoid parallel JLD2 access
            println("Preloading $(length(traj_keys)) trajectories from $src_file...")
            flush(stdout)
            valid_traj_keys = String[]
            valid_trajs = Matrix{Float64}[]
            n_atoms, n_times = nothing, nothing
            for key in traj_keys
                try
                    sz_traj = sz_group[key]
                    dims = size(sz_traj)
                    println("  $key: dimensions $dims")
                    if isnothing(n_atoms)
                        n_atoms, n_times = dims
                        push!(valid_traj_keys, key)
                        push!(valid_trajs, sz_traj)
                    elseif dims == (n_atoms, n_times)
                        push!(valid_traj_keys, key)
                        push!(valid_trajs, sz_traj)
                    else
                        println("  Skipping $key: invalid dimensions $dims (expected ($n_atoms, $n_times))")
                    end
                catch e
                    println("  Skipping $key: error reading ($e)")
                end
            end
            if isempty(valid_trajs)
                error("No valid trajectories found in $src_file")
            end
            # Preallocate sz_mean
            sz_mean = zeros(Float64, n_atoms, n_times)
            # Thread-local sums
            thread_sums = [zeros(Float64, n_atoms, n_times) for _ in 1:Threads.nthreads()]
            println("Processing $(length(valid_trajs)) valid trajectories across $(Threads.nthreads()) threads...")
            flush(stdout)
            # Parallel loop over preloaded trajectories
            Threads.@threads for i in 1:length(valid_trajs)
                thread_id = Threads.threadid()
                sz_traj = valid_trajs[i]
                @inbounds for j in 1:n_times
                    for k in 1:n_atoms
                        thread_sums[thread_id][k, j] += sz_traj[k, j]
                    end
                end
            end
            # Combine thread results
            println("Combining results from threads...")
            flush(stdout)
            @inbounds for t in 1:Threads.nthreads()
                for j in 1:n_times
                    for k in 1:n_atoms
                        sz_mean[k, j] += thread_sums[t][k, j]
                    end
                end
            end
            sz_mean ./= length(valid_trajs)
            t_save = file["tSave"]
            if length(t_save) != n_times
                println("Warning: tSave length $(length(t_save)) does not match n_times $n_times")
            end
            println("Writing to $dest_file...")
            flush(stdout)
            jldopen(dest_file, "w", compress=true) do out_file
                out_file["sz_mean"] = sz_mean
                out_file["tSave"] = t_save
            end
        end
    end
    println("Saved mean sz to $dest_file")
    flush(stdout)
end

function computeTWA(nAtoms, tf, nT, nTraj, Ω, Δ, V, Γ, γ, case)
    tspan = (0, tf)
    tSave = LinRange(0, tf, nT)
    u0 = Vector{Float64}(undef, 2 * nAtoms)
    neighbors = get_or_compute_neighbors(nAtoms, case)
    p = (Ω, Δ, V, Γ, γ, nAtoms, neighbors)

    prob = SDEProblem(drift!, diffusion!, u0, tspan, p)
    ensemble_prob = EnsembleProblem(prob; prob_func=prob_func, output_func=output_func)

    solve(ensemble_prob, SRIW1(), EnsembleThreads();
          saveat=tSave,
          trajectories=nTraj,
          maxiters=5e9,
          abstol=1e-3,
          reltol=1e-3,
          dtmax=0.0001)

    return nothing
end

let
    global Ω = parse(Float64, ARGS[1])
    global γ = parse(Float64, ARGS[2])
    global Γ = parse(Float64, ARGS[3])
    global Δ = parse(Float64, ARGS[4])
    global V = parse(Float64, ARGS[5])
    global nAtoms = parse(Int, ARGS[6])
    global tf = parse(Float64, ARGS[7])
    global nT = parse(Int, ARGS[8])
    global nTraj = parse(Int, ARGS[9])
    global case = parse(Int, ARGS[10])
    global sol_filename = joinpath(dirname(@__DIR__), "results_data", "atoms=$(nAtoms),Δ=$(Δ),γ=$(γ)", "sz_ss_$(case)D,Ω=$(Ω),Δ=$(Δ),γ=$(γ).jld2")

    # Ensure output directory exists
    sol_dir = dirname(sol_filename)
    if isdir(sol_dir)
        println("Output directory already exists: $sol_dir")
    else
        mkpath(sol_dir)
        println("Created output directory: $sol_dir")
    end

    @assert case in [1, 2, 3] "case must be 1 (1D), 2 (2D), or 3 (3D)"
    traj_counter[] = 0
    existing_traj = isfile(sol_filename) ? jldopen(sol_filename, "r") do file
        haskey(file, "sz") ? length([k for k in keys(file["sz"]) if occursin(r"traj_\d+", k)]) : 0
    end : 0
    println("Starting ensemble solve with $nTraj new trajectories (existing: $existing_traj) on $(Threads.nthreads()) threads...")
    flush(stdout)
    @time computeTWA(nAtoms, tf, nT, nTraj, Ω, Δ, V, Γ, γ, case)
    println("Finished computeTWA.")
    flush(stdout)

    # Ensure mean output directory exists
    data_folder = joinpath(@__DIR__, "results_data_mean", "atoms=$(nAtoms),Δ=$(Δ),γ=$(γ)")
    if isdir(data_folder)
        println("Mean output directory already exists: $data_folder")
    else
        mkpath(data_folder)
        println("Created mean output directory: $data_folder")
    end
    
    Ω_str = string(Ω)
    γ_str = string(γ)
    src_file = sol_filename
    dest_file = joinpath(data_folder, "ρ_ss_$(case)D,Ω=$(Ω_str),Δ=$(Δ),γ=$(γ_str).jld2")
    println("Computing mean sz and saving to $dest_file...")
    flush(stdout)
    # Call optimized compute_and_save_sz_mean
    compute_and_save_sz_mean(src_file, dest_file)
end
