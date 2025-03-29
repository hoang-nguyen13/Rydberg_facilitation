using JLD2
using LinearAlgebra

# Function to compute normalized Sz from the 3D array
function compute_spin_Sz(sol, nAtoms)
    nTraj = size(sol, 3)  # Get number of trajectories
    θ = sol[1:nAtoms, :, :]  # Extract θ values (first nAtoms rows)
    Szs = sum(sqrt(3) * cos.(θ), dims=1) / (nAtoms * nTraj)  # Sum over atoms, normalize
    return dropdims(Szs, dims=1)  # Convert from 1×nT×nTraj to nT×nTraj
end

# Check command-line arguments
if length(ARGS) < 3
    println("Error: Please provide 3 arguments: sol_file nAtoms nTraj")
    println("Usage: julia compute_sz.jl <sol_file> <nAtoms> <nTraj>")
    exit(1)
end

sol_filename = ARGS[1]
nAtoms = parse(Int, ARGS[2])
# nTraj from ARGS[3] is ignored; we'll use sol's dimension

if !isfile(sol_filename)
    println("Error: File $sol_filename not found.")
    exit(1)
end

# Load the saved t and sol directly
data = jldopen(sol_filename, "r") do file
    t = file["t"]
    sol = file["sol"]  # Load the 3D array directly
    (t, sol)
end
t, sol = data

# Debug
println("Loaded sol type: $(typeof(sol))")
println("Loaded sol dimensions: $(size(sol))")
nTraj = size(sol, 3)
println("Using nTraj = $nTraj from sol dimensions (ignoring ARGS[3])")

# Compute normalized Sz
println("Computing normalized Sz from $sol_filename...")
Szs = compute_spin_Sz(sol, nAtoms)

# Save to new filename
sz_filename = replace(sol_filename, "temp_sol_" => "sz_mean_steady_for_")
try
    jldsave(sz_filename; t=t, Szs=Szs)
    println("Normalized Sz saved successfully to $sz_filename")
catch e
    println("Error saving Sz: $e")
end

# Delete temporary sol file
try
    rm(sol_filename)
    println("Temporary sol file deleted: $sol_filename")
catch e
    println("Error deleting sol file: $e")
end

# Clean up
sol = nothing
Szs = nothing
GC.gc()
println("Sz computation and cleanup complete.")
