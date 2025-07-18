using JLD2

function inspect_jld2_dimensions(src_file, dest_file)
    println("Inspecting src_file: $src_file")
    if !isfile(src_file)
        println("Error: $src_file does not exist")
        return
    end

    jldopen(src_file, "r") do file
        println("Contents of $src_file:")
        if haskey(file, "sz")
            sz_group = file["sz"]
            traj_keys = [k for k in keys(sz_group) if occursin(r"traj_\d+", k)]
            println("Found $(length(traj_keys)) trajectories")
            for key in traj_keys
                try
                    sz_traj = sz_group[key]
                    dims = size(sz_traj)
                    println("  $key: dimensions $dims")
                catch e
                    println("  $key: error reading ($e)")
                end
            end
        else
            println("No 'sz' group found in $src_file")
        end
        if haskey(file, "tSave")
            t_save = file["tSave"]
            println("  tSave: length $(length(t_save))")
        else
            println("  tSave: not found")
        end
    end

    println("\nInspecting dest_file: $dest_file")
    if !isfile(dest_file)
        println("Error: $dest_file does not exist")
        return
    end

    jldopen(dest_file, "r") do file
        println("Contents of $dest_file:")
        if haskey(file, "sz_mean")
            sz_mean = file["sz_mean"]
            dims = size(sz_mean)
            println("  sz_mean: dimensions $dims")
        else
            println("  sz_mean: not found")
        end
        if haskey(file, "tSave")
            t_save = file["tSave"]
            println("  tSave: length $(length(t_save))")
        else
            println("  tSave: not found")
        end
    end
end

# Define file paths from your log
src_file = "/home/quw51vuk/results_data/atoms=9,Δ=2000.0,γ=0.1/sz_ss_2D,Ω=10.5,Δ=2000.0,γ=0.1.jld2"
dest_file = "/home/quw51vuk/Rydberg_facilitation/results_data_mean/atoms=9,Δ=2000.0,γ=0.1/ρ_ss_2D,Ω=10.5,Δ=2000.0,γ=0.1.jld2"

# Run inspection
inspect_jld2_dimensions(src_file, dest_file)
