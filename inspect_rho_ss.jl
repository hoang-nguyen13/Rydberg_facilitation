using JLD2
using CodecZlib

function inspect_rho_ss_dimensions(root_dir)
    println("Inspecting ρ_ss_*.jld2 files in $root_dir and subdirectories...")
    files = String[]
    
    # Walk through subdirectories
    for (dirpath, dirnames, filenames) in walkdir(root_dir)
        for filename in filenames
            if occursin(r"ρ_ss_.*\.jld2$", filename)
                push!(files, joinpath(dirpath, filename))
            end
        end
    end
    
    if isempty(files)
        println("No ρ_ss_*.jld2 files found in $root_dir")
        return
    end
    println("Found $(length(files)) ρ_ss_*.jld2 files")

    for file in files
        println("\nInspecting file: $file")
        try
            jldopen(file, "r") do f
                println("Contents of $file:")
                if haskey(f, "sz_mean")
                    sz_mean = f["sz_mean"]
                    dims = size(sz_mean)
                    println("  sz_mean: dimensions $dims")
                else
                    println("  sz_mean: not found")
                end
                if haskey(f, "tSave")
                    t_save = f["tSave"]
                    println("  tSave: length $(length(t_save))")
                else
                    println("  tSave: not found")
                end
            end
        catch e
            println("  Error reading $file: $e")
        end
    end
end

# Define the root directory
root_dir = "/home/quw51vuk/Rydberg_facilitation/results_data_mean"

# Run inspection
inspect_rho_ss_dimensions(root_dir)
