include("utils.jl")
include("params.jl")
include("execute.jl")

start!()

# Convert jld2 file to nc file
include("jld2_to_nc.jl")
convert_to_nc()

# Convert the restart to already made nc file
# include("jld2_to_nc_restarts.jl")
# convert_to_nc()