# See https://github.com/pseastham/jld2h5 for source code

using HDF5, JLD2, CUDA

# include function which converts jld2 to hdf5
include("jld2h5.jl")

# include and import parameters
include("params.jl")
import .Params

# Get path
expt_name = Params.expt_name             
file = "../../output" * expt_name

# Convert jld2 to hdf5
jld2h5(file)