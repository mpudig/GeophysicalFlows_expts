using HDF5

expt_name = "/kappa01_kt25_h1"              
path_name = "/scratch/mp6191/GeophysicalFlows_expts" * expt_name * "/output" * expt_name

# Open the JLD2 file
jld2_file = open(path_name * ".jld2", "r")

# Convert the JLD2 file to an HDF5 file
h5_file = open(path_name * ".h5", "w")
jld2h5(jld2_file, h5_file)

# Close the files
close(jld2_file)
close(h5_file)