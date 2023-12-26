using NCDatasets, JLD2

# include and import parameters

include("params.jl")
import .Params

function convert_to_nc()
    # Get path and open jld2 file

    expt_name = Params.expt_name             
    file_path = "../../output" * expt_name * ".jld2"
    file = jldopen(file_path)

    # Get necessary key information from file
    # Clock

    dt = file["clock/dt"]

    # Grid

    nx = file["grid/nx"]
    ny = file["grid/ny"]
    Lx = file["grid/Lx"]
    Ly = file["grid/Ly"]
    x = file["grid/x"]
    x = -x[1] .+ x
    y = file["grid/y"]
    y = -y[1] .+ y

    # Params

    f0 = file["params/f₀"]
    beta = file["params/β"]
    rho = file["params/ρ"]
    rho1 = rho[1]
    rho2 = rho[2]
    U = file["params/U"][1,1,:]
    U1 = U[1]
    U2 = U[2]
    H = file["params/H"]
    delta = H[1] / H[2]
    H0 = sum(H)
    kappa = file["params/μ"]
    gp = file["params/g′"]
    eta = file["params/eta"]
    htop = H[1] / f0 .* eta
    Qx = file["params/Qx"]
    Qy = file["params/Qy"]

    # Time and diagnostics

    iterations = parse.(Int, keys(file["snapshots/t"]))
    t = [file["snapshots/t/$iteration"] for iteration in iterations]
    KE = [file["snapshots/KE/$iteration"] for iteration in iterations]
    KE = reduce(hcat, KE)
    APE = [file["snapshots/APE/$iteration"] for iteration in iterations]
    D = [file["snapshots/D/$iteration"] for iteration in iterations]
    V = [file["snapshots/V/$iteration"] for iteration in iterations]
    Lmix = [file["snapshots/Lmix/$iteration"] for iteration in iterations]

    # This creates a new NetCDF file
    # The mode "c" stands for creating a new file (clobber); the mode "a" stands for opening in write mode

    ds = NCDataset("../../output" * expt_name * ".nc", "c")
    ds = NCDataset("../../output" * expt_name * ".nc", "a")

    # Define attributes

    ds.attrib["title"] = expt_name
    ds.attrib["dt"] = dt
    ds.attrib["f0"] = f0
    ds.attrib["beta"] = beta
    ds.attrib["kappa"] = kappa
    ds.attrib["rho1"] = rho1
    ds.attrib["rho2"] = rho2
    ds.attrib["gp"] = gp
    ds.attrib["U1"] = U1
    ds.attrib["U2"] = U2
    ds.attrib["H"] = H0
    ds.attrib["delta"] = delta

    # Define the dimensions, with names and sizes

    defDim(ds, "x", size(x)[1])
    defDim(ds, "y", size(y)[1])
    defDim(ds, "lev", 2)
    defDim(ds, "t", size(t)[1])

    # Define coordinates (i.e., variables with the same name as dimensions)

    defVar(ds, "x", Float64, ("x",))
    ds["x"][:] = x

    defVar(ds, "y", Float64, ("y",))
    ds["y"][:] = y

    defVar(ds, "lev", Int64, ("lev",))
    ds["lev"][:] = [1, 2]

    defVar(ds, "t", Float64, ("t",))
    ds["t"][:] = t

    # Define variables: fields, diagnostics, snapshots

    defVar(ds, "htop", Float64, ("x", "y"))
    ds["htop"][:,:] = htop

    defVar(ds, "KE", Float64, ("lev", "t"))
    ds["KE"][:,:] = KE

    defVar(ds, "APE", Float64, ("t",))
    ds["APE"][:] = APE

    defVar(ds, "D", Float64, ("t",))
    ds["D"][:] = D

    defVar(ds, "V", Float64, ("t",))
    ds["V"][:] = V

    defVar(ds, "Lmix", Float64, ("t",))
    ds["Lmix"][:] = Lmix

    defVar(ds, "q", Float64, ("x", "y", "lev", "t"))
    for i in 1:length(iterations)
        iter = iterations[i]
        ds["q"][:,:,:,i] = file["snapshots/q/$iter"]
    end

    # Finally, after all the work is done, we can close the file and the dataset
    close(file)
    close(ds)
end