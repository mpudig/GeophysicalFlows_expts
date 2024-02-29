using NCDatasets, JLD2

# include and import parameters

include("params.jl")
import .Params

function convert_to_nc()
    # Get path and open jld2 file

    expt_name = Params.expt_name         
    file_path = "../../output" * expt_name * ".jld2"
    file = jldopen(file_path)

    # Open already created .nc file in write mode 
    ds = NCDataset("../../output" * expt_name * ".nc", "a")

    # Extract fields from original nc file
    q_nc = ds["q"]
    KE_nc = ds["KE"]
    APE_nc = ds["APE"]
    D_nc = ds["D"]
    V_nc = ds["V"]
    Lmix_nc = ds["Lmix"]
    #KEFlux1_nc = ds["KEFlux1"]
    #APEFlux1_nc = ds["APEFlux1"]
    #ShearFlux1_nc = ds["ShearFlux1"]
    #KEFlux2_nc = ds["KEFlux2"]
    #APEFlux2_nc = ds["APEFlux2"]
    #TopoFlux2_nc = ds["TopoFlux2"]
    #DragFlux2_nc = ds["DragFlux2"]

    # Make new time axis for restarts
    iterations = parse.(Int, keys(file["snapshots/t"]))[2:end] # don't take first snapshot as this is same as in file that was restarted from
    t_restart = [file["snapshots/t/$iteration"] for iteration in iterations]
    t_nc = ds["t"][:]
    t_nc_id = length(t_nc)
    t_restart = t_restart .+ last(t_nc)
    t = [t_nc; t_restart]

    # Diagnostics from restart
    KE_jld2 = [file["snapshots/KE/$iteration"] for iteration in iterations]
    KE_jld2 = reduce(hcat, KE)
    APE_jld2 = [file["snapshots/APE/$iteration"] for iteration in iterations]
    D_jld2 = [file["snapshots/D/$iteration"] for iteration in iterations]
    V_jld2 = [file["snapshots/V/$iteration"] for iteration in iterations]
    Lmix_jld2 = [file["snapshots/Lmix/$iteration"] for iteration in iterations]

    KEFlux1_jld2 = [file["snapshots/KEFlux1/$iteration"] for iteration in iterations]
    APEFlux1_jld2 = [file["snapshots/APEFlux1/$iteration"] for iteration in iterations]
    ShearFlux1_jld2 = [file["snapshots/ShearFlux1/$iteration"] for iteration in iterations]

    KEFlux2_jld2 = [file["snapshots/KEFlux2/$iteration"] for iteration in iterations]
    APEFlux2_jld2 = [file["snapshots/APEFlux2/$iteration"] for iteration in iterations]
    TopoFlux2_jld2 = [file["snapshots/TopoFlux2/$iteration"] for iteration in iterations]
    DragFlux2_jld2 = [file["snapshots/DragFlux2/$iteration"] for iteration in iterations]

    # Update time coordinates (i.e., variables with the same name as dimensions)

    defVar(ds, "t", Float64, ("t",))
    ds["t"][:] = t

    # Make isotropic wavenumber grid if not already made
    nx = file["grid/nx"]
    Lx = file["grid/Lx"]
    nk = Int(nx / 2 + 1)
	nl = nx
	dk = 2 * pi / Lx
	dl = dk
    k = reshape( rfftfreq(nx, dk * nx), (nk, 1) )
	l = reshape( fftfreq(nx, dl * nx), (1, nl) )
    kmax = maximum(k)
	lmax = maximum(abs.(l))
	Kmax = sqrt(kmax^2 + lmax^2)
	Kmin = 0.
	dK = sqrt(dk^2 + dl^2)
    K = Kmin:dK:Kmax-1

    defVar(ds, "K", Float64, ("K",))
    ds["K"][:] = K

    # Define variables: fields, diagnostics, snapshots
    # Append new restart fields to old fields

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

    # Energy budget diagnostics
    defVar(ds, "KEFlux1", Float64, ("K", "t"))
    ds["KEFlux1"][:,:] = KEFlux1

    defVar(ds, "PEFlux1", Float64, ("K", "t"))
    ds["PEFlux1"][:,:] = PEFlux1

    defVar(ds, "ShearFlux1", Float64, ("K", "t"))
    ds["ShearFlux1"][:,:] = ShearFlux1

    defVar(ds, "KEFlux2", Float64, ("K", "t"))
    ds["KEFlux2"][:,:] = KEFlux2

    defVar(ds, "PEFlux2", Float64, ("K", "t"))
    ds["PEFlux2"][:,:] = PEFlux2

    defVar(ds, "TopoFlux2", Float64, ("K", "t"))
    ds["TopoFlux2"][:,:] = TopoFlux2

    defVar(ds, "DragFlux2", Float64, ("K", "t"))
    ds["DragFlux2"][:,:] = DragFlux2

    # Finally, after all the work is done, we can close the file and the dataset
    close(file)
    close(ds)

    # Delete jld2 file
    rm(file_path)
end