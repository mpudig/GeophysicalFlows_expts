"""
Goal: take the last snapshot for PV from a run,
initialize a model instance with this state,
use my energy budget diagnostics to calculate
the energy budget terms given this snapshot,
then save these terms in a .nc file.

I am doing this because for some reason 
the budget terms are not saving as model
diagnostics when I run the model forward.
Ultimately, I'd like to fix this bug but
at present I cannot figure out how to.
"""

using GeophysicalFlows, FFTW, Statistics, Random, Printf, JLD2, NCDatasets, CUDA, CUDA_Driver_jll, CUDA_Runtime_jll, GPUCompiler, GPUArrays, NCDatasets;

# local import
import .Utils
import .Params

# include and import parameters

include("params.jl")
import .Params

function calc_and_save_energy_budget():

    # Get path, open nc file, get final snapshot of q

    expt_name = Params.expt_name         
    path = "../../output" * expt_name * ".nc"
    ds = NCDataset(path, "r")
    qi = ds["q"][:, :, :, end]
    close(ds)

    # Create model instance using this qi and also parameters from params

     ### Grid ###

    nx = Params.nx
    Lx = Params.Lx

    nlayers = Params.nz
    H = Params.H

    dev = GPU()

    ### Planetary parameters ###

    f₀ = Params.f0
    β = Params.beta
    g = Params.g
    μ = Params.kappa

    ρ = Params.rho
    U = Params.U
    eta = Params.eta

    kappa_star = Params.kappa_star
    h_star = Params.h_star

    ### Create model instance ###

    prob = MultiLayerQG.Problem(nlayers, dev; nx, Lx, f₀, β, g, U, H, ρ, eta, μ)
    vars, params, grid, sol = prob.vars, prob.params, prob.grid, prob.sol
    A = device_array(grid.device)
    MultiLayerQG.set_q!(prob, A(qi))

    ### Create isotropic wavenumber grid ###
    nk = Int(nx / 2 + 1)
	nl = nx

	kr = prob.grid.kr
	l = prob.grid.l
	Kr = @. sqrt(kr^2 + l^2)

	krmax = maximum(kr)
	lmax = maximum(abs.(l))
	Kmax = sqrt(krmax^2 + lmax^2)
	Kmin = 0.

	dkr = 2 * pi / Lx
	dl = dkr
	dKr = sqrt(dkr^2 + dl^2)
 
	K = Kmin:dKr:Kmax-1
	K_id = lastindex(K)

    ##### ENERGY BUDGET TERMS BELOW ###

    ### KEFlux1 ###

    KEFlux1 = zeros(K_id)

    # Get stream functions and vorticity
	psih = view(vars.ψh, :, :, :)
	uh = view(vars.uh, :, :, :)
	vh = view(vars.vh, :, :, :)
	zeta = view(vars.ψ, :, :, :)   # use as scratch variable
	invtransform!(zeta, -grid.Krsq .* vars.ψh, params)

    for j = 1:K_id
		# Define high-pass filter matrix
		hpf = ifelse.(Kr .> K[j], Kr ./ Kr, 0 .* Kr)

		# Filter the Fourier transformed fields
		psih_hpf = view(hpf, :, :) .* psih
		uh_lpf = uh .- view(hpf, :, :) .* uh
		vh_lpf = vh .- view(hpf, :, :) .* vh

		# Inverse transform the filtered fields
		psi_hpf = view(vars.ψ, :, :, :)          # use as scratch variable
		invtransform!(psi_hpf, psih_hpf, params)

		u_lpf = view(vars.ψ, :, :, :)            # use as scratch variable
		invtransform!(u_lpf, uh_lpf, params)

		v_lpf = view(vars.ψ, :, :, :)            # use as scratch variable
		invtransform!(v_lpf, vh_lpf, params)

		# Multiply for spectral products
		uzeta = u_lpf .* zeta
		uzetah = view(vars.ψh, :, :, :)          # use as scratch variable
		fwdtransform!(uzetah, uzeta, params) 

		vzeta = v_lpf .* zeta
		vzetah = view(vars.ψh, :, :, :)          # use as scratch variable
		fwdtransform!(vzetah, vzeta, params)

		# Calculate spectral derivatives
		uzetah_ik = im .* grid.kr .* uzetah
		vzetah_il = im .* grid.l .* vzetah
		
		# Inverse transform spectral derivatives
		uzeta_dx = view(vars.ψ, :, :, :)         # use as scratch variable
		invtransform!(uzeta_dx, uzetah_ik, params)

		vzeta_dy = view(vars.ψ, :, :, :)         # use as scratch variable
		invtransform!(vzeta_dy, vzetah_il, params)

		# Create views of only upper layer fields
		psi_hpf_1 = view(psi_hpf, :, :, 1)
		uzeta_dx_1 = view(uzeta_dx, :, :, 1)
		vzeta_dy_1 = view(vzeta_dy, :, :, 1)

		view(KEFlux1, j) .= mean(psi_hpf_1 .* uzeta_dx_1 + psi_hpf_1 .* vzeta_dy_1)
	end

    ### APEFlux1 ###

    APEFlux1 = zeros(K_id)

    # Get stream functions and velocity
	psih = view(vars.ψh, :, :, :)
	uh = view(vars.uh, :, :, :)
	vh = view(vars.vh, :, :, :)
	psi2 = view(vars.ψ, :, :, 2)

    for j = 1:K_id
		# Define high-pass filter matrix
		hpf = ifelse.(Kr .> K[j], Kr ./ Kr, 0 .* Kr)

		# Filter the Fourier transformed fields
		psih_hpf = view(hpf, :, :) .* psih
		uh_lpf = uh .- view(hpf, :, :) .* uh
		vh_lpf = vh .- view(hpf, :, :) .* vh

		# Inverse transform the filtered fields
		psi_hpf = view(vars.ψ, :, :, :)          # use as scratch variable
		invtransform!(psi_hpf, psih_hpf, params)

		u_lpf = view(vars.ψ, :, :, :)            # use as scratch variable
		invtransform!(u_lpf, uh_lpf, params)

		v_lpf = view(vars.ψ, :, :, :)            # use as scratch variable
		invtransform!(v_lpf, vh_lpf, params)

		# Multiply for spectral products
		upsi2 = u_lpf .* psi2
		upsi2h = view(vars.ψh, :, :, :)          # use as scratch variable
		fwdtransform!(upsi2h, upsi2, params) 

		vpsi2 = v_lpf .* psi2
		vpsi2h = view(vars.ψh, :, :, :)          # use as scratch variable
		fwdtransform!(vpsi2h, vpsi2, params)

		# Calculate spectral derivatives
		upsi2h_ik = im .* grid.kr .* upsi2h
		vpsi2h_il = im .* grid.l .* vpsi2h
		
		# Inverse transform spectral derivatives
		upsi2_dx = view(vars.ψ, :, :, :)         # use as scratch variable
		invtransform!(upsi2_dx, upsi2h_ik, params)

		vpsi2_dy = view(vars.ψ, :, :, :)         # use as scratch variable
		invtransform!(vpsi2_dy, vpsi2h_il, params)

		# Create views of only upper layer fields
		psi_hpf_1 = view(psi_hpf, :, :, 1)
		upsi2_dx_1 = view(upsi2_dx, :, :, 1)
		vpsi2_dy_1 = view(vpsi2_dy, :, :, 1)

		view(APEFlux1, j) .= mean(0.5 .* psi_hpf_1 .* upsi2_dx_1 + 0.5 .* psi_hpf_1 .* vpsi2_dy_1)
	end
	
	### ShearFlux1 ###

	ShearFlux1 = zeros(K_id)

    # Get stream functions and velocity
	psih = view(vars.ψh, :, :, :)
	psi2 = view(vars.ψ, :, :, 2)

	for j = 1:K_id
		# Define high-pass filter matrix
		hpf = ifelse.(Kr .> K[j], Kr ./ Kr, 0 .* Kr)

		# Filter the Fourier transformed fields
		psih_hpf = view(hpf, :, :) .* psih

		# Calculate spectral derivative
		psih_hpf_ik = im .* grid.kr .* psih_hpf

		# Inverse transform
		psi_hpf = view(vars.ψ, :, :, :)          # use as scratch variable
		invtransform!(psi_hpf, psih_hpf, params)

		psi_hpf_dx = view(vars.ψ, :, :, :)       # use as scratch variable
		invtransform!(psi_hpf_dx, psih_hpf_ik, params)

		# Views of necessary upper and lower layer fields
		psi_hpf_1 = view(psi_hpf, :, :, 1)
		psi_hpf_dx_2 = view(psi_hpf_dx, :, :, 2)

		# Calculate flux
		view(ShearFlux1, j) .= mean(psi_hpf_1 .* psi_hpf_dx_2)
	end

    ### KEFlux2 ###

    KEFlux2 = zeros(K_id)

    # Get stream functions and vorticity
	psih = view(vars.ψh, :, :, :)
	uh = view(vars.uh, :, :, :)
	vh = view(vars.vh, :, :, :)
	zeta = view(vars.ψ, :, :, :)   # use as scratch variable
	invtransform!(zeta, -grid.Krsq .* vars.ψh, params)

    for j = 1:K_id
		# Define high-pass filter matrix
		hpf = ifelse.(Kr .> K[j], Kr ./ Kr, 0 .* Kr)

		# Filter the Fourier transformed fields
		psih_hpf = view(hpf, :, :) .* psih
		uh_lpf = uh .- view(hpf, :, :) .* uh
		vh_lpf = vh .- view(hpf, :, :) .* vh

		# Inverse transform the filtered fields
		psi_hpf = view(vars.ψ, :, :, :)          # use as scratch variable
		invtransform!(psi_hpf, psih_hpf, params)

		u_lpf = view(vars.ψ, :, :, :)            # use as scratch variable
		invtransform!(u_lpf, uh_lpf, params)

		v_lpf = view(vars.ψ, :, :, :)            # use as scratch variable
		invtransform!(v_lpf, vh_lpf, params)

		# Multiply for spectral products
		uzeta = u_lpf .* zeta
		uzetah = view(vars.ψh, :, :, :)          # use as scratch variable
		fwdtransform!(uzetah, uzeta, params) 

		vzeta = v_lpf .* zeta
		vzetah = view(vars.ψh, :, :, :)          # use as scratch variable
		fwdtransform!(vzetah, vzeta, params)

		# Calculate spectral derivatives
		uzetah_ik = im .* grid.kr .* uzetah
		vzetah_il = im .* grid.l .* vzetah
		
		# Inverse transform spectral derivatives
		uzeta_dx = view(vars.ψ, :, :, :)         # use as scratch variable
		invtransform!(uzeta_dx, uzetah_ik, params)

		vzeta_dy = view(vars.ψ, :, :, :)         # use as scratch variable
		invtransform!(vzeta_dy, vzetah_il, params)

		# Create views of only lower layer fields
		psi_hpf_2 = view(psi_hpf, :, :, 2)
		uzeta_dx_2 = view(uzeta_dx, :, :, 2)
		vzeta_dy_2 = view(vzeta_dy, :, :, 2)

		view(KEFlux2, j) .= mean(psi_hpf_2 .* uzeta_dx_2 + psi_hpf_2 .* vzeta_dy_2)
	end

    ### APEFlux2 ###

    APEFlux2 = zeros(K_id)

    # Get stream functions and velocity
	psih = view(vars.ψh, :, :, :)
	uh = view(vars.uh, :, :, :)
	vh = view(vars.vh, :, :, :)
	psi1 = view(vars.ψ, :, :, 1)

    for j = 1:K_id
		# Define high-pass filter matrix
		hpf = ifelse.(Kr .> K[j], Kr ./ Kr, 0 .* Kr)

		# Filter the Fourier transformed fields
		psih_hpf = view(hpf, :, :) .* psih
		uh_lpf = uh .- view(hpf, :, :) .* uh
		vh_lpf = vh .- view(hpf, :, :) .* vh

		# Inverse transform the filtered fields
		psi_hpf = view(vars.ψ, :, :, :)          # use as scratch variable
		invtransform!(psi_hpf, psih_hpf, params)

		u_lpf = view(vars.ψ, :, :, :)            # use as scratch variable
		invtransform!(u_lpf, uh_lpf, params)

		v_lpf = view(vars.ψ, :, :, :)            # use as scratch variable
		invtransform!(v_lpf, vh_lpf, params)

		# Multiply for spectral products
		upsi1 = u_lpf .* psi1
		upsi1h = view(vars.ψh, :, :, :)          # use as scratch variable
		fwdtransform!(upsi1h, upsi1, params) 

		vpsi1 = v_lpf .* psi1
		vpsi1h = view(vars.ψh, :, :, :)          # use as scratch variable
		fwdtransform!(vpsi1h, vpsi1, params)

		# Calculate spectral derivatives
		upsi1h_ik = im .* grid.kr .* upsi1h
		vpsi1h_il = im .* grid.l .* vpsi1h
		
		# Inverse transform spectral derivatives
		upsi1_dx = view(vars.ψ, :, :, :)         # use as scratch variable
		invtransform!(upsi1_dx, upsi1h_ik, params)

		vpsi1_dy = view(vars.ψ, :, :, :)         # use as scratch variable
		invtransform!(vpsi1_dy, vpsi1h_il, params)

		# Create views of only upper layer fields
		psi_hpf_2 = view(psi_hpf, :, :, 2)
		upsi1_dx_2 = view(upsi1_dx, :, :, 2)
		vpsi1_dy_2 = view(vpsi1_dy, :, :, 2)

		view(APEFlux2, j) .= mean(0.5 .* psi_hpf_2 .* upsi1_dx_2 + 0.5 .* psi_hpf_2 .* vpsi1_dy_2)
	end

    ### TopoFlux2 ###

    TopoFlux2 = zeros(K_id)

    # Get stream functions and topography
	psih = view(vars.ψh, :, :, :)
	uh = view(vars.uh, :, :, :)
	vh = view(vars.vh, :, :, :)

	f0 = params.f₀
	H0 = sum(params.H)
	htop = H0 / f0 .* params.eta


	for j = 1:K_id
		# Define high-pass filter matrix
		hpf = ifelse.(Kr .> K[j], Kr ./ Kr, 0 .* Kr)

		# Filter the Fourier transformed fields
		psih_hpf = view(hpf, :, :) .* psih
		uh_lpf = uh .- view(hpf, :, :) .* uh
		vh_lpf = vh .- view(hpf, :, :) .* vh

		# Inverse transform the filtered fields
		psi_hpf = view(vars.ψ, :, :, :)          # use as scratch variable
		invtransform!(psi_hpf, psih_hpf, params)

		u_lpf = view(vars.ψ, :, :, :)            # use as scratch variable
		invtransform!(u_lpf, uh_lpf, params)

		v_lpf = view(vars.ψ, :, :, :)            # use as scratch variable
		invtransform!(v_lpf, vh_lpf, params)

		# Multiply for spectral products
		uhtop = u_lpf .* htop
		uhtoph = view(vars.ψh, :, :, :)          # use as scratch variable
		fwdtransform!(uhtoph, uhtop, params) 

		vhtop = v_lpf .* htop
		vhtoph = view(vars.ψh, :, :, :)          # use as scratch variable
		fwdtransform!(vhtoph, vhtop, params)

		# Calculate spectral derivatives
		uhtoph_ik = im .* grid.kr .* uhtoph
		vhtoph_il = im .* grid.l .* vhtoph
		
		# Inverse transform spectral derivatives
		uhtop_dx = view(vars.ψ, :, :, :)         # use as scratch variable
		invtransform!(uhtop_dx, uhtoph_ik, params)

		vhtop_dy = view(vars.ψ, :, :, :)         # use as scratch variable
		invtransform!(vhtop_dy, vhtoph_il, params)

		# Create views of only lower layer fields
		psi_hpf_2 = view(psi_hpf, :, :, 2)
		uhtop_dx_2 = view(uhtop_dx, :, :, 2)
		vhtop_dy_2 = view(vhtop_dy, :, :, 2)

		view(TopoFlux2, j) .= mean(2 * h_star .* psi_hpf_2 .* uhtop_dx_2 + 2 * h_star .* psi_hpf_2 .* vhtop_dy_2)
	end

    ### DragFlux2 ###

    DragFlux2 = zeros(K_id)

    # Get velocities
	uh = view(vars.uh, :, :, :)
	vh = view(vars.vh, :, :, :)

	for j = 1:K_id
		# Define high-pass filter matrix
		hpf = ifelse.(Kr .> K[j], Kr ./ Kr, 0 .* Kr)

		# Filter the Fourier transformed fields
		uh_hpf = view(hpf, :, :) .* uh
		vh_hpf = view(hpf, :, :) .* vh

		# Inverse transform the filtered fields
		u_hpf = view(vars.ψ, :, :, :)            # use as scratch variable
		invtransform!(u_hpf, uh_hpf, params)

		v_hpf = view(vars.ψ, :, :, :)            # use as scratch variable
		invtransform!(v_hpf, vh_hpf, params)

		# Create views of only lower layer fields
		u_hpf_2 = view(u_hpf, :, :, 2)
		v_hpf_2 = view(v_hpf, :, :, 2)

		# Calculate drag flux
		view(DragFlux2, j) .= mean(-2 * kappa_star .* u_hpf.^2 - 2 * kappa_star .* v_hpf.^2)
	end

    ##### Here, I save the energy budget terms in a new .nc file #####

    path = "../../output" * expt_name * "energy_budget.nc"
    ds = NCDataset(file_path_nc, "c")
    ds = NCDataset(file_path_nc, "a")

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

    defDim(ds, "K", size(K)[1])

    # Define coordinates (i.e., variables with the same name as dimensions)

    defVar(ds, "K", Float64, ("K",))
    ds["K"][:] = K

    # Define variables: fields, diagnostics, snapshots

    defVar(ds, "KEFlux1", Float64, ("K",))
    ds["KEFlux1"][:] = KEFlux1
    
    defVar(ds, "APEFlux1", Float64, ("K",))
    ds["APEFlux1"][:] = APEFlux1

    defVar(ds, "ShearFlux1", Float64, ("K",))
    ds["ShearFlux1"][:] = ShearFlux1

    defVar(ds, "KEFlux2", Float64, ("K",))
    ds["KEFlux2"][:] = KEFlux2

    defVar(ds, "APEFlux2", Float64, ("K",))
    ds["APEFlux2"][:] = APEFlux2

    defVar(ds, "TopoFlux2", Float64, ("K",))
    ds["TopoFlux2"][:] = TopoFlux2

    defVar(ds, "DragFlux2", Float64, ("K",))
    ds["DragFlux2"][:] = DragFlux2

    # Finally, after all the work is done, we can close the dataset
    close(ds)

end