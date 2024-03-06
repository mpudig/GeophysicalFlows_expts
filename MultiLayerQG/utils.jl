# Some useful extra functions for running my experiments in GeophysicalFlows

module Utils

# compile other packages
using GeophysicalFlows, FFTW, Statistics, Random, CUDA, CUDA_Driver_jll, CUDA_Runtime_jll, GPUCompiler
using FourierFlows: parsevalsum

"""
	monoscale_random(hrms, Ktopo, Lx, nx, dev, T)

Returns a 2D topography field defined by a single length scale with random phases. 
"""

function monoscale_random(hrms, Ktopo, Lx, nx, dev)

	 # Wavenumber grid
	 nk = Int(nx / 2 + 1)
	 nl = nx
	
	 dk = 2 * pi / Lx
	 dl = dk
	 
	 k = reshape( rfftfreq(nx, dk * nx), (nk, 1) )
	 l = reshape( fftfreq(nx, dl * nx), (1, nl) )

	 K = @. sqrt(k^2 + l^2)

	 # Isotropic Gaussian in wavenumber space about mean, Ktopo, with standard deviation, sigma
	 # with random Fourier phases
	 sigma = sqrt(2) * dk

	 Random.seed!(1234)
	 hh = exp.(-(K .- Ktopo).^2 ./ (2 * sigma^2)) .* exp.(2 * pi * im .* rand(nk, nl))

	 # Recover h from hh
	 h = irfft(hh, nx)

	 c = hrms / sqrt.(mean(h.^2))
	 h = c .* h

	 return h
end

"""
	goff_jordan_iso(hrms, Ktopo, Lx, nx, dev, T)

Returns a 2D, isotropic topography field defined by the Goff Jordan spectrum with random phases. 
"""

function goff_jordan_iso(hrms, Ktopo, Lx, nx, dev)

	 # Wavenumber grid
	 nk = Int(nx / 2 + 1)
	 nl = nx
	
	 dk = 2 * pi / Lx
	 dl = dk
	 
	 k = reshape( rfftfreq(nx, dk * nx), (nk, 1) )
	 l = reshape( fftfreq(nx, dl * nx), (1, nl) )

	 K = @. sqrt(k^2 + l^2)

	 # Goff Jordan spectrum assuming isotropy
	 mu = 3.5
	 k0 = 1.8e-4
	 l0 = 1.8e-4

	 Random.seed!(1234)
	 hspec =  2 * pi * hrms^2 * (mu - 2) / (k0 * l0) .* (1 + (kk ./ k0).^2 + (ll ./ l0).^2)^(-mu / 2)
	 hh = hspec .* exp.(2 * pi * im .* rand(nk, nl))

	 # Recover h from hh
	 h = irfft(hh, nx)

	 c = hrms / sqrt.(mean(h.^2))
	 h = c .* h

	 return h
end


"""
	set_initial_condition!(prob, grid, K0, E0)

	Sets the initial condition of MultiLayerQG to be a random q(x,y) field with baroclinic structure
	and with energy localized in spectral space about K = K0 and with total kinetic energy equal to E0
"""


"""
	set_initial_condition!(prob, grid, K0, E0)

	Sets the initial condition of MultiLayerQG to be a random q(x,y) field with baroclinic structure
	and with energy localized in spectral space about K = K0 and with total kinetic energy equal to E0
"""

function set_initial_condition!(prob, E0, K0, Kd)
	params = prob.params
	grid = prob.grid
	vars = prob.vars
	dev = grid.device
	T = eltype(grid)
	A = device_array(dev)

	# Grid
	nx = grid.nx
	Lx = grid.Lx

	nk = Int(nx / 2 + 1)
	nl = nx

	dk = 2 * pi / Lx
	dl = dk

	k = reshape( rfftfreq(nx, dk * nx), (nk, 1) )
	l = reshape( fftfreq(nx, dl * nx), (1, nl) )

	K2 = @. k^2 + l^2
	K = @. sqrt(K2)

	# Isotropic Gaussian in wavenumber space about mean, K0, with standard deviation, sigma
	sigma = sqrt(2) * dk

	Random.seed!(4321)
	psihmag = exp.(-(K .- K0).^2 ./ (2 * sigma^2)) .* exp.(2 * pi * im .* rand(nk, nl))

	psih = zeros(nk, nl, 2) .* im
	psih[:,:,1] = psihmag
	psih[:,:,2] = -1 .* psihmag

	# Calculate KE and APE, and prescribe mean total energy
	H = params.H
	V = grid.Lx * grid.Ly * sum(H)
	f0, gp = params.f₀, params.g′
	
	abs²∇𝐮h = zeros(nk, nl, 2) .* im
    abs²∇𝐮h[:,:,1] = K2 .* abs2.(psih[:,:,1])
    abs²∇𝐮h[:,:,2] = K2 .* abs2.(psih[:,:,2])

    KE = 1 / (2 * V) * (parsevalsum(abs²∇𝐮h[:,:,1], grid) * H[1] + parsevalsum(abs²∇𝐮h[:,:,1], grid) * H[2])
    APE = 1 / (2 * V) * f0^2 / gp * parsevalsum(abs2.(psih[:,:,1] .- psih[:,:,2]), grid)
    E = KE + APE
    c = sqrt(E0 / E)
    psih = @. c * psih

    # Invert psih to get qh, then transform qh to real space qh
    qh = zeros(nk, nl, 2) .* im
    qh[:,:,1] = - K2 .* psih[:,:,1] .+ f0^2 / (gp * H[1]) .* (psih[:,:,2] .- psih[:,:,1])
    qh[:,:,2] = - K2 .* psih[:,:,2] .+ f0^2 / (gp * H[2]) .* (psih[:,:,1] .- psih[:,:,2])

    q = zeros(nx, nx, 2)
    q[:,:,1] = irfft(qh[:,:,1], nx)
    q[:,:,2] = irfft(qh[:,:,2], nx)

    # Set as initial condition
    MultiLayerQG.set_q!(prob, A(q))
end



### Calculate diagnostics ###

function calc_KE(prob)
		vars, params, grid, sol = prob.vars, prob.params, prob.grid, prob.sol
		nlayers = 2
		KE = zeros(nlayers)
			
		@. vars.qh = sol
		MultiLayerQG.streamfunctionfrompv!(vars.ψh, vars.qh, params, grid)
			
		abs²∇𝐮h = vars.uh        # use vars.uh as scratch variable
		@. abs²∇𝐮h = grid.Krsq * abs2(vars.ψh)
			
		V = grid.Lx * grid.Ly * sum(params.H)
			
		ψ1h, ψ2h = view(vars.ψh, :, :, 1), view(vars.ψh, :, :, 2)
			
		for j = 1:nlayers
			  view(KE, j) .= 1 / (2 * V) * parsevalsum(view(abs²∇𝐮h, :, :, j), grid) * params.H[j]
		end
			  
		return KE
end
  
function calc_APE(prob)
		vars, params, grid, sol = prob.vars, prob.params, prob.grid, prob.sol
			
		V = grid.Lx * grid.Ly * sum(params.H)

		ψ1h, ψ2h = view(vars.ψh, :, :, 1), view(vars.ψh, :, :, 2)
			
		APE = 1 / (2 * V) * params.f₀^2 / params.g′ * parsevalsum(abs2.(ψ1h .- ψ2h), grid)
			  
		return APE
end

function calc_meridiff(prob)
	vars, params, grid, sol = prob.vars, prob.params, prob.grid, prob.sol

	psi1, psi2 = view(vars.ψ, :, :, 1), view(vars.ψ, :, :, 2)
	v1, v2 = view(vars.v, :, :, 1), view(vars.v, :, :, 2)

	psi_bc = 0.5 .* (psi1 - psi2)
	v_bt = 0.5 .* (v1 + v2)

	U1 = view(params.U, 1, 1, 1)
    U2 = view(params.U, 1, 1, 2)
    U0 = 0.5 * (U1 - U2)

	D = mean(psi_bc .* v_bt ./ U0)
		  
	return D
end

function calc_meribarovel(prob)
	vars, params, grid, sol = prob.vars, prob.params, prob.grid, prob.sol

	v1, v2 = view(vars.v, :, :, 1), view(vars.v, :, :, 2)
	v_bt = 0.5 .* (v1 + v2)

	V = sqrt(mean(v_bt.^2))
		  
	return V
end

function calc_mixlen(prob)
	vars, params, grid, sol = prob.vars, prob.params, prob.grid, prob.sol

	psi1, psi2 = view(vars.ψ, :, :, 1), view(vars.ψ, :, :, 2)
	psi_bc = 0.5 .* (psi1 - psi2)

	U1 = view(params.U, 1, 1, 1)
    U2 = view(params.U, 1, 1, 2)
    U0 = 0.5 * (U1 - U2)

	Lmix = sqrt(mean(psi_bc.^2) ./ U0.^2) 
	  
	return Lmix
end

function calc_KEFlux_1(prob)
	vars, params, grid, sol = prob.vars, prob.params, prob.grid, prob.sol

	nx = grid.nx
	Lx = grid.Lx

	nk = Int(nx / 2 + 1)
	nl = nx

	# Make isotropic wavenumber grid
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
	
	# Get stream functions and vorticity
	psih = view(vars.ψh, :, :, :)
	uh = view(vars.uh, :, :, :)
	vh = view(vars.vh, :, :, :)
	zeta = view(vars.ψ, :, :, :)   # use as scratch variable
	invtransform!(zeta, -grid.Krsq .* vars.ψh, params)
	
	# Loop over filters and calculate KE flux
	KEflux = zeros(K_id)

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

		view(KEflux, j) .= mean(psi_hpf_1 .* uzeta_dx_1 + psi_hpf_1 .* vzeta_dy_1)
	end

	return KEflux
end

function calc_APEFlux_1(prob)
	vars, params, grid, sol = prob.vars, prob.params, prob.grid, prob.sol

	nx = grid.nx
	Lx = grid.Lx

	nk = Int(nx / 2 + 1)
	nl = nx

	# Make isotropic wavenumber grid
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
	
	# Get stream functions and velocity
	psih = view(vars.ψh, :, :, :)
	uh = view(vars.uh, :, :, :)
	vh = view(vars.vh, :, :, :)
	psi2 = view(vars.ψ, :, :, 2)
	
	# Loop over filters and calculate PE flux
	PEflux = zeros(K_id)

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

		view(PEflux, j) .= mean(0.5 .* psi_hpf_1 .* upsi2_dx_1 + 0.5 .* psi_hpf_1 .* vpsi2_dy_1)
	end

	return PEflux
end

function calc_ShearFlux_1(prob)
	vars, params, grid, sol = prob.vars, prob.params, prob.grid, prob.sol

	nx = grid.nx
	Lx = grid.Lx

	nk = Int(nx / 2 + 1)
	nl = nx

	# Make isotropic wavenumber grid=
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

	# Get stream functions and velocity
	psih = view(vars.ψh, :, :, :)
	psi2 = view(vars.ψ, :, :, 2)
	
	# Loop over filters and calculate shear forcing flux
	ShearFlux = zeros(K_id)

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
		view(ShearFlux, j) .= mean(psi_hpf_1 .* psi_hpf_dx_2)
	end

	return ShearFlux
end

function calc_KEFlux_2(prob)
	vars, params, grid, sol = prob.vars, prob.params, prob.grid, prob.sol

	nx = grid.nx
	Lx = grid.Lx

	nk = Int(nx / 2 + 1)
	nl = nx

	# Make isotropic wavenumber grid
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
	
	# Get stream functions and vorticity
	psih = view(vars.ψh, :, :, :)
	uh = view(vars.uh, :, :, :)
	vh = view(vars.vh, :, :, :)
	zeta = view(vars.ψ, :, :, :)   # use as scratch variable
	invtransform!(zeta, -grid.Krsq .* vars.ψh, params)
	
	# Loop over filters and calculate KE flux
	KEflux = zeros(K_id)

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

		view(KEflux, j) .= mean(psi_hpf_2 .* uzeta_dx_2 + psi_hpf_2 .* vzeta_dy_2)
	end

	return KEflux
end

function calc_APEFlux_2(prob)
	vars, params, grid, sol = prob.vars, prob.params, prob.grid, prob.sol

	nx = grid.nx
	Lx = grid.Lx

	nk = Int(nx / 2 + 1)
	nl = nx

	# Make isotropic wavenumber grid
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
	
	# Get stream functions and velocity
	psih = view(vars.ψh, :, :, :)
	uh = view(vars.uh, :, :, :)
	vh = view(vars.vh, :, :, :)
	psi1 = view(vars.ψ, :, :, 1)
	
	# Loop over filters and calculate PE flux
	PEflux = zeros(K_id)

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

		view(PEflux, j) .= mean(0.5 .* psi_hpf_2 .* upsi1_dx_2 + 0.5 .* psi_hpf_2 .* vpsi1_dy_2)
	end

	return PEflux
end

function calc_TopoFlux_2(prob)
	vars, params, grid, sol = prob.vars, prob.params, prob.grid, prob.sol

	nx = grid.nx
	Lx = grid.Lx

	nk = Int(nx / 2 + 1)
	nl = nx

	# Make isotropic wavenumber grid
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
	
	# Get stream functions and topography
	psih = view(vars.ψh, :, :, :)
	uh = view(vars.uh, :, :, :)
	vh = view(vars.vh, :, :, :)

	f0 = params.f₀
	H0 = sum(params.H)
	htop = H0 / f0 .* params.eta
	
	# Loop over filters and calculate topographic flux
	TopoFlux = zeros(K_id)

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

		view(TopoFlux, j) .= mean(2 .* psi_hpf_2 .* uhtop_dx_2 + 2 .* psi_hpf_2 .* vhtop_dy_2)
	end

	return TopoFlux
end

function calc_DragFlux_2(prob)
	vars, params, grid, sol = prob.vars, prob.params, prob.grid, prob.sol

	nx = grid.nx
	Lx = grid.Lx

	nk = Int(nx / 2 + 1)
	nl = nx

	# Make isotropic wavenumber grid
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
	
	# Get velocities
	uh = view(vars.uh, :, :, :)
	vh = view(vars.vh, :, :, :)
	
	# Loop over filters and calculate drag flux
	DragFlux = zeros(K_id)

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
		view(DragFlux, j) .= mean(-2 .* u_hpf.^2 - 2 .* v_hpf.^2)
	end

	return DragFlux
end

end