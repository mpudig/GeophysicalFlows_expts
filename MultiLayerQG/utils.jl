# Some useful extra functions for running my experiments in GeophysicalFlows

module Utils

# compile other packages
using GeophysicalFlows, FFTW, Statistics, Random, CUDA, CUDA_Driver_jll, CUDA_Runtime_jll, GPUCompiler

"""
	monoscale_random(hrms, Ktopo, Lx, nx, dev, T)

Returns a 2D topography field defined by a single length scale with random phases. 
"""

function monoscale_random(hrms, Ktopo, Lx, nx, dev)
	# Random seed for reproducability purposes
	if dev == CPU()
		Random.seed!(1234)
	else
		CUDA.seed!(1234)
	end

	 # Wavenumber grid
	 nk = Int(nx / 2 + 1)
	 nl = nx
	
	 dk = 2 * pi / Lx
	 dl = dk
	 
	 k = reshape( rfftfreq(nx, dk * nx), (nk, 1) )
	 l = reshape( fftfreq(nx, dl * nx), (1, nl) )

	 K = @. sqrt(k^2 + l^2)

	 # Isotropic Gaussian in wavenumber space about mean, Ktopo, with standard deviation, sigma 
	 sigma = sqrt(2) * dk
	 hh = exp.(-(K .- Ktopo).^2 ./ (2 * sigma^2)) .* exp.(2 * pi * im .* rand(nk, nl))

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

function set_initial_condition!(prob, E0, K0, Kd)
	params = prob.params
	grid = prob.grid
	dev = grid.device
	T = eltype(grid)
	A = device_array(dev)

	# Random seed for reproducability purposes
	if dev == CPU()
		Random.seed!(4321)
	else
		CUDA.seed!(4321)
	end

	# Grid
	nx = grid.nx
	Lx = grid.Lx

	nk = Int(nx / 2 + 1)
	nl = nx
   
	dk = 2 * pi / Lx
	dl = dk
	
	k = A(reshape( rfftfreq(nx, dk * nx), (nk, 1) ))
	l = A(reshape( fftfreq(nx, dl * nx), (1, nl) ))

	K2 = @. k^2 + l^2
	K = @. sqrt(K2)

	# Isotropic Gaussian in wavenumber space about mean, K0, with standard deviation, sigma 
	sigma = sqrt(2) * dk
	phases = 2 * pi * im .* A(rand(nk, nl))
	psihmag = exp.(-(K .- K0).^2 ./ (2 * sigma^2)) .* exp.(phases)
	psih = A(zeros(nk, nl, 2) .* im)
	CUDA.@allowscalar psih[:,:,1] = psihmag
	CUDA.@allowscalar psih[:,:,2] = -1 .* psihmag

	# Calculate average energy and scaling factor so that average energy is prescribed
	M = nx^2
	H = params.H

	KE = 1 / (2 * Lx^2 * sum(H)) * sum(H[1] .* K2 .* abs.(psih[:,:,1] ./ M).^2) + sum((H[2] .* K2 .* abs.(psih[:,:,2] ./ M).^2))
	APE = 1 / (2 * Lx^2) * Kd^2 / 4 * sum( abs.(psih[:,:,1] ./ M - psih[:,:,2] ./ M).^2  )
	E = KE + APE
	c = sqrt(E0 / E)
	psih = @. c * psih

	# Invert psih to get qh, then transform to real space 
	f0, gp = params.f₀, params.g′

	qh = A(zeros(nk, nl, 2) .* im)
	CUDA.@allowscalar qh[:,:,1] = - K .* psih[:,:,1] .+ f0^2 / (gp * H[1]) .* (psih[:,:,2] .- psih[:,:,1])
	CUDA.@allowscalar qh[:,:,2] = - K .* psih[:,:,2] .+ f0^2 / (gp * H[2]) .* (psih[:,:,1] .- psih[:,:,2])

	q = A(zeros(nx, nx, 2))
	CUDA.@allowscalar q[:,:,1] = A(irfft(qh[:,:,1], nx))
	CUDA.@allowscalar q[:,:,2] = A(irfft(qh[:,:,2], nx))

	MultiLayerQG.set_q!(prob, q)
end
end