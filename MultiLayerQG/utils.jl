# Some useful extra functions for running my experiments in GeophysicalFlows

module Utils

# compile other packages
using GeophysicalFlows, FFTW, Statistics, Random, CUDA

"""
	monoscale_random(hrms, Ktopo, Lx, nx, dev, T)

Returns a 2D topography field defined by a single length scale with random phases. 
"""

function monoscale_random(hrms, Ktopo, Lx, nx, dev)
	A = device_array(dev)

	# Random seed for reproducability purposes
	if dev == CPU(); Random.seed!(1234); else; CUDA.seed!(1234)

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
	 hh = exp.(-(K .- Ktopo).^2 ./ (2 * sigma^2) .* exp.(2 * pi * im .* rand(nk, nl)))

	 # Recover h from hh
	 h = irfft(hh, nx)

	 c = hrms / sqrt.(mean(h.^2))
	 h = c .* h

	 return A(h)
end


"""
	set_initial_condition!(prob, grid, K0, E0)

	Sets the initial condition of MultiLayerQG to be a random q(x,y) field with baroclinic structure
	and with energy localized in spectral space about K = K0 and with total kinetic energy equal to E0
"""

function set_initial_condition!(prob, grid, K0, E0)
	dev = grid.device
	T = eltype(grid)
	A = device_array(dev)
	if dev == CPU(); Random.seed!(4321); else; CUDA.seed!(4321)
	const newaxis = [CartesianIndex()]

	q0 = peakedisotropicspectrum(grid, kpeak = K0, E₀ = E0)
	q0 = q0[:, :, newaxis] .* [1, -1][newaxis, newaxis, :]
	q0 = A(q0)

	MultiLayerQG.set_q!(prob, q0)
 end


#=
"""
	set_psih(K0, E0, Lx, nx, Kd, H)

Returns 3D psih field with energy localized in spectral space about K = K0, with baroclinic structure, and with total energy equal to E0.
You should then use GeophysicalFlows built-in function to get qh from psih and then q from qh. 

NOTE: I just realised that GeophysicalFlows already has their built-in "peakedisotropicspectrum" function which calculates q(x,y) for me...
"""

function set_q(K0, E0, Lx, nx, Kd, H, S)

	 const newaxis = [CartesianIndex()]
	 
	 # Wavenumber grid
         nk = Int(nx / 2 + 1)
         nl = nx

         dk = 2 * pi / Lx
         dl = dk

         k = reshape( rfftfreq(nx, dk * nx), (nk, 1) )
         l = reshape( fftfreq(nx, dl * nx), (1, nl) )

	 K2 = @. k^2 + l^2
         K = @. sqrt(K2)


         # Isotropic Gaussian in wavenumber space about mean, K0, with standard deviation, sigma, and with baroclinic structure
         sigma = sqrt(2) * dk
         psih = psih = exp.(-(K .- K0).^2 ./ (2 * sigma^2)) .* exp.(2 * pi * im .* rand(nk, nl))
	 psih = psih[newaxis, :, :] .* [1, -1][:, newaxis, newaxis]


	 # Calculate total energy and scaling factor so that energy of nondimensional system is unity
	 M = nx^2

	 KE = Lx^2 / (2 * sum(H)) * sum( H[:, newaxis, newaxis] .* K2[newaxis, :, :] .* abs.(psih ./ M).^2 )
	 APE = Lx^2 / (2 * sum(H) ) * Kd^2 / 4 * sum( abs.(psih[1, :, :] - psih[2, :, :]).^2 ./ M^2  )
	 E = KE + APE
	 c = sqrt(E0 / E)
	 psih = c * psih

	 return psih
end
=#