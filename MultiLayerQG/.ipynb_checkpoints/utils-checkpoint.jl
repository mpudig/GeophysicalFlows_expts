"""
	monoscale_random(hrms, Ktopo, Lx, nx)

Returns a 2D topography field defined by a single length scale with random phases. 
"""

function monoscale_random(hrms, Ktopo, Lx, nx)

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

	 return h
end



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
	 



    I = np.eye(2)[:, :, np.newaxis, np.newaxis]
    M = S[:, :, np.newaxis, np.newaxis] - I * K2

    # Get qh from psih, and q from qh                                                                                                                                             
    qh = np.zeros_like(psih)
    qh[0] = M[0,0] * psih[0] + M[0,1] * psih[1]
    qh[1] = M[1,0] * psih[0] + M[1,1] * psih[1]

    # Get q from qh                                                                                                                                                               
    q = np.real(np.fft.ifftn(qh, axes = (-2, -1)))

    return q

