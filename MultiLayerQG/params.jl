module Params

  	 ### Save path and device ###

expt_name = ""
path_name = "" * expt_name * "/output"

dev = GPU() # or CPU()

	 ### Grid ###

nx = 1024           # number of x grid cells
Ld = 15e3           # deformation radius
ld = 2 * pi * Ld    # deformation wavelength
Kd = 1 / Ld         # deformation wavenumber
Lx = 25 * ld        # side length of square domain

nz = 2                         # number of z layers
H0 = 4000.                     # total rest depth of fluid
delta = 1.                     # ratio of layer 1 to layer 2
H1 = delta / (1 + delta) * H0  # rest depth of layer 1
H2 = 1 / (1 + delta) * H0      # rest depth of layer 2
H = [H1, H2]                   # rest depth of each layer


    	 ### Control parameters ###

beta_star = 0.
kappa_star = 0.1
h_star = 5.



    	 ### Planetary parameters ###

U0 = 0.01
U = [2 * U0, 0 * U0]

f0 = 1e-4
beta = U0 / Ld^2 * beta_star

g = 9.81
rho0 = 1000.
rho1 = rho0 + 25.
rho2 = rho1 + (rho1 * f0^2 Ld^2) / (g * H0)
rho = [rho1, rho2]
# b = -g / rho0 .* [rho1, rho2]

kappa = U0 / Ld * kappa_star



      	   ### Topography ###
	   
Ktopo = 25 * (2 * pi / Lx)
hrms = h_star * U0 * H0 * Ktopo / f0
h = utils.monoscale_random(hrms, Ktopo, Lx, nx)
eta = f0 / H2 .* h



      	   ### Time stepping ###

Ti = Ld / U0
dt = 300.
nsteps = ceil(Int, 750 * Ti / dt)
nsnaps = ???