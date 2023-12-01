# include all modules
include("utils.jl")
include("params.jl")

# compile other packages
using GeophysicalFlows, FFTW, Statistics
using Random: seed!

# local import
import .Utils
import .Params



      ### Save path and device ###

path_name = params.path_name
dev = params.dev



      ### Grid ###

nx = params.nx
Lx = params.Lx

nz = params.nz
H = params.H



      ### Planetary parameters ###

f0 = Params.f0
beta = Params.beta
g = Params.g
kappa = Params.kappa

rho = params.rho
U = params.U

eta = params.eta



      ### Time stepping ###

dt = params.dt
dtsnap = params.dtsnap
nsteps = params.nsteps



      ### Initiate model ###

prob = MultiLayerQG.Problem(nlayers = nz, dev; nx = nx, Lx = Lx, f₀ = f0, H, ρ, U, μ, β, dt, stepper = "FilteredRK4")

sol, clock, params, vars, grid = prob.sol, prob.clock, prob.params, prob.vars, prob.grid
x, y = grid.x, grid.y

