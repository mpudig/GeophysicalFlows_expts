# This is the driver: set up, run, and save the model

# include all modules
include("utils.jl")
include("params.jl")

# compile other packages
using GeophysicalFlows, FFTW, Statistics, Random, CUDA, Printf, JLD2;

# local import
import .Utils
import .Params



      ### Save path and device ###

path_name = Params.path_name
dev = Params.dev



      ### Grid ###

nx = Params.nx
Lx = Params.Lx

nlayers = Params.nz
H = Params.H



      ### Planetary parameters ###

f₀ = Params.f0
β = Params.beta
g = Params.g
μ = Params.kappa

ρ = Params.rho
U = Params.U

eta = Params.eta



      ### Time stepping ###

dt = Params.dt
nsubs = Params.nsubs
nsteps = Params.nsteps
stepper = Params.stepper



      ### Step the model forward ###

function simulate!(nsteps, nsubs, grid, prob, out, diags, E)
      saveproblem(out)
      sol, clock, params, vars, grid = prob.sol, prob.clock, prob.params, prob.vars, prob.grid
      
      startwalltime = time()
      frames = 0:round(Int, nsteps / nsubs)
      
      for j = frames
            if j % 5 == 0
                  cfl = clock.dt * maximum([maximum(vars.u) / grid.dx, maximum(vars.v) / grid.dy])
      
                  log = @sprintf("step: %04d, t: %.1f, cfl: %.3f, KE₁: %.3e, KE₂: %.3e, PE: %.3e, walltime: %.2f min",
                   clock.step, clock.t, cfl, E.data[E.i][1][1], E.data[E.i][1][2], E.data[E.i][2][1], (time()-startwalltime)/60)
      
                  println(log)
                  flush(stdout)
            end

            stepforward!(prob, diags, nsubs)
            MultiLayerQG.updatevars!(prob)
            saveoutput(out)
      end 
end

      ### Get real space solution ###

function get_q(prob)
      sol, grid = prob.sol, prob.grid
      dev = grid.device
      A = device_array(dev)

      q = A(irfft(prob.sol, grid.nx))

      return q
end

      ### Initialize and then call step forward function ###

function start!()
      prob = MultiLayerQG.Problem(nlayers, dev; nx, Lx, f₀, β, g, U, H, ρ, eta, μ, 
                                  dt, stepper, aliased_fraction=0)

      sol, clock, params, vars, grid = prob.sol, prob.clock, prob.params, prob.vars, prob.grid
      x, y = grid.x, grid.y

      E = Diagnostic(MultiLayerQG.energies, prob; nsteps)
      diags = [E]

      filename = Params.path_name
      if isfile(filename); rm(filename); end

      out = Output(prob, filename, (:q, get_q), (:E, MultiLayerQG.energies))

      Utils.set_initial_condition!(prob, Params.E0, Params.K0, Params.Kd)

      simulate!(nsteps, nsubs, grid, prob, out, diags, E)
end