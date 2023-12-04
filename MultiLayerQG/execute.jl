# This is the driver: set up, run, and save the model

module Execute

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

nz = Params.nz
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

function simulate!(nsteps, nsubs, grid, prob, out, diags)
      saveproblem(out)
      sol, clock, params, vars, grid = prob.sol, prob.clock, prob.params, prob.vars, prob.grid
      
      startwalltime = time()
      frames = 0:round(Int, nsteps / nsubs)
      
      for j = frames
            if j % (1000 / nsubs) == 0
                  cfl = clock.dt * maximum([maximum(vars.u) / grid.dx, maximum(vars.v) / grid.dy])
      
                  log = @sprintf("step: %04d, t: %.1f, cfl: %.2f, walltime: %.2f min",
                              clock.step, clock.t, cfl, (time()-startwalltime)/60)
      
                  println(log)
            end

            stepforward!(prob, diags, nsubs);
            MultiLayerQG.updatevars!(prob);
            saveoutput(out);
      end 
end
        

        ### Initialize and then call step forward function ###

function start!()
      prob = MultiLayerQG.Problem(nlayers, dev; nx, Lx, f₀, β, g, U, H, ρ, eta, μ, 
                                  dt, stepper)

      sol, clock, params, vars, grid = prob.sol, prob.clock, prob.params, prob.vars, prob.grid
      x, y = grid.x, grid.y

      E = Diagnostic(MultiLayerQG.energies, prob; nsteps)
      diags = [E]

      filename = Params.pathname
      out = Output(prob, filename, (:sol, get_sol), (:E, MultiLayerQG.energies))

      set_initial_condition!(prob, grid, Params.K0, Params.E0)

      simulate!(nsteps, nsubs, grid, prob, out, diags)
end

# Now, run everything
start!()