# This is the driver: set up, run, and save the model

# include all modules
include("utils.jl")
include("params.jl")

# compile other packages
using GeophysicalFlows, FFTW, Statistics, Random, CUDA, Printf, JLD2, CUDA_Driver_jll, CUDA_Runtime_jll, GPUCompiler, GPUArrays;

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

function simulate!(nsteps, nsubs, grid, prob, out, diags, KE, APE)
      saveproblem(out)
      sol, clock, params, vars, grid = prob.sol, prob.clock, prob.params, prob.vars, prob.grid
      
      startwalltime = time()
      frames = 0:round(Int, nsteps / nsubs)
      
      for j = frames
            if j % 5 == 0
                  cfl = clock.dt * maximum([maximum(vars.u) / grid.dx, maximum(vars.v) / grid.dy])
      
                  log = @sprintf("step: %04d, t: %.1f, cfl: %.3f, KE₁: %.3e, KE₂: %.3e, PE: %.3e, walltime: %.2f min",
                   clock.step, clock.t, cfl, KE.data[KE.i][1], KE.data[KE.i][2], APE.data[APE.i][1], (time()-startwalltime)/60)
      
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
      sol, params, vars, grid = prob.sol, prob.params, prob.vars, prob.grid

      # We want to save CPU arrays not GPU arrays
      A = device_array(GPU())
      B = device_array(CPU())

      q = A(zeros(size(vars.q)))
      qh = prob.sol
      MultiLayerQG.invtransform!(q, qh, params)

      return B(q)
end

      ### Initialize and then call step forward function ###

function start!()
      prob = MultiLayerQG.Problem(nlayers, dev; nx, Lx, f₀, β, g, U, H, ρ, eta, μ, 
                                  dt, stepper, aliased_fraction=0)

      sol, clock, params, vars, grid = prob.sol, prob.clock, prob.params, prob.vars, prob.grid
      x, y = grid.x, grid.y

      KE = Diagnostic(Utils.calc_KE, prob; nsteps)
      APE = Diagnostic(Utils.calc_APE, prob; nsteps)
      D = Diagnostic(Utils.calc_meridiff, prob; nsteps)
      V = Diagnostic(Utils.calc_meribarovel, prob; nsteps)
      Lmix = Diagnostic(Utils.calc_mixlen, prob; nsteps)
      diags = [KE, APE, D, V, Lmix]

      filename = Params.path_name
      if isfile(filename); rm(filename); end

      out = Output(prob, filename, (:q, get_q),
                  (:KE, Utils.calc_KE), (:APE, Utils.calc_APE), (:D, Utils.calc_meridiff), (:V, Utils.calc_meribarovel), (:Lmix, Utils.calc_mixlen))

      Utils.set_initial_condition!(prob, Params.E0, Params.K0, Params.Kd)

      simulate!(nsteps, nsubs, grid, prob, out, diags, KE, APE)
end