import Gen
import Gen: @gen, Unfold, normal, uniform, choicemap, get_args, get_retval, get_traces
using PyPlot

include("mixture.jl")


### data types ###

struct Hypers
  outlier_min :: Float64
  outlier_max :: Float64
  inlier_stdev :: Float64
  prob_outlier :: Float64
  robust_likelihood :: Bool
end

const robust_noisy_likelihood = Mixture{Float64}([normal, uniform])


### model ###

@gen function step_forward(t::Int, y_prev::Float64, hypers::Hypers)
  y ~ normal(y_prev, 1)
  if hypers.robust_likelihood
    obs ~ robust_noisy_likelihood([1 - hypers.prob_outlier, hypers.prob_outlier],
                                  [(y, hypers.inlier_stdev), (hypers.outlier_min, hypers.outlier_max)])
  else
    obs ~ normal(y, hypers.inlier_stdev)
  end
  return y
end

steps = Unfold(step_forward)

@gen function model(T::Int, hypers::Hypers)
  y_init ~ normal(0, 1)
  if hypers.robust_likelihood
    obs_init ~ robust_noisy_likelihood([1 - hypers.prob_outlier, hypers.prob_outlier],
                                       [(y_init, hypers.inlier_stdev), (hypers.outlier_min, hypers.outlier_max)])
  else
    obs_init ~ normal(y_init, hypers.inlier_stdev)
  end
  ys ~ steps(T-1, y_init, hypers)
  return [y_init, ys...]
end

# utility functions

get_obs(trace) = vcat(trace[:obs_init], [trace[:ys => i => :obs] for i in 1:first(get_args(trace))-1])
get_latents(trace) = get_retval(trace)
function make_hypers(;robust_likelihood::Bool=true)
  Hypers(
    -25,  # outlier_min
    25,   # outlier_max
    0.1,  # inlier_stdev 
    0.1,  # prob_outlier 
    robust_likelihood
  )
end

Gen.load_generated_functions()

if false
  # make and plot prior sample
  trace, = Gen.generate(model, (20, make_hypers(robust_likelihood=true)), choicemap())
  plot(get_latents(trace))
  scatter(0:length(get_obs(trace))-1, get_obs(trace))
end


### inference ###

# utility functions

@gen function random_walk_proposal(trace, address, width)
    prev_value = trace[address]
    {address} ~ normal(prev_value, width)
end

function rejuv_kernel!(state)
  t = first(get_args(first(get_traces(state))))
  traces = get_traces(state)
  for i in 1:length(traces)
    trace = traces[i]
    for rejuv_step in 1:100
      trace, = Gen.mh(trace, random_walk_proposal, (addr(t), 0.1))
    end
    traces[i] = trace
  end
  return state
end

function make_obs(data; t::Int=1)
  obs_choicemap = choicemap()
  if t > 1
    obs_choicemap[:ys => t-1 => :obs] = data
  else
    obs_choicemap[:obs_init] = data
  end
  return obs_choicemap
end

addr(t) = t > 1 ? (:ys => t-1 => :y) : (:y_init)

# make data
synthetic_data = get_obs(Gen.simulate(model, (20, make_hypers(robust_likelihood=true))))

fig, axs = subplots(2)
for (ax, robust_likelihood) in zip(axs, [false, true])
  if false
    # plot synthetic data
    scatter(0:length(synthetic_data)-1, synthetic_data)
  end

  # initialize PF

  init_obs = make_obs(synthetic_data[1])
  model_args = (1, make_hypers(robust_likelihood=robust_likelihood))
  state = Gen.initialize_particle_filter(model, model_args, init_obs, 1)

  # step forward PF

  argsdiff = (Gen.UnknownChange(), Gen.NoChange())
  for t in 2:length(synthetic_data)
    rejuv_kernel!(state)
    new_obs = make_obs(synthetic_data[t]; t=t)
    new_model_args = (t, model_args[2])
    Gen.particle_filter_step!(state, new_model_args, argsdiff, new_obs)
  end
  rejuv_kernel!(state)  # last timestep inference


  # plotting results

  trace = first(get_traces(state))
  latents = get_latents(trace)
  obs = get_obs(trace)
  ax.plot(get_latents(trace))
  label_string = robust_likelihood ? "Robust Likelihood" : "Non-robust likelihood"
  ax.scatter(0:length(get_obs(trace))-1, get_obs(trace), label=label_string)
  ax.legend()
end
