"""
    train!(loss, ps, preparesamples, opt, iterations; cb = () -> (), cby = (y) -> ())
"""
function train!(loss, ps, preparesamples, opt, iterations; cb = () -> (), cby = (y) -> ())
  ps = Flux.Params(ps)
  cb = Flux.Optimise.runall(cb)
  for i in 1:iterations
  		xy = preparesamples()
        y, gs = _pgradient(loss, ps, (xy,))
        Flux.Optimise.update!(opt, ps, gs)
        cb()
        cby(y)
  end
end

"""
  autotrain!(loss, ps, preparesamples, opt, iterations; cb = () -> (), cby = (y) -> ())
  """
function autotrain!(loss, ps, preparesamples, opt, iterations; cb = () -> (), cby = (y) -> ())
  perfs = paralelization_stats(loss, ps, prepare_minibatch, minibatch_size)
  parts = perfs.parts[argmin(perfs.time)]
  ptrain!(loss, ps, () -> preparesamples(minibatch_size, parts), opt, iterations; cby )
end

"""
  ptrain!(loss, ps, preparesamples, opt, iterations; cb = () -> (), cby = (y) -> ())

  parallel training using threads of one computer
  preparesamples() --- create tuples of arguments of loss. Each item of the tuple 
        is dispatched to a separate thread. In practice, this is tuple of tuples 

"""
function ptrain!(loss, ps, preparesamples, opt, iterations; cb = () -> (), cby = (y) -> ())
  ps = Flux.Params(ps)
  cb = Flux.Optimise.runall(cb)
  maxy = 0
  for i in 1:iterations
      ds = preparesamples()
      y, gs = pgradient(loss, ps, ds)
      Flux.Optimise.update!(opt, ps, gs)
      cb()
      cby(y)
  end
end

"""
  dividebatch(bs::Int, xs...)

  divide minibatch `xs` into `bs` chunks of similar size
"""
function dividebatch(bs::Int, xs...)
  n = div(nobs(xs), bs)
  xs = map(Iterators.partition(1:nobs(xs), n)) do i 
    tuple(map(x -> LearnBase.getobs(x, i), xs)...)
  end 
  tuple(xs...)
end

"""
  paralelization_stats(loss, ps, prepare_minibatch, minibatch_size; steps , try_parts)

  loss --- implicitly contains the `model` and should accepts as an arguments outputs of `prepare_minibatch`
  prepare_minibatch --- should accept as input a `minibatch_size` and the number of items to which it should be partitioned
"""
function paralelization_stats(loss, ps, prepare_minibatch, minibatch_size; steps = 10, try_parts = 0:round(Int,log(minibatch_size)/log(2)))
  perfs = map(try_parts) do sc
    parts = 2^sc
    dss = prepare_minibatch(minibatch_size, parts)
    pgradient(loss, ps, dss)
    t = @elapsed for i in 1:steps
      dss = prepare_minibatch(minibatch_size, parts)
      pgradient(loss, ps, dss)
    end
    @show (;parts, time = t)
    (;parts, time = t)
  end |> DataFrame

  ds = prepare_minibatch(minibatch_size)
  gradient(() -> loss(ds...), ps)
  t = @elapsed for i in 1:steps
    ds = prepare_minibatch(minibatch_size)
    gradient(() -> loss(ds...), ps)
  end
  push!(perfs, (;parts = 1, time = t))
  perfs
end
