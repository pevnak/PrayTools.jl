
"""
    tgradient(loss, ps, preparesamples)

    gradient of `() -> loss(preparesamples()...)` with 
    respect to `ps` calculated using all available threads
"""    
function tgradient(loss, ps, preparesamples, n = Threads.nthreads()) 
    y, gs = _tgradient(loss, ps, preparesamples, n)
    for p in ps
        isnothing(gs[p]) && continue
        gs[p] ./= n
    end
    y/n, gs 
end
    
function _tgradient(loss, ps, preparesamples, nchilds)
    if nchilds == 1
      x = preparesamples()
      y, back = Zygote.pullback(() -> loss(x...), ps)
      gs = back(Zygote.sensitivity(y))
      return(y, gs)
    else 
        i = div(nchilds,2)
        ref2 = Threads.@spawn _tgradient(loss, ps, preparesamples, nchilds - i)
        gs1 = _tgradient(loss, ps, preparesamples, i)
        return(addgrad!(gs1, fetch(ref2), ps))
    end
end

"""
    function ttrain!(loss, ps, preparesamples, opt, iterations; cb = () -> ())
"""
function ttrain!(loss, ps, preparesamples, opt, iterations; cb = () -> (), cby = (y) -> ())
  ps = Flux.Params(ps)
  cb = Flux.Optimise.runall(cb)
  for i in 1:iterations
      y, gs = tgradient(loss, ps, preparesamples)
      Flux.Optimise.update!(opt, ps, gs)
      cb()
      cby(y)
  end
end

function pgradient(loss, ps, samples::Tuple) 
    y, gs = _pgradient(loss, ps, samples)
    n = length(samples)
    normalize!(gs, ps, n)
    y/n, gs 
end

function _pgradient(loss, ps, samples)
    if length(samples) == 1
        x = samples[1]
        y, back = Zygote.pullback(() -> loss(x...), ps)
        gs = back(Zygote.sensitivity(y))
        return(y, gs)
    else 
        i = div(length(samples),2)
        ref1 = Threads.@spawn _pgradient(loss, ps, samples[1:i])
        gs2 = _pgradient(loss, ps, samples[i+1:end])
        return(addgrad!(fetch(ref1), gs2, ps))
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
    function ptrain!(loss, ps, preparesamples, opt, iterations; cb = () -> (), bs = Threads.nthreads())
"""
function ptrain!(loss, ps, preparesamples, opt, iterations; cb = () -> (), cby = (y) -> (), debugmode = false, bs = Threads.nthreads())
  dataprovider = () ->  dividebatch(bs, preparesamples()...)
  ps = Flux.Params(ps)
  cb = Flux.Optimise.runall(cb)
  for i in 1:iterations
        y, gs = pgradient(loss, ps, dataprovider())
        Flux.Optimise.update!(opt, ps, gs)
        cb()
        cby(y)
  end
end

function ptraind!(loss, ps, preparesamples, opt, iterations; cb = () -> (), cby = (y) -> ())
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
