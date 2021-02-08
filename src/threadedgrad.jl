
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

function addgrad!(x::AbstractArray, y::AbstractArray)
  x .+= y
  x
end

function addgrad!(gs1::NamedTuple, gs2::NamedTuple)
  for p in keys(x)
    if gs1[p] != nothing && gs2[p] != nothing 
        addgrad!(gs1[p], gs2[p])
    else
      error("cannot joint two NamedTuples with non-equal keys")
    end
  end
  gs1
end

function addgrad!!(gs1, gs2, ps)
  for p in ps 
      if gs1[p] != nothing && gs2[p] != nothing 
          addgrad!(gs1[p], gs2[p])
      elseif gs2[p] != nothing
          gs1[p].grads = gs2[p]
      end
  end
  gs1
end

function addgrad!(gs1::Zygote.Grads, gs2::Zygote.Grads, ps::Zygote.Params)
  for p in ps 
      if gs1[p] != nothing && gs2[p] != nothing 
          addgrad!(gs1[p], gs2[p])
      elseif gs2[p] != nothing
          gs1[p].grads = gs2[p]
      end
  end
  gs1
end

function addgrad!(gs1::Tuple{T,Zygote.Grads}, gs2::Tuple{T,Zygote.Grads}, ps::Zygote.Params) where {T}
    (gs1[1] + gs2[1], addgrad!(gs1[2], gs2[2], ps))
end

function normalize!(gs, ps, n::Int)
  for p in ps
      isnothing(gs[p]) && continue
      normalize!(gs[p], n)
  end
end

normalize!(x::AbstractArray, n) = x ./= n
normalize!(x::NamedTuple, n) = normalize!(x, keys(x), n)

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
      # if y > maxy
      #   debug_info = ds
      #   maxy = y
      # end
      # println("y = ", y, gradinfo(gs))
      Flux.Optimise.update!(opt, ps, gs)
      cb()
      cby(y)
  end
end

function gradinfo(gs)
  gs = filter(x -> isa(x, Array),  collect(values(gs.grads)))
  (nan = mapreduce(x -> sum(isnan.(x)), +,  gs),
    inf = mapreduce(x -> sum(isinf.(x)), +, gs),
    max = mapreduce(x -> maximum(abs.(x)), max, gs),
    l2 = mapreduce(x -> sum(x.^2), +, gs),
    )
end
