using Flux, Zygote
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
    y, gs 
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

function addgrad!(gs1, gs2, ps)
    for p in ps 
        if gs1[p] != nothing && gs2[p] != nothing 
            gs1.grads[p] .+= gs2[p]
        elseif gs2[p] != nothing
            gs1.grads[p] = gs2[p]
        end
    end
    gs1
end

function addgrad!(gs1::Tuple, gs2::Tuple, ps)
    (gs1[1] + gs2[1], addgrad!(gs1[2], gs2[2], ps))
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
    gs = _pgradient(loss, ps, samples)
    for p in ps
        isnothing(gs[p]) && continue
        gs[p] ./= length(samples)
    end
    gs 
end

function _pgradient(loss, ps, samples)
    if length(samples) == 1
        x = samples[1]
        return(gradient(() -> loss(x...), ps))
    else 
        i = div(length(samples),2)
        ref1 = Threads.@spawn _pgradient(loss, ps, samples[1:i])
        gs2 = _pgradient(loss, ps, samples[i+1:end])
        return(addgrad!(fetch(ref1), gs2, ps))
    end
end

"""
    function ptrain!(loss, ps, preparesamples, opt, iterations; cb = () -> ())
"""
function ptrain!(loss, ps, preparesamples, opt, iterations; cb = () -> ())
  ps = Flux.Params(ps)
  cb = Flux.Optimise.runall(cb)
  for i in 1:iterations
      gs = pgradient(loss, ps, preparesamples())
      Flux.Optimise.update!(opt, ps, gs)
      cb()
  end
end
