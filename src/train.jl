"""
    function ptrain!(loss, ps, preparesamples, opt, iterations; cb = () -> ())
"""
function train!(loss, ps, preparesamples, opt, iterations; cb = () -> (), cby = (y) -> (), debugmode = false)
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
