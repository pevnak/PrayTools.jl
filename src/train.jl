"""
    function ptrain!(loss, ps, preparesamples, opt, iterations; cb = () -> ())
"""
function train!(loss, ps, preparesamples, opt, iterations; cb = () -> (), cby = (y) -> (), debugmode = false)
  ps = Flux.Params(ps)
  cb = Flux.Optimise.runall(cb)
  for i in 1:iterations
        y, gs = _pgradient(loss, ps, (preparesamples(),))
        Flux.Optimise.update!(opt, ps, gs)
        cb()
        cby(y)
  end
end

