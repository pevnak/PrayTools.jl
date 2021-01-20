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

"""
	trainy!(loss, ps, preparesamples, opt, iterations; cb = () -> (), cby = (y) -> ())
"""
function trainy!(loss, ps, preparesamples, opt, iterations; cb = () -> (), cby = (y) -> ())
  ps = Flux.Params(ps)
  cb = Flux.Optimise.runall(cb)
  for i in 1:iterations
	  	x = preparesamples()
        y, gs = gradient(() -> loss(x...), ps)
        Flux.Optimise.update!(opt, ps, gs)
        cb()
        cby(y)
  end
end

