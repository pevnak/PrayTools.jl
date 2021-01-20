############
#  Learning explanations that are hard to vary
#  https://arxiv.org/pdf/2009.00329.pdf
############

function signalign(xs, τ)
	xs = filter(x -> x != nothing, xs)
	m = mean(sign.(x) for x in  xs)
	m = abs.(m) .>= τ
	m .* mean(xs)
end

function maskedgrad(gss::Vector{Zygote.Grads}, τ)
	length(gss) == 0 && error("zero number of gradient, nothing to aggregate")
	length(gss) == 1 && return(gss[1])
	ps = gss[1].params
	gs = map(ps) do p 
		xs = [gs[p] for gs in gss]
		p => signalign(xs, τ)
	end |> IdDict
	Zygote.Grads(gs, ps)
end

function igradient(loss, ps, xs, τ)
	ygss = ThreadPools.qmap(xs) do x
	    y, back = Zygote.pullback(() -> loss(x...), ps)
	    y, back(Zygote.sensitivity(y))
	end
	y = mean(y[1] for y in ygss)
	gs = maskedgrad([y[2] for y in ygss], τ)
	y, gs
end

"""
    function itrain!(loss, ps, preparesamples, opt, iterations; cb = () -> ())
"""
function itrain!(loss, ps, preparesamples, opt, iterations; cb = () -> (), cby = (y) -> (), τ = 0.5)
  ps = Flux.Params(ps)
  cb = Flux.Optimise.runall(cb)
  for i in 1:iterations
        y, gs = igradient(loss, ps, preparesamples(), τ)
        Flux.Optimise.update!(opt, ps, gs)
        cb()
        cby(y)
  end
end