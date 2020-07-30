using TrainTools, Flux, Test

function approxgrads(gs1, gs2, ps)
	for p in ps 
		!(gs1[p] ≈ gs2[p]) && return(false)
	end
	return(true)
end

l, d = 100, 4 
l = div(l, 2)
m = Chain(Dense(d,d,relu), Dense(d,2))
x = randn(d,2*l)
x[:,l+1:end] .+= 3
ps = Flux.params(m)
y = Flux.onehotbatch(vcat(fill(1,l),fill(2,l)), 1:2)

loss(m, x, y) = Flux.logitcrossentropy(m(x), y)
gs = Flux.gradient(() -> loss(m, x, y), ps)
gs1 = tgradient((x...) -> loss(m, x...), ps, () -> (x,y)) 

@test approxgrads(gs, gs1[2], ps)

#Let's do some benchmarking
using BenchmarkTools
@btime TrainTools.addgrad!(gs, gs1, ps)
# 2.843 μs (8 allocations: 192 bytes)

@btime Flux.gradient(() -> loss(m, x, y), ps)
# 113.857 μs (4039 allocations: 109.42 KiB)
@btime tgradient((x...) -> loss(m, x...), ps, () -> (x,y)) 
# 166.652 μs (16300 allocations: 442.47 KiB)

opt = ADAM()
ttrain!((x...) -> loss(m, x...), ps, () -> (x,y), opt, 100, cby = y -> println(y))