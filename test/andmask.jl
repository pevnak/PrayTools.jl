using PrayTools
using Test
using PrayTools: maskedgrad, signalign, igradient
using Flux

@testset "signalign" begin 
	xs = [[1, 1, -1], [1, -1, -1], [1, 1, - 1]]
	xsm = [[1, 1, -1], [1, -1, -1], [1, 1, - 1], nothing]

	@test signalign(xs, 0.5) ≈ [1, 0, -1]
	@test signalign(xs, 1) ≈ [1, 0, -1]
	@test signalign(xs, 0) ≈ [1, 0.3333333333333333, -1]

	@test signalign(xsm, 0.5) ≈ [1, 0, -1]
	@test signalign(xsm, 1) ≈ [1, 0, -1]
	@test signalign(xsm, 0) ≈ [1, 0.3333333333333333, -1]
end


@testset "maskedgrad" begin 
	xs = [[1, 1, -1], [1, -1, -1], [1, 1, - 1]]
	W = [1 2 3; 4 5 6; 7 8 9]
	b = [0,0,0]
	m₁ = Dense(W,b)
	m₂ = Dense(deepcopy(W),deepcopy(b))
	ps = Flux.params((m₁, m₂))
	ps₁ = Flux.params(m₁)
	ps₂ = Flux.params(m₂)
	τ = 0.5
	gss₁ = map(x -> gradient(() -> sum(m₁(x)), ps), xs)
	gss₂ = map(x -> gradient(() -> sum(m₂(x)), ps), xs)
	@test all(all(gs[p] == nothing for p in ps₁) for gs in gss₂)
	@test all(all(gs[p] == nothing for p in ps₂) for gs in gss₁)
	gs = maskedgrad(vcat(gss₁, gss₂), 0.5)
	@test gs[ps[1]] == gs[ps[3]]
	@test gs[ps[2]] == gs[ps[4]]

	@test gs[ps[1]] ≈  [1.0 0.0 -1.0; 1.0 0.0 -1.0; 1.0 0.0 -1.0]
	@test gs[ps[3]] ≈  [1.0 0.0 -1.0; 1.0 0.0 -1.0; 1.0 0.0 -1.0]
	@test gs[ps[2]] ≈  [1,1,1]
	@test gs[ps[4]] ≈  [1,1,1]
end

@testset "igradient" begin 
	xs = [[1, 1, -1], [1, -1, -1], [1, 1, - 1]]
	W = [1 2 3; 4 5 6; 7 8 9]
	b = [0,0,0]
	m = Dense(W,b)
	ps = Flux.params(m)
	τ = 0.5

	loss = x -> sum(sin.(m(x))) 
	gs₁ = maskedgrad(map(x -> gradient(() -> loss(x), ps), xs), τ)
	y₁ = mean(loss(x) for x in xs)
	y₂, gs₂ = igradient(loss, ps, map(x -> (x,), xs), τ)
	@test y₁ ≈ y₂
	@test gs₁[W] ≈ gs₂[W]
	@test gs₁[b] ≈ gs₂[b]
end