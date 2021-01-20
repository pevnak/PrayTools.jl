############
#  Learning explanations that are hard to vary
#  https://arxiv.org/pdf/2009.00329.pdf
############
using Statistics

function signalign(xs::Vector{T}, τ) where {T<:AbstractArray}
	m = mean(sign.(x) for x in  xs)
	m = abs.(m) .>= τ
	m .* mean(xs)
end

function maskedgrad(gss::Vector{Zygote.Grads}, τ)
	length(gss) == 0 && error("zero number of gradient, nothing to aggregate")
	length(gss) == 1 && return(gss[1])
	map(p -> p => signalign(gs[p] for p in gs), keys(gss[1]))
end