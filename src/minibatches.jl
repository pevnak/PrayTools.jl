using StatsBase

function classindexes(targets)
	d = Dict{Int,Vector{Int}}()
	for (i, v) in enumerate(targets)
		if haskey(d, v)
			push!(d[v], i)
		else 
			d[v] = [i]
		end
	end
	return(d)
end

function StatsBase.sample(class_indexes::Dict, n::Int)
	n = div(n, length(keys(class_indexes)))
	ii = [sample(class_indexes[k], n, replace = false) for k in keys(class_indexes)]
	ii = vcat(ii...);
end

function makebatch(data::AbstractMatrix, targets, target_set, class_indexes, n)
	ii = sample(classindexes, n)
	data[:, ii], Flux.onehotbatch(targets[ii], target_set)
end

function makebatch(data::AbstractVector, targets, target_set, class_indexes, n)
	ii = sample(classindexes, n)
	data[ii], Flux.onehotbatch(targets[ii], target_set)
end

"""
	initbatchprovider(data, targets, n)

	clusuer, which upon a call returns a minibatch with approximately `n` samples, 
	such that each class contains `div(n,k)` samples, where `k` is the number of classes
"""
function initbatchprovider(data, targets, n)
	ci = classindexes(targets)
	target_set = sort(collect(keys(ci)))
	() -> makebatch(data, targets, target_set, ci, n)
end