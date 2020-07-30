using StatsBase, PooledArrays

function classindexes(targets::Vector{T}) where {T}
	d = Dict{T,Vector{Int}}()
	for (i, v) in enumerate(targets)
		if haskey(d, v)
			push!(d[v], i)
		else 
			d[v] = [i]
		end
	end
	return(d)
end

classindexes(targets::PooledArray) = classindexes(Vector(targets))

function StatsBase.sample(class_indices::Dict, n::Int)
	n = div(n, length(keys(class_indices)))
	ii = [sample(class_indices[k], n, replace = false) for k in keys(class_indices)]
	ii = vcat(ii...);
end

function makebatch(data::AbstractMatrix, targets, target_set, class_indices, n)
	ii = sample(class_indices, n)
	data[:, ii], Flux.onehotbatch(targets[ii], target_set)
end

function makebatch(data::AbstractVector, targets, target_set, class_indices, n)
	ii = sample(class_indices, n)
	data[ii], Flux.onehotbatch(targets[ii], target_set)
end

"""
	initbatchprovider(data, targets, n)
	initbatchprovider(data, targets, train_indices, n)

	clusuer, which upon a call returns a minibatch with approximately `n` samples, 
	such that each class contains `div(n,k)` samples, where `k` is the number of classes
"""
function initbatchprovider(data, targets, n)
	ci = classindexes(targets)
	target_set = sort(collect(keys(ci)))
	() -> makebatch(data, targets, target_set, ci, n)
end

function initbatchprovider(data, targets, train_indices, n)
	ci = classindexes(targets)
	for k in keys(ci)
		ci[k] = intersect(ci[k], train_indices)
	end
	target_set = sort(collect(keys(ci)))
	() -> makebatch(data, targets, target_set, ci, n)
end