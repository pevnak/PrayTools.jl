"""
	ffnn(idim, nneurons, nlayers, fun, bnfun = nothing)
	ffnn(idim, nneurons, nlayers, fun; last_linear = false)

	creates a feedforward neural network with 
	`idim` --- input dimention
	`nlayers` --- layers, 
	`nneurons` --- hidden dimension,
	`fun` --- transfer functions,
	`bnfun` --- batch normalization (default to `nothing` meaning disabled)


```juliadoc

julia> PrayTools.ffnn(3, 5, 2, tanh, nothing)
Chain(Dense(3, 5, tanh), Dense(5, 5, tanh))

julia> PrayTools.ffnn(3, 5, 2, tanh, identity)
Chain(Dense(3, 5, tanh), BatchNorm(5), Dense(5, 5, tanh), BatchNorm(5))
"""
function ffnn(idim, nneurons, nlayers, fun, bnfun, dr)
	c = []
	for i in 1:nlayers
		idim = i == 1 ? idim : nneurons
		push!(c, Dense(idim, nneurons, fun))
		if bnfun != nothing
			push!(c, BatchNorm(nneurons, bnfun))
		end

		if dr > 0 
			push!(c, Flux.Dropout(dr))
		end
	end
	Chain(c...)
end


function ffnn(idim, nneurons, nlayers, fun; last_linear = false)
	layers = []
	for i in 1:nlayers
		if nlayers == i && last_linear
			push!(layers, Dense(idim, nneurons))
		else
			push!(layers, Dense(idim, nneurons, fun))
		end
		idim = nneurons
	end
	length(layers) > 1 ? Chain(layers...) : layers[1]
end

