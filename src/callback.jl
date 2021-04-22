using BSON: @save

function initevalcby(; steps =1000, accuracy = () -> () , resultdir = nothing, model = nothing)
	i = 0
	ts = time()
	history = MVHistory()
	y, ynorm = 0.0, 0
	cby = function evalcb(_y)
		i += 1
		y += _y
		ynorm += 1
		if mod(i, steps) == 0
			evaltime = @elapsed acc = accuracy()
			l = y / ynorm
			y, ynorm = 0.0, 0
			push!(history, :loss, i, l)
			for k in keys(acc)
				push!(history, k, i, acc[k])
			end
			push!(history, :time, i, time() - ts)
			println(i,": loss: ", l, " accuracy: ", acc," time per step: ",round((time() - ts)/steps, sigdigits = 2), "s evaluation time: ", round(evaltime, sigdigits = 2),"s")
			!isnothing(resultdir) && !isnothing(model) && @save joinpath(resultdir,"model_$(i).bson") model
			ts = time()
		end
		if 0 < i < 200 && mod(i, 20) == 0
			l = y / ynorm
			y, ynorm = 0.0, 0
			println(i,": loss: ",l, ": time per step: ",round((time() - ts)/i, sigdigits = 2), "s ")

		end
	end
	cby, history
end

# function throttle(cb; steps = typemax(Int), t = typemax(Int))
# 	steps != typemax(Int) && return(throttle_steps(cb, steps = 1000))
# 	t != typemax(Int) && return(throttle_steps(cb, steps = 1000))
# end

function throttle_steps(cb, steps = 1000)
	i = 0
	() -> begin
		i += 1
		if mod(i, steps) == 0
			cb()
		end
	end
end


function bestmodelselector(model, accuracy; showaccuracy::Bool = false)
	best_accuracy = 0.0
	best_model = Ref{Any}(model)

	cb = PrayTools.throttle_steps() do 
		acc = accuracy()
		showaccuracy && println("accuracy = ",acc)
		if acc > best_accuracy
			best_accuracy = acc
			best_model[] = deepcopy(model)
			println("improved best model ", acc)
		end
	end
	cb, best_model
end
