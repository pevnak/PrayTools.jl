function initevalcby(steps; accuracy = () -> (,), resultdir, model)
	i = 0
	ts = time()
	history = MVHistory()
	y, ynorm = 0.0, 0
	cby = function evalcb(_y)
		i += 1
		y += _y
		ynorm += 1
		if mod(i, steps) == 0
			evaltime = @elapsed acc = accuracy(i)
			l = y / ynorm
			y, ynorm = 0.0, 0
			push!(history, :loss, i, l)
			for k in keys(acc)
				push!(history, k, i, acc[v])
			end
			push!(history, :time, i, time() - ts)
			println(i,": loss: ", l, " accuracy: ", acc," time per step: ",round((time() - ts)/steps, sigdigits = 2), "s evaluation time: ", round(evaltime, sigdigits = 2),"s")
			serialize(resultdir("model_$(i).jls"), model)
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
