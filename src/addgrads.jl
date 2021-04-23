function addgrad!(x::AbstractArray, y::AbstractArray)
  x .+= y
  x
end

function addgrad!(gs1::NamedTuple, gs2::NamedTuple)
  ks = keys(gs1)
  for p in ks
    if gs1[p] != nothing && gs2[p] != nothing 
        addgrad!(gs1[p], gs2[p])
    elseif gs2[p] != nothing
          gs1[p] = gs2[p]
      end
  end
  gs1
end

function addgrad!(gs1, gs2, ps)
  for p in ps 
      if gs1[p] != nothing && gs2[p] != nothing 
          addgrad!(gs1[p], gs2[p])
      elseif gs2[p] != nothing
          gs1.grads[p] = gs2[p]
      end
  end
  gs1
end

function addgrad!(gs1::Zygote.Grads, gs2::Zygote.Grads, ps::Zygote.Params)
  for p in ps 
      if gs1[p] != nothing && gs2[p] != nothing 
          addgrad!(gs1[p], gs2[p])
      elseif gs2[p] != nothing
          gs1.grads[p] = gs2[p]
      end
  end
  gs1
end

function addgrad!(gs1::Tuple{T,Zygote.Grads}, gs2::Tuple{T,Zygote.Grads}, ps::Zygote.Params) where {T}
    (gs1[1] + gs2[1], addgrad!(gs1[2], gs2[2], ps))
end

function normalize!(gs, ps, n::Int)
  for p in ps
      isnothing(gs[p]) && continue
      normalize!(gs[p], n)
  end
end

normalize!(x::AbstractArray, n) = x ./= n
normalize!(x::NamedTuple, n) = normalize!(x, keys(x), n)

function gradinfo(gs)
  gs = filter(x -> isa(x, Array),  collect(values(gs.grads)))
  (nan = mapreduce(x -> sum(isnan.(x)), +,  gs),
    inf = mapreduce(x -> sum(isinf.(x)), +, gs),
    max = mapreduce(x -> maximum(abs.(x)), max, gs),
    l2 = mapreduce(x -> sum(x.^2), +, gs),
    )
end
