module TrainTools

include("threadedgrad.jl")
include("minibatches.jl")
include("callback.jl")

export tgradient, ttrain!, ptrain!, classindexes, initbatchprovider

end # module
