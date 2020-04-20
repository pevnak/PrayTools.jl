module TrainTools

include("threadedgrad.jl")
include("minibatches.jl")

export tgradient, ttrain!, ptrain!, classindexes, initbatchprovider

end # module
