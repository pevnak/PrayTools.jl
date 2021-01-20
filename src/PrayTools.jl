module PrayTools

using Flux
using Zygote
using ValueHistories
using Statistics
using ThreadPools

include("threadedgrad.jl")
include("andmask.jl")
include("train.jl")
include("minibatches.jl")
include("callback.jl")

export tgradient, ttrain!, ptrain!, classindexes, initbatchprovider

end # module
