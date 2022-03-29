module PrayTools

using Flux
using Zygote
using ValueHistories
using Statistics
using ThreadPools
using LearnBase

include("ffnn.jl")
export ffnn

# include("backgroundloader.jl")
# export BackgroundDataLoader

include("addgrads.jl")
include("threadedgrad.jl")
export tgradient, ttrain!, ptrain!

include("andmask.jl")
include("train.jl")
export trainy
include("minibatches.jl")
include("callback.jl")
export initevalcby

export classindexes, initbatchprovider

end # module
