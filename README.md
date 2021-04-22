# PrayTools.jl
**PrayTools.jl** are former **TrainTools**, that I had to rename because someone has created an internal package of the same name and I was developing a split personality.

**PrayTools.jl** is a collection of routines to simplify boring stuff and my messing around of training NNs, mainly about various versions of distributed training.

### parallel training
```julia
ptrain!(loss, ps, preparesamples, opt, iterations; cb = () -> (), cby = (y) ->(), bs = Threads.nthreads())
ttrain!(loss, ps, preparesamples, opt, iterations; cb = () -> (), cby = (y) ->())
```


performs parallel training assuming that the loss function is additive differing in *where* `preparesamples` is called. In `ptrain!`, a single thread calls `preparesamples` to prepare minibatch, then `dividebatch(bs::Int, xs...)` divides it to `bs` sub-batches, which are then dispatched to a separated thread to calculate gradient. In `ttrain!`, each thread calls `preparesamples` and calculate the gradient immediately. This means that in `ptrain!`, `preparesamples` should return the full minibatch, in `ttrain!` it should return just the sub-batch that would be used by one thread. Both functions use tree-reduction algorithm, such that the complexity of reducing gradients is log(bs). The `iterations` is the number of iterations, after which the loop stops, `cb` is the callback function similar to that of `Flux.train!` and `cby` is a callback taking the output of the loss function as an argument, which is convenient for floating averages.

### Loading data on Background
`BackgroundDataLoader(loadfun, n)` executes function `loadfun()` `n`-times in background. Loaded items can be retrieved by `take!`, which is blocking, i.e. if there is no minibatch loaded, it will wait till one is available. If a minibatch is retrieved, a new call to `loadfun()` is issued. At the moment, there is no way to stop, may-be if the structure is destroyed.

To add loading samples on background, we can do
```julia
preparesamples() = (randn(2,100), rand(1:2,100))
bdl = BackgroundDataLoader(preparesamples, 10)
train!(loss, ps, bdl, opt, iterations)
```

Just do not forget to run julia with more then single threads.

### PrayTools.initevalcby
initializes a very simple callback function
```
cby, history = PrayTools.initevalcby(;accuracy = () -> accuracy(model))
```
