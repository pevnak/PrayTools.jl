# PrayTools.jl
PrayTools.jl are former TrainTools, that I have renamed because someone has created an internal package of the same name and I got afraid of schizophrenia.

A collection of routines to simplify boring stuff around training NNs.


Currently the main tools are around multi-threaded calculation of gradient. The first function is a variation of a `Flux.train!`, where the data parameter is a function (instead of an iterator) which provides a minibatch for every invokation. ttrain! executes this minibatchprovider in every thread and then calculates a gradient on it. 
```
minibatchprovider() = prepare_minibatch(options)
 ttrain!(loss, ps, minibatchprovider, opt, settings.iterations; cb = () -> ())
```


Alternative is `ptrain` where a `minibatchprovider` is invoked once out of the multithreading. It is assumed that in this case minibatchprovider() returns a tuple of minibatches, and gradient of each minibatch in the tuple is executed on a thread.

Assuming the `prepeare_dataset
```
minibatchprovider() = prepare_minibatch(options)
 ttrain!(loss, ps, minibatchprovider, opt, settings.iterations; cb = () -> ())
```
