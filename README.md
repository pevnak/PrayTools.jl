# TrainTools.jl
A collection of routines to simplify boring stuff around training NNs.


```
 ttrain!(loss, ps, () -> prepare_dataset(options), opt, settings.iterations; cb = () -> ())
```
