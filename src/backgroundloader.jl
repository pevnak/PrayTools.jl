"""
    BackgroundDataLoader(loadfun, n)
    BackgroundDataLoader(loadfun, Channel{Any}(n))

    executes function `loadfun()` `n`-times in background. Loaded items
    can be retrieved by `take!`, which is blocking, i.e. if there is no
    minibatch loaded, it will wait till one is available. If a minibatch
    is retrieved, a new call to `loadfun()` is issued. At the moment, there
    is no way to stop, may-be if the structure is destroyed. Alternatively,
    instead of `take!`, one can call it as a functor

    Example of use

```julia
mbprovider() = (randn(2,100), rand(1:2,100))

bdl = BackgroundDataLoader(mbprovider, 10)

for i in 1:30
  x, y = take!(bdl)
  # or
  x, y = bdl()
  #calculate gradient
end


"""
struct BackgroundDataLoader{F,P}
  fun::F 
  c::P
end

function BackgroundDataLoader(loadfun, n::Int)
  c = Channel(n);
  function fun()
    x = loadfun()
    put!(c, x)
  end
  for _ in 1:n 
    Threads.@spawn fun()
  end
  BackgroundDataLoader(fun, c)
end

function Base.take!(pl::BackgroundDataLoader)
  !isready(pl.c) && wait(pl.c)
  o = take!(pl.c)
  Threads.@spawn pl.fun()
  o
end

function (bdl::BackgroundDataLoader)
  take!(bdl)
end


# function TKFDataLoader(loadfun, n::Int)
#     ch = Channel(n) do ch
#       @sync for _ in 1:n
#           @spawn put!(ch, loadfun())
#       end
#       while true
#           wait(@spawn put!(ch, loadfun()))
#       end
#   end
# end
