using PrayTools
using Test
using PrayTools: maskedgrad, signalign

xs = [[1, 1, -1], [1, -1, -1], [1, 1, - 1]]

@test signalign(xs, 0.5) ≈ [1, 0, -1]
@test signalign(xs, 1) ≈ [1, 0, -1]
@test signalign(xs, 0) ≈ [1, 0.3333333333333333, -1]