module StochasticVolatility

include("HighFrequency.jl")
include("RangeBased.jl")
include("Models.jl")
include("RandomNumbers.jl")
include("Simulation.jl")

export Heston,
       Bates,
       SABR,
       FullTruncation,
       QuadraticExponential,
       Euler,
       PseudoRandom,
       QuasiRandom,
       SimulatedPaths,
       simulate,
       simulate_terminal,
       bipower_variation,
       realized_variance,
       realized_volatility,
       realized_semivariance,
       realized_quarticity,
       two_scales_realized_variance,
       realized_kernel,
       jump_variation,
       bns_test,
       ams_noise_test,
       hurst_exponent,
       historical_variance,
       historical_volatility,
       meilijson,
       kunitomo,
       alizadeh_brandt_diebold,
       parkinson,
       garman_klass,
       rogers_satchell,
       yang_zhang

end # module StochasticVolatility
