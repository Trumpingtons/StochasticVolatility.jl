# StochasticVolatility.jl

`StochasticVolatility.jl` is a Julia package that simulates the Heston, Bates and SABR stochastic volatility models, and includes implementations of historical volatility, range-based volatility measures, realized variance estimators, and jump detection tests.

## Features

- `Heston`, `Bates`, `SABR` — the models
- `simulate(model, T; paths, steps, scheme, numbers, threaded)` — simulated paths of the price and of the variance or volatility
- `simulate_terminal(model, T; ...)` — the terminal prices only
- `QuadraticExponential()`, `FullTruncation()` — schemes for Heston and Bates; `Euler()` for SABR
- `PseudoRandom(seed=...)`, `QuasiRandom(seed=..., replicates=...)` — random and quasi-random numbers
- `european_price(model, K, T; rate, call=true)` — closed-form European prices under Heston and Bates
- `monte_carlo_price(model, K, T; rate, call=true, paths=65536, steps=252, scheme, numbers, threaded=true)` — Monte Carlo European prices with a standard error under Heston, Bates and SABR, for one strike or a vector of strikes from one simulation
- `longstaff_schwartz_price(model, K, T; rate, call=true, degree=3, paths=65536, steps=252, scheme, numbers, threaded=true)` — American prices by least-squares Monte Carlo with a standard error under Heston, Bates and SABR, for one strike or a vector of strikes from one simulation
- `hagan_implied_vol(model, K, T)` and `hagan_price(model, K, T; rate, call=true)` — the approximation of Hagan et al. (2002) under SABR

- `historical_volatility(close; scale=1)` — close-to-close historical volatility estimator
- `historical_variance(close; scale=1)` — close-to-close historical variance estimator
- `parkinson(high, low)`
- `garman_klass(open, high, low, close)`
- `rogers_satchell(open, high, low, close)`
- `yang_zhang(open, high, low, close)`
- `meilijson(open, high, low, close)` — Meilijson OHLC volatility estimator
- `kunitomo(bridge_high, bridge_low)` — Kunitomo Brownian-bridge range volatility estimator
- `alizadeh_brandt_diebold(high, low; location=0.43)` — ABD log-range volatility proxy

- `realized_variance(r)` — realized variance from return squares
- `realized_volatility(r)` — square root of realized variance
- `realized_semivariance(r)` — negative and positive realized semivariance
- `realized_quarticity(r)` — realized quarticity estimator
- `bipower_variation(r)` — bipower variation estimator
- `jump_variation(r)` — jump variation estimated from realized variance and bipower variation
- `two_scales_realized_variance(r; k=10)` — two-scales realized variance estimator
- `realized_kernel(r, H)` — realized kernel estimator with Bartlett kernel

- `bns_test(r)` — Barndorff-Nielsen & Shephard jump detection test
- `ams_noise_test`
- `hurst_exponent`

## Installation

```julia
using Pkg
Pkg.add(path="/path/to/StochasticVolatility.jl")
```

## Usage

Simulate a model, here on all the threads Julia was started with (`julia -t auto`). The same seed gives the same paths on any number of threads.

```julia
using StochasticVolatility

model = Heston(S0=100, v0=0.04, mu=0.05, kappa=2, theta=0.04, xi=0.3, rho=-0.7)
result = simulate(model, 1.0; paths=10_000, steps=252, numbers=PseudoRandom(seed=1))
result.time, result.price, result.state  # times, prices and variances, one column per path

# Quasi-random numbers: paths must be replicates times a power of 2
terminal = simulate_terminal(model, 1.0; paths=16 * 2^12, steps=252, numbers=QuasiRandom(seed=1, replicates=16))
```

Measure volatility from data:

```julia
using StochasticVolatility

r = randn(1000) .* 0.001
close = 100 .* exp.(cumsum(r))
println("Historical Variance: ", historical_variance(close))
println("Historical Volatility: ", historical_volatility(close))
println("ABD Volatility Proxy: ", alizadeh_brandt_diebold(close .* 1.001, close .* 0.999))
println("Bipower Variation: ", bipower_variation(r))
println("Realized Variance: ", realized_variance(r))
println("Realized Volatility: ", realized_volatility(r))
println("Realized Semivariance: ", realized_semivariance(r))
println("Realized Quarticity: ", realized_quarticity(r))
println("Jump Variation: ", jump_variation(r))
println("TSRV: ", two_scales_realized_variance(r, k=10))
println("BNS Z-statistic: ", bns_test(r))
println("Realized Kernel: ", realized_kernel(r, Int(floor(sqrt(length(r))))))
```

## Testing

Run the package tests with:

```bash
julia --project=. test/runtests.jl
```

## Documentation

Build the documentation locally with:

```bash
julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
julia --project=docs docs/make.jl
```
