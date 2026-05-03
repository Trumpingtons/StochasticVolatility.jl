# StochasticVolatility.jl

## Overview

`StochasticVolatility.jl` is a Julia package that includes implementations of range-based volatility measures, realized variance estimators, and jump detection tests:

- `historical_volatility(close; scale=1)`
- `historical_variance(close; scale=1)`
- `parkinson(high, low)`
- `garman_klass(open, high, low, close)`
- `rogers_satchell(open, high, low, close)`
- `yang_zhang(open, high, low, close)`
- `meilijson(open, high, low, close)`
- `kunitomo(bridge_high, bridge_low)`
- `alizadeh_brandt_diebold(high, low; location=0.43)`
- 
- `realized_variance(r)`
- `realized_volatility(r)`
- `realized_semivariance(r)`
- `realized_quarticity(r)`
- `bipower_variation(r)`
- `jump_variation(r)`
- `two_scales_realized_variance(r; k=10)`
- `realized_kernel(r, H)`
- 
- `bns_test(r)`
- `ams_noise_test(r)`
- `hurst_exponent(rv; q=1.0, max_lag=10)`


## Quick Start

```julia
using StochasticVolatility

r = randn(1000) * 0.001
println("Bipower Variation: ", bipower_variation(r))
println("Realized Variance: ", realized_variance(r))
```

## Price-Based Estimators from OHLC Prices

The following example simulates a Geometric Brownian Motion stock price path, builds synthetic OHLC bars, and compares close-to-close historical volatility with range-based volatility estimators.

```julia
using Random
using StochasticVolatility

Random.seed!(123)
S0 = 100.0
σ = 0.2
μ = 0.05
dt = 1 / 252
n = 252
ε = randn(n)
logret = (μ - 0.5 * σ^2) * dt .+ σ * sqrt(dt) .* ε
close = S0 * exp.(cumsum(logret))
open = [S0; close[1:end-1]]

# Synthetic high/low values from the observed open-close range
high = max.(open, close)
low = min.(open, close)

println("Historical variance: ", historical_variance(close))
println("Historical volatility: ", historical_volatility(close))
println("Meilijson volatility: ", meilijson(open, high, low, close))
println("ABD volatility proxy: ", alizadeh_brandt_diebold(high, low))
println("Parkinson volatility: ", parkinson(high, low))
println("Garman-Klass volatility: ", garman_klass(open, high, low, close))
println("Rogers-Satchell volatility: ", rogers_satchell(open, high, low, close))
println("Yang-Zhang volatility: ", yang_zhang(open, high, low, close))
```

This comparison illustrates the close-to-close baseline and how range-based estimators use more information from OHLC price bars.

Kunitomo's estimator requires the high and low of a detrended log-price Brownian bridge rather than ordinary OHLC high and low prices, so it is documented in the API reference but omitted from this simple OHLC example.

## Realized Measures from Intraday Returns

The following example simulates 30 trading days of intraday stock returns using 10-minute intervals and compares realized variance, realized volatility, semivariance, quarticity, bipower variation, jump variation, two-scales realized variance, and a realized kernel estimate.

```julia
using Random
using StochasticVolatility

Random.seed!(456)
S0 = 100.0
σ = 0.2
μ = 0.0
intervals_per_day = 39  # 6.5 trading hours / 10-minute interval
days = 30
n = intervals_per_day * days

dt = 1.0 / (252 * intervals_per_day)
ε = randn(n)
logret = (μ - 0.5 * σ^2) * dt .+ σ * sqrt(dt) .* ε
r = logret

RV = realized_variance(r)
RVol = realized_volatility(r)
RSV = realized_semivariance(r)
RQ = realized_quarticity(r)
BPV = bipower_variation(r)
JV = jump_variation(r)
TSRV = two_scales_realized_variance(r; k = 10)
H = Int(floor(sqrt(n)))
RK = realized_kernel(r, H)

println("Realized Variance: ", RV)
println("Realized Volatility: ", RVol)
println("Realized Semivariance: ", RSV)
println("Realized Quarticity: ", RQ)
println("Bipower Variation: ", BPV)
println("Jump Variation: ", JV)
println("Two-Scales Realized Variance: ", TSRV)
println("Realized Kernel (H=", H, "): ", RK)
```

This example uses 10-minute returns over 30 trading days to compare estimators built directly from intraday returns. For many applications, 5-to-15 minute intervals are a practical compromise: shorter intervals use more data but may be more sensitive to microstructure noise, while longer intervals reduce noise at the cost of fewer observations.



## Documentation

See the [API Reference](api.md) for the public function list and formulas.
