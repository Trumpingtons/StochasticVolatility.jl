# StochasticVolatility.jl

`StochasticVolatility.jl` is a Julia package that includes implementations of historical volatility, range-based volatility measures, realized variance estimators, and jump detection tests.

## Features

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
Pkg.add(path="/path/to/StochasticVolatility")
```

## Usage

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
