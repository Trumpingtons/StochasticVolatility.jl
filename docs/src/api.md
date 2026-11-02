# API Reference

This page documents the public functions exported by the `StochasticVolatility` module.

## Models

```@docs
Heston
Bates
SABR
```

## Simulation

```@docs
simulate
simulate_terminal
SimulatedPaths
QuadraticExponential
FullTruncation
Euler
PseudoRandom
QuasiRandom
```

## Prices

```@docs
european_price
monte_carlo_price
longstaff_schwartz_price
finite_difference_price
hagan_implied_vol
hagan_price
```

## Price-Based Volatility

```@docs
historical_variance
historical_volatility
parkinson
garman_klass
rogers_satchell
yang_zhang
meilijson
kunitomo
alizadeh_brandt_diebold
```

## Intraday Realized Measures

```@docs
realized_variance
realized_volatility
realized_semivariance
realized_quarticity
bipower_variation
jump_variation
two_scales_realized_variance
realized_kernel
bns_test
ams_noise_test
hurst_exponent
```
