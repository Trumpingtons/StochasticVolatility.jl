"""Supertype of the stochastic volatility models: [`Heston`](@ref), [`Bates`](@ref) and [`SABR`](@ref)."""
abstract type StochasticVolatilityModel end

function _require_model_positive(value::Real, name::AbstractString)
    value > 0 || throw(DomainError(value, "$name must be positive"))
    return nothing
end

function _require_model_nonnegative(value::Real, name::AbstractString)
    value >= 0 || throw(DomainError(value, "$name must not be negative"))
    return nothing
end

function _require_correlation(value::Real, name::AbstractString)
    -1 <= value <= 1 || throw(DomainError(value, "$name must lie in [-1, 1]"))
    return nothing
end

"""Heston (1993) stochastic volatility model.

```
dS = mu S dt + sqrt(v) S dW1
dv = kappa (theta - v) dt + xi sqrt(v) dW2,    corr(dW1, dW2) = rho
```

# Arguments
- `S0`: initial price.
- `v0`: initial variance (not volatility).
- `mu`: risk-neutral drift, the risk-free rate minus the continuous dividend yield.
- `kappa`: mean-reversion speed of the variance.
- `theta`: long-run mean of the variance.
- `xi`: volatility of the variance.
- `rho`: correlation between the price and variance Brownian motions.

# Reference
Heston, S. L. (1993). A closed-form solution for options with stochastic volatility
with applications to bond and currency options. Review of Financial Studies 6(2),
327-343. doi:10.1093/rfs/6.2.327
"""
struct Heston <: StochasticVolatilityModel
    S0::Float64
    v0::Float64
    mu::Float64
    kappa::Float64
    theta::Float64
    xi::Float64
    rho::Float64

    function Heston(S0::Real, v0::Real, mu::Real, kappa::Real, theta::Real, xi::Real, rho::Real)
        _require_model_positive(S0, "S0")
        _require_model_nonnegative(v0, "v0")
        _require_model_positive(kappa, "kappa")
        _require_model_positive(theta, "theta")
        _require_model_positive(xi, "xi")
        _require_correlation(rho, "rho")
        return new(S0, v0, mu, kappa, theta, xi, rho)
    end
end

Heston(; S0::Real, v0::Real, mu::Real, kappa::Real, theta::Real, xi::Real, rho::Real) =
    Heston(S0, v0, mu, kappa, theta, xi, rho)

"""Bates (1996) model: the [`Heston`](@ref) model with compound Poisson log-normal jumps in the price.

```
dS / S = (mu - lambda m) dt + sqrt(v) dW1 + (J - 1) dN,    log J ~ Normal(muJ, deltaJ^2)
m = exp(muJ + deltaJ^2 / 2) - 1
```

The drift is compensated by `lambda m`, so `E[S(T)] = S0 exp(mu T)` as under Heston.

# Arguments
- `S0`, `v0`, `mu`, `kappa`, `theta`, `xi`, `rho`: as in [`Heston`](@ref).
- `lambda`: jump intensity, the expected number of jumps per year.
- `muJ`: mean of the log jump.
- `deltaJ`: standard deviation of the log jump.

# Reference
Bates, D. S. (1996). Jumps and stochastic volatility: exchange rate processes implicit
in Deutsche Mark options. Review of Financial Studies 9(1), 69-107.
doi:10.1093/rfs/9.1.69
"""
struct Bates <: StochasticVolatilityModel
    diffusion::Heston
    lambda::Float64
    muJ::Float64
    deltaJ::Float64

    function Bates(diffusion::Heston, lambda::Real, muJ::Real, deltaJ::Real)
        _require_model_nonnegative(lambda, "lambda")
        _require_model_nonnegative(deltaJ, "deltaJ")
        return new(diffusion, lambda, muJ, deltaJ)
    end
end

Bates(; S0::Real, v0::Real, mu::Real, kappa::Real, theta::Real, xi::Real, rho::Real, lambda::Real, muJ::Real, deltaJ::Real) =
    Bates(Heston(S0, v0, mu, kappa, theta, xi, rho), lambda, muJ, deltaJ)

"""SABR model of Hagan, Kumar, Lesniewski and Woodward (2002), for a forward price `F`.

```
dF = sigma F^beta dW1
dsigma = nu sigma dW2,    corr(dW1, dW2) = rho
```

# Arguments
- `F0`: initial forward price.
- `alpha`: initial stochastic volatility `sigma(0)`.
- `beta`: CEV exponent, from 0 (normal) to 1 (log-normal).
- `rho`: correlation between the forward and volatility Brownian motions.
- `nu`: volatility of the volatility.

# Reference
Hagan, P. S., Kumar, D., Lesniewski, A. S. and Woodward, D. E. (2002). Managing smile
risk. Wilmott Magazine.
"""
struct SABR <: StochasticVolatilityModel
    F0::Float64
    alpha::Float64
    beta::Float64
    rho::Float64
    nu::Float64

    function SABR(F0::Real, alpha::Real, beta::Real, rho::Real, nu::Real)
        _require_model_positive(F0, "F0")
        _require_model_positive(alpha, "alpha")
        0 <= beta <= 1 || throw(DomainError(beta, "beta must lie in [0, 1]"))
        _require_correlation(rho, "rho")
        _require_model_nonnegative(nu, "nu")
        return new(F0, alpha, beta, rho, nu)
    end
end

SABR(; F0::Real, alpha::Real, beta::Real, rho::Real, nu::Real) = SABR(F0, alpha, beta, rho, nu)

_diffusion(model::Heston) = model
_diffusion(model::Bates) = model.diffusion

_mean_jump(model::Bates) = expm1(model.muJ + abs2(model.deltaJ) / 2)

# Drift of the log price before the variance term, net of the jump compensator.
_net_drift(model::Heston) = model.mu
_net_drift(model::Bates) = model.diffusion.mu - model.lambda * _mean_jump(model)
