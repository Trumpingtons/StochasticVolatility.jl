using LinearAlgebra: ColumnNorm, qr
using QuadGK: quadgk

# Log characteristic function of log(S(T) / F), F the forward, in the form of Albrecher,
# Mayer, Schoutens and Tistaert (2007), whose principal complex logarithm stays continuous.
function _log_characteristic(model::Heston, z::Complex, T::Real)
    xi2 = abs2(model.xi)
    b = model.kappa - im * model.rho * model.xi * z
    d = sqrt(b * b + xi2 * (z * z + im * z))
    minus = b - d
    g = minus / (b + d)
    decay = exp(-d * T)
    C = model.kappa * model.theta / xi2 * (minus * T - 2 * log((1 - g * decay) / (1 - g)))
    D = minus / xi2 * (1 - decay) / (1 - g * decay)
    return C + D * model.v0
end

function _log_characteristic(model::Bates, z::Complex, T::Real)
    jump = exp(im * z * model.muJ - z * z * abs2(model.deltaJ) / 2) - 1 - im * z * _mean_jump(model)
    return _log_characteristic(model.diffusion, z, T) + T * model.lambda * jump
end

function _black_call(F::Real, K::Real, total_variance::Real)
    root = sqrt(total_variance)
    d1 = log(F / K) / root + root / 2
    return F * _normal_cdf(d1) - K * _normal_cdf(d1 - root)
end

"""Closed-form price of a European option under [`Heston`](@ref) or [`Bates`](@ref).

The price is a single integral of the characteristic function, in the form of Lewis, taken
numerically. The integrand is the difference from a Black price at the average expected
variance, which is added back, so that it decays quickly at short maturities too.

# Arguments
- `model`: a [`Heston`](@ref) or [`Bates`](@ref) model.
- `K`: the strike.
- `T`: the maturity in years.
- `rate`: the risk-free rate that discounts the payoff. The model's `mu` is this rate
  minus the dividend yield.
- `call`: `true` for a call, `false` for a put.

# Returns
- `Float64`: the price.

# References
Lewis, A. L. (2000). Option Valuation under Stochastic Volatility: With Mathematica Code.
Finance Press.

Albrecher, H., Mayer, P., Schoutens, W. and Tistaert, J. (2007). The little Heston trap.
Wilmott 2007(1), 83-92.
"""
function european_price(model::Union{Heston, Bates}, K::Real, T::Real; rate::Real, call::Bool = true)
    K > 0 || throw(DomainError(K, "K must be positive"))
    T > 0 || throw(DomainError(T, "T must be positive"))
    h = _diffusion(model)
    F = h.S0 * exp(h.mu * T)
    log_moneyness = log(F / K)
    total_variance = h.theta * T + (h.v0 - h.theta) * (-expm1(-h.kappa * T)) / h.kappa
    function integrand(u)
        w = abs2(u) + 0.25
        difference = exp(-total_variance * w / 2) - exp(_log_characteristic(model, complex(u, -0.5), T))
        return real(cis(u * log_moneyness) * difference) / w
    end
    integral, _ = quadgk(integrand, 0.0, Inf; rtol = 1e-12, atol = 1e-14)
    forward_call = _black_call(F, K, total_variance) + sqrt(F * K) / pi * integral
    return exp(-rate * T) * (call ? forward_call : forward_call - F + K)
end

_mean(values) = sum(values) / length(values)

function _standard_error(values, ::PseudoRandom)
    center = _mean(values)
    return sqrt(sum(value -> abs2(value - center), values) / (length(values) - 1) / length(values))
end

# Replicates are independent, while the paths within one are not.
function _standard_error(values, numbers::QuasiRandom)
    per_replicate = length(values) ÷ numbers.replicates
    means = [_mean(view(values, ((replicate - 1) * per_replicate + 1):(replicate * per_replicate))) for replicate in 1:numbers.replicates]
    return _standard_error(means, PseudoRandom(0))
end

"""Monte Carlo price of a European option under any of the models.

# Arguments
- `model`: a [`Heston`](@ref), [`Bates`](@ref) or [`SABR`](@ref) model.
- `K`: the strike, or a vector of strikes, all priced from one simulation.
- `T`: the maturity in years.
- `rate`: the risk-free rate that discounts the payoff.
- `call`: `true` for calls, `false` for puts.
- `paths`, `steps`, `scheme`, `numbers`, `threaded`: as in [`simulate`](@ref). `paths` must
  be at least 2.

# Returns
- A named tuple `(; price, standard_error)`, or a vector of them for a vector of strikes,
  in the same order. The standard error comes from the spread of
  the paths under [`PseudoRandom`](@ref), and of the replicate means under
  [`QuasiRandom`](@ref). It measures the sampling error only, not the bias of the scheme.
"""
monte_carlo_price(model::StochasticVolatilityModel, K::Real, T::Real; arguments...) =
    only(monte_carlo_price(model, [K], T; arguments...))

function monte_carlo_price(
    model::StochasticVolatilityModel,
    K::AbstractVector{<:Real},
    T::Real;
    rate::Real,
    call::Bool = true,
    paths::Integer = 65536,
    steps::Integer = 252,
    scheme::SimulationScheme = _default_scheme(model),
    numbers::NumberSource = PseudoRandom(),
    threaded::Bool = true,
)
    all(>(0), K) || throw(DomainError(K, "K must be positive"))
    paths >= 2 || throw(ArgumentError("paths must be at least 2"))
    terminal = simulate_terminal(model, T; paths, steps, scheme, numbers, threaded)
    discount = exp(-rate * T)
    return map(K) do strike
        payoffs = [max(call ? price - strike : strike - price, 0.0) for price in terminal]
        return (; price = discount * _mean(payoffs), standard_error = discount * _standard_error(payoffs, numbers))
    end
end

# The terms of total degree up to `degree` in x and y: 1, x, y, x^2, x y, y^2, ...
function _regressors(x::AbstractVector, y::AbstractVector, degree::Integer)
    X = Matrix{Float64}(undef, length(x), (degree + 1) * (degree + 2) ÷ 2)
    column = 0
    for total in 0:degree, power in 0:total
        column += 1
        X[:, column] .= x .^ (total - power) .* y .^ power
    end
    return X
end

function _longstaff_schwartz(paths::SimulatedPaths, K::Real, rate::Real, call::Bool, degree::Integer)
    (; time, price, state) = paths
    exercise(value) = max(call ? value - K : K - value, 0.0)
    step_discount = exp(-rate * (time[2] - time[1]))
    terms = (degree + 1) * (degree + 2) ÷ 2
    values = exercise.(price[end, :])
    for row in (lastindex(time) - 1):-1:2
        values .*= step_discount
        candidates = findall(column -> exercise(price[row, column]) > 0, axes(price, 2))
        length(candidates) > terms || continue
        second = state[row, candidates]
        scale = _mean(second)
        X = _regressors(price[row, candidates] ./ K, scale > 0 ? second ./ scale : second, degree)
        continuation = X * (qr(X, ColumnNorm()) \ values[candidates])
        for (index, column) in enumerate(candidates)
            now = exercise(price[row, column])
            if now > continuation[index]
                values[column] = now
            end
        end
    end
    values .*= step_discount
    immediate = exercise(price[1, 1])
    center = _mean(values)
    immediate >= center && return (; price = immediate, standard_error = 0.0)
    return (; price = center, standard_error = _standard_error(values, paths.numbers))
end

"""American price under any of the models, by the least-squares Monte Carlo of Longstaff and Schwartz.

Exercise is possible at time 0 and at the end of every step, so the price is that of a
Bermudan option with `steps` dates after time 0, which approaches the American price as
`steps` grows. At each date, the value of continuing is regressed, on the paths in the
money, on polynomials in the price and in the variance or volatility. Under [`SABR`](@ref)
the option is on the forward.

# Arguments
- `model`: a [`Heston`](@ref), [`Bates`](@ref) or [`SABR`](@ref) model.
- `K`: the strike, or a vector of strikes, all priced from one simulation.
- `T`: the maturity in years.
- `rate`: the risk-free rate that discounts the payoff.
- `call`: `true` for calls, `false` for puts.
- `degree`: the highest total degree of the polynomials, at least 1. The default of 3 gives
  10 terms.
- `paths`, `steps`, `scheme`, `numbers`, `threaded`: as in [`simulate`](@ref). `paths` must
  be at least 2. Every path is kept during the regressions, as in [`simulate`](@ref).

# Returns
- A named tuple `(; price, standard_error)`, or a vector of them for a vector of strikes,
  in the same order. Where exercising at time 0 is worth at least the simulated value, the
  price is that exercise value, with a standard error of 0. The standard error measures
  the sampling error with the exercise rule held fixed. It leaves out the bias of the
  scheme, the bias from the limited dates and polynomials, which lowers the price, and the
  bias from fitting the rule on the paths it then prices, which raises it.

# Reference
Longstaff, F. A. and Schwartz, E. S. (2001). Valuing American options by simulation: a
simple least-squares approach. Review of Financial Studies 14(1), 113-147.
doi:10.1093/rfs/14.1.113
"""
longstaff_schwartz_price(model::StochasticVolatilityModel, K::Real, T::Real; arguments...) =
    only(longstaff_schwartz_price(model, [K], T; arguments...))

function longstaff_schwartz_price(
    model::StochasticVolatilityModel,
    K::AbstractVector{<:Real},
    T::Real;
    rate::Real,
    call::Bool = true,
    degree::Integer = 3,
    paths::Integer = 65536,
    steps::Integer = 252,
    scheme::SimulationScheme = _default_scheme(model),
    numbers::NumberSource = PseudoRandom(),
    threaded::Bool = true,
)
    all(>(0), K) || throw(DomainError(K, "K must be positive"))
    paths >= 2 || throw(ArgumentError("paths must be at least 2"))
    degree >= 1 || throw(ArgumentError("degree must be at least 1"))
    simulated = simulate(model, T; paths, steps, scheme, numbers, threaded)
    return map(strike -> _longstaff_schwartz(simulated, strike, rate, call, degree), K)
end

# z / x(z) of Hagan et al. (2002), with its series near z = 0, where the ratio is 0 / 0.
function _hagan_ratio(z::Real, rho::Real)
    abs(z) < 1e-6 && return 1 - rho * z / 2 + (2 - 3 * abs2(rho)) * abs2(z) / 12
    y = abs2(z) - 2 * rho * z
    return z / log1p((y / (sqrt(1 + y) + 1) + z) / (1 - rho))
end

"""Black implied volatility of a European option under [`SABR`](@ref), by the formula of Hagan et al.

An asymptotic approximation, not the exact implied volatility of the model. At small
strikes and long maturities it can give wrong prices and imply a negative density, as
Obloj (2008) notes.

# Arguments
- `model`: a [`SABR`](@ref) model.
- `K`: the strike.
- `T`: the maturity in years.

# Returns
- `Float64`: the implied volatility.

# References
Hagan, P. S., Kumar, D., Lesniewski, A. S. and Woodward, D. E. (2002). Managing smile risk.
Wilmott Magazine, 84-108.

Obloj, J. (2008). Fine-tune your smile: Correction to Hagan et al. arXiv:0708.0998
"""
function hagan_implied_vol(model::SABR, K::Real, T::Real)
    K > 0 || throw(DomainError(K, "K must be positive"))
    T > 0 || throw(DomainError(T, "T must be positive"))
    (; F0, alpha, beta, rho, nu) = model
    complement = 1 - beta
    scale = (F0 * K)^(complement / 2)
    log_moneyness = log(F0 / K)
    denominator = 1 + abs2(complement * log_moneyness) / 24 + (complement * log_moneyness)^4 / 1920
    correction = abs2(complement * alpha / scale) / 24 + rho * beta * nu * alpha / scale / 4 + (2 - 3 * abs2(rho)) * abs2(nu) / 24
    z = nu / alpha * scale * log_moneyness
    return alpha / (scale * denominator) * _hagan_ratio(z, rho) * (1 + correction * T)
end

"""Price of a European option under [`SABR`](@ref): the Black price at [`hagan_implied_vol`](@ref).

An approximation, for the reasons given there.

# Arguments
- `model`: a [`SABR`](@ref) model.
- `K`: the strike.
- `T`: the maturity in years.
- `rate`: the risk-free rate that discounts the payoff.
- `call`: `true` for a call, `false` for a put.

# Returns
- `Float64`: the price.
"""
function hagan_price(model::SABR, K::Real, T::Real; rate::Real, call::Bool = true)
    forward_call = _black_call(model.F0, K, abs2(hagan_implied_vol(model, K, T)) * T)
    return exp(-rate * T) * (call ? forward_call : forward_call - model.F0 + K)
end
