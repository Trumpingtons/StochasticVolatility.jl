"""Supertype of the discretisation schemes: [`FullTruncation`](@ref), [`QuadraticExponential`](@ref) and [`Euler`](@ref)."""
abstract type SimulationScheme end

"""Euler scheme for [`Heston`](@ref) and [`Bates`](@ref) with full truncation of the variance.

The log price follows an Euler step, and negative values of the variance count as zero
in its drift and diffusion.

# Reference
Lord, R., Koekkoek, R. and van Dijk, D. (2010). A comparison of biased simulation
schemes for stochastic volatility models. Quantitative Finance 10(2), 177-194.
doi:10.1080/14697680802392496
"""
struct FullTruncation <: SimulationScheme end

"""Quadratic-exponential scheme of Andersen (2008) for [`Heston`](@ref) and [`Bates`](@ref).

The variance matches the conditional mean and variance of the exact process, and never
goes negative. The price uses the central discretisation of the integrated variance
(`gamma1 = gamma2 = 1/2`). The default scheme for these two models.

# Arguments
- `psi_critical`: where the scheme moves from quadratic to exponential, from 1 to 2.
- `martingale_correction`: whether the drift is set so that the simulated price stays a
  martingale, step by step, net of `mu`. When the moment it needs does not exist, the
  step falls back to the uncorrected drift.

# Reference
Andersen, L. (2008). Simple and efficient simulation of the Heston stochastic volatility
model. Journal of Computational Finance 11(3), 1-42. doi:10.21314/JCF.2008.189
"""
struct QuadraticExponential <: SimulationScheme
    psi_critical::Float64
    martingale_correction::Bool

    function QuadraticExponential(psi_critical::Real, martingale_correction::Bool)
        1 <= psi_critical <= 2 || throw(DomainError(psi_critical, "psi_critical must lie in [1, 2]"))
        return new(psi_critical, martingale_correction)
    end
end

QuadraticExponential(; psi_critical::Real = 1.5, martingale_correction::Bool = true) =
    QuadraticExponential(psi_critical, martingale_correction)

"""Euler scheme for the forward of [`SABR`](@ref), absorbed at zero, with an exact log-normal step for the volatility."""
struct Euler <: SimulationScheme end

_default_scheme(::Union{Heston, Bates}) = QuadraticExponential()
_default_scheme(::SABR) = Euler()

_factors(::Heston) = 2
_factors(::Bates) = 4
_factors(::SABR) = 2

_validate_scheme(::Union{Heston, Bates}, ::Union{FullTruncation, QuadraticExponential}) = nothing
_validate_scheme(::SABR, ::Euler) = nothing
_validate_scheme(model::StochasticVolatilityModel, scheme::SimulationScheme) =
    throw(ArgumentError("$(nameof(typeof(scheme))) does not apply to $(nameof(typeof(model)))"))

"""Simulated paths of a stochastic volatility model.

# Fields
- `time`: the `steps + 1` times, from 0 to the horizon.
- `price`: a matrix with one column per path, the forward price under [`SABR`](@ref).
- `state`: a matrix of the same size, the variance under [`Heston`](@ref) and [`Bates`](@ref),
  the volatility under [`SABR`](@ref).
- `model`, `scheme`, `numbers`: what produced the paths, including the seed.
"""
struct SimulatedPaths{M <: StochasticVolatilityModel, S <: SimulationScheme, N <: NumberSource}
    time::Vector{Float64}
    price::Matrix{Float64}
    state::Matrix{Float64}
    model::M
    scheme::S
    numbers::N
end

_record!(::Nothing, ::Nothing, ::Integer, ::Integer, ::Real, ::Real) = nothing

function _record!(price::Matrix{Float64}, state::Matrix{Float64}, row::Integer, column::Integer, value::Real, second::Real)
    price[row, column] = value
    state[row, column] = second
    return nothing
end

_log_jump(::Heston, Z, i) = 0.0

function _log_jump(model::Bates, Z, i)
    jumps = Z[3, i]
    jumps == 0 && return 0.0
    return jumps * model.muJ + sqrt(jumps) * model.deltaJ * Z[4, i]
end

function _poisson_quantile(u::Real, rate::Real)
    probability = exp(-rate)
    total = probability
    count = 0
    while u > total && probability > 0
        count += 1
        probability *= rate / count
        total += probability
    end
    return count
end

_jump_rate(::Union{Heston, SABR}, ::Real) = 0.0
_jump_rate(model::Bates, dt::Real) = model.lambda * dt

function _coefficients(model::Union{Heston, Bates}, ::FullTruncation, dt::Real)
    return (; dt, drift = _net_drift(model), rho_complement = sqrt(1 - abs2(_diffusion(model).rho)))
end

function _path!(price, state, column, model::Union{Heston, Bates}, ::FullTruncation, c, Z)
    h = _diffusion(model)
    log_price = log(h.S0)
    v = h.v0
    _record!(price, state, 1, column, h.S0, v)
    for i in axes(Z, 2)
        positive = max(v, 0.0)
        root = sqrt(positive * c.dt)
        log_price += (c.drift - positive / 2) * c.dt + root * Z[1, i] + _log_jump(model, Z, i)
        v += h.kappa * (h.theta - positive) * c.dt + h.xi * root * (h.rho * Z[1, i] + c.rho_complement * Z[2, i])
        _record!(price, state, i + 1, column, exp(log_price), max(v, 0.0))
    end
    return exp(log_price)
end

function _coefficients(model::Union{Heston, Bates}, ::QuadraticExponential, dt::Real)
    h = _diffusion(model)
    decay = exp(-h.kappa * dt)
    slope = dt * (h.kappa * h.rho / h.xi - 0.5) / 2
    K0 = -h.rho * h.kappa * h.theta * dt / h.xi
    K1 = slope - h.rho / h.xi
    K2 = slope + h.rho / h.xi
    K3 = dt * (1 - abs2(h.rho)) / 2
    K4 = K3
    return (; dt, drift = _net_drift(model), decay, K0, K1, K2, K3, K4, A = K2 + K4 / 2)
end

function _path!(price, state, column, model::Union{Heston, Bates}, scheme::QuadraticExponential, c, Z)
    h = _diffusion(model)
    log_price = log(h.S0)
    v = h.v0
    _record!(price, state, 1, column, h.S0, v)
    for i in axes(Z, 2)
        mean_next = h.theta + (v - h.theta) * c.decay
        variance_next = abs2(h.xi) * (1 - c.decay) * (v * c.decay + h.theta * (1 - c.decay) / 2) / h.kappa
        psi = variance_next / abs2(mean_next)
        next, log_moment = _quadratic_exponential(mean_next, psi, Z[2, i], scheme.psi_critical, c.A)
        K0 = (scheme.martingale_correction && isfinite(log_moment)) ? -log_moment - (c.K1 + c.K3 / 2) * v : c.K0
        log_price += c.drift * c.dt + K0 + c.K1 * v + c.K2 * next + sqrt(c.K3 * v + c.K4 * next) * Z[1, i] + _log_jump(model, Z, i)
        v = next
        _record!(price, state, i + 1, column, exp(log_price), v)
    end
    return exp(log_price)
end

# The next variance, with log E[exp(A next)], or Inf where that moment does not exist.
function _quadratic_exponential(mean_next::Real, psi::Real, z::Real, psi_critical::Real, A::Real)
    if psi <= psi_critical
        ratio = 2 / psi
        b2 = ratio - 1 + sqrt(ratio * (ratio - 1))
        a = mean_next / (1 + b2)
        room = 1 - 2 * A * a
        log_moment = room > 0 ? A * b2 * a / room - log(room) / 2 : Inf
        return a * abs2(sqrt(b2) + z), log_moment
    end
    p = (psi - 1) / (psi + 1)
    beta = (1 - p) / mean_next
    log_moment = A < beta ? log(p + (1 - p) * beta / (beta - A)) : Inf
    upper_tail = _normal_cdf(-z)
    next = upper_tail >= 1 - p ? 0.0 : log((1 - p) / upper_tail) / beta
    return next, log_moment
end

function _coefficients(model::SABR, ::Euler, dt::Real)
    return (; root_dt = sqrt(dt), rho_complement = sqrt(1 - abs2(model.rho)), volatility_drift = -abs2(model.nu) * dt / 2)
end

function _path!(price, state, column, model::SABR, ::Euler, c, Z)
    F = model.F0
    sigma = model.alpha
    _record!(price, state, 1, column, F, sigma)
    for i in axes(Z, 2)
        if F > 0
            F = max(F + sigma * F^model.beta * c.root_dt * Z[1, i], 0.0)
        end
        sigma *= exp(model.nu * c.root_dt * (model.rho * Z[1, i] + c.rho_complement * Z[2, i]) + c.volatility_drift)
        _record!(price, state, i + 1, column, F, sigma)
    end
    return F
end

function _run_chunk!(terminal, price, state, model, scheme, c, numbers, chunk, factors, steps, bridge, jump_rate)
    Z = Matrix{Float64}(undef, factors, steps)
    source = _source(numbers, chunk, factors, steps, bridge)
    for column in chunk.first_path:chunk.last_path
        _fill_normals!(Z, source, jump_rate)
        terminal[column] = _path!(price, state, column, model, scheme, c, Z)
    end
    return nothing
end

function _run!(terminal, price, state, model, T, steps, paths, scheme, numbers, threaded)
    T > 0 || throw(DomainError(T, "T must be positive"))
    steps >= 1 || throw(ArgumentError("steps must be at least 1"))
    paths >= 1 || throw(ArgumentError("paths must be at least 1"))
    _validate_scheme(model, scheme)
    factors = _factors(model)
    _validate_numbers(numbers, paths, factors, steps)
    c = _coefficients(model, scheme, T / steps)
    jump_rate = _jump_rate(model, T / steps)
    bridge = _bridge_for(numbers, steps)
    chunks = _chunks(numbers, paths)
    if threaded
        Threads.@threads for chunk in chunks
            _run_chunk!(terminal, price, state, model, scheme, c, numbers, chunk, factors, steps, bridge, jump_rate)
        end
    else
        for chunk in chunks
            _run_chunk!(terminal, price, state, model, scheme, c, numbers, chunk, factors, steps, bridge, jump_rate)
        end
    end
    return nothing
end

"""Simulate paths of a stochastic volatility model.

# Arguments
- `model`: a [`Heston`](@ref), [`Bates`](@ref) or [`SABR`](@ref) model.
- `T`: the horizon in years.
- `paths`: the number of paths.
- `steps`: the number of time steps.
- `scheme`: [`QuadraticExponential`](@ref) or [`FullTruncation`](@ref) for Heston and Bates,
  [`Euler`](@ref) for SABR.
- `numbers`: [`PseudoRandom`](@ref) or [`QuasiRandom`](@ref).
- `threaded`: whether to use the threads Julia was started with (`julia -t auto`). The
  result does not depend on the number of threads.

# Returns
- [`SimulatedPaths`](@ref): the times, prices and variances or volatilities of every path.

Keeping every path takes `2 * (steps + 1) * paths` numbers; [`simulate_terminal`](@ref)
keeps only the last price.
"""
function simulate(
    model::StochasticVolatilityModel,
    T::Real;
    paths::Integer = 1000,
    steps::Integer = 252,
    scheme::SimulationScheme = _default_scheme(model),
    numbers::NumberSource = PseudoRandom(),
    threaded::Bool = true,
)
    steps >= 1 || throw(ArgumentError("steps must be at least 1"))
    paths >= 1 || throw(ArgumentError("paths must be at least 1"))
    terminal = Vector{Float64}(undef, paths)
    price = Matrix{Float64}(undef, steps + 1, paths)
    state = Matrix{Float64}(undef, steps + 1, paths)
    _run!(terminal, price, state, model, T, steps, paths, scheme, numbers, threaded)
    return SimulatedPaths(collect(range(0.0, float(T); length = steps + 1)), price, state, model, scheme, numbers)
end

"""Simulate only the terminal prices of a stochastic volatility model.

Takes the same arguments as [`simulate`](@ref), and gives the same terminal prices for the
same `numbers`.

# Returns
- `Vector{Float64}`: the price at the horizon on each path.
"""
function simulate_terminal(
    model::StochasticVolatilityModel,
    T::Real;
    paths::Integer = 1000,
    steps::Integer = 252,
    scheme::SimulationScheme = _default_scheme(model),
    numbers::NumberSource = PseudoRandom(),
    threaded::Bool = true,
)
    paths >= 1 || throw(ArgumentError("paths must be at least 1"))
    terminal = Vector{Float64}(undef, paths)
    _run!(terminal, nothing, nothing, model, T, steps, paths, scheme, numbers, threaded)
    return terminal
end
