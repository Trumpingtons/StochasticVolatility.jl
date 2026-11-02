using LinearAlgebra: mul!

# The mesh in the price of Section 2 of Haentjens and in 't Hout (2014): uniform on
# [left, right], which holds the strike and the exercise boundary, and sinh outside it.
function _price_mesh(K::Real, left::Real, right::Real, points::Integer, top::Real = 14K)
    d = K / 20
    lowest = asinh(-left / d)
    inner = (right - left) / d
    highest = inner + asinh((top - right) / d)
    mesh = map(range(lowest, highest; length = points + 1)) do z
        z < 0 ? left + d * sinh(z) : z <= inner ? left + d * z : right + d * sinh(z - inner)
    end
    mesh[1] = 0.0
    mesh[end] = top
    return mesh
end

# The mesh in the variance of the same section, with many points near the boundary v = 0.
function _variance_mesh(points::Integer)
    largest = 5.0
    d = largest / 500
    return d .* sinh.(asinh(largest / d) .* (0:points) ./ points)
end

# The payoff on the mesh, with the point nearest the strike replaced by its cell average.
function _payoff_on_mesh(s::AbstractVector, K::Real, call::Bool)
    payoff = call ? max.(s .- K, 0.0) : max.(K .- s, 0.0)
    nearest = argmin(abs.(s .- K))
    low, high = (s[nearest-1] + s[nearest]) / 2, (s[nearest] + s[nearest+1]) / 2
    payoff[nearest] = abs2(call ? max(high - K, 0.0) : max(K - low, 0.0)) / (2 * (high - low))
    return payoff
end

# A tridiagonal matrix per line, and its factors for I - weight * matrix.
struct _Lines{T<:AbstractArray}
    lower::T
    diagonal::T
    upper::T
    denominator::T
    ratio::T
end

function _factor(lower::AbstractArray, diagonal::AbstractArray, upper::AbstractArray, weight::Real)
    denominator, ratio = similar(diagonal), similar(diagonal)
    points = size(diagonal, 1)
    for line in axes(diagonal, 2)
        denominator[1, line] = 1 - weight * diagonal[1, line]
        ratio[1, line] = -weight * upper[1, line] / denominator[1, line]
        for i in 2:points
            denominator[i, line] = 1 - weight * diagonal[i, line] + weight * lower[i, line] * ratio[i-1, line]
            ratio[i, line] = -weight * upper[i, line] / denominator[i, line]
        end
    end
    return _Lines(lower, diagonal, upper, denominator, ratio)
end

# out = A1 U, along the price.
function _apply_price!(out::Matrix, A::_Lines, U::Matrix)
    points = size(U, 1)
    @inbounds for j in axes(U, 2)
        out[1, j] = 0.0
        for i in 2:points-1
            out[i, j] = A.lower[i, j] * U[i-1, j] + A.diagonal[i, j] * U[i, j] + A.upper[i, j] * U[i+1, j]
        end
        out[points, j] = A.lower[points, j] * U[points-1, j] + A.diagonal[points, j] * U[points, j]
    end
    return out
end

# out = A2 U, along the variance, the same on every line of the price after the first.
function _apply_variance!(out::Matrix, A::_Lines, U::Matrix)
    columns = size(U, 2)
    @inbounds for j in 1:columns
        out[1, j] = 0.0
        low, middle, high = A.lower[j, 1], A.diagonal[j, 1], A.upper[j, 1]
        for i in 2:size(U, 1)
            value = middle * U[i, j]
            j > 1 && (value += low * U[i, j-1])
            j < columns && (value += high * U[i, j+1])
            out[i, j] = value
        end
    end
    return out
end

# out = A0 U, the mixed derivative on its 9-point stencil, zero on the boundaries.
function _apply_mixed!(out::Matrix, price_weights::Matrix, variance_weights::Matrix, scale::Matrix, U::Matrix)
    fill!(out, 0.0)
    @inbounds for j in 2:size(U, 2)-1, i in 2:size(U, 1)-1
        total = 0.0
        for l in -1:1, k in -1:1
            total += price_weights[k+2, i] * variance_weights[l+2, j] * U[i+k, j+l]
        end
        out[i, j] = scale[i, j] * total
    end
    return out
end

# Solves (I - weight A1) out = right along the price, in place.
function _solve_price!(right::Matrix, A::_Lines, weight::Real)
    points = size(right, 1)
    @inbounds for j in axes(right, 2)
        right[1, j] /= A.denominator[1, j]
        for i in 2:points
            right[i, j] = (right[i, j] + weight * A.lower[i, j] * right[i-1, j]) / A.denominator[i, j]
        end
        for i in points-1:-1:1
            right[i, j] -= A.ratio[i, j] * right[i+1, j]
        end
    end
    return right
end

# Solves (I - weight A2) out = right along the variance, in place, leaving the first line of the price.
function _solve_variance!(right::Matrix, A::_Lines, weight::Real)
    columns = size(right, 2)
    @inbounds begin
        for i in 2:size(right, 1)
            right[i, 1] /= A.denominator[1, 1]
        end
        for j in 2:columns, i in 2:size(right, 1)
            right[i, j] = (right[i, j] + weight * A.lower[j, 1] * right[i, j-1]) / A.denominator[j, 1]
        end
        for j in columns-1:-1:1, i in 2:size(right, 1)
            right[i, j] -= A.ratio[j, 1] * right[i, j+1]
        end
    end
    return right
end

# The weights of the central first derivative on a nonuniform mesh, per interior point.
function _central_weights(mesh::AbstractVector)
    weights = zeros(3, length(mesh))
    for i in 2:length(mesh)-1
        before, after = mesh[i] - mesh[i-1], mesh[i+1] - mesh[i]
        weights[1, i] = -after / (before * (before + after))
        weights[2, i] = (after - before) / (before * after)
        weights[3, i] = before / (after * (before + after))
    end
    return weights
end

# The jump integral of Bates on the price mesh: its weights, with the value between two mesh
# points linear in the price, and beyond the top of the mesh linear with the given slope. The
# integral over the piece from a to b is exact under the log-normal law of the jump.
_jump_weights(::Heston, s::AbstractVector, slope::Real) = nothing

function _jump_weights(model::Bates, s::AbstractVector, slope::Real)
    model.lambda > 0 || return nothing
    rows = length(s)
    weights, beyond = zeros(rows, rows), zeros(rows)
    mean_jump = exp(model.muJ + abs2(model.deltaJ) / 2)
    for i in 2:rows
        score(x) = (log(x / s[i]) - model.muJ) / model.deltaJ
        for k in 1:rows-1
            a, b = s[k], s[k+1]
            low, high = a > 0 ? score(a) : -Inf, score(b)
            probability = _normal_cdf(high) - _normal_cdf(low)
            expectation = s[i] * mean_jump * (_normal_cdf(high - model.deltaJ) - _normal_cdf(low - model.deltaJ))
            weights[i, k] += (b * probability - expectation) / (b - a)
            weights[i, k+1] += (expectation - a * probability) / (b - a)
        end
        top = score(s[rows])
        weights[i, rows] += _normal_cdf(-top)
        beyond[i] = slope * (s[i] * mean_jump * _normal_cdf(model.deltaJ - top) - s[rows] * _normal_cdf(-top))
    end
    return (; weights, beyond)
end

# out += lambda (integral of U - U), explicit in time like the mixed derivative, zero at s = 0.
_add_jump!(out::Matrix, ::Bates, ::Nothing, U::Matrix, work::Matrix) = out
_add_jump!(out::Matrix, ::Heston, ::Nothing, U::Matrix, work::Matrix) = out

function _add_jump!(out::Matrix, model::Bates, jump::NamedTuple, U::Matrix, work::Matrix)
    mul!(work, jump.weights, U)
    @inbounds for j in axes(U, 2), i in 2:size(U, 1)
        out[i, j] += model.lambda * (work[i, j] - U[i, j])
    end
    return out
end

# The values at time 0 on the meshes, by the MCS-IT method of Haentjens and in 't Hout (2014),
# with the jump integral of Bates in the explicit part of the scheme.
function _finite_difference_surface(model::Union{Heston, Bates}, K::Real, T::Real, rate::Real, call::Bool, price_points::Integer, variance_points::Integer, steps::Integer)
    left, right = call ? (K, min(2.0, exp(T / 4)) * K) : (max(0.5, exp(-T / 4)) * K, K)
    s = _price_mesh(K, left, right, price_points)
    v = _variance_mesh(variance_points)
    columns, rows = length(v), length(s)
    dt = T / steps
    theta = 1 / 3
    slope = call ? 1.0 : 0.0
    base = _diffusion(model)
    net_drift = _net_drift(model)

    # Along the price, with half of the discounting.
    lower1, diagonal1, upper1 = zeros(rows, columns), zeros(rows, columns), zeros(rows, columns)
    boundary = zeros(rows, columns)
    for j in 1:columns
        for i in 2:rows-1
            before, after = s[i] - s[i-1], s[i+1] - s[i]
            diffusion = abs2(s[i]) * v[j] / 2
            lower1[i, j] = diffusion * 2 / (before * (before + after))
            upper1[i, j] = diffusion * 2 / (after * (before + after))
            diagonal1[i, j] = -diffusion * 2 / (before * after) - rate / 2
            drift = net_drift * s[i]
            if drift >= 0
                upper1[i, j] += drift / after
                diagonal1[i, j] -= drift / after
            else
                lower1[i, j] -= drift / before
                diagonal1[i, j] += drift / before
            end
        end
        last_step = s[rows] - s[rows-1]
        curvature = abs2(s[rows]) * v[j] / abs2(last_step)
        lower1[rows, j] = curvature
        diagonal1[rows, j] = -curvature - rate / 2
        boundary[rows, j] = (curvature * last_step + net_drift * s[rows]) * slope
    end

    # Along the variance, with the other half.
    lower2, diagonal2, upper2 = zeros(columns, 1), zeros(columns, 1), zeros(columns, 1)
    for j in 2:columns-1
        before, after = v[j] - v[j-1], v[j+1] - v[j]
        diffusion = abs2(base.xi) * v[j] / 2
        lower2[j] = diffusion * 2 / (before * (before + after))
        upper2[j] = diffusion * 2 / (after * (before + after))
        diagonal2[j] = -diffusion * 2 / (before * after) - rate / 2
        drift = base.kappa * (base.theta - v[j])
        if v[j] <= base.theta
            upper2[j] += drift / after
            diagonal2[j] -= drift / after
        else
            lower2[j] -= drift / before
            diagonal2[j] += drift / before
        end
    end
    upper2[1] = base.kappa * base.theta / (v[2] - v[1])
    diagonal2[1] = -upper2[1] - rate / 2
    lower2[columns] = abs2(base.xi) * v[columns] / abs2(v[columns] - v[columns-1])
    diagonal2[columns] = -lower2[columns] - rate / 2

    A1 = _factor(lower1, diagonal1, upper1, theta * dt)
    A2 = _factor(lower2, diagonal2, upper2, theta * dt)
    price_weights, variance_weights = _central_weights(s), _central_weights(v)
    mixed_scale = [base.rho * base.xi * s[i] * v[j] for i in 1:rows, j in 1:columns]

    payoff = _payoff_on_mesh(s, K, call) * ones(columns)'
    U = copy(payoff)
    multiplier = zeros(rows, columns)
    A0U, A1U, A2U = similar(U), similar(U), similar(U)
    Y0, Y, difference, A0D, A1D, A2D = similar(U), similar(U), similar(U), similar(U), similar(U), similar(U)
    jump = _jump_weights(model, s, slope)
    jump === nothing || (boundary .+= model.lambda .* jump.beyond)
    work = similar(U)
    for _ in 1:steps
        _apply_mixed!(A0U, price_weights, variance_weights, mixed_scale, U)
        _add_jump!(A0U, model, jump, U, work)
        _apply_price!(A1U, A1, U)
        _apply_variance!(A2U, A2, U)
        @. Y0 = U + dt * (A0U + A1U + A2U + boundary + multiplier)

        @. Y = Y0 - theta * dt * A1U
        _solve_price!(Y, A1, theta * dt)
        @. Y -= theta * dt * A2U
        _solve_variance!(Y, A2, theta * dt)

        @. difference = Y - U
        _apply_mixed!(A0D, price_weights, variance_weights, mixed_scale, difference)
        _add_jump!(A0D, model, jump, difference, work)
        _apply_price!(A1D, A1, difference)
        _apply_variance!(A2D, A2, difference)
        @. Y = Y0 + theta * dt * A0D + (1 / 2 - theta) * dt * (A0D + A1D + A2D) - theta * dt * A1U
        _solve_price!(Y, A1, theta * dt)
        @. Y -= theta * dt * A2U
        _solve_variance!(Y, A2, theta * dt)

        # The Ikonen-Toivanen step, equation (3.5), with the multiplier of the step before.
        @. U = max(Y - dt * multiplier, payoff)
        @. multiplier = max(0.0, multiplier + (payoff - Y) / dt)
    end
    return s, v, U
end

# The values at time 0 under SABR, by the same scheme, in the forward and y = log(volatility).
# The forward has no drift and is absorbed at zero, where the value stays at the payoff. The
# mesh in y is uniform over 6 standard deviations of y at T on each side of log(alpha), with no
# slope in y at its ends. Without early exercise, the caller passes rate 0 and discounts.
function _finite_difference_surface(model::SABR, K::Real, T::Real, rate::Real, call::Bool, american::Bool, price_points::Integer, volatility_points::Integer, steps::Integer)
    (; F0, alpha, beta, rho, nu) = model
    s = _price_mesh(K, 0.9 * min(K, F0), 1.1 * max(K, F0), price_points, 14 * max(K, F0))
    half_width = 6 * nu * sqrt(T)
    y = range(log(alpha) - half_width, log(alpha) + half_width; length = volatility_points + 1)
    volatility = exp.(y)
    columns, rows = length(y), length(s)
    dt = T / steps
    theta = 1 / 3
    slope = call ? 1.0 : 0.0
    local_scale = s .^ beta

    # Along the forward, with half of the discounting.
    lower1, diagonal1, upper1 = zeros(rows, columns), zeros(rows, columns), zeros(rows, columns)
    boundary = zeros(rows, columns)
    for j in 1:columns
        for i in 2:rows-1
            before, after = s[i] - s[i-1], s[i+1] - s[i]
            diffusion = abs2(volatility[j] * local_scale[i]) / 2
            lower1[i, j] = diffusion * 2 / (before * (before + after))
            upper1[i, j] = diffusion * 2 / (after * (before + after))
            diagonal1[i, j] = -diffusion * 2 / (before * after) - rate / 2
        end
        last_step = s[rows] - s[rows-1]
        curvature = abs2(volatility[j] * local_scale[rows]) / abs2(last_step)
        lower1[rows, j] = curvature
        diagonal1[rows, j] = -curvature - rate / 2
        boundary[rows, j] = curvature * last_step * slope
    end

    # Along y, with the other half: nu^2 / 2 (second derivative - first derivative).
    lower2, diagonal2, upper2 = zeros(columns, 1), zeros(columns, 1), zeros(columns, 1)
    step_y = step(y)
    curvature_y = abs2(nu / step_y)
    drift_y = abs2(nu) / (4 * step_y)
    for j in 2:columns-1
        lower2[j] = curvature_y / 2 + drift_y
        upper2[j] = curvature_y / 2 - drift_y
        diagonal2[j] = -curvature_y - rate / 2
    end
    upper2[1] = curvature_y
    diagonal2[1] = -curvature_y - rate / 2
    lower2[columns] = curvature_y
    diagonal2[columns] = -curvature_y - rate / 2

    A1 = _factor(lower1, diagonal1, upper1, theta * dt)
    A2 = _factor(lower2, diagonal2, upper2, theta * dt)
    price_weights, variance_weights = _central_weights(s), _central_weights(collect(y))
    mixed_scale = [rho * nu * volatility[j] * local_scale[i] for i in 1:rows, j in 1:columns]

    payoff = _payoff_on_mesh(s, K, call) * ones(columns)'
    U = copy(payoff)
    multiplier = zeros(rows, columns)
    A0U, A1U, A2U = similar(U), similar(U), similar(U)
    Y0, Y, difference, A0D, A1D, A2D = similar(U), similar(U), similar(U), similar(U), similar(U), similar(U)
    for _ in 1:steps
        _apply_mixed!(A0U, price_weights, variance_weights, mixed_scale, U)
        _apply_price!(A1U, A1, U)
        _apply_variance!(A2U, A2, U)
        @. Y0 = U + dt * (A0U + A1U + A2U + boundary + multiplier)

        @. Y = Y0 - theta * dt * A1U
        _solve_price!(Y, A1, theta * dt)
        @. Y -= theta * dt * A2U
        _solve_variance!(Y, A2, theta * dt)

        @. difference = Y - U
        _apply_mixed!(A0D, price_weights, variance_weights, mixed_scale, difference)
        _apply_price!(A1D, A1, difference)
        _apply_variance!(A2D, A2, difference)
        @. Y = Y0 + theta * dt * A0D + (1 / 2 - theta) * dt * (A0D + A1D + A2D) - theta * dt * A1U
        _solve_price!(Y, A1, theta * dt)
        @. Y -= theta * dt * A2U
        _solve_variance!(Y, A2, theta * dt)

        if american
            @. U = max(Y - dt * multiplier, payoff)
            @. multiplier = max(0.0, multiplier + (payoff - Y) / dt)
        else
            U .= Y
        end
    end
    return s, y, U
end

# The index of the first of the 4 mesh points around x, and their Lagrange weights.
function _lagrange_cubic(mesh::AbstractVector, x::Real)
    first_point = clamp(searchsortedlast(mesh, x) - 1, 1, length(mesh) - 3)
    points = view(mesh, first_point:first_point+3)
    weights = map(1:4) do a
        prod((x - points[b]) / (points[a] - points[b]) for b in 1:4 if b != a)
    end
    return first_point, weights
end

"""American price under [`Heston`](@ref) or [`Bates`](@ref), by finite differences.

The price and the variance are discretized on the nonuniform meshes of Haentjens and in 't
Hout (2014), with first-order upwind convection, and time by their MCS-IT method: the
modified Craig-Sneyd ADI scheme with `theta = 1/3` and the Ikonen-Toivanen splitting for the
early exercise. Exercise is possible at every time step. The price at `S0` and `v0` is
interpolated between the mesh points by cubic polynomials in each direction.

Under Bates, the jump integral is applied in the explicit part of the scheme, with the value
linear in the price between the mesh points. The paper does not treat jumps.

The paper treats puts and a positive drift. For calls here, the mesh is uniform from `K` to
`min(2, exp(T / 4)) K`, the slope at the largest price is 1, and a negative drift is upwinded
the other way.

# Arguments
- `model`: a [`Heston`](@ref) or [`Bates`](@ref) model.
- `K`: the strike, or a vector of strikes.
- `T`: the maturity in years.
- `rate`: the risk-free rate.
- `call`: `true` for calls, `false` for puts.
- `price_points`, `variance_points`, `steps`: the numbers of mesh intervals and of time
  steps. The defaults are the finest grid of Tables 2 and 3 of the paper.

# Returns
- The price, or a vector of prices for a vector of strikes, at least the value of
  exercising at time 0.

# Reference
Haentjens, T. and in 't Hout, K. J. (2015). ADI schemes for pricing American options under
the Heston model. Applied Mathematical Finance 22(3), 207-237.
arXiv:1309.0110
"""
finite_difference_price(model::Union{Heston, Bates}, K::Real, T::Real; arguments...) =
    only(finite_difference_price(model, [K], T; arguments...))

function finite_difference_price(
    model::Union{Heston, Bates},
    K::AbstractVector{<:Real},
    T::Real;
    rate::Real,
    call::Bool = true,
    price_points::Integer = 300,
    variance_points::Integer = 150,
    steps::Integer = 75,
)
    all(>(0), K) || throw(DomainError(K, "K must be positive"))
    T > 0 || throw(DomainError(T, "T must be positive"))
    min(price_points, variance_points) >= 4 || throw(ArgumentError("price_points and variance_points must be at least 4"))
    steps >= 1 || throw(ArgumentError("steps must be at least 1"))
    base = _diffusion(model)
    model isa Bates && model.lambda > 0 && model.deltaJ == 0 && throw(ArgumentError("deltaJ must be positive when lambda is"))
    base.v0 <= 5 || throw(DomainError(base.v0, "v0 must be at most 5, the top of the mesh"))
    return map(K) do strike
        base.S0 <= 14 * strike || throw(DomainError(base.S0, "S0 must be at most 14 K, the top of the mesh"))
        s, v, U = _finite_difference_surface(model, strike, T, rate, call, price_points, variance_points, steps)
        first_price, price_weights = _lagrange_cubic(s, base.S0)
        first_variance, variance_weights = _lagrange_cubic(v, base.v0)
        value = sum(price_weights[a] * variance_weights[b] * U[first_price+a-1, first_variance+b-1] for a in 1:4, b in 1:4)
        return max(value, call ? base.S0 - strike : strike - base.S0)
    end
end

"""Price under [`SABR`](@ref) by finite differences, American or European.

The forward has no drift and is absorbed at zero. The scheme is that of the method for
[`Heston`](@ref), MCS with `theta = 1/3` and the Ikonen-Toivanen splitting for the early
exercise, in the forward and the logarithm of the volatility. Derivatives are central, as
the forward has no drift. Exercise is possible at every time step. The price at `F0` and
`alpha` is interpolated between the mesh points by cubic polynomials in each direction.

# Arguments
- `model`: a [`SABR`](@ref) model with `nu > 0`.
- `K`: the strike, or a vector of strikes.
- `T`: the maturity in years.
- `rate`: the risk-free rate.
- `call`: `true` for calls, `false` for puts.
- `american`: `false` for the European price.
- `price_points`, `volatility_points`, `steps`: the numbers of mesh intervals and of time steps.

# Returns
- The price, or a vector of prices for a vector of strikes.
"""
finite_difference_price(model::SABR, K::Real, T::Real; arguments...) =
    only(finite_difference_price(model, [K], T; arguments...))

function finite_difference_price(
    model::SABR,
    K::AbstractVector{<:Real},
    T::Real;
    rate::Real,
    call::Bool = true,
    american::Bool = true,
    price_points::Integer = 300,
    volatility_points::Integer = 150,
    steps::Integer = 75,
)
    all(>(0), K) || throw(DomainError(K, "K must be positive"))
    T > 0 || throw(DomainError(T, "T must be positive"))
    min(price_points, volatility_points) >= 4 || throw(ArgumentError("price_points and volatility_points must be at least 4"))
    steps >= 1 || throw(ArgumentError("steps must be at least 1"))
    model.nu > 0 || throw(DomainError(model.nu, "nu must be positive"))
    return map(K) do strike
        s, y, U = _finite_difference_surface(model, strike, T, american ? rate : 0.0, call, american, price_points, volatility_points, steps)
        first_price, price_weights = _lagrange_cubic(s, model.F0)
        first_volatility, volatility_weights = _lagrange_cubic(y, log(model.alpha))
        value = sum(price_weights[a] * volatility_weights[b] * U[first_price+a-1, first_volatility+b-1] for a in 1:4, b in 1:4)
        american && return max(value, call ? model.F0 - strike : strike - model.F0)
        return exp(-rate * T) * value
    end
end
