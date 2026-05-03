using SpecialFunctions: gamma

function _require_min_length(r::AbstractVector, min_length::Integer, name::AbstractString)
    length(r) >= min_length || throw(ArgumentError("$name must have at least $min_length observations."))
    return nothing
end

"""Compute realized variance from a vector of returns.

# Arguments
- `r`: `AbstractVector{<:Real}` of returns.

# Returns
- `Real`: the realized variance estimate.

This estimator is the sum of squared returns and serves as a basic high-frequency variance measure.
"""
function realized_variance(r::AbstractVector{<:Real})
    _require_min_length(r, 1, "r")
    return sum(abs2, r)
end

"""Compute realized volatility from a vector of returns.

# Arguments
- `r`: `AbstractVector{<:Real}` of returns.

# Returns
- `Real`: the square root of realized variance.
"""
realized_volatility(r::AbstractVector{<:Real}) = sqrt(realized_variance(r))

"""Compute negative and positive realized semivariance from returns.

# Arguments
- `r`: `AbstractVector{<:Real}` of returns.

# Returns
- `NamedTuple`: `(negative = ..., positive = ...)`, where negative semivariance
  sums squared negative returns and positive semivariance sums squared positive returns.
"""
function realized_semivariance(r::AbstractVector{<:Real})
    _require_min_length(r, 1, "r")

    negative = zero(float(first(r)))
    positive = zero(float(first(r)))

    for x in r
        if x < 0
            negative += abs2(x)
        elseif x > 0
            positive += abs2(x)
        end
    end

    return (negative = negative, positive = positive)
end

"""Compute realized quarticity from a vector of returns.

# Arguments
- `r`: `AbstractVector{<:Real}` of returns.

# Returns
- `Real`: the realized quarticity estimate.

The estimator is ``n / 3 * sum(r_i^4)`` for `n` intraday returns.
"""
function realized_quarticity(r::AbstractVector{<:Real})
    n = length(r)
    _require_min_length(r, 1, "r")

    total = zero(float(first(r)))
    for x in r
        total += x^4
    end

    (n / 3) * total
end


"""Compute the Two-Scales Realized Variance (TSRV).

# Arguments
- `r`: `AbstractVector{<:Real}` of returns.
- `k`: `Integer` subsampling frequency for the coarse scale (default: `10`).

# Returns
- `Real`: the TSRV estimate.

The estimator averages coarse-scale realized variances across `k` sampling grids and
subtracts the matching fine-scale bias correction:

    mean(RV_coarse) - (mean(number of coarse returns) / n) * RV_fine
"""
function two_scales_realized_variance(r::AbstractVector{<:Real}; k::Integer = 10)
    n = length(r)
    _require_min_length(r, 2, "r")
    k > 0 || throw(ArgumentError("k must be positive."))
    k = min(k, n)

    RV_fine = realized_variance(r)
    RV_coarse_total = zero(float(first(r)))
    coarse_count = 0
    grids = 0

    for offset in 1:k
        RV_grid = zero(float(first(r)))
        grid_count = 0

        for start in offset:k:(n - k + 1)
            coarse_return = zero(float(first(r)))

            for i in start:(start + k - 1)
                coarse_return += r[i]
            end

            RV_grid += abs2(coarse_return)
            grid_count += 1
        end

        if grid_count > 0
            RV_coarse_total += RV_grid
            coarse_count += grid_count
            grids += 1
        end
    end

    RV_coarse = RV_coarse_total / grids
    RV_coarse - (coarse_count / grids / n) * RV_fine
end

"""Estimate bipower variation from a vector of returns.

# Arguments
- `r`: `AbstractVector{<:Real}` of intraday returns.

# Returns
- `Real`: the bipower variation estimate.

Bipower variation is computed with the usual ``μ_1^{-2} = π / 2`` scaling:

    (π / 2) * sum(abs(r[i]) * abs(r[i - 1]) for i in 2:length(r))

This estimator is robust to jumps and is often used to separate continuous variance from discontinuous jumps.
"""
function bipower_variation(r::AbstractVector{<:Real})
    _require_min_length(r, 2, "r")

    total = zero(float(first(r)))
    for i in 2:length(r)
        total += abs(r[i]) * abs(r[i - 1])
    end

    (π / 2) * total
end

"""Estimate jump variation from realized variance and bipower variation.

# Arguments
- `r`: `AbstractVector{<:Real}` of intraday returns.

# Returns
- `Real`: `max(realized_variance(r) - bipower_variation(r), 0)`.
"""
function jump_variation(r::AbstractVector{<:Real})
    _require_min_length(r, 2, "r")
    max(realized_variance(r) - bipower_variation(r), zero(float(first(r))))
end




"""Estimate variance using the realized kernel with the Bartlett kernel.

# Arguments
- `r`: `AbstractVector{<:Real}` of returns.
- `H`: `Integer` bandwidth parameter.

# Returns
- `Real`: the realized kernel estimate.

The estimator uses lagged autocovariances up to lag `H` and Bartlett weights to reduce noise bias.
"""
function realized_kernel(r::AbstractVector{<:Real}, H::Integer)
    n = length(r)
    _require_min_length(r, 1, "r")
    H >= 0 || throw(ArgumentError("H must be non-negative."))

    function autocovariance(h::Integer)
        total = zero(float(first(r)))
        for i in (h + 1):n
            total += r[i] * r[i - h]
        end
        total
    end

    kernel(x) = 1 - abs(x)
    RK = autocovariance(0)

    for h in 1:min(H, n - 1)
        weight = kernel(h / (H + 1))
        RK += 2 * weight * autocovariance(h)
    end

    RK
end



"""Perform the Ait-Sahalia, Mykland & Zhang (AMS) microstructure noise test.

# Arguments
- `r`: `AbstractVector{<:Real}` of intraday returns.

# Returns
- `Real`: the Z statistic for noise detection.

Microstructure noise induces negative first-order autocorrelation in observed returns.
The test statistic is:

    Z = Σ(r[i] * r[i-1]) / sqrt(Σ(r[i] * r[i-1])²)

Under the null of no noise Z is asymptotically N(0, 1). A significantly negative value
(e.g. Z < -1.645 for a 5% one-sided test) rejects the null in favour of noise.
"""
function ams_noise_test(r::AbstractVector{<:Real})
    _require_min_length(r, 2, "r")

    numerator = zero(float(first(r)))
    denominator = zero(float(first(r)))

    for i in 2:length(r)
        p = r[i] * r[i - 1]
        numerator += p
        denominator += abs2(p)
    end

    denominator > 0 || throw(DomainError(denominator, "AMS variance estimate must be positive."))
    numerator / sqrt(denominator)
end

"""Estimate the Hurst exponent of log-realized variance (Gatheral, Jaisson & Rosenbaum 2018).

# Arguments
- `rv`: `AbstractVector{<:Real}` of realized variance estimates (one per period, e.g. daily).
- `q`: moment order used in the scaling regression (default: `1.0`).
- `max_lag`: maximum lag `Δ` included in the regression (default: `10`).

# Returns
- `Real`: the Hurst exponent estimate `H ∈ (0, 1)`.

The estimator exploits the power-law scaling of absolute log-RV increments:

    m(q, Δ) = mean(|log RV(t+Δ) − log RV(t)|^q) ≈ K(q) · Δ^{qH}

Taking logs gives a linear relation in log Δ with slope qH. H is recovered by OLS
across lags `Δ = 1, …, max_lag` and dividing the estimated slope by `q`.

A value H ≈ 0.1 is consistent with rough volatility; H = 0.5 corresponds to
standard Brownian motion.
"""
function hurst_exponent(rv::AbstractVector{<:Real}; q::Real = 1.0, max_lag::Integer = 10)
    T = length(rv)
    _require_min_length(rv, max_lag + 1, "rv")
    q > 0 || throw(ArgumentError("q must be positive."))
    max_lag >= 2 || throw(ArgumentError("max_lag must be at least 2."))

    for v in rv
        v > 0 || throw(DomainError(v, "rv values must be positive."))
    end

    log_rv = [log(v) for v in rv]

    sum_x  = zero(float(first(rv)))
    sum_y  = zero(float(first(rv)))
    sum_xx = zero(float(first(rv)))
    sum_xy = zero(float(first(rv)))

    for Δ in 1:max_lag
        moment = zero(float(first(rv)))
        for t in 1:(T - Δ)
            moment += abs(log_rv[t + Δ] - log_rv[t])^q
        end
        moment /= (T - Δ)
        moment > 0 || throw(DomainError(moment, "log-RV moment is zero at lag $Δ."))

        log_Δ = log(float(Δ))
        log_m = log(moment)

        sum_x  += log_Δ
        sum_y  += log_m
        sum_xx += log_Δ * log_Δ
        sum_xy += log_Δ * log_m
    end

    denom = sum_xx - sum_x * sum_x / max_lag
    denom > 0 || throw(DomainError(denom, "Hurst OLS denominator must be positive."))
    slope = (sum_xy - sum_x * sum_y / max_lag) / denom

    slope / q
end

"""Perform the Barndorff-Nielsen & Shephard (BNS) jump test.

# Arguments
- `r`: `AbstractVector{<:Real}` of returns.

# Returns
- `Real`: the Z statistic for jump detection.

The BNS test compares realized variance and bipower variation and computes the asymptotic variance using tripower quarticity.
"""
function bns_test(r::AbstractVector{<:Real})
    n = length(r)
    _require_min_length(r, 3, "r")

    RV = realized_variance(r)
    BPV = bipower_variation(r)

    μ43 = (2^(2/3)) * gamma(7/6) / gamma(1/2)
    tripower_sum = zero(float(first(r)))
    for i in 3:n
        tripower_sum += abs(r[i])^(4 / 3) * abs(r[i - 1])^(4 / 3) * abs(r[i - 2])^(4 / 3)
    end
    TPQ = n * μ43^(-3) * tripower_sum

    σ2 = ((π / 2)^2 + π - 5) * TPQ / n
    σ2 > 0 || throw(DomainError(σ2, "BNS asymptotic variance must be positive."))
    (RV - BPV) / sqrt(σ2)
end
