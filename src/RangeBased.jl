function _require_same_length(series::Pair{<:AbstractString,<:AbstractVector}...)
    lengths = length.(last.(series))
    all(==(first(lengths)), lengths) || throw(ArgumentError("Input series must have the same length."))
    return first(lengths)
end

function _require_positive(name::AbstractString, x::Real)
    x > 0 || throw(DomainError(x, "$name prices must be positive."))
    return nothing
end

function _sqrt_variance(variance::Real, name::AbstractString)
    variance >= 0 || throw(DomainError(variance, "$name variance estimate must be non-negative."))
    return sqrt(variance)
end

function _validate_high_low(high::AbstractVector{<:Real}, low::AbstractVector{<:Real})
    for i in eachindex(high, low)
        _require_positive("high", high[i])
        _require_positive("low", low[i])
        high[i] >= low[i] || throw(ArgumentError("high must be greater than or equal to low."))
    end

    return nothing
end

function _validate_ohlc(open::AbstractVector{<:Real}, high::AbstractVector{<:Real}, low::AbstractVector{<:Real}, close::AbstractVector{<:Real})
    for i in eachindex(open, high, low, close)
        _require_positive("open", open[i])
        _require_positive("high", high[i])
        _require_positive("low", low[i])
        _require_positive("close", close[i])

        high[i] >= max(open[i], close[i]) || throw(ArgumentError("high must be greater than or equal to open and close."))
        low[i] <= min(open[i], close[i]) || throw(ArgumentError("low must be less than or equal to open and close."))
    end

    return nothing
end

function _sample_variance(sum_x::Real, sumsq_x::Real, n::Integer)
    variance = (sumsq_x - abs2(sum_x) / n) / (n - 1)
    return max(variance, zero(variance))
end

"""Close-to-close historical volatility estimator from closing prices.

# Arguments
- `close`: `AbstractVector{<:Real}` of closing prices.
- `scale`: optional variance scaling factor, for example `252` for annualized daily volatility.

# Returns
- `Real`: the close-to-close historical volatility estimate.

This estimator is the sample standard deviation of close-to-close log returns,
optionally scaled by `sqrt(scale)`.
"""
function historical_volatility(close::AbstractVector{<:Real}; scale::Real = 1)
    sqrt(historical_variance(close; scale = scale))
end

"""Close-to-close historical variance estimator from closing prices.

# Arguments
- `close`: `AbstractVector{<:Real}` of closing prices.
- `scale`: optional variance scaling factor, for example `252` for annualized daily variance.

# Returns
- `Real`: the close-to-close historical variance estimate.

This estimator is the sample variance of close-to-close log returns,
optionally multiplied by `scale`.
"""
function historical_variance(close::AbstractVector{<:Real}; scale::Real = 1)
    _require_min_length(close, 3, "close")
    scale > 0 || throw(DomainError(scale, "scale must be positive."))

    for price in close
        _require_positive("close", price)
    end

    n = length(close) - 1
    sum_r = zero(float(first(close)))
    sumsq_r = zero(float(first(close)))

    for i in 2:length(close)
        r = log(close[i] / close[i - 1])
        sum_r += r
        sumsq_r += abs2(r)
    end

    scale * _sample_variance(sum_r, sumsq_r, n)
end

"""Meilijson volatility estimator from open, high, low, and close prices.

# Arguments
- `open`: `AbstractVector{<:Real}` of opening prices.
- `high`: `AbstractVector{<:Real}` of intraday high prices.
- `low`: `AbstractVector{<:Real}` of intraday low prices.
- `close`: `AbstractVector{<:Real}` of closing prices.

# Returns
- `Real`: the Meilijson volatility estimate.

This estimator is the square root of the average Meilijson variance estimate.
It is an improved zero-drift Brownian-motion OHLC estimator built from a
compressed sufficient statistic of the open-normalized high, low, and close.
"""
function meilijson(open::AbstractVector{<:Real}, high::AbstractVector{<:Real}, low::AbstractVector{<:Real}, close::AbstractVector{<:Real})
    n = _require_same_length("open" => open, "high" => high, "low" => low, "close" => close)
    _require_min_length(open, 1, "open")
    _validate_ohlc(open, high, low, close)

    total = zero(float(first(open)))
    for i in eachindex(open, high, low, close)
        c = log(close[i] / open[i])
        h = log(high[i] / open[i])
        l = log(low[i] / open[i])

        C = abs(c)
        H, L = c >= 0 ? (h, l) : (-l, -h)

        sigma1 = 2 * (abs2(H - C) + abs2(L))
        sigma2 = abs2(C)
        sigma3 = 2 * (H - C - L) * C
        sigma4 = -((H - C) * L) / (2 * log(2) - 5 / 4)

        total += 0.273520 * sigma1 + 0.160358 * sigma2 +
                 0.365212 * sigma3 + 0.200910 * sigma4
    end

    _sqrt_variance(total / n, "Meilijson")
end

"""Kunitomo volatility estimator from Brownian-bridge high and low values.

# Arguments
- `bridge_high`: `AbstractVector{<:Real}` of highs of the detrended log-price path.
- `bridge_low`: `AbstractVector{<:Real}` of lows of the detrended log-price path.

# Returns
- `Real`: the Kunitomo volatility estimate.

Kunitomo's estimator is based on the range of a Brownian bridge. The inputs
are not ordinary OHLC high and low prices; they are the high and low of the
log-price path after subtracting the open-to-close linear trend within each
period.
"""
function kunitomo(bridge_high::AbstractVector{<:Real}, bridge_low::AbstractVector{<:Real})
    n = _require_same_length("bridge_high" => bridge_high, "bridge_low" => bridge_low)
    _require_min_length(bridge_high, 1, "bridge_high")

    total = zero(float(first(bridge_high)))
    for i in eachindex(bridge_high, bridge_low)
        bridge_high[i] >= bridge_low[i] || throw(ArgumentError("bridge_high must be greater than or equal to bridge_low."))
        total += (6 / π^2) * abs2(bridge_high[i] - bridge_low[i])
    end

    sqrt(total / n)
end

"""Alizadeh-Brandt-Diebold log-range volatility proxy.

# Arguments
- `high`: `AbstractVector{<:Real}` of high prices.
- `low`: `AbstractVector{<:Real}` of low prices.
- `location`: constant in the approximate relation `log(range) ≈ location + log(volatility)`.

# Returns
- `Real`: the ABD range-based volatility proxy.

The proxy is `exp(mean(log(log(high / low))) - location)`. The default
`location = 0.43` follows the approximation in Alizadeh, Brandt, and Diebold
that the log range is approximately Gaussian with mean `0.43 + log(volatility)`,
so it is the mean of the log range that estimates `location + log(volatility)`.
"""
function alizadeh_brandt_diebold(high::AbstractVector{<:Real}, low::AbstractVector{<:Real}; location::Real = 0.43)
    n = _require_same_length("high" => high, "low" => low)
    _require_min_length(high, 1, "high")
    _validate_high_low(high, low)

    total = zero(float(first(high)))
    for i in eachindex(high, low)
        high[i] > low[i] || throw(DomainError(high[i], "high must exceed low, since the log of a zero range is not finite."))
        total += log(log(high[i] / low[i]))
    end

    exp(total / n - location)
end

"""Parkinson volatility estimator from high and low prices.

# Arguments
- `high`: `AbstractVector{<:Real}` of intraday high prices.
- `low`: `AbstractVector{<:Real}` of intraday low prices.

# Returns
- `Real`: the Parkinson volatility estimate.

This range-based estimator uses only the high and low prices and is efficient for continuous volatility measurement.
"""
function parkinson(high::AbstractVector{<:Real}, low::AbstractVector{<:Real})
    n = _require_same_length("high" => high, "low" => low)
    _require_min_length(high, 1, "high")
    _validate_high_low(high, low)

    total = zero(float(first(high)))
    for i in eachindex(high, low)
        total += abs2(log(high[i] / low[i]))
    end

    sqrt(total / (4 * n * log(2)))
end

"""Garman-Klass volatility estimator from open, high, low, and close prices.

# Arguments
- `open`: `AbstractVector{<:Real}` of opening prices.
- `high`: `AbstractVector{<:Real}` of intraday high prices.
- `low`: `AbstractVector{<:Real}` of intraday low prices.
- `close`: `AbstractVector{<:Real}` of closing prices.

# Returns
- `Real`: the Garman-Klass volatility estimate.

This estimator improves on Parkinson by incorporating opening and closing prices to reduce bias.
"""
function garman_klass(open::AbstractVector{<:Real}, high::AbstractVector{<:Real}, low::AbstractVector{<:Real}, close::AbstractVector{<:Real})
    n = _require_same_length("open" => open, "high" => high, "low" => low, "close" => close)
    _require_min_length(open, 1, "open")
    _validate_ohlc(open, high, low, close)

    total = zero(float(first(open)))
    for i in eachindex(open, high, low, close)
        log_hl = log(high[i] / low[i])
        log_co = log(close[i] / open[i])
        total += 0.5 * abs2(log_hl) - (2 * log(2) - 1) * abs2(log_co)
    end

    _sqrt_variance(total / n, "Garman-Klass")
end

"""Rogers-Satchell volatility estimator from open, high, low, and close prices.

# Arguments
- `open`: `AbstractVector{<:Real}` of opening prices.
- `high`: `AbstractVector{<:Real}` of intraday high prices.
- `low`: `AbstractVector{<:Real}` of intraday low prices.
- `close`: `AbstractVector{<:Real}` of closing prices.

# Returns
- `Real`: the Rogers-Satchell volatility estimate.

This estimator is suitable when returns may have drift and does not require zero mean.
"""
function rogers_satchell(open::AbstractVector{<:Real}, high::AbstractVector{<:Real}, low::AbstractVector{<:Real}, close::AbstractVector{<:Real})
    n = _require_same_length("open" => open, "high" => high, "low" => low, "close" => close)
    _require_min_length(open, 1, "open")
    _validate_ohlc(open, high, low, close)

    total = zero(float(first(open)))
    for i in eachindex(open, high, low, close)
        log_hc = log(high[i] / close[i])
        log_ho = log(high[i] / open[i])
        log_lc = log(low[i] / close[i])
        log_lo = log(low[i] / open[i])
        total += log_hc * log_ho + log_lc * log_lo
    end

    _sqrt_variance(total / n, "Rogers-Satchell")
end

"""Yang-Zhang volatility estimator from open, high, low, and close prices.

# Arguments
- `open`: `AbstractVector{<:Real}` of opening prices.
- `high`: `AbstractVector{<:Real}` of intraday high prices.
- `low`: `AbstractVector{<:Real}` of intraday low prices.
- `close`: `AbstractVector{<:Real}` of closing prices.

# Returns
- `Real`: the Yang-Zhang volatility estimate.

This estimator combines overnight, open-to-close, and Rogers-Satchell components for a robust variance estimate.
"""
function yang_zhang(open::AbstractVector{<:Real}, high::AbstractVector{<:Real}, low::AbstractVector{<:Real}, close::AbstractVector{<:Real})
    _require_same_length("open" => open, "high" => high, "low" => low, "close" => close)
    _require_min_length(open, 3, "open")
    _validate_ohlc(open, high, low, close)

    n = length(open) - 1
    sum_o = zero(float(first(open)))
    sumsq_o = zero(float(first(open)))
    sum_c = zero(float(first(open)))
    sumsq_c = zero(float(first(open)))
    sum_rs = zero(float(first(open)))

    for i in 2:length(open)
        overnight = log(open[i] / close[i - 1])
        open_close = log(close[i] / open[i])
        rs = log(high[i] / close[i]) * log(high[i] / open[i]) +
             log(low[i] / close[i]) * log(low[i] / open[i])

        sum_o += overnight
        sumsq_o += abs2(overnight)
        sum_c += open_close
        sumsq_c += abs2(open_close)
        sum_rs += rs
    end

    sigma_o2 = _sample_variance(sum_o, sumsq_o, n)
    sigma_c2 = _sample_variance(sum_c, sumsq_c, n)
    sigma_rs2 = sum_rs / n

    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    _sqrt_variance(sigma_o2 + k * sigma_c2 + (1 - k) * sigma_rs2, "Yang-Zhang")
end
