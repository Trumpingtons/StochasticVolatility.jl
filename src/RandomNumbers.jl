using Random: Xoshiro, randn!
using SpecialFunctions: erfc, erfcinv
import Sobol

# Paths per stream. Fixed, so that a seed gives the same paths on any number of threads.
const _CHUNK_PATHS = 1024
const _STREAM_STEP = 0x9e3779b97f4a7c15
const _SOBOL_MAX_DIMENSION = 21201

"""Supertype of the sources of random numbers: [`PseudoRandom`](@ref) and [`QuasiRandom`](@ref)."""
abstract type NumberSource end

"""Pseudo-random numbers from independent Xoshiro streams, one per fixed block of paths.

# Arguments
- `seed`: an integer. The same seed gives the same paths on any number of threads.
  By default a new one is drawn, and kept in the result's `numbers` field.
"""
struct PseudoRandom <: NumberSource
    seed::UInt64
end

PseudoRandom(; seed::Integer = rand(UInt64)) = PseudoRandom(UInt64(seed))

"""Quasi-random numbers: a digitally shifted Sobol sequence with a Brownian bridge.

Each replicate is the same Sobol sequence under its own random digital shift, so the
replicates are independent and the spread of their means gives a standard error. The
Brownian bridge gives the first coordinates of the sequence, the most evenly spread, to
the largest moves of the price and of its variance or volatility.

# Arguments
- `seed`: an integer, as in [`PseudoRandom`](@ref).
- `replicates`: the number of shifted sequences, at least 2. `paths` must be `replicates`
  times a power of 2, which is where a Sobol sequence is balanced.
"""
struct QuasiRandom <: NumberSource
    seed::UInt64
    replicates::Int

    function QuasiRandom(seed::UInt64, replicates::Integer)
        replicates >= 2 || throw(DomainError(replicates, "replicates must be at least 2"))
        return new(seed, replicates)
    end
end

QuasiRandom(; seed::Integer = rand(UInt64), replicates::Integer = 16) = QuasiRandom(UInt64(seed), replicates)

_normal_cdf(z::Real) = erfc(-z / sqrt(2.0)) / 2
_normal_quantile(u::Real) = -sqrt(2.0) * erfcinv(2 * u)

struct _Chunk
    first_path::Int
    last_path::Int
    replicate::Int
    offset::Int
end

function _chunks(::PseudoRandom, paths::Integer)
    return [_Chunk(first, min(first + _CHUNK_PATHS - 1, paths), 1, first - 1) for first in 1:_CHUNK_PATHS:paths]
end

function _chunks(numbers::QuasiRandom, paths::Integer)
    per_replicate = paths ÷ numbers.replicates
    chunks = _Chunk[]
    for replicate in 1:numbers.replicates, offset in 0:_CHUNK_PATHS:(per_replicate - 1)
        first = (replicate - 1) * per_replicate + offset + 1
        last = first + min(_CHUNK_PATHS, per_replicate - offset) - 1
        push!(chunks, _Chunk(first, last, replicate, offset))
    end
    return chunks
end

function _validate_numbers(numbers::QuasiRandom, paths::Integer, factors::Integer, steps::Integer)
    per_replicate, remainder = divrem(paths, numbers.replicates)
    (remainder == 0 && ispow2(per_replicate)) || throw(ArgumentError(
        "paths must be replicates times a power of 2; got paths = $paths and replicates = $(numbers.replicates)"))
    factors * steps <= _SOBOL_MAX_DIMENSION || throw(ArgumentError(
        "QuasiRandom supports at most $_SOBOL_MAX_DIMENSION numbers per path; got $(factors * steps)"))
    return nothing
end

_validate_numbers(::PseudoRandom, ::Integer, ::Integer, ::Integer) = nothing

struct _BrownianBridge
    left::Vector{Int}
    right::Vector{Int}
    middle::Vector{Int}
    left_weight::Vector{Float64}
    right_weight::Vector{Float64}
    deviation::Vector{Float64}
    terminal_deviation::Float64
end

# Fills the time grid from the end point inward, widest gaps first, on unit time steps.
function _brownian_bridge(steps::Integer)
    left, right, middle = Int[], Int[], Int[]
    left_weight, right_weight, deviation = Float64[], Float64[], Float64[]
    gaps = [(0, Int(steps))]
    position = 1
    while position <= length(gaps)
        l, r = gaps[position]
        position += 1
        r - l >= 2 || continue
        m = (l + r) ÷ 2
        push!(left, l)
        push!(right, r)
        push!(middle, m)
        push!(left_weight, (r - m) / (r - l))
        push!(right_weight, (m - l) / (r - l))
        push!(deviation, sqrt((m - l) * (r - m) / (r - l)))
        push!(gaps, (l, m), (m, r))
    end
    return _BrownianBridge(left, right, middle, left_weight, right_weight, deviation, sqrt(float(steps)))
end

# Turns the standard normals of one factor into its standardised increments, column by step.
function _bridge_increments!(Z::AbstractMatrix, factor::Integer, normal::AbstractVector, factors::Integer, bridge::_BrownianBridge, W::AbstractVector)
    steps = size(Z, 2)
    W[1] = 0.0
    W[steps + 1] = bridge.terminal_deviation * normal[factor]
    for k in eachindex(bridge.middle)
        z = normal[k * factors + factor]
        W[bridge.middle[k] + 1] = bridge.left_weight[k] * W[bridge.left[k] + 1] +
            bridge.right_weight[k] * W[bridge.right[k] + 1] + bridge.deviation[k] * z
    end
    for i in 1:steps
        Z[factor, i] = W[i + 1] - W[i]
    end
    return nothing
end

struct _PseudoSource
    rng::Xoshiro
end

mutable struct _QuasiSource
    sequence::Sobol.SobolSeq
    shift::Vector{UInt32}
    point::Vector{Float64}
    normal::Vector{Float64}
    W::Vector{Float64}
    bridge::_BrownianBridge
    at_origin::Bool
end

_bridge_for(::PseudoRandom, ::Integer) = nothing
_bridge_for(::QuasiRandom, steps::Integer) = _brownian_bridge(steps)

function _source(numbers::PseudoRandom, chunk::_Chunk, ::Integer, ::Integer, ::Nothing)
    return _PseudoSource(Xoshiro(numbers.seed + UInt64(chunk.first_path ÷ _CHUNK_PATHS) * _STREAM_STEP))
end

function _source(numbers::QuasiRandom, chunk::_Chunk, factors::Integer, steps::Integer, bridge::_BrownianBridge)
    dimension = factors * steps
    sequence = Sobol.SobolSeq(dimension)
    point = zeros(dimension)
    chunk.offset > 1 && Sobol.skip!(sequence, chunk.offset - 1, point; exact = true)
    shift = rand(Xoshiro(numbers.seed + UInt64(chunk.replicate) * _STREAM_STEP), UInt32, dimension)
    return _QuasiSource(sequence, shift, point, zeros(dimension), zeros(steps + 1), bridge, chunk.offset == 0)
end

_fill_normals!(Z::AbstractMatrix, source::_PseudoSource) = randn!(source.rng, Z)

function _fill_normals!(Z::AbstractMatrix, source::_QuasiSource)
    # Sobol.jl leaves out the origin, without which the first 2^k points are not balanced.
    source.at_origin ? fill!(source.point, 0.0) : Sobol.next!(source.sequence, source.point)
    source.at_origin = false
    for k in eachindex(source.point)
        bits = xor(trunc(UInt32, ldexp(source.point[k], 32)), source.shift[k])
        source.normal[k] = _normal_quantile(ldexp(bits + 0.5, -32))
    end
    factors, steps = size(Z)
    _bridge_increments!(Z, 1, source.normal, factors, source.bridge, source.W)
    _bridge_increments!(Z, 2, source.normal, factors, source.bridge, source.W)
    for i in 1:steps, factor in 3:factors
        Z[factor, i] = source.normal[(i - 1) * factors + factor]
    end
    return Z
end
