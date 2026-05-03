using Test
using Statistics
using SpecialFunctions: gamma
using StochasticVolatility

@testset "StochasticVolatility" begin
    r = [0.01, -0.02, 0.015, -0.005, 0.007]

    @testset "High-frequency estimators" begin
        @test realized_variance(r) == sum(abs2, r)
        @test realized_volatility(r) ≈ sqrt(realized_variance(r))
        @test realized_semivariance(r) == (
            negative = sum(abs2, x for x in r if x < 0),
            positive = sum(abs2, x for x in r if x > 0),
        )
        @test sum(realized_semivariance(r)) ≈ realized_variance(r)
        @test realized_semivariance([0.01, 0.02, 0.03]).negative == 0.0
        @test realized_semivariance([-0.01, -0.02, -0.03]).positive == 0.0
        @test realized_semivariance([0.01, 0.0, -0.02]) == (negative = abs2(-0.02), positive = abs2(0.01))
        @test realized_quarticity(r) ≈ (length(r) / 3) * sum(x^4 for x in r)
        @test bipower_variation(r) ≈ (π / 2) * sum(abs(r[i]) * abs(r[i - 1]) for i in 2:length(r))
        @test jump_variation(r) ≈ max(realized_variance(r) - bipower_variation(r), 0)
        r_jump = [0.001, -0.001, 0.001, -0.001, 0.1]
        @test jump_variation(r_jump) > 0
        @test jump_variation(r_jump) ≈ realized_variance(r_jump) - bipower_variation(r_jump)

        rv_grid_1 = abs2(r[1] + r[2]) + abs2(r[3] + r[4])
        rv_grid_2 = abs2(r[2] + r[3]) + abs2(r[4] + r[5])
        coarse_average = (rv_grid_1 + rv_grid_2) / 2
        average_coarse_returns = 2
        @test two_scales_realized_variance(r, k = 2) ≈ coarse_average - (average_coarse_returns / length(r)) * realized_variance(r)
        @test two_scales_realized_variance(r) == two_scales_realized_variance(r, k = 10)

        @test realized_kernel(r, 0) == realized_variance(r)
        @test realized_kernel(r, 1) ≈ sum(abs2, r) + sum(r[i] * r[i - 1] for i in 2:length(r))
        @test realized_kernel(r, 2) ≈ sum(abs2, r) +
            (4/3) * sum(r[i] * r[i - 1] for i in 2:length(r)) +
            (2/3) * sum(r[i] * r[i - 2] for i in 3:length(r))
        @test bns_test(r) ≈ let
            n = length(r)
            RV  = sum(abs2, r)
            BPV = (π / 2) * sum(abs(r[i]) * abs(r[i - 1]) for i in 2:n)
            μ43 = (2^(2/3)) * gamma(7/6) / gamma(1/2)
            TPQ = n * μ43^(-3) * sum(abs(r[i])^(4/3) * abs(r[i-1])^(4/3) * abs(r[i-2])^(4/3) for i in 3:n)
            (RV - BPV) / sqrt(((π/2)^2 + π - 5) * TPQ / n)
        end
        @test bns_test(r_jump) > 0

        @test ams_noise_test(r) ≈ let
            n = length(r)
            num = sum(r[i] * r[i - 1] for i in 2:n)
            den = sqrt(sum((r[i] * r[i - 1])^2 for i in 2:n))
            num / den
        end
        @test ams_noise_test([0.01, -0.01, 0.01, -0.01, 0.01]) < 0

        rv_linear = [exp(0.1 * t) for t in 0:14]
        @test hurst_exponent(rv_linear) ≈ 1.0
    end

    open = [100.0, 101.0, 102.0, 103.0]
    high = [101.0, 102.0, 103.0, 104.0]
    low = [99.0, 100.0, 101.0, 102.0]
    close = [100.5, 101.5, 102.5, 103.5]

    @testset "Range-based estimators" begin
        close_returns = [log(close[i] / close[i - 1]) for i in 2:length(close)]
        @test historical_variance(close) ≈ var(close_returns)
        @test historical_variance(close; scale = 252) ≈ 252 * historical_variance(close)
        @test historical_volatility(close) ≈ std(close_returns)
        @test historical_volatility(close; scale = 252) ≈ sqrt(252) * historical_volatility(close)
        @test alizadeh_brandt_diebold(high, low) ≈ sqrt(mean((log(high[i] / low[i]) / exp(0.43))^2 for i in eachindex(high, low)))
        @test alizadeh_brandt_diebold(high, low; location = 0.0) ≈ sqrt(mean(log(high[i] / low[i])^2 for i in eachindex(high, low)))
        @test kunitomo([0.02, 0.03], [-0.01, -0.02]) ≈ sqrt(mean((6 / π^2) * (h - l)^2 for (h, l) in zip([0.02, 0.03], [-0.01, -0.02])))
        @test meilijson(open, high, low, close) ≈ let total = 0.0
            for i in eachindex(open, high, low, close)
                c = log(close[i] / open[i])
                h = log(high[i] / open[i])
                l = log(low[i] / open[i])
                C = abs(c)
                H, L = c >= 0 ? (h, l) : (-l, -h)
                s1 = 2 * ((H - C)^2 + L^2)
                s2 = C^2
                s3 = 2 * (H - C - L) * C
                s4 = -((H - C) * L) / (2 * log(2) - 5/4)
                total += 0.273520 * s1 + 0.160358 * s2 + 0.365212 * s3 + 0.200910 * s4
            end
            sqrt(total / length(open))
        end
        @test parkinson(high, low) ≈ sqrt(sum(log(high[i] / low[i])^2 for i in eachindex(high, low)) / (4 * length(high) * log(2)))
        @test garman_klass(open, high, low, close) ≈ sqrt(sum(
            0.5 * log(high[i] / low[i])^2 - (2 * log(2) - 1) * log(close[i] / open[i])^2
            for i in eachindex(open, high, low, close)
        ) / length(open))
        @test rogers_satchell(open, high, low, close) ≈ sqrt(sum(
            log(high[i] / close[i]) * log(high[i] / open[i]) +
            log(low[i] / close[i]) * log(low[i] / open[i])
            for i in eachindex(open, high, low, close)
        ) / length(open))

        n = length(open) - 1
        overnight = [log(open[i] / close[i - 1]) for i in 2:length(open)]
        open_close = [log(close[i] / open[i]) for i in 2:length(open)]
        rs = [
            log(high[i] / close[i]) * log(high[i] / open[i]) +
            log(low[i] / close[i]) * log(low[i] / open[i])
            for i in 2:length(open)
        ]
        k = 0.34 / (1.34 + (n + 1) / (n - 1))
        @test yang_zhang(open, high, low, close) ≈ sqrt(var(overnight) + k * var(open_close) + (1 - k) * mean(rs))
    end


    @testset "Input validation" begin
        @test_throws ArgumentError realized_variance(Float64[])
        @test_throws ArgumentError realized_volatility(Float64[])
        @test_throws ArgumentError realized_semivariance(Float64[])
        @test_throws ArgumentError realized_quarticity(Float64[])
        @test_throws ArgumentError bipower_variation([0.01])
        @test_throws ArgumentError jump_variation([0.01])
        @test_throws ArgumentError two_scales_realized_variance(r, k = 0)
        @test_throws ArgumentError realized_kernel(r, -1)
        @test_throws DomainError bns_test([0.0, 0.0, 0.0])

        @test_throws ArgumentError parkinson([101.0], [99.0, 100.0])
        @test_throws DomainError parkinson([101.0], [0.0])
        @test_throws ArgumentError parkinson([99.0], [101.0])
        @test_throws ArgumentError historical_volatility([100.0, 101.0])
        @test_throws DomainError historical_volatility([100.0, 0.0, 101.0])
        @test_throws DomainError historical_volatility(close; scale = 0)
        @test_throws ArgumentError historical_variance([100.0, 101.0])
        @test_throws DomainError historical_variance([100.0, 0.0, 101.0])
        @test_throws DomainError historical_variance(close; scale = 0)
        @test_throws ArgumentError alizadeh_brandt_diebold([101.0], [99.0, 100.0])
        @test_throws DomainError alizadeh_brandt_diebold([101.0], [0.0])
        @test_throws ArgumentError alizadeh_brandt_diebold([99.0], [101.0])
        @test_throws ArgumentError kunitomo([0.01], [0.0, -0.01])
        @test_throws ArgumentError kunitomo([0.0], [0.01])

        @test_throws ArgumentError garman_klass(open, high[1:3], low, close)
        @test_throws DomainError garman_klass([0.0], [1.0], [0.5], [0.8])
        @test_throws ArgumentError garman_klass([100.0], [99.0], [98.0], [100.5])
        @test_throws ArgumentError meilijson(open, high[1:3], low, close)
        @test_throws DomainError meilijson([0.0], [1.0], [0.5], [0.8])
        @test_throws ArgumentError meilijson([100.0], [99.0], [98.0], [100.5])
        @test_throws ArgumentError rogers_satchell([100.0], [101.0], [100.5], [100.5])
        @test_throws ArgumentError rogers_satchell(open, high[1:3], low, close)
        @test_throws DomainError rogers_satchell([0.0], [1.0], [0.5], [0.8])
        @test_throws ArgumentError yang_zhang(open[1:2], high[1:2], low[1:2], close[1:2])
        @test_throws ArgumentError yang_zhang(open, high[1:3], low, close)
        @test_throws DomainError yang_zhang([0.0, 101.0, 102.0], [1.0, 102.0, 103.0], [0.5, 100.0, 101.0], [0.8, 101.5, 102.5])

        @test_throws ArgumentError ams_noise_test([0.01])
        @test_throws DomainError ams_noise_test([0.0, 0.0, 0.0])

        @test_throws ArgumentError hurst_exponent(fill(0.001, 10))
        @test_throws DomainError hurst_exponent([0.001, 0.002, 0.0, 0.001]; max_lag = 2)
        @test_throws ArgumentError hurst_exponent(fill(0.001, 15); max_lag = 1)
        @test_throws ArgumentError hurst_exponent(fill(0.001, 15); q = 0.0)
    end
end
