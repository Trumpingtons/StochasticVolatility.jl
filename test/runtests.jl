using Test
using Statistics
using SpecialFunctions: gamma
using LinearAlgebra: I
using Random: Xoshiro
using StochasticVolatility
using StochasticVolatility: _brownian_bridge, _bridge_increments!, _chunks, _diffusion, _mean_jump, _normal_cdf, _poisson_quantile

# How many standard errors a sample mean lies from its exact value.
zscore(x, exact) = abs(mean(x) - exact) / (std(x) / sqrt(length(x)))

@testset "StochasticVolatility.jl" begin
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
        @test alizadeh_brandt_diebold(high, low) ≈ exp(mean(log(log(high[i] / low[i])) for i in eachindex(high, low)) - 0.43)
        @test alizadeh_brandt_diebold(high, low; location = 0.0) ≈ exp(mean(log(log(high[i] / low[i])) for i in eachindex(high, low)))
        @test_throws DomainError alizadeh_brandt_diebold([100.0], [100.0])
        let rng = Xoshiro(20261101), sigma = 0.02, days = 5000, grid = 2000
            bridge_high, bridge_low = zeros(days), zeros(days)
            for day in 1:days
                x = cumsum(randn(rng, grid)) .* (sigma / sqrt(grid))
                bridge_high[day] = exp(max(0, maximum(x)))
                bridge_low[day] = exp(min(0, minimum(x)))
            end
            @test isapprox(alizadeh_brandt_diebold(bridge_high, bridge_low), sigma; rtol = 0.05)
        end
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

    @testset "Simulation" begin
        heston = Heston(S0 = 100, v0 = 0.09, mu = 0.03, kappa = 1.5, theta = 0.04, xi = 0.8, rho = -0.7)
        bates = Bates(S0 = 100, v0 = 0.09, mu = 0.03, kappa = 1.5, theta = 0.04, xi = 0.8, rho = -0.7, lambda = 0.5, muJ = -0.1, deltaJ = 0.2)
        sabr = SABR(F0 = 100, alpha = 2, beta = 0.5, rho = -0.3, nu = 0.4)
        T = 2.0

        # Exact moments of the variance and of the price, against every scheme and number source.
        decay = exp(-heston.kappa * T)
        mean_variance = heston.theta + (heston.v0 - heston.theta) * decay
        variance_variance = heston.v0 * heston.xi^2 * decay * (1 - decay) / heston.kappa +
            heston.theta * heston.xi^2 * (1 - decay)^2 / (2 * heston.kappa)
        mean_price = heston.S0 * exp(heston.mu * T)

        @testset "Heston moments, $(nameof(typeof(scheme)))" for scheme in (QuadraticExponential(), FullTruncation())
            result = simulate(heston, T; paths = 2^17, steps = 200, scheme, numbers = PseudoRandom(seed = 11))
            @test result.time == collect(range(0.0, T; length = 201))
            @test size(result.price) == size(result.state) == (201, 2^17)
            @test all(>=(0), result.state)
            @test zscore(result.price[end, :], mean_price) < 4
            @test zscore(result.state[end, :], mean_variance) < 4
            @test isapprox(var(result.state[end, :]), variance_variance; rtol = 0.05)
        end

        @testset "Bates moments, $(nameof(typeof(scheme)))" for scheme in (QuadraticExponential(), FullTruncation())
            result = simulate_terminal(bates, T; paths = 2^17, steps = 200, scheme, numbers = PseudoRandom(seed = 12))
            @test zscore(result, mean_price) < 4
            integrated_variance = heston.theta * T + (heston.v0 - heston.theta) * (1 - decay) / heston.kappa
            mean_log = log(heston.S0) + (heston.mu - bates.lambda * _mean_jump(bates)) * T - integrated_variance / 2 + bates.lambda * bates.muJ * T
            @test zscore(log.(result), mean_log) < 4
        end

        @testset "SABR moments" begin
            result = simulate(sabr, T; paths = 2^17, steps = 200, numbers = PseudoRandom(seed = 13))
            @test zscore(result.price[end, :], sabr.F0) < 4
            @test zscore(result.state[end, :], sabr.alpha) < 4
            @test result.time[end] == T
        end

        @testset "SABR absorbs the forward at zero" begin
            for beta in (0.0, 0.5)
                result = simulate(SABR(F0 = 1, alpha = 3, beta = beta, rho = 0, nu = 0.1), 1.0; paths = 2000, steps = 100, numbers = PseudoRandom(seed = 14))
                absorbed = 0
                for column in axes(result.price, 2)
                    hit = findfirst(==(0.0), result.price[:, column])
                    if hit !== nothing
                        absorbed += 1
                        @test all(==(0.0), result.price[hit:end, column])
                    end
                end
                @test absorbed > 0
            end
        end

        @testset "Quasi-random numbers" begin
            numbers = QuasiRandom(seed = 21, replicates = 16)
            replicate_means(x) = [mean(x[((r - 1) * 2^12 + 1):(r * 2^12)]) for r in 1:16]
            @testset "$(nameof(typeof(model)))" for (model, exact) in ((heston, mean_price), (bates, mean_price), (sabr, sabr.F0))
                result = replicate_means(simulate_terminal(model, T; paths = 16 * 2^12, steps = 64, numbers))
                @test zscore(result, exact) < 4
            end
            # The reason to have them: a smaller spread than the same number of pseudo-random paths.
            call(x) = max.(x .- heston.S0, 0.0)
            quasi = replicate_means(call(simulate_terminal(heston, T; paths = 16 * 2^12, steps = 64, numbers)))
            pseudo = replicate_means(call(simulate_terminal(heston, T; paths = 16 * 2^12, steps = 64, numbers = PseudoRandom(seed = 22))))
            @test std(quasi) < std(pseudo)
            @test_throws ArgumentError simulate_terminal(heston, T; paths = 100, numbers)
            @test_throws ArgumentError simulate_terminal(bates, T; paths = 32, steps = 6000, numbers)
        end

        @testset "The Brownian bridge is an orthogonal transformation" begin
            steps = 37
            bridge = _brownian_bridge(steps)
            B = zeros(steps, steps)
            for k in 1:steps
                normal = zeros(steps)
                normal[k] = 1.0
                Z = zeros(1, steps)
                _bridge_increments!(Z, 1, normal, 1, bridge, zeros(steps + 1))
                B[:, k] = Z[1, :]
            end
            @test B' * B ≈ Matrix(I, steps, steps)
        end

        @testset "Each block of paths is balanced and has its own stream" begin
            @test [(c.first_path, c.last_path, c.offset) for c in _chunks(PseudoRandom(seed = 1), 2500)] == [(1, 1024, 0), (1025, 2048, 1024), (2049, 2500, 2048)]
            @test [(c.first_path, c.last_path, c.replicate, c.offset) for c in _chunks(QuasiRandom(seed = 1, replicates = 2), 4096)] ==
                [(1, 1024, 1, 0), (1025, 2048, 1, 1024), (2049, 3072, 2, 0), (3073, 4096, 2, 1024)]
            # One step at beta 0 and nu 0 gives F = F0 + alpha z, so the first Sobol coordinate
            # can be read back, and balance means one path in each of 2^k equal bins.
            model = SABR(F0 = 1, alpha = 0.05, beta = 0, rho = 0, nu = 0)
            x = simulate_terminal(model, 1.0; paths = 2 * 2^12, steps = 1, numbers = QuasiRandom(seed = 3, replicates = 2))
            for r in 1:2
                part = x[((r - 1) * 2^12 + 1):(r * 2^12)]
                @test sort(floor.(Int, _normal_cdf.((part .- 1) ./ 0.05) .* 2^12)) == 0:(2^12 - 1)
            end
        end

        @testset "Reproducibility" begin
            @testset "$(nameof(typeof(numbers)))" for numbers in (PseudoRandom(seed = 31), QuasiRandom(seed = 32, replicates = 4))
                @testset "$(nameof(typeof(model)))" for model in (heston, bates, sabr)
                    threaded = simulate_terminal(model, T; paths = 4 * 2^11, steps = 16, numbers, threaded = true)
                    serial = simulate_terminal(model, T; paths = 4 * 2^11, steps = 16, numbers, threaded = false)
                    full = simulate(model, T; paths = 4 * 2^11, steps = 16, numbers)
                    @test threaded == serial == full.price[end, :]
                end
            end
            @test simulate_terminal(heston, T; paths = 8, steps = 4, numbers = PseudoRandom(seed = 1)) != simulate_terminal(heston, T; paths = 8, steps = 4, numbers = PseudoRandom(seed = 2))
            @test PseudoRandom().seed != PseudoRandom().seed
        end

        @testset "Poisson quantile" begin
            @test [_poisson_quantile(u, 1.0) for u in (0.1, exp(-1.0), 0.5, 0.9, 0.99)] == [0, 0, 1, 2, 4]
            @test _poisson_quantile(0.5, 0.0) == 0
        end

        @testset "Input validation" begin
            @test _diffusion(bates) == heston
            @test_throws DomainError Heston(S0 = 0, v0 = 0.04, mu = 0, kappa = 1, theta = 0.04, xi = 0.3, rho = 0)
            @test_throws DomainError Heston(S0 = 100, v0 = -1, mu = 0, kappa = 1, theta = 0.04, xi = 0.3, rho = 0)
            @test_throws DomainError Heston(S0 = 100, v0 = 0.04, mu = 0, kappa = 1, theta = 0.04, xi = 0.3, rho = 1.5)
            @test_throws DomainError Bates(S0 = 100, v0 = 0.04, mu = 0, kappa = 1, theta = 0.04, xi = 0.3, rho = 0, lambda = -1, muJ = 0, deltaJ = 0.1)
            @test_throws DomainError SABR(F0 = 100, alpha = 2, beta = 1.5, rho = 0, nu = 0.4)
            @test_throws DomainError simulate_terminal(heston, 0.0)
            @test_throws ArgumentError simulate_terminal(heston, T; steps = 0)
            @test_throws ArgumentError simulate(heston, T; paths = 0)
            @test_throws ArgumentError simulate_terminal(heston, T; scheme = Euler())
            @test_throws ArgumentError simulate_terminal(sabr, T; scheme = FullTruncation())
            @test_throws DomainError QuadraticExponential(psi_critical = 3)
            @test_throws DomainError QuasiRandom(replicates = 1)
        end
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
