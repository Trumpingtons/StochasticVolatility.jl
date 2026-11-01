using Test
using Statistics
using SpecialFunctions: gamma, loggamma
using LinearAlgebra: I
using Random: Xoshiro
using StochasticVolatility
using StochasticVolatility: _binomial_quantile, _black_call, _brownian_bridge, _bridge_increments!, _Chunk, _chunks, _fill_normals!, _source, _diffusion, _mean_jump, _normal_cdf, _poisson_quantile

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

        @testset "Jump bridge" begin
            @test [_binomial_quantile(u, 4, 0.5) for u in (0.05, 0.0625, 0.5, 0.9, 1.0)] == [0, 0, 2, 3, 4]
            @test _binomial_quantile(0.5, 0, 0.5) == 0

            # Each step's count is Poisson and independent of the others, and the jump normals standard.
            paths, steps, rate = 2^14, 8, 0.25
            Z = Matrix{Float64}(undef, 4, steps)
            source = _source(QuasiRandom(seed = 61, replicates = 2), _Chunk(1, paths, 1, 0), 4, steps, _brownian_bridge(steps))
            counts, sizes = Matrix{Float64}(undef, steps, paths), Float64[]
            for path in 1:paths
                _fill_normals!(Z, source, rate)
                counts[:, path] = Z[3, :]
                append!(sizes, Z[4, Z[3, :] .> 0])
            end
            @test all(isapprox.(mean(counts; dims = 2), rate; rtol = 0.02))
            @test all(isapprox.(var(counts; dims = 2), rate; rtol = 0.05))
            @test abs(cov(counts[1, :], counts[steps, :])) < 0.01
            @test isapprox(mean(sizes), 0; atol = 0.02)
            @test isapprox(var(sizes), 1; rtol = 0.05)
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

    @testset "Closed-form prices" begin
        @testset "Heston against the reference values of Alan Lewis, in the QuantLib tests" begin
            model = Heston(S0 = 100, v0 = 0.04, mu = -0.01, kappa = 4, theta = 0.25, xi = 1, rho = -0.5)
            @testset "K = $K" for (K, put, call) in (
                (80, 7.958878113256768, 26.774758743998854),
                (90, 12.017966707346305, 20.933349000596710),
                (100, 17.055270961270109, 16.070154917028834),
                (110, 23.017825898442801, 12.132211516709845),
                (120, 29.811026202682472, 9.024913483457836),
            )
                @test isapprox(european_price(model, K, 1.0; rate = 0.01), call; rtol = 1e-12)
                @test isapprox(european_price(model, K, 1.0; rate = 0.01, call = false), put; rtol = 1e-12)
            end
        end

        @testset "Heston against the other values in the QuantLib tests" begin
            kahl_jaeckel = Heston(S0 = 100, v0 = 0.16, mu = 0, kappa = 1, theta = 0.16, xi = 2, rho = -0.8)
            @test isapprox(european_price(kahl_jaeckel, 200, 10.0; rate = 0), 4.95212; atol = 2e-5)
            model = Heston(S0 = 100, v0 = 0.1, mu = -0.025, kappa = 4, theta = 0.05, xi = 0.4, rho = -0.75)
            @test isapprox(european_price(model, 100, 1.0; rate = 0.05, call = false), 10.147041515497; atol = 1e-8)
        end

        @testset "Bates without jumps is Heston" begin
            heston = Heston(S0 = 100, v0 = 0.09, mu = 0.03, kappa = 1.5, theta = 0.04, xi = 0.8, rho = -0.7)
            @test european_price(Bates(heston, 0, -0.1, 0.2), 100, 2.0; rate = 0.03) == european_price(heston, 100, 2.0; rate = 0.03)
        end

        @testset "Bates with almost constant variance is the Merton series, K = $K" for K in (80, 100, 120)
            S0, variance, mu, lambda, muJ, deltaJ, T, rate = 100.0, 0.04, 0.03, 2.0, -0.2, 0.2, 1.0, 0.05
            model = Bates(Heston(S0, variance, mu, 2, variance, 1e-4, 0), lambda, muJ, deltaJ)
            log_moment = muJ + deltaJ^2 / 2
            series = sum(0:80) do n
                weight = exp(-lambda * T + n * log(lambda * T) - loggamma(n + 1))
                F = S0 * exp(mu * T - lambda * expm1(log_moment) * T + n * log_moment)
                return weight * _black_call(F, K, variance * T + n * deltaJ^2)
            end
            @test isapprox(european_price(model, K, T; rate), exp(-rate * T) * series; atol = 1e-8)
        end

        @testset "Against simulation, K = $K" for K in (80, 100, 120)
            heston = Heston(S0 = 100, v0 = 0.09, mu = 0.03, kappa = 1.5, theta = 0.04, xi = 0.8, rho = -0.7)
            @testset "$(nameof(typeof(model)))" for model in (heston, Bates(heston, 0.5, -0.1, 0.2))
                terminal = simulate_terminal(model, 2.0; paths = 2^18, steps = 200, numbers = PseudoRandom(seed = 41))
                @test zscore(exp(-0.06) .* max.(terminal .- K, 0), european_price(model, K, 2.0; rate = 0.03)) < 4
                replicates = 32
                quasi = simulate_terminal(model, 2.0; paths = replicates * 2^13, steps = 200, numbers = QuasiRandom(seed = 42; replicates))
                means = [mean(max.(block .- K, 0)) for block in Iterators.partition(quasi, 2^13)]
                @test zscore(exp(-0.06) .* means, european_price(model, K, 2.0; rate = 0.03)) < 4
            end
        end

        @testset "Input validation" begin
            model = Heston(S0 = 100, v0 = 0.04, mu = 0, kappa = 1, theta = 0.04, xi = 0.3, rho = 0)
            @test_throws DomainError european_price(model, 0, 1.0; rate = 0)
            @test_throws DomainError european_price(model, 100, 0.0; rate = 0)
        end
    end

    @testset "Monte Carlo prices" begin
        heston = Heston(S0 = 100, v0 = 0.09, mu = 0.03, kappa = 1.5, theta = 0.04, xi = 0.8, rho = -0.7)
        bates = Bates(heston, 0.5, -0.1, 0.2)
        sabr = SABR(100, 0.2, 1.0, -0.3, 0.4)
        pseudo = PseudoRandom(seed = 51)
        quasi = QuasiRandom(seed = 52, replicates = 32)

        @testset "Against the closed form, $(nameof(typeof(model))), $(nameof(typeof(numbers)))" for model in (heston, bates), numbers in (pseudo, quasi)
            @testset "call = $call" for call in (true, false)
                strikes = [80, 100, 120]
                results = monte_carlo_price(model, strikes, 2.0; rate = 0.05, call, paths = 2^18, steps = 200, numbers)
                @test all(abs(result.price - european_price(model, K, 2.0; rate = 0.05, call)) < 4 * result.standard_error for (K, result) in zip(strikes, results))
            end
        end

        @testset "SABR with almost constant volatility is Black" begin
            model = SABR(100, 0.2, 1.0, 0.0, 1e-4)
            result = monte_carlo_price(model, 110, 1.0; rate = 0.02, paths = 2^18, numbers = pseudo)
            @test abs(result.price - exp(-0.02) * _black_call(100, 110, 0.04)) < 4 * result.standard_error
        end

        @testset "Discounted payoffs and standard errors, $(nameof(typeof(model)))" for model in (heston, bates, sabr)
            paths = 32 * 2^8
            discount = exp(-0.05 * 2.0)
            payoffs = discount .* max.(simulate_terminal(model, 2.0; paths, steps = 16, numbers = pseudo) .- 100, 0)
            result = monte_carlo_price(model, 100, 2.0; rate = 0.05, paths, steps = 16, numbers = pseudo)
            @test collect(result) ≈ [mean(payoffs), std(payoffs) / sqrt(paths)]
            payoffs = discount .* max.(simulate_terminal(model, 2.0; paths, steps = 16, numbers = quasi) .- 100, 0)
            means = [mean(block) for block in Iterators.partition(payoffs, 2^8)]
            result = monte_carlo_price(model, 100, 2.0; rate = 0.05, paths, steps = 16, numbers = quasi)
            @test collect(result) ≈ [mean(payoffs), std(means) / sqrt(32)]
        end

        @testset "Put-call parity on the same paths, $(nameof(typeof(model)))" for model in (heston, bates, sabr)
            terminal = simulate_terminal(model, 1.0; paths = 2^10, steps = 16, numbers = pseudo)
            arguments = (; rate = 0.05, paths = 2^10, steps = 16, numbers = pseudo)
            difference = monte_carlo_price(model, 100, 1.0; arguments...).price - monte_carlo_price(model, 100, 1.0; call = false, arguments...).price
            @test difference ≈ exp(-0.05) * (mean(terminal) - 100)
        end

        @testset "Input validation" begin
            @test_throws DomainError monte_carlo_price(heston, 0, 1.0; rate = 0)
            @test_throws DomainError monte_carlo_price(heston, 100, 0.0; rate = 0)
            @test_throws ArgumentError monte_carlo_price(heston, 100, 1.0; rate = 0, paths = 1)
            @test_throws ArgumentError monte_carlo_price(heston, 100, 1.0; rate = 0, paths = 1000, numbers = quasi)
        end
    end

    @testset "Hagan formula against QuantLib" begin
        rows = [parse.(Float64, split(line)) for line in eachline(joinpath(@__DIR__, "hagan_quantlib.txt")) if !startswith(line, "#")]
        differences = map(rows) do (F0, alpha, beta, rho, nu, T, K, volatility, call, put)
            model = SABR(F0, alpha, beta, rho, nu)
            return (
                abs(hagan_implied_vol(model, K, T) / volatility - 1),
                abs(hagan_price(model, K, T; rate = 0.05) - call) / F0,
                abs(hagan_price(model, K, T; rate = 0.05, call = false) - put) / F0,
            )
        end
        @test maximum(maximum, differences) < 1e-13
    end

    @testset "Heston against QuantLib and NMOF" begin
        rows = [parse.(Float64, split(line)) for line in eachline(joinpath(@__DIR__, "heston_references.txt")) if !startswith(line, "#")]
        differences = map(rows) do (S0, v0, mu, kappa, theta, xi, rho, rate, T, K, quantlib_call, quantlib_put, nmof_call)
            model = Heston(S0, v0, mu, kappa, theta, xi, rho)
            call = european_price(model, K, T; rate)
            return (max(abs(call - quantlib_call), abs(european_price(model, K, T; rate, call = false) - quantlib_put)), abs(call - nmof_call))
        end
        @test maximum(first, differences) < 1e-11
        @test maximum(last, differences) < 2e-6
    end

    # BatesEngine of QuantLib has not converged at xi of 1 or more, so there only NMOF is the reference.
    @testset "Bates against QuantLib and NMOF" begin
        rows = [parse.(Float64, split(line)) for line in eachline(joinpath(@__DIR__, "bates_references.txt")) if !startswith(line, "#")]
        differences = map(rows) do (S0, v0, mu, kappa, theta, xi, rho, lambda, muJ, deltaJ, rate, T, K, quantlib_call, quantlib_put, nmof_call)
            model = Bates(Heston(S0, v0, mu, kappa, theta, xi, rho), lambda, muJ, deltaJ)
            call = european_price(model, K, T; rate)
            quantlib = xi < 1 ? max(abs(call - quantlib_call), abs(european_price(model, K, T; rate, call = false) - quantlib_put)) : 0.0
            return (quantlib, abs(call - nmof_call))
        end
        @test maximum(first, differences) < 2e-8
        @test maximum(last, differences) < 2e-6
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
