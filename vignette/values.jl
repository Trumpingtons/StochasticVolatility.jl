# The Julia values of the vignette, from the reference files of test/ and the stored results of benchmark/. No simulation here.
# Run from the package folder: julia --project=. vignette/values.jl
using StochasticVolatility, Printf
const Root = dirname(@__DIR__)
rows(file) = [split(line) for line in eachline(joinpath(Root, file)) if !startswith(line, "#")]
numbers(file) = [parse.(Float64, row) for row in rows(file)]

println("American puts under Heston, MCS-IT: table v0 S0 grid published ours")
heston = numbers("test/finite_difference_heston_published.txt")
kink = 0
same = 0
furthest = 0.0
for row in heston
	S0, v0, kappa, theta, xi, rho, rate, T, K, m1, m2, N, published = row
	price = finite_difference_price(Heston(S0, v0, rate, kappa, theta, xi, rho), K, T; rate, call = false, price_points = Int(m1), variance_points = Int(m2), steps = Int(N))
	round(price, digits = 4) == published || @printf("differs: v0 %.4f S0 %.0f %d x %d x %d: %.4f %.6f\n", v0, S0, m1, m2, N, published, price)
	global same += round(price, digits = 4) == published
	at_kink = abs(published - (K - S0)) < 0.01
	global kink += at_kink
	at_kink || (global furthest = max(furthest, abs(price - published)))
	(m1, N) in ((300.0, 75.0), (300.0, 60.0)) && @printf("v0 %.4f S0 %.0f %d x %d x %d: %.4f %.4f %s\n", v0, S0, m1, m2, N, published, price, at_kink ? "at the kink" : "")
end
println(length(heston), " values, ", same, " to every printed digit, ", kink, " within a cent of the exercise value, furthest of the others ", furthest)

println("\nAmerican calls under Bates: S0 published fd_600 fd_1200 (stored) quantlib_100x100x50 (stored)")
quantlib = Dict(parse(Float64, row[4]) => row[5] for row in rows("benchmark/quantlib_bates_american.txt") if row[1] == "100")
for row in rows("benchmark/finite_difference_bates_published.txt")
	S0 = parse(Float64, row[1])
	model = Bates(S0 = S0, v0 = 0.04, mu = -0.02, kappa = 2, theta = 0.04, xi = 0.25, rho = -0.5, lambda = 0.2, muJ = -0.58, deltaJ = 0.4)
	price = finite_difference_price(model, 100, 0.5; rate = 0.03, call = true, price_points = 600, variance_points = 300, steps = 150)
	@printf("%.0f %s %.6f %s %s\n", S0, row[2], price, row[5], quantlib[S0])
end

println("\nAmerican under SABR, 400 x 50 x 64: type K quantlib ours gap")
model = SABR(100.0, 2.0, 0.5, -0.7, 0.8)
largest = 0.0
for (rate, type, K, reference) in rows("test/finite_difference_sabr_quantlib.txt")
	price = finite_difference_price(model, parse(Float64, K), 0.25; rate = parse(Float64, rate), call = type == "call", price_points = 400, volatility_points = 50, steps = 64)
	gap = price - parse(Float64, reference)
	global largest = max(largest, abs(gap))
	@printf("%s %s %s %.6f %.6f\n", type, K, reference, price, gap)
end
println("largest gap ", largest)

println("\nClosed forms and the Hagan formula: rows, largest differences")
hagan = numbers("test/hagan_quantlib.txt")
println("Hagan, relative: ", length(hagan), " ", maximum(hagan) do (F0, alpha, beta, rho, nu, T, K, volatility, call, put)
	sabr = SABR(F0, alpha, beta, rho, nu)
	return max(abs(hagan_implied_vol(sabr, K, T) / volatility - 1), abs(hagan_price(sabr, K, T; rate = 0.05) - call) / F0, abs(hagan_price(sabr, K, T; rate = 0.05, call = false) - put) / F0)
end)
stored = numbers("test/heston_references.txt")
gaps = map(stored) do (S0, v0, mu, kappa, theta, xi, rho, rate, T, K, ql_call, ql_put, nmof)
	m = Heston(S0, v0, mu, kappa, theta, xi, rho)
	call = european_price(m, K, T; rate)
	return (max(abs(call - ql_call), abs(european_price(m, K, T; rate, call = false) - ql_put)), abs(call - nmof))
end
println("Heston: ", length(stored), " QuantLib ", maximum(first, gaps), " NMOF ", maximum(last, gaps))
stored = numbers("test/bates_references.txt")
gaps = map(stored) do (S0, v0, mu, kappa, theta, xi, rho, lambda, muJ, deltaJ, rate, T, K, ql_call, ql_put, nmof)
	m = Bates(Heston(S0, v0, mu, kappa, theta, xi, rho), lambda, muJ, deltaJ)
	call = european_price(m, K, T; rate)
	return (xi < 1 ? max(abs(call - ql_call), abs(european_price(m, K, T; rate, call = false) - ql_put)) : 0.0, abs(call - nmof))
end
println("Bates: ", length(stored), " QuantLib ", maximum(first, gaps), " (", count(row -> row[6] < 1, stored), " rows with xi < 1) NMOF ", maximum(last, gaps))

println("\nThe worked example")
heston_model = Heston(S0 = 10, v0 = 0.0625, mu = 0.1, kappa = 5, theta = 0.16, xi = 0.9, rho = 0.1)
@printf("Heston put K 10: American %.4f European %.4f\n", finite_difference_price(heston_model, 10, 0.25; rate = 0.1, call = false), european_price(heston_model, 10, 0.25; rate = 0.1, call = false))
sabr_model = SABR(100.0, 2.0, 0.5, -0.7, 0.8)
@printf("SABR put K 100: American %.4f European %.4f Hagan %.4f\n", finite_difference_price(sabr_model, 100, 0.25; rate = 0.05, call = false), finite_difference_price(sabr_model, 100, 0.25; rate = 0.05, call = false, american = false), hagan_price(sabr_model, 100, 0.25; rate = 0.05, call = false))
