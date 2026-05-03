using Documenter
using StochasticVolatility

DocMeta.setdocmeta!(
    StochasticVolatility,
    :DocTestSetup,
    :(using StochasticVolatility);
    recursive = true,
)

makedocs(
    modules = [StochasticVolatility],
    sitename = "StochasticVolatility.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        edit_link = nothing,
        repolink = nothing,
    ),
    pages = [
        "Home" => "index.md",
        "API Reference" => "api.md",
    ],
    checkdocs = :exports,
    remotes = nothing,
)
