#!/usr/bin/env julia
# Cross-check worker reproducing IdeaBio.jl/src/omics function bodies verbatim
# (numeric core), operating on a plain matrix with NaN as the missing marker.
# Loading the full IdeaBio module is impractical (Makie/RCall/CairoMakie dep
# tree), so we reproduce the exact operations from:
#   normalisations.jl::normalise_zscore   (per-column, skip missing; StatsBase.zscore
#                                           == (v .- mean(v)) ./ std(v), corrected n-1)
#   transformations.jl::transform_log2     (x>0 ? log2(x) : missing)
#   transformations.jl::transform_log10    (x>0 ? log10(x) : missing)
#   imputations.jl::impute_default_metaboanalyst (0.2 * min(non-missing) per column)
using Statistics
using DelimitedFiles

d = ARGS[1]
read_m(f) = readdlm(joinpath(d, f), ',', Float64)

function zscore_cols(x)             # normalise_zscore
    y = copy(x)
    for j in 1:size(x, 2)
        col = x[:, j]
        ids = .!isnan.(col)
        v = col[ids]
        y[ids, j] = (v .- mean(v)) ./ std(v)   # std corrected (n-1) == StatsBase.zscore
    end                                         # NaN entries are left untouched
    return y
end

function log2t(x)                   # transform_log2
    y = copy(x)
    for i in eachindex(y)
        if !isnan(y[i])
            y[i] = y[i] > 0 ? log2(y[i]) : NaN
        end
    end
    return y
end

function log10t(x)                  # transform_log10
    y = copy(x)
    for i in eachindex(y)
        if !isnan(y[i])
            y[i] = y[i] > 0 ? log10(y[i]) : NaN
        end
    end
    return y
end

function metaboanalyst(x)           # impute_default_metaboanalyst
    y = copy(x)
    for j in 1:size(x, 2)
        col = x[:, j]
        mn = minimum(col[.!isnan.(col)])        # min of NON-missing (not positive-filtered)
        for i in 1:size(x, 1)
            if isnan(y[i, j])
                y[i, j] = 0.2 * mn
            end
        end
    end
    return y
end

xc = read_m("x_complete_plain.csv")
xm = read_m("x_missing_plain.csv")
writedlm(joinpath(d, "jl_zscore_complete.csv"), zscore_cols(xc), ',')
writedlm(joinpath(d, "jl_zscore_missing.csv"),  zscore_cols(xm), ',')
writedlm(joinpath(d, "jl_log2_complete.csv"),   log2t(xc), ',')
writedlm(joinpath(d, "jl_log10_complete.csv"),  log10t(xc), ',')
writedlm(joinpath(d, "jl_metaboanalyst.csv"),   metaboanalyst(xm), ',')
println("Julia reference outputs written.")
