library(purrr)
library(tidyr)
library(MASS)
library(reticulate)

tf <- import("tensorflow")
SpatialDE <- import_from_path("SpatialDE", "~/SpatialDE/")
ad <- import("anndata")
gc <- import("gc")

set.seed(42)

npoints <- 2e4
repeats <- 200
dispersion <- 0.1
m <- 5

X <- matrix(runif(npoints*2, -100, 100), nrow=npoints)

sizefactors <- 1 + runif(npoints) * 50
means <- m * sizefactors

Y <- replicate(repeats, rnegbin(npoints, means, 1/dispersion))

times <- map_dfr(c(200, 500, 1000, 2500, 5000, 7500, 10000, 15000, 20000), function(n) {
    idx <- sample(npoints, n)

    currX <- X[idx,]
    currY <- Y[idx,]

    adata <- ad$AnnData(X=currY)
    py_set_item(adata$obsm, "spatial", currX)

    omnibus <- system.time(SpatialDE$test(adata, omnibus=TRUE, use_cache=n < 1.1e4))
    gc$collect()
    cauchy <- system.time(SpatialDE$test(adata, omnibus=FALSE, use_cache=n < 1.1e4))
    gc$collect()
    tibble(npoints=n, omnibus=omnibus[3], cauchy=cauchy[3])
}) %>%
    pivot_longer(-npoints, names_to="method", values_to="time")

saveRDS(times, paste0("tests_runtime_gpu.rds"))
