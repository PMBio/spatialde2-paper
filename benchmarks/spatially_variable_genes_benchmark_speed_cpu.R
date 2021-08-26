library(purrr)
library(tidyr)
library(MASS)
library(reticulate)
library(SPARK)

tf <- import("tensorflow")
SpatialDE <- import_from_path("SpatialDE", "~/SpatialDE/")
SpatialDE1 <- import_from_path("SpatialDE1", "~/SpatialDE-1.1.0/Python-module/")
ad <- import("anndata")

set.seed(42)

ncores <- as.integer(Sys.getenv("LSB_DJOB_NUMPROC"))
tf$config$threading$set_inter_op_parallelism_threads(ncores)
tf$config$threading$set_intra_op_parallelism_threads(ncores)

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

    omnibus <- system.time(SpatialDE$test(adata, omnibus=TRUE))
    cauchy <- system.time(SpatialDE$test(adata, omnibus=FALSE))

    loc <- as.data.frame(currX)
    rownames(currY) <- rownames(loc)
    colnames(currY) <- 1:ncol(currY)
    spatialde1 <- system.time(SpatialDE1$run(loc, as.data.frame(currY)))

    spark <- CreateSPARKObject(counts = t(currY), location = as.data.frame(currX),
                               percentage = 0, min_total_counts = 1)
    spark@lib_size <- apply(spark@counts, 2, sum)
    sparktime <- system.time({
        spark <- spark.vc(spark, covariates = NULL, lib_size = spark@lib_size,
                          verbose = F, fit.maxiter = 500, num_core=ncores)
        spark <- spark.test(spark, check_positive = T, verbose = F)
    })
    sparktime_nocheck <- system.time({
        spark <- spark.vc(spark, covariates = NULL, lib_size = spark@lib_size,
                          verbose = F, fit.maxiter = 500, num_core=ncores)
        spark <- spark.test(spark, check_positive = F, verbose = F)
    })
    tibble(npoints=n, omnibus=omnibus[3], cauchy=cauchy[3], spark=sparktime[3], spark_nocheckpositive=sparktime_nocheck[3], spatialde1=spatialde1[3])
}) %>%
    pivot_longer(-npoints, names_to="method", values_to="time")

saveRDS(times, paste0("tests_runtime_", ncores, "cores.rds"))
