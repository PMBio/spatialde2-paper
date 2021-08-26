library(Matrix)
library(dplyr)
library(purrr)
library(tidyr)
library(magrittr)
library(tibble)
library(reticulate)
library(SPARK)

tf <- import("tensorflow")
NaiveDE <- import("NaiveDE")
SpatialDE <- import_from_path("SpatialDE", "~/SpatialDE/")
SpatialDE1 <- import_from_path("SpatialDE1", "~/SpatialDE-1.1.0/Python-module/")
ad <- import("anndata")

ncores <- as.integer(Sys.getenv("LSB_DJOB_NUMPROC"))
tf$config$threading$set_inter_op_parallelism_threads(ncores)
tf$config$threading$set_intra_op_parallelism_threads(ncores)

dir.create("benchmarks")
results <- map_dfr(1:10, function(rep){
    res <- map_dfr(1:10, function(lengthscaleidx) {
        map_dfr(c(2,4,6,8,10), function(foldchange) {
            message(sprintf("BENCHMARKING rep %d lengthscale %d foldchange %d", rep, lengthscaleidx, foldchange))

            adata <- ad$read_h5ad(file.path("simulated", paste0("c2l_mouse_brain_lengthscale_", lengthscaleidx, "_fc_", foldchange, "_rep_", rep, ".h5ad")))

            spark <- CreateSPARKObject(counts = set_rownames(set_colnames(t(as.matrix(adata$X)), as.character(adata$obs_names$to_numpy())), adata$var_names$to_numpy()), location = as.data.frame(set_rownames(adata$obsm[["spatial"]], adata$obs_names$to_numpy())),
                                       percentage = 0, min_total_counts = 0)
            spark@lib_size <- apply(spark@counts, 2, sum)
            spark <- spark.vc(spark, covariates = NULL, lib_size = spark@lib_size,
                              verbose = F, fit.maxiter = 500, num_core=ncores)
            spark1 <- spark.test(spark, check_positive = T, verbose = F)
            spark2 <- spark.test(spark, check_positive = F, verbose = F)

            sparkres1 <- rownames_to_column(spark1@res_mtest, 'gene') %>%
                select(gene, pval=combined_pvalue, p.adj=adjusted_pvalue) %>%
                mutate(method='spark')
            sparkres2 <- rownames_to_column(spark2@res_mtest, 'gene') %>%
                select(gene, pval=combined_pvalue, p.adj=adjusted_pvalue) %>%
                mutate(method='spark_nocheckpositive')

            omnibus <- SpatialDE$test(adata, omnibus=TRUE)[[1]] %>%
                select(gene, pval, padj) %>%
                mutate(method='omnibus')
            cauchy <- SpatialDE$test(adata, omnibus=FALSE)[[1]] %>%
                select(gene, pval, padj) %>%
                mutate(method='cauchy')

            dfm <- t(as.matrix(adata$X)) %>%
                NaiveDE$stabilize()
            res <- NaiveDE$regress_out(data.frame(total_counts=rowSums(adata$X)), dfm, 'np.log(total_counts)') %>%
                t()
            colnames(res) <- adata$var_names$to_numpy()
            rownames(res) <- adata$obs_names$to_numpy()
            res <- as.data.frame(res)
            spde1 <- SpatialDE1$run(adata$obsm[["spatial"]], res) %>%
                select(gene=g, pval, p.adj=qval) %>%
                mutate(method="spatialde1")

            bind_rows(sparkres1, sparkres2, omnibus, cauchy, spde1) %>%
                separate(gene, c(NA, 'gene'), sep='gene', convert=TRUE)
        }, .id="foldchange")
    }, .id="lengthscale")
    saveRDS(res, file.path("benchmarks", paste0("sparsim_benchmark_rep_", rep, ".rds")))
    res
}, .id="rep")

saveRDS(results, "sparsim_benchmark.rds")
