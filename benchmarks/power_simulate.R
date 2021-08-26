library(Seurat)
library(SPARSim)
library(MASS)
library(doMC)
library(reticulate)

ncores <- as.integer(Sys.getenv("LSB_DJOB_NUMPROC"))
if (is.na(ncores))
    ncores <- 1
registerDoMC(cores=ncores)

ad <- import("anndata")
sp <- import("scipy.sparse")

brain <- Load10X_Spatial("data/c2l_mouse_brain/ST8059048/")
brain[["percent.mt"]] <- PercentageFeatureSet(brain, pattern = "^MT-")
brain <- subset(brain, nCount_Spatial > 4000 & percent.mt < 20)
brain <- brain[rowSums(brain) > 100,]
data <- GetAssayData(GetAssay(brain))
norm_data <- scran_normalization(data)
sparsim_params <- SPARSim_estimate_parameter_from_data(raw_data=as.matrix(data), norm_data=norm_data, conditions=list(A=1:ncol(data)))
coords <- GetTissueCoordinates(brain)

rescale_pattern <- function(pattern, fc) {
    # x * exp(max(pattern)) / (x * exp(min(pattern))) == fc
    exp(pattern / diff(range(pattern)) * log(fc))
}

ngenes <- nrow(data)
nse_genes <- 1000

distances <- dist(coords)
min_dist <- 2 * min(distances)
max_dist <- max(distances)

lengthscales <- 10^(seq(from=log10(min_dist), to=log10(max_dist), length.out=10))
distances <- as.matrix(distances)^2

message("starting simulation...")

dir.create("simulated")
foreach (rep=1:10, .combine=function(...)NULL) %:%
    foreach (lengthscale=lengthscales, lengthscaleidx=1:length(lengthscales), .combine=function(...)NULL) %:%
        foreach (foldchange=c(2, 4, 6, 8, 10), .combine=function(...)NULL) %dopar% {
            set.seed(rep)

            covmat <- exp(-distances / lengthscale^2)
            pattern <- rescale_pattern(mvrnorm(mu=rep(0, nrow(distances)), Sigma=exp(-distances/lengthscale^2)), foldchange)

            params <- sparsim_params[[1]]
            idx <- sample(ngenes)
            params$intensity <- params$intensity[idx]
            params$variability <- params$variability[idx]
            names(params$intensity) <- names(params$variability) <- paste0("gene", 1:ngenes)

            params <- mapply(function(p, n) {
                multiplier <- c(rep(p, nse_genes), rep(1, ngenes - nse_genes))
                SPARSim_create_DE_genes_parameter(sim_param=params, fc_multiplier=multiplier, N_cells=1, condition_name=n)
            }, pattern, rownames(coords), SIMPLIFY=FALSE)

            simulated <- SPARSim_simulation(params)
            #colnames(simulated$count_matrix) <- gsub("_cell1", "", colnames(simulated$count_matrix), fixed=TRUE)

            adata <- ad$AnnData(X=sp$csr_matrix(t(simulated$count_matrix)))
            adata$obs_names <- rownames(coords)
            adata$var_names <- rownames(simulated$count_matrix)
            adata$var[["spatially_variable"]] <- c(rep(TRUE, nse_genes), rep(FALSE, ngenes - nse_genes))
            py_set_item(adata$obsm, "spatial", as.matrix(coords))
            py_set_item(adata$uns, "simulation_params", dict(lengthscale=lengthscale, foldchange=foldchange))
            py_set_item(adata$obsm, "pattern", as.matrix(pattern))

            fname <- file.path("simulated", paste0("c2l_mouse_brain_lengthscale_", lengthscaleidx, "_fc_", foldchange, "_rep_", rep, ".h5ad"))
            message(paste0("Writing to file ", fname))
            adata$write_h5ad(fname, compression="gzip", compression_opts=9)
        }

