import anndata as ad
import harmonypy
import numpy as np
import pandas as pd
import pooch
import rapids_singlecell as rsc
from scipy.stats import pearsonr

X_pca_file = pooch.retrieve(
    "https://github.com/slowkow/harmonypy/raw/3cb114c/data/pbmc_3500_pcs.tsv.gz",
    known_hash="md5:27e319b3ddcc0c00d98e70aa8e677b10",
)
X_pca = pd.read_csv(X_pca_file, delimiter="\t")

meta_file = pooch.retrieve(
    "https://github.com/slowkow/harmonypy/raw/3cb114c/data/pbmc_3500_meta.tsv.gz",
    known_hash="md5:8c7ca20e926513da7cf0def1211baecb",
)
meta = pd.read_csv(meta_file, delimiter="\t")

adata = ad.AnnData(X=None, obs=meta, obsm={"X_pca": X_pca.values})

X_pca_harmony_ref = harmonypy.run_harmony(X_pca.values, meta, "donor", max_iter_harmony=20).Z_corr
if X_pca_harmony_ref.shape != X_pca.shape:
    X_pca_harmony_ref = X_pca_harmony_ref.T
rsc.pp.harmony_integrate(adata, key="donor", max_iter_harmony=20)

# Compare against harmonypy reference embedding per PC
corr = np.array(
    [pearsonr(adata.obsm["X_pca_harmony"][:, i], X_pca_harmony_ref[:, i])[0] for i in range(X_pca_harmony_ref.shape[1])]
)
l2 = np.linalg.norm(adata.obsm["X_pca_harmony"] - X_pca_harmony_ref, axis=0) / np.linalg.norm(X_pca_harmony_ref, axis=0)
assert corr.min() > 0.95
assert l2.max() < 0.1
