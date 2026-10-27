from __future__ import annotations

import anndata as ad
import harmonypy
import numpy as np
import pandas as pd
import pooch
import rapids_singlecell as rsc
from _report import capture, package_version, write_report

METHOD = "scanpy_core_harmony"

# Pinned to the commit before harmonypy replaced the harmonized file with an R harmony2 reference.
DATA = "https://github.com/slowkow/harmonypy/raw/3cb114c/data"
pcs_path = pooch.retrieve(f"{DATA}/pbmc_3500_pcs.tsv.gz", known_hash="md5:27e319b3ddcc0c00d98e70aa8e677b10")
meta_path = pooch.retrieve(f"{DATA}/pbmc_3500_meta.tsv.gz", known_hash="md5:8c7ca20e926513da7cf0def1211baecb")
harmonized_path = pooch.retrieve(
    f"{DATA}/pbmc_3500_pcs_harmonized.tsv.gz", known_hash="md5:a7c4ce4b98c390997c66d63d48e09221"
)
pcs = pd.read_csv(pcs_path, delimiter="\t").to_numpy()
meta = pd.read_csv(meta_path, delimiter="\t")


def run_rsc(flavor: str, **kwargs) -> np.ndarray:
    adata = ad.AnnData(X=None, obs=meta.copy(), obsm={"X_pca": pcs.copy()})
    rsc.pp.harmony_integrate(
        adata, key="donor", basis="X_pca", adjusted_basis="X_pca_harmony", flavor=flavor, random_state=0, **kwargs
    )
    rsc.get.anndata_to_CPU(adata)
    return adata.obsm["X_pca_harmony"]


# harmony1: the stored R harmony output harmonypy 0.x tested against; harmonypy 2.x no longer implements it.
capture(
    METHOD,
    "harmony1",
    reference=pd.read_csv(harmonized_path, delimiter="\t").to_numpy(),
    candidate=run_rsc("harmony1", max_iter_harmony=20, max_iter_clustering=20),
)

# harmony2: the rsc default, against harmonypy 2.x run now, both at their default stopping rules.
# harmonypy before 2.0 implements harmony1 only, so there is no CPU harmony2 to compare against.
if int(package_version("harmonypy").split(".")[0]) >= 2:
    result = harmonypy.run_harmony(pcs.astype(np.float64), meta, "donor", random_state=0, verbose=False)
    reference = np.asarray(result.Z_corr)
    if reference.shape != pcs.shape:
        reference = reference.T
    capture(METHOD, "harmony2", reference=reference, candidate=run_rsc("harmony2"))

write_report(
    METHOD,
    "Harmonypy PBMC 3,500-cell donor benchmark: harmony1 against the stored R harmony output, "
    "harmony2 (the rsc default) against harmonypy 2.x",
    "iterative",
    [],
    reference_package="harmonypy",
    packages=("harmonypy",),
    shape=pcs.shape,
)
