import numpy as np
import rapids_singlecell as rsc
import scanpy as sc

from benchmarks.comparison.constants import ASSERT_ALLCLOSE_KWARGS

adata_sc = sc.datasets.pbmc3k()
adata_rsc = adata_sc.copy()
rsc.get.anndata_to_GPU(adata_rsc)
sc.pp.normalize_total(adata_sc)
rsc.pp.normalize_total(adata_rsc)
rsc.get.anndata_to_CPU(adata_rsc)
np.testing.assert_allclose(adata_sc.X.toarray(), adata_rsc.X.toarray(), **ASSERT_ALLCLOSE_KWARGS)
