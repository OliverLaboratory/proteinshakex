# -*- coding: utf-8 -*-
"""
CFOLD multi-conformation dataset.

Binary conformational ensembles — each protein has exactly 2 experimental
structures representing different conformational states, stored as separate
PDB files grouped by ensemble ID.
"""
from proteinshake.datasets.dance import DANCEDataset


class CFOLDDataset(DANCEDataset):
    """Multi-conformation dataset from the CFOLD benchmark.

    Each protein ensemble contains exactly 2 experimental structures
    representing different conformational states of the same protein.
    Inherits all parsing logic from DANCEDataset since the data format
    is identical.

    Parameters
    ----------
    root : str
        Data root directory for processed output.
    cfold_path : str
        Path to the CFOLD data directory containing ``input_pdbs/``
        and ``input_dataset.csv``.
    use_filtered : bool
        If True, only include ensembles from ``filtered_ensembles.csv``.
    """

    exlude_args_from_signature = ['cfold_path']

    def __init__(self,
                 root='data',
                 cfold_path='../ConformationDiscoveryBenchmark/data/inputs/CFOLD',
                 use_filtered=True,
                 use_precomputed=True,
                 n_jobs=1,
                 verbosity=2,
                 **kwargs):
        # Pass cfold_path as dance_path to the parent
        super().__init__(
            root=root,
            dance_path=cfold_path,
            use_filtered=use_filtered,
            use_precomputed=use_precomputed,
            n_jobs=n_jobs,
            verbosity=verbosity,
            **kwargs,
        )

    @property
    def name(self):
        return 'CFOLDDataset'
