"""PyTorch-compatible data loader for ProteinShake datasets.

Provides ``ProteinShakeLoader``, a ``torch.utils.data.Dataset`` that serves
protein records from an indexed :class:`ProteinStore` with O(1) random access.
Supports splits, transforms, and multi-worker DataLoaders out of the box.

Usage::

    from proteinshake.loader import ProteinShakeLoader

    loader = ProteinShakeLoader.from_dataset(
        dataset,                      # any ProteinShake Dataset instance
        resolution='residue',
    )
    protein = loader[0]               # dict with 'protein', 'residue' keys
    train = loader.split('train')     # subset for training
    dl = train.dataloader(batch_size=32, num_workers=4)
"""

import os
from pathlib import Path

import numpy as np

from proteinshake.utils.protein_store import ProteinStore


class ProteinShakeLoader:
    """PyTorch Dataset backed by an indexed ProteinStore.

    Parameters
    ----------
    store : ProteinStore
        The underlying data store.
    indices : np.ndarray or None
        Subset of indices into the store (for splits).  ``None`` = all.
    transform : callable or None
        Applied to each protein dict on access.
    """

    def __init__(self, store, indices=None, transform=None):
        self._store = store
        self._indices = indices
        self._transform = transform

    # ------------------------------------------------------------------
    # torch.utils.data.Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        if self._indices is not None:
            return len(self._indices)
        return len(self._store)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]

        idx = int(idx)
        if idx < 0:
            idx += len(self)

        real_idx = int(self._indices[idx]) if self._indices is not None else idx
        protein = self._store[real_idx]

        if self._transform is not None:
            protein = self._transform(protein)

        return protein

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    # ------------------------------------------------------------------
    # Splits
    # ------------------------------------------------------------------

    def split(self, split_name, split_key=None):
        """Return a new loader restricted to proteins in the given split.

        Scans the store once to find indices matching ``split_name``.
        The result is cached so subsequent calls are instant.

        Parameters
        ----------
        split_name : str
            One of ``'train'``, ``'val'``, ``'test'``.
        split_key : str, optional
            The key in ``protein['protein']`` containing the split label.
            If ``None``, auto-detects from common patterns
            (``random_split``, ``sequence_split_0.7``, etc.).

        Returns
        -------
        ProteinShakeLoader
            A new loader over the subset.
        """
        cache_attr = f'_split_cache_{split_name}_{split_key}'
        if hasattr(self, cache_attr):
            return getattr(self, cache_attr)

        indices = []
        for i in range(len(self._store)):
            protein = self._store[i]
            prot_meta = protein.get('protein', {})

            if split_key:
                if prot_meta.get(split_key) == split_name:
                    indices.append(i)
            else:
                # Auto-detect split key
                for key in sorted(prot_meta.keys()):
                    if 'split' in key and prot_meta[key] == split_name:
                        indices.append(i)
                        break

        result = ProteinShakeLoader(
            self._store,
            indices=np.array(indices, dtype=np.int64),
            transform=self._transform,
        )
        setattr(self, cache_attr, result)
        return result

    def subset(self, indices):
        """Return a new loader restricted to the given indices.

        Parameters
        ----------
        indices : array-like
            Integer indices into this loader.

        Returns
        -------
        ProteinShakeLoader
        """
        indices = np.asarray(indices, dtype=np.int64)
        if self._indices is not None:
            indices = self._indices[indices]
        return ProteinShakeLoader(self._store, indices=indices, transform=self._transform)

    def with_transform(self, transform):
        """Return a new loader with the given transform applied on access.

        Parameters
        ----------
        transform : callable
            Function that takes a protein dict and returns a transformed version.

        Returns
        -------
        ProteinShakeLoader
        """
        return ProteinShakeLoader(self._store, indices=self._indices, transform=transform)

    # ------------------------------------------------------------------
    # DataLoader convenience
    # ------------------------------------------------------------------

    def dataloader(self, batch_size=32, num_workers=0, shuffle=True, **kwargs):
        """Create a ``torch.utils.data.DataLoader`` for this loader.

        Parameters
        ----------
        batch_size : int
        num_workers : int
        shuffle : bool
        **kwargs
            Passed to ``DataLoader``.

        Returns
        -------
        torch.utils.data.DataLoader
        """
        from torch.utils.data import DataLoader

        return DataLoader(
            self,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            collate_fn=kwargs.pop('collate_fn', None),
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Construction from ProteinShake Dataset
    # ------------------------------------------------------------------

    @classmethod
    def from_dataset(cls, dataset, resolution='residue', transform=None, verbosity=2):
        """Build a loader from a ProteinShake Dataset instance.

        On first call, converts the dataset's avro file(s) into an indexed
        ProteinStore (cached to disk).  Subsequent calls load the existing
        store instantly.

        Parameters
        ----------
        dataset : proteinshake.datasets.Dataset
            Any ProteinShake dataset.
        resolution : str
            ``'residue'`` or ``'atom'``.
        transform : callable, optional
            Applied to each protein on access.
        verbosity : int
            Logging verbosity.

        Returns
        -------
        ProteinShakeLoader
        """
        avro_files = dataset._find_avro_files(resolution)
        if not avro_files:
            dataset.download_precomputed(resolution=resolution)
            avro_files = dataset._find_avro_files(resolution)
        if not avro_files:
            raise FileNotFoundError(
                f"No avro files found for {dataset.name}.{resolution}"
            )

        store_path = os.path.join(dataset.root, f'{dataset.name}.{resolution}.store')
        store = ProteinStore.from_avro(avro_files, store_path, verbosity=verbosity)

        return cls(store, transform=transform)

    @classmethod
    def from_store(cls, store_path, transform=None):
        """Load from an existing ProteinStore.

        Parameters
        ----------
        store_path : str or Path
            Path to the ``.store`` file.
        transform : callable, optional

        Returns
        -------
        ProteinShakeLoader
        """
        return cls(ProteinStore(store_path), transform=transform)
