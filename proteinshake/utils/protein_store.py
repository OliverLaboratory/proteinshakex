"""Indexed binary store for fast random access to protein records.

Converts avro files into a pair of binary files:
  - ``<name>.store`` — concatenated pickle records
  - ``<name>.store.idx`` — numpy array of (byte_offset, byte_length) per record

Random access is O(1): read offset from index, seek in data file, unpickle
one record.  Multiple DataLoader workers can read concurrently since each
opens its own file handle.

Usage::

    store = ProteinStore.from_avro(avro_files, 'data/my_dataset.store')
    store[0]       # first protein dict
    store[100000]  # instant random access
    len(store)     # total count
"""

import os
import pickle
import struct
from pathlib import Path

import numpy as np


class ProteinStore:
    """Indexed binary store for protein dictionaries.

    Parameters
    ----------
    store_path : str or Path
        Path to the ``.store`` file.  The index file is at
        ``<store_path>.idx``.
    """

    def __init__(self, store_path):
        self._store_path = str(store_path)
        self._index_path = self._store_path + '.idx.npy'

        if not os.path.exists(self._index_path):
            raise FileNotFoundError(
                f"Index file not found: {self._index_path}. "
                "Build the store with ProteinStore.from_avro() first."
            )

        # File handle opened lazily per-process (for multiprocessing safety)
        self._fh = None
        self._pid = None

        # Load index: Nx2 array of (offset, length)
        self._index = np.load(self._index_path)
        self._len = len(self._index)

    def _get_fh(self):
        """Get a file handle, opening a new one if we're in a new process."""
        pid = os.getpid()
        if self._fh is None or self._pid != pid:
            if self._fh is not None:
                self._fh.close()
            self._fh = open(self._store_path, 'rb')
            self._pid = pid
        return self._fh

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        if isinstance(idx, (list, tuple, np.ndarray)):
            return [self[i] for i in idx]
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(self._len))]

        idx = int(idx)
        if idx < 0:
            idx += self._len
        if idx < 0 or idx >= self._len:
            raise IndexError(f"Index {idx} out of range for {self._len} proteins")

        offset, length = self._index[idx]
        fh = self._get_fh()
        fh.seek(int(offset))
        data = fh.read(int(length))
        return pickle.loads(data)

    def __iter__(self):
        """Sequential iteration — reads records in order, fast."""
        fh = self._get_fh()
        for offset, length in self._index:
            fh.seek(int(offset))
            data = fh.read(int(length))
            yield pickle.loads(data)

    def __del__(self):
        if self._fh is not None:
            self._fh.close()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_avro(cls, avro_files, store_path, verbosity=2):
        """Build a ProteinStore from one or more avro files.

        Streams through all avro files, serializes each protein dict with
        pickle, and writes to a single binary file with an index.

        Parameters
        ----------
        avro_files : list[str]
            Ordered list of avro file paths.
        store_path : str or Path
            Output path for the ``.store`` file.
        verbosity : int
            Logging verbosity.

        Returns
        -------
        ProteinStore
            Ready-to-use store instance.
        """
        from fastavro import reader as avro_reader

        store_path = str(store_path)
        index_path = store_path + '.idx.npy'

        if os.path.exists(store_path) and os.path.exists(index_path):
            if verbosity > 0:
                print(f'ProteinStore already exists at {store_path}')
            return cls(store_path)

        offsets = []  # list of (offset, length)

        if verbosity > 0:
            print(f'Building ProteinStore from {len(avro_files)} avro file(s)...')

        with open(store_path, 'wb') as out:
            for fi, avro_path in enumerate(avro_files):
                if verbosity > 0:
                    print(f'  Processing {os.path.basename(avro_path)}...')
                with open(avro_path, 'rb') as f:
                    for record in avro_reader(f):
                        blob = pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL)
                        offset = out.tell()
                        out.write(blob)
                        offsets.append((offset, len(blob)))

        # Save index as numpy array
        index_arr = np.array(offsets, dtype=np.int64)
        np.save(index_path, index_arr)

        if verbosity > 0:
            store_mb = os.path.getsize(store_path) / 1024**2
            print(f'  Store: {len(offsets)} proteins, {store_mb:.0f} MB')

        return cls(store_path)

    @classmethod
    def from_proteins(cls, proteins_iter, store_path, total=None, verbosity=2):
        """Build a ProteinStore from an iterable of protein dicts.

        Parameters
        ----------
        proteins_iter : iterable
            Iterable of protein dictionaries.
        store_path : str or Path
            Output path.
        total : int, optional
            Total count for progress display.
        verbosity : int
            Logging verbosity.

        Returns
        -------
        ProteinStore
        """
        store_path = str(store_path)
        index_path = store_path + '.idx.npy'

        if os.path.exists(store_path) and os.path.exists(index_path):
            if verbosity > 0:
                print(f'ProteinStore already exists at {store_path}')
            return cls(store_path)

        offsets = []

        if verbosity > 0:
            print(f'Building ProteinStore...')

        with open(store_path, 'wb') as out:
            for i, record in enumerate(proteins_iter):
                blob = pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL)
                offset = out.tell()
                out.write(blob)
                offsets.append((offset, len(blob)))
                if verbosity > 0 and (i + 1) % 10000 == 0:
                    print(f'  {i+1} proteins written' + (f'/{total}' if total else ''))

        index_arr = np.array(offsets, dtype=np.int64)
        np.save(index_path, index_arr)

        if verbosity > 0:
            store_mb = os.path.getsize(store_path) / 1024**2
            print(f'  Store: {len(offsets)} proteins, {store_mb:.0f} MB')

        return cls(store_path)
