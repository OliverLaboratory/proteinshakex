"""Lazy, disk-backed protein collection for large avro datasets.

Builds a lightweight byte-offset index on first access so that individual
proteins can be read from disk on demand via ``__getitem__`` without loading
the entire dataset into memory.
"""

import json
import os
import pickle
from pathlib import Path

from fastavro import reader as avro_reader, block_reader as avro_block_reader


class LazyProteins:
    """Random-access protein collection backed by avro files on disk.

    On first instantiation the class scans the avro file(s) and builds a list
    of ``(file_path, block_index, record_index_in_block)`` tuples — one per
    protein.  This index is cached to ``<root>/<name>.<resolution>.index.pkl``
    so subsequent loads are instant.

    Parameters
    ----------
    avro_files : list[str]
        Ordered list of avro file paths (single file or chunks).
    cache_path : str, optional
        Where to cache the offset index.  If ``None``, no caching.
    """

    def __init__(self, avro_files, cache_path=None):
        self._avro_files = avro_files
        self._cache_path = cache_path
        self._index = None  # list of (file_idx, block_idx, rec_idx)
        self._block_offsets = None  # list of (file_idx, block_byte_offset, num_records)
        self._build_index()

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def _build_index(self):
        """Build index from avro file metadata.

        Stores ``_file_ranges``: a list of ``(file_idx, start_global, end_global)``
        so that ``__getitem__`` can determine which file and local offset a
        global index maps to.  This only reads file headers, not record data.
        """
        if self._cache_path and os.path.exists(self._cache_path):
            try:
                with open(self._cache_path, 'rb') as f:
                    cached = pickle.load(f)
                if cached.get('files') == self._avro_files and cached.get('version', 0) >= 3:
                    self._file_ranges = cached['file_ranges']
                    self._total = cached['total']
                    return
            except Exception:
                pass

        self._file_ranges = []  # (file_idx, global_start, global_end)
        offset = 0

        for file_idx, path in enumerate(self._avro_files):
            with open(path, 'rb') as f:
                reader = avro_reader(f)
                meta = reader.metadata
                if 'number_of_proteins' in meta:
                    count = int(meta['number_of_proteins'])
                else:
                    # Fallback: count by scanning block headers
                    f.seek(0)
                    count = sum(block.num_records for block in avro_block_reader(f))
            self._file_ranges.append((file_idx, offset, offset + count))
            offset += count

        self._total = offset

        if self._cache_path:
            try:
                os.makedirs(os.path.dirname(self._cache_path) or '.', exist_ok=True)
                with open(self._cache_path, 'wb') as f:
                    pickle.dump({
                        'files': self._avro_files,
                        'file_ranges': self._file_ranges,
                        'total': self._total,
                        'version': 3,
                    }, f, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def __len__(self):
        return self._total

    def _resolve_idx(self, idx):
        """Map a global index to (file_path, local_offset)."""
        for file_idx, start, end in self._file_ranges:
            if start <= idx < end:
                return self._avro_files[file_idx], idx - start
        raise IndexError(f"Index {idx} out of range for {self._total} proteins")

    def __getitem__(self, idx):
        if isinstance(idx, (list, tuple)):
            return [self[i] for i in idx]
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]

        idx = int(idx)
        if idx < 0:
            idx += self._total
        if idx < 0 or idx >= self._total:
            raise IndexError(f"Index {idx} out of range for {self._total} proteins")

        path, local_idx = self._resolve_idx(idx)

        # Stream through the file to reach the target record
        with open(path, 'rb') as f:
            for i, record in enumerate(avro_reader(f)):
                if i == local_idx:
                    return record

        raise IndexError(f"Could not read protein at index {idx}")

    def __iter__(self):
        """Stream all proteins in order without loading all into memory."""
        for path in self._avro_files:
            with open(path, 'rb') as f:
                for record in avro_reader(f):
                    yield record


class EagerProteins:
    """In-memory protein collection (original behavior). Used for small datasets."""

    def __init__(self, proteins):
        self.proteins = list(proteins)

    def __len__(self):
        return len(self.proteins)

    def __getitem__(self, idx):
        if isinstance(idx, (list, tuple)):
            return [self[i] for i in idx]
        if isinstance(idx, slice):
            return self.proteins[idx]
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            return [self[i] for i in idx]
        if idx >= len(self.proteins):
            raise StopIteration
        return self.proteins[idx]

    def __iter__(self):
        return iter(self.proteins)
