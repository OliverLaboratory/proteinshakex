# -*- coding: utf-8 -*-
"""
UNICORNE multi-conformation dataset.

Each protein has multiple experimental structures (from different PDB entries)
stored as separate mmCIF files, pre-aligned to a reference.
"""
import os
import glob
import csv

import numpy as np
import pandas as pd
from Bio.Align import PairwiseAligner

from proteinshake.datasets.multi_conf import MultiConfDataset
from proteinshake.datasets.dataset import AA_THREE_TO_ONE


def _parse_cif(path):
    """Parse a mmCIF file into a DataFrame of ATOM records.

    Parameters
    ----------
    path : str
        Path to a .cif file.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: atom_type, residue_type, residue_number,
        x, y, z, chain_id, b_factor.
    """
    with open(path, 'r') as f:
        lines = f.readlines()

    cols = []
    data_start = None
    in_loop = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == 'loop_':
            in_loop = True
            cols = []
            continue
        if in_loop and stripped.startswith('_atom_site.'):
            cols.append(stripped.replace('_atom_site.', ''))
        elif in_loop and not stripped.startswith('_'):
            data_start = i
            break

    if data_start is None or not cols:
        return pd.DataFrame()

    rows = []
    for line in lines[data_start:]:
        stripped = line.strip()
        if stripped == '' or stripped.startswith('#'):
            break
        rows.append(stripped.split())

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=cols)

    # Filter to ATOM records only
    if 'group_PDB' in df.columns:
        df = df[df['group_PDB'] == 'ATOM']

    # Map columns to standard names
    col_map = {
        'label_atom_id': 'atom_type',
        'label_comp_id': 'residue_type_3',
        'label_seq_id': 'residue_number',
        'Cartn_x': 'x',
        'Cartn_y': 'y',
        'Cartn_z': 'z',
        'label_asym_id': 'chain_id',
        'B_iso_or_equiv': 'b_factor',
    }
    # Use auth columns if available
    if 'auth_asym_id' in df.columns:
        col_map['auth_asym_id'] = 'chain_id'
        del col_map['label_asym_id']

    df = df.rename(columns=col_map)
    df['residue_number'] = df['residue_number'].astype(int)
    df['x'] = df['x'].astype(float)
    df['y'] = df['y'].astype(float)
    df['z'] = df['z'].astype(float)

    # Convert 3-letter to 1-letter residue codes
    df['residue_type'] = df['residue_type_3'].map(
        lambda x: AA_THREE_TO_ONE.get(x, None)
    )

    df = df.sort_values(by=['chain_id', 'residue_number'])
    return df


def _align_seq_to_ref(ref_seq, query_seq):
    """Align a query sequence to a reference using global pairwise alignment.

    Returns a list of length len(query_seq) where each element is the
    0-based index into ref_seq that the query residue aligns to, or -1
    if the query residue aligns to a gap in the reference.

    Parameters
    ----------
    ref_seq : str
        The reference (UniProt) sequence.
    query_seq : str
        The conformation sequence to align.

    Returns
    -------
    list of int
        Mapping from query position -> reference position (-1 for gaps).
    """
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -5
    aligner.extend_gap_score = -0.5

    alignment = aligner.align(ref_seq, query_seq)[0]

    # Build mapping: query_pos -> ref_pos
    # alignment.aligned gives pairs of (ref_intervals, query_intervals)
    ref_intervals, query_intervals = alignment.aligned
    mapping = [-1] * len(query_seq)

    for (ref_start, ref_end), (query_start, query_end) in zip(ref_intervals, query_intervals):
        for offset in range(ref_end - ref_start):
            mapping[query_start + offset] = ref_start + offset

    return mapping


class UNICORNEDataset(MultiConfDataset):
    """Multi-conformation dataset from the UNICORNE_BENCH benchmark.

    Each protein entry contains multiple experimental structures (from
    different PDB depositions) that have been pre-aligned. Conformations
    are stored as separate mmCIF files grouped by UniProt accession.

    The protein dictionary extends the MultiConfDataset format with::

        protein['protein']['uniprot']           -> str (UniProt accession)
        protein['protein']['num_clusters']      -> int
        protein['protein']['source_pdbs']       -> list of str (PDB_chain IDs)
        protein['protein']['num_conformations'] -> int

    Parameters
    ----------
    root : str
        Data root directory for processed output.
    unicorne_path : str
        Path to the UNICORNE_BENCH data directory containing
        ``input_pdbs/`` and ``input_dataset.csv``. Defaults to the
        UNICORNE_BENCH directory shipped alongside this package's
        sibling ConformationDiscoveryBenchmark repo.
    use_filtered : bool
        If True, only include conformations that passed UNICORNE_BENCH
        quality filtering (from ``filtered_ensembles_v2.csv``).
    apply_blacklist : bool
        If True, exclude UniProts listed in ``fold_quality_blacklist.csv``
        (artifactual / IDP / fibril GTs that aren't bona fide multi-state
        ensembles — Lysozyme C amyloid fibril, IDP-NMR cases, TFE-induced
        opening, etc.).
    """

    exlude_args_from_signature = ['unicorne_path']

    def __init__(self,
                 root='data',
                 unicorne_path='../ConformationDiscoveryBenchmark/data/inputs/UNICORNE_BENCH',
                 use_filtered=True,
                 apply_blacklist=True,
                 use_precomputed=True,
                 n_jobs=1,
                 verbosity=2,
                 **kwargs):
        self.unicorne_path = unicorne_path
        self.use_filtered = use_filtered
        self.apply_blacklist = apply_blacklist
        self._metadata = None
        self._filtered_indices = None
        self._blacklist = None
        super().__init__(
            root=root,
            use_precomputed=use_precomputed,
            n_jobs=n_jobs,
            verbosity=verbosity,
            **kwargs,
        )

    @property
    def name(self):
        return 'UNICORNEDataset'

    def _load_metadata(self):
        """Load and cache input_dataset.csv (or benchmarking-dataset-13.csv).

        The UNICORNE_BENCH source-of-truth for sequence + representative
        structures is `benchmarking-dataset-13.csv`; older drops shipped a
        `input_dataset.csv`. Try both.
        """
        if self._metadata is not None:
            return self._metadata
        self._metadata = {}
        for fname in ('benchmarking-dataset-13.csv', 'input_dataset.csv'):
            csv_path = os.path.join(self.unicorne_path, fname)
            if os.path.exists(csv_path):
                with open(csv_path, 'r') as f:
                    for row in csv.DictReader(f):
                        eid = row.get('uniprot') or row.get('ensemble_ID')
                        if eid:
                            self._metadata[eid] = row
                break
        return self._metadata

    def _load_filtered_indices(self):
        """Load filtered_ensembles_v2.csv (preferred) or v1 fallback."""
        if self._filtered_indices is not None:
            return self._filtered_indices
        self._filtered_indices = {}
        for fname in ('filtered_ensembles_v2.csv', 'filtered_ensembles.csv'):
            csv_path = os.path.join(self.unicorne_path, fname)
            if os.path.exists(csv_path):
                with open(csv_path, 'r') as f:
                    for row in csv.DictReader(f):
                        indices = [int(x) for x in row['kept_indices'].split(';')]
                        self._filtered_indices[row['ensemble_ID']] = {
                            'kept_indices': indices,
                            'avg_pairwise_tm': float(row['avg_pairwise_tm']),
                            'min_pairwise_tm': float(row['min_pairwise_tm']),
                        }
                break
        return self._filtered_indices

    def _load_blacklist(self):
        """Load fold_quality_blacklist.csv — UniProts whose GT structures
        are artifactual / IDP / fibril and shouldn't be evaluated as
        multi-state ensembles. See data/inputs/UNICORNE_BENCH/
        fold_quality_blacklist.csv for the curated list and reasons."""
        if self._blacklist is not None:
            return self._blacklist
        self._blacklist = set()
        f = os.path.join(self.unicorne_path, 'fold_quality_blacklist.csv')
        if os.path.exists(f):
            with open(f, 'r') as fh:
                for row in csv.DictReader(fh):
                    self._blacklist.add((row.get('uniprot') or '').strip())
        return self._blacklist

    def download(self):
        """Verify that the UNICORNE data directory exists.

        UNICORNE data is expected to already be available locally.
        This method creates symlinks from the raw directory to the
        UNICORNE input_pdbs.
        """
        input_pdbs = os.path.join(self.unicorne_path, 'input_pdbs')
        if not os.path.isdir(input_pdbs):
            raise FileNotFoundError(
                f"UNICORNE input_pdbs directory not found at {input_pdbs}. "
                f"Set unicorne_path to the directory containing input_pdbs/."
            )
        # Symlink each protein directory into raw/files/
        raw_dir = os.path.join(self.root, 'raw', 'files')
        os.makedirs(raw_dir, exist_ok=True)
        for d in os.listdir(input_pdbs):
            src = os.path.join(input_pdbs, d)
            dst = os.path.join(raw_dir, d)
            if os.path.isdir(src) and not os.path.exists(dst):
                os.symlink(os.path.abspath(src), dst)

    def get_raw_files(self):
        """Return list of protein directories (one per UniProt ID).

        Each directory path contains multiple CIF files representing
        different conformations of the same protein.
        """
        raw_dir = os.path.join(self.root, 'raw', 'files')
        dirs = sorted(glob.glob(os.path.join(raw_dir, '*')))
        return [d for d in dirs if os.path.isdir(d)]

    def get_id_from_filename(self, filename):
        """Extract UniProt ID from the directory name."""
        return os.path.basename(filename)

    def parse_pdb(self, path):
        """Parse a protein directory containing multiple CIF conformations.

        Parameters
        ----------
        path : str
            Path to a directory containing CIF files for one protein.

        Returns
        -------
        dict or None
            A protein dictionary with multi-conformation coordinates.
        """
        uniprot_id = self.get_id_from_filename(path)
        if uniprot_id in self.exclude_ids:
            return None

        # Fold-quality blacklist (artifactual / IDP / fibril GTs).
        if self.apply_blacklist and uniprot_id in self._load_blacklist():
            return None

        cif_files = sorted(glob.glob(os.path.join(path, '*-aligned.cif')))
        if not cif_files:
            # Fall back to .pdb for ensembles whose canonical files are PDB.
            cif_files = sorted(glob.glob(os.path.join(path, '*-aligned.pdb')))
        if not cif_files:
            return None

        # Load metadata
        metadata = self._load_metadata()
        meta = metadata.get(uniprot_id, {})
        rep_structures = meta.get('representative_structures', '')
        source_pdbs = rep_structures.split(';') if rep_structures else []

        # Apply filtering if requested
        if self.use_filtered:
            filtered = self._load_filtered_indices()
            if uniprot_id in filtered:
                kept = filtered[uniprot_id]['kept_indices']
                # Map indices to the sorted CIF file list
                # The representative_structures order matches the CIF file order
                if kept and max(kept) < len(cif_files):
                    cif_files = [cif_files[i] for i in kept]
            elif filtered:
                # Protein not in filtered set, skip it
                return None

        # Parse all CIF files
        conf_dfs = []
        for cif_path in cif_files:
            df = _parse_cif(cif_path)
            if df.empty:
                continue
            # Filter to standard amino acids
            df = df[df['residue_type'].notna()]
            if len(df) == 0:
                continue
            conf_dfs.append(df)

        if not conf_dfs:
            return None

        # Use first conformation as reference for shared attributes
        ref_df = conf_dfs[0]
        ref_ca = ref_df[ref_df['atom_type'] == 'CA'].copy()

        # Validate length
        n_residues = len(ref_ca)
        if n_residues < self.minimum_length or n_residues > self.maximum_length:
            return None

        # Check single chain filter
        if self.only_single_chain and len(ref_df['chain_id'].unique()) > 1:
            return None

        # Build sequence from reference
        sequence = ''.join(ref_ca['residue_type'].tolist())

        # Collect coordinates from all conformations (keep all, even mismatched lengths)
        residue_x, residue_y, residue_z = [], [], []
        atom_x, atom_y, atom_z = [], [], []
        conf_residue_numbers, conf_residue_types = [], []
        conf_num_residues = []

        for df in conf_dfs:
            ca = df[df['atom_type'] == 'CA']
            residue_x.append(ca['x'].tolist())
            residue_y.append(ca['y'].tolist())
            residue_z.append(ca['z'].tolist())
            atom_x.append(df['x'].tolist())
            atom_y.append(df['y'].tolist())
            atom_z.append(df['z'].tolist())
            conf_residue_numbers.append(ca['residue_number'].tolist())
            conf_residue_types.append(ca['residue_type'].tolist())
            conf_num_residues.append(len(ca))

        # Build global residue index by aligning each conformation to the
        # UniProt reference sequence from the metadata CSV.
        ref_sequence = meta.get('sequence', '')
        if not ref_sequence:
            # Fallback: use the longest conformation as reference
            longest_idx = max(range(len(conf_dfs)), key=lambda i: conf_num_residues[i])
            ref_sequence = ''.join(conf_residue_types[longest_idx])

        n_global = len(ref_sequence)
        global_residue_type = list(ref_sequence)

        # Align each conformation's sequence to the reference.
        # conf_global_idx[c][j] = position in reference for residue j of conformation c.
        # -1 if the residue doesn't align to any reference position.
        conf_global_idx = []
        for c in range(len(conf_dfs)):
            conf_seq = ''.join(conf_residue_types[c])
            mapping = _align_seq_to_ref(ref_sequence, conf_seq)
            conf_global_idx.append(mapping)

        # Extract PDB IDs from CIF filenames
        conf_pdb_ids = []
        for cif_path in cif_files:
            fname = os.path.basename(cif_path)
            # e.g. "7KO4_V-aligned.cif" -> "7KO4_V"
            pdb_chain = fname.replace('-aligned.cif', '')
            conf_pdb_ids.append(pdb_chain)

        protein = {
            'protein': {
                'ID': uniprot_id,
                'sequence': ref_sequence,
                'num_conformations': len(conf_dfs),
                'num_residues_per_conf': conf_num_residues,
                'num_global_residues': n_global,
                'uniprot': uniprot_id,
                'num_clusters': int(meta.get('num_clusters', len(conf_dfs))),
                'source_pdbs': conf_pdb_ids,
            },
            'residue': {
                'global_residue_type': global_residue_type,
                'conf_global_idx': conf_global_idx,
                'residue_number': conf_residue_numbers,
                'residue_type': conf_residue_types,
                'x': residue_x,
                'y': residue_y,
                'z': residue_z,
            },
            'atom': {
                'x': atom_x,
                'y': atom_y,
                'z': atom_z,
            },
        }

        protein = self.add_protein_attributes(protein)
        return protein
