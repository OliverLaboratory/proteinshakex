# -*- coding: utf-8 -*-
"""
DANCE multi-conformation dataset.

Each protein ensemble has multiple experimental structures (from different
PDB depositions) stored as separate PDB files grouped by ensemble ID.
"""
import os
import glob
import csv

import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb

from proteinshake.datasets.multi_conf import MultiConfDataset
from proteinshake.datasets.dataset import AA_THREE_TO_ONE


class DANCEDataset(MultiConfDataset):
    """Multi-conformation dataset from the DANCE benchmark.

    Each protein ensemble contains multiple experimental structures stored
    as individual PDB files (backbone atoms: N, CA, C, O) grouped by
    ensemble directory.

    The protein dictionary extends the MultiConfDataset format with::

        protein['protein']['ensemble_id']       -> str
        protein['protein']['source_pdbs']       -> list of str
        protein['protein']['num_conformations'] -> int
        protein['protein']['num_residues_per_conf'] -> list of int
        protein['protein']['num_global_residues']   -> int
        residue['conf_global_idx']              -> list of list of int
        residue['global_residue_type']          -> list of str

    Parameters
    ----------
    root : str
        Data root directory for processed output.
    dance_path : str
        Path to the DANCE data directory containing ``input_pdbs/``
        and ``input_dataset.csv``.
    use_filtered : bool
        If True, only include conformations that passed DANCE quality
        filtering (from ``filtered_ensembles.csv``).
    """

    exlude_args_from_signature = ['dance_path']

    def __init__(self,
                 root='data',
                 dance_path='../ConformationDiscoveryBenchmark/data/inputs/DANCE',
                 use_filtered=True,
                 use_precomputed=True,
                 n_jobs=1,
                 verbosity=2,
                 **kwargs):
        self.dance_path = dance_path
        self.use_filtered = use_filtered
        self._metadata = None
        self._filtered_indices = None
        super().__init__(
            root=root,
            use_precomputed=use_precomputed,
            n_jobs=n_jobs,
            verbosity=verbosity,
            **kwargs,
        )

    @property
    def name(self):
        return 'DANCEDataset'

    def _load_metadata(self):
        """Load and cache the input_dataset.csv metadata."""
        if self._metadata is not None:
            return self._metadata
        csv_path = os.path.join(self.dance_path, 'input_dataset.csv')
        self._metadata = {}
        with open(csv_path, 'r') as f:
            for row in csv.DictReader(f):
                self._metadata[row['ensemble_ID']] = row
        return self._metadata

    def _load_filtered_indices(self):
        """Load and cache the filtered_ensembles.csv data."""
        if self._filtered_indices is not None:
            return self._filtered_indices
        csv_path = os.path.join(self.dance_path, 'filtered_ensembles.csv')
        self._filtered_indices = {}
        if os.path.exists(csv_path):
            with open(csv_path, 'r') as f:
                for row in csv.DictReader(f):
                    indices = [int(x) for x in row['kept_indices'].split(';')]
                    self._filtered_indices[row['ensemble_ID']] = {
                        'kept_indices': indices,
                        'avg_pairwise_tm': float(row['avg_pairwise_tm']),
                        'min_pairwise_tm': float(row['min_pairwise_tm']),
                    }
        return self._filtered_indices

    def download(self):
        """Verify that the DANCE data directory exists and symlink into raw/."""
        input_pdbs = os.path.join(self.dance_path, 'input_pdbs')
        if not os.path.isdir(input_pdbs):
            raise FileNotFoundError(
                f"DANCE input_pdbs directory not found at {input_pdbs}. "
                f"Set dance_path to the directory containing input_pdbs/."
            )
        raw_dir = os.path.join(self.root, 'raw', 'files')
        os.makedirs(raw_dir, exist_ok=True)
        for d in os.listdir(input_pdbs):
            src = os.path.join(input_pdbs, d)
            dst = os.path.join(raw_dir, d)
            if os.path.isdir(src) and not os.path.exists(dst):
                os.symlink(os.path.abspath(src), dst)

    def get_raw_files(self):
        """Return list of ensemble directories."""
        raw_dir = os.path.join(self.root, 'raw', 'files')
        dirs = sorted(glob.glob(os.path.join(raw_dir, '*')))
        return [d for d in dirs if os.path.isdir(d)]

    def get_id_from_filename(self, filename):
        """Extract ensemble ID from the directory name."""
        return os.path.basename(filename)

    @staticmethod
    def _parse_pdb_file(path):
        """Parse a single PDB file into a DataFrame.

        Parameters
        ----------
        path : str
            Path to a .pdb file.

        Returns
        -------
        pd.DataFrame
            DataFrame with standard column names.
        """
        IONS = ['ZN', 'MG']
        df = PandasPdb().read_pdb(path).df['ATOM']
        if len(df) == 0:
            return df
        df = df.loc[~df['residue_name'].isin(IONS)]
        df['residue_name'] = df['residue_name'].map(
            lambda x: AA_THREE_TO_ONE.get(x, None)
        )
        df = df.rename(columns={
            'atom_name': 'atom_type',
            'residue_name': 'residue_type',
            'x_coord': 'x',
            'y_coord': 'y',
            'z_coord': 'z',
        })
        df = df.sort_values(by=['chain_id', 'residue_number', 'atom_number'])
        return df

    def parse_pdb(self, path):
        """Parse an ensemble directory containing multiple PDB conformations.

        Parameters
        ----------
        path : str
            Path to a directory containing PDB files for one ensemble.

        Returns
        -------
        dict or None
            A protein dictionary with multi-conformation coordinates.
        """
        ensemble_id = self.get_id_from_filename(path)
        if ensemble_id in self.exclude_ids:
            return None

        pdb_files = sorted(glob.glob(os.path.join(path, '*.pdb')))
        if not pdb_files:
            return None

        # Load metadata
        metadata = self._load_metadata()
        meta = metadata.get(ensemble_id, {})

        # Apply filtering if requested
        if self.use_filtered:
            filtered = self._load_filtered_indices()
            if ensemble_id in filtered:
                kept = filtered[ensemble_id]['kept_indices']
                if kept and max(kept) < len(pdb_files):
                    pdb_files = [pdb_files[i] for i in kept]
            elif filtered:
                return None

        # Parse all PDB files
        conf_dfs = []
        for pdb_path in pdb_files:
            try:
                df = self._parse_pdb_file(pdb_path)
                if len(df) == 0:
                    continue
                df = df[df['residue_type'].notna()]
                if len(df) == 0:
                    continue
                conf_dfs.append(df)
            except Exception:
                continue

        if not conf_dfs:
            return None

        # Validate using first conformation
        ref_df = conf_dfs[0]
        ref_ca = ref_df[ref_df['atom_type'] == 'CA'].copy()
        n_residues = len(ref_ca)

        if n_residues < self.minimum_length or n_residues > self.maximum_length:
            return None
        if self.only_single_chain and len(ref_df['chain_id'].unique()) > 1:
            return None

        # Collect per-conformation data
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

        # DANCE ensembles can contain sequence-divergent structural homologs,
        # so sequence alignment is not meaningful for a global index.
        # Use the ref_sequence from metadata as the canonical sequence,
        # and store per-conformation sequences independently.
        ref_sequence = meta.get('ref_sequence', '')
        if not ref_sequence:
            longest_idx = max(range(len(conf_dfs)), key=lambda i: conf_num_residues[i])
            ref_sequence = ''.join(conf_residue_types[longest_idx])

        n_global = len(ref_sequence)
        global_residue_type = list(ref_sequence)

        # Extract PDB IDs from filenames
        conf_pdb_ids = []
        for pdb_path in pdb_files:
            fname = os.path.basename(pdb_path)
            pdb_id = fname.replace('.pdb', '')
            conf_pdb_ids.append(pdb_id)

        protein = {
            'protein': {
                'ID': ensemble_id,
                'sequence': ref_sequence,
                'num_conformations': len(conf_dfs),
                'num_residues_per_conf': conf_num_residues,
                'num_global_residues': n_global,
                'ensemble_id': ensemble_id,
                'source_pdbs': conf_pdb_ids,
            },
            'residue': {
                'global_residue_type': global_residue_type,
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
