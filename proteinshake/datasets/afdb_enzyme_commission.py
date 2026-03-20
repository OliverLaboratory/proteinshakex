# -*- coding: utf-8 -*-
"""
Dataset for AlphaFold DB predicted structures with EC number annotations.

Combines AlphaFold predicted structures from AFDB with enzyme commission (EC)
annotations from UniProt (Swiss-Prot reviewed entries). This provides ~273k
enzyme structures — far more than the ~15k experimental structures in PDB.

EC annotations are fetched from the UniProt REST API and structures are
downloaded individually from AlphaFold DB.
"""
import csv
import glob
import io
import os
import re
import time

import requests

from proteinshake.datasets import Dataset
from proteinshake.utils import download_url, progressbar, warning, error


UNIPROT_STREAM_URL = (
    "https://rest.uniprot.org/uniprotkb/stream"
    "?query=(ec:*)%20AND%20(reviewed:true)%20AND%20(database:alphafolddb)"
    "&format=tsv"
    "&fields=accession,ec,organism_name,protein_name,length"
)

AFDB_PDB_URL = "https://alphafold.ebi.ac.uk/files/AF-{accession}-F1-model_{version}.pdb"


class AFDBEnzymeCommissionDataset(Dataset):
    """Enzymes with EC annotations using AlphaFold DB predicted structures.

    This dataset combines:
    - **Structures**: AlphaFold DB predicted structures (single-chain, high
      confidence).
    - **Labels**: Enzyme Commission (EC) numbers from UniProt (Swiss-Prot
      reviewed entries only).

    The dataset first downloads a UniProt index of all reviewed proteins with
    EC annotations and AlphaFold structures, then downloads each PDB file from
    AFDB individually.

    .. admonition:: Please cite

        Jumper, J. et al. "Highly accurate protein structure prediction with
        AlphaFold." Nature 596 (2021): 583-589.

        Varadi, M. et al. "AlphaFold Protein Structure Database: massively
        expanding the structural coverage of protein-sequence space with
        high-accuracy models." Nucleic Acids Research 50.D1 (2022): D439-D444.

    .. admonition:: Source

        EC annotations from `UniProt <https://www.uniprot.org/>`_ (CC-BY 4.0).
        Structures from `AlphaFold DB <https://alphafold.ebi.ac.uk/>`_ (CC-BY 4.0).

    .. list-table:: Dataset stats
        :widths: 100
        :header-rows: 1

        * - # proteins
        * - ~273,000 (Swiss-Prot reviewed enzymes with AFDB structures)

    .. list-table:: Annotations
        :widths: 25 35 45
        :header-rows: 1

        * - Attribute
          - Key
          - Sample value
        * - Enzyme Commission
          - :code:`protein['protein']['EC']`
          - :code:`'2.7.7.4'`
        * - All EC numbers (multi-functional enzymes)
          - :code:`protein['protein']['EC_all']`
          - :code:`['2.7.7.49', '2.7.7.7']`
        * - UniProt accession
          - :code:`protein['protein']['uniprot_id']`
          - :code:`'P00520'`
        * - Organism
          - :code:`protein['protein']['organism']`
          - :code:`'Homo sapiens (Human)'`

    Parameters
    ----------
    max_proteins : int, optional
        Maximum number of proteins to include. If None, includes all available
        (~273k). Useful for testing or creating smaller subsets.
    min_length : int, default 30
        Minimum sequence length to include.
    max_length : int, default 2048
        Maximum sequence length to include.
    afdb_version : str, default 'v4'
        AlphaFold DB model version.
    download_workers : int, default 4
        Number of parallel workers for downloading PDB files.
    """

    description = 'Enzymes (AlphaFold DB structures)'

    def __init__(
        self,
        max_proteins=None,
        min_length=30,
        max_length=2048,
        afdb_version='v6',
        download_workers=4,
        **kwargs,
    ):
        self.max_proteins = max_proteins
        self.min_length = min_length
        self.max_length = max_length
        self.afdb_version = afdb_version
        self.download_workers = download_workers
        self.ec_index = {}  # accession -> {ec, ec_all, organism, protein_name, length}
        kwargs.setdefault('only_single_chain', True)
        kwargs.setdefault('minimum_length', min_length)
        kwargs.setdefault('maximum_length', max_length)
        super().__init__(**kwargs)

    def get_id_from_filename(self, filename):
        """Extract UniProt accession from AFDB PDB filename."""
        match = re.search(r'AF-(.+?)-F\d+-model', os.path.basename(filename))
        if match:
            return match.group(1)
        return os.path.basename(filename).replace('.pdb', '')

    def get_raw_files(self):
        return sorted(glob.glob(f'{self.root}/raw/files/*.pdb'))

    # ------------------------------------------------------------------
    # UniProt index
    # ------------------------------------------------------------------

    def _index_path(self):
        return os.path.join(self.root, 'raw', 'uniprot_ec_index.tsv')

    def _download_uniprot_index(self):
        """Download the UniProt index of EC-annotated, AFDB-available proteins."""
        index_path = self._index_path()
        if os.path.exists(index_path):
            if self.verbosity > 0:
                print(f'UniProt EC index already exists at {index_path}')
            return

        os.makedirs(os.path.dirname(index_path), exist_ok=True)

        if self.verbosity > 0:
            print('Downloading UniProt EC index (this may take a few minutes)...')

        response = requests.get(UNIPROT_STREAM_URL, timeout=600, stream=True)
        response.raise_for_status()

        with open(index_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)

        if self.verbosity > 0:
            print(f'Saved UniProt EC index to {index_path}')

    def _load_ec_index(self):
        """Load the UniProt EC index TSV into self.ec_index."""
        if self.ec_index:
            return

        index_path = self._index_path()
        if not os.path.exists(index_path):
            self._download_uniprot_index()

        self.ec_index = {}
        with open(index_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                accession = row.get('Entry', '').strip()
                ec_raw = row.get('EC number', '').strip()
                organism = row.get('Organism', '').strip()
                protein_name = row.get('Protein names', '').strip()
                length_str = row.get('Length', '0').strip()

                if not accession or not ec_raw:
                    continue

                try:
                    length = int(length_str)
                except ValueError:
                    length = 0

                # Filter by length
                if length < self.min_length or length > self.max_length:
                    continue

                # Parse EC numbers (semicolon-separated for multi-functional)
                ec_list = [e.strip() for e in ec_raw.split(';') if e.strip()]
                # Filter out partial EC numbers (containing '-')
                ec_complete = [e for e in ec_list if '-' not in e and e.count('.') == 3]

                if not ec_complete:
                    continue

                self.ec_index[accession] = {
                    'ec': ec_complete[0],      # Primary EC
                    'ec_all': ec_complete,      # All complete EC numbers
                    'organism': organism,
                    'protein_name': protein_name,
                    'length': length,
                }

        if self.verbosity > 0:
            print(f'Loaded {len(self.ec_index)} proteins from UniProt EC index')

    # ------------------------------------------------------------------
    # Structure download
    # ------------------------------------------------------------------

    def _download_single_pdb(self, accession):
        """Download a single AFDB PDB file. Returns True on success.

        Tries the configured version first, then falls back to v4 and v3.
        """
        out_path = os.path.join(self.root, 'raw', 'files', f'AF-{accession}-F1-model_{self.afdb_version}.pdb')
        if os.path.exists(out_path):
            return True

        # Also check if downloaded with a different version name
        for existing in glob.glob(os.path.join(self.root, 'raw', 'files', f'AF-{accession}-F1-model_*.pdb')):
            return True

        versions_to_try = [self.afdb_version]
        for fallback in ('v4', 'v3', 'v2'):
            if fallback not in versions_to_try:
                versions_to_try.append(fallback)

        for version in versions_to_try:
            url = AFDB_PDB_URL.format(accession=accession, version=version)
            try:
                resp = requests.get(url, timeout=30)
                if resp.status_code == 200:
                    vpath = os.path.join(self.root, 'raw', 'files', f'AF-{accession}-F1-model_{version}.pdb')
                    with open(vpath, 'wb') as f:
                        f.write(resp.content)
                    return True
                elif resp.status_code == 429:
                    time.sleep(2)
                    continue
                # 404 → try next version
            except Exception:
                continue

        return False

    def download(self):
        """Download UniProt index and AFDB PDB files."""
        self._download_uniprot_index()
        self._load_ec_index()

        accessions = sorted(self.ec_index.keys())
        if self.max_proteins is not None:
            accessions = accessions[:self.max_proteins]

        os.makedirs(os.path.join(self.root, 'raw', 'files'), exist_ok=True)

        if self.verbosity > 0:
            print(f'Downloading {len(accessions)} AFDB structures...')

        # Download with parallel workers
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=self.download_workers)(
            delayed(self._download_single_pdb)(acc)
            for acc in progressbar(accessions, desc='Downloading AFDB PDBs', verbosity=self.verbosity)
        )

        success = sum(results)
        if self.verbosity > 0:
            print(f'Downloaded {success}/{len(accessions)} structures')

    # ------------------------------------------------------------------
    # Parsing (skip freesasa for speed)
    # ------------------------------------------------------------------

    def parse_pdb(self, path):
        """Parse a PDB file, skipping freesasa computation for speed.

        AFDB structures are single-chain predicted models. We store pLDDT
        (from B-factor column) instead of SASA.
        """
        import numpy as np

        pdbid = self.get_id_from_filename(os.path.basename(path))
        if pdbid in self.exclude_ids:
            return None
        atom_df = self.pdb2df(path)
        residue_df = atom_df[atom_df['atom_type'] == 'CA']
        if not self.validate(atom_df):
            return None

        protein = {
            'protein': {
                'ID': pdbid,
                'sequence': ''.join(residue_df['residue_type']),
            },
            'residue': {
                'residue_number': residue_df['residue_number'].tolist(),
                'residue_type': residue_df['residue_type'].tolist(),
                'x': residue_df['x'].tolist(),
                'y': residue_df['y'].tolist(),
                'z': residue_df['z'].tolist(),
                'pLDDT': residue_df['b_factor'].tolist(),
            },
            'atom': {
                'atom_number': atom_df['atom_number'].tolist(),
                'atom_type': atom_df['atom_type'].tolist(),
                'residue_number': atom_df['residue_number'].tolist(),
                'residue_type': atom_df['residue_type'].tolist(),
                'x': atom_df['x'].tolist(),
                'y': atom_df['y'].tolist(),
                'z': atom_df['z'].tolist(),
                'pLDDT': atom_df['b_factor'].tolist(),
            },
        }

        protein = self.add_protein_attributes(protein)
        return protein

    # ------------------------------------------------------------------
    # Protein attributes
    # ------------------------------------------------------------------

    def add_protein_attributes(self, protein):
        """Add EC annotation to the protein dict."""
        pdb_id = protein['protein']['ID']

        # pdb_id may be the UniProt accession (extracted from AF-{acc}-F1-model)
        accession = pdb_id

        if not self.ec_index:
            self._load_ec_index()

        if accession not in self.ec_index:
            # Skip proteins without EC annotation
            return None

        entry = self.ec_index[accession]
        protein['protein']['EC'] = entry['ec']
        protein['protein']['EC_all'] = entry['ec_all']
        protein['protein']['uniprot_id'] = accession
        protein['protein']['organism'] = entry['organism']

        return protein

    def describe(self):
        desc = super().describe()
        desc['property'] = 'Enzyme Commission (AlphaFold DB)'
        desc['type'] = 'Classification (multi-level)'
        return desc
