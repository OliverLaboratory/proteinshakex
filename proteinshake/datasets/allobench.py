# -*- coding: utf-8 -*-
"""
Dataset for proteins with allosteric and active site annotations from AlloBench.

AlloBench is a benchmark dataset of proteins with known allosteric and active sites,
derived from the AlloSteric Database (ASD) and enriched with structural information
from UniProt and PDB.

Reference: https://github.com/djmaity/allobench
"""
import os
import ast
import csv
import re

import numpy as np

from proteinshake.datasets import RCSBDataset
from proteinshake.utils import download_url, progressbar, warning, error


class AlloBenchDataset(RCSBDataset):
    """Proteins with allosteric and active site residue annotations from AlloBench.

    This dataset includes PDB structures annotated with allosteric site residues
    (where allosteric modulators bind) and active/catalytic site residues. The
    annotations originate from the AlloSteric Database (ASD), enriched with
    UniProt and PDB data.

    Each protein entry may appear multiple times in the source CSV (once per
    modulator). This dataset deduplicates by PDB ID, merging all allosteric site
    residues and keeping a single set of active site residues per structure.

    .. admonition:: Please cite

        Maity, D. "AlloBench: a pipeline to create dataset of proteins with known
        allosteric and active sites." GitHub, 2024. https://github.com/djmaity/allobench

    .. admonition:: Source

        Raw data obtained from `AlloBench <https://github.com/djmaity/allobench>`_,
        licensed under MIT.

    .. list-table:: Annotations
        :widths: 25 35 45
        :header-rows: 1

        * - Attribute
          - Key
          - Sample value
        * - Allosteric sites (list of residue lists per modulator)
          - :code:`protein['sites']['allosteric']`
          - :code:`[[{'chain': 'A', 'residue_number': 45}, ...], ...]`
        * - Allosteric sites info (modulator metadata)
          - :code:`protein['sites']['allosteric_info']`
          - :code:`[{'modulator': 'PHE', 'class': 'Inhibitor', 'type': 'Lig'}, ...]`
        * - Active sites (list with one residue list)
          - :code:`protein['sites']['active']`
          - :code:`[[{'chain': '', 'residue_number': 63}, ...]]`
        * - Active sites info
          - :code:`protein['sites']['active_info']`
          - :code:`[{'type': 'active_site'}]`
        * - Allosteric site residues (binary per residue)
          - :code:`protein['residue']['allosteric_site']`
          - :code:`[0, 0, 1, 0, 1, 0, ...]`
        * - Active site residues (binary per residue)
          - :code:`protein['residue']['active_site']`
          - :code:`[0, 0, 1, 0, 1, 0, ...]`
        * - Allosteric or active site (binary)
          - :code:`protein['residue']['functional_site']`
          - :code:`[0, 0, 1, 0, 1, 0, ...]`
        * - Site overlap flag
          - :code:`protein['protein']['site_overlap']`
          - :code:`True`
        * - UniProt ID
          - :code:`protein['protein']['uniprot_id']`
          - :code:`'Q9K169'`
        * - Organism
          - :code:`protein['protein']['organism']`
          - :code:`'Human'`
        * - Target gene
          - :code:`protein['protein']['target_gene']`
          - :code:`'PDPK1'`

    Parameters
    ----------
    csv_file : str, optional
        Path to AlloBench.csv. If None, downloads from GitHub.
    """

    description = 'Allosteric and Active Site Residues (AlloBench)'
    CSV_URL = 'https://raw.githubusercontent.com/djmaity/allobench/main/AlloBench.csv'

    def __init__(self, csv_file=None, **kwargs):
        self.csv_file = csv_file
        self.annotations = {}  # pdb_id -> merged annotation dict
        super().__init__(query=[], from_list=None, only_single_chain=False, **kwargs)

        # Ensure annotations are loaded even if download was skipped
        if not self.annotations:
            self._load_csv()

    def get_id_from_filename(self, filename):
        return os.path.basename(filename)[:4].upper()

    # ------------------------------------------------------------------
    # CSV loading and parsing
    # ------------------------------------------------------------------

    def _resolve_csv_path(self):
        """Return path to AlloBench.csv, downloading if needed."""
        if self.csv_file and os.path.exists(self.csv_file):
            return self.csv_file
        local_path = os.path.join(self.root, 'raw', 'AlloBench.csv')
        if not os.path.exists(local_path):
            os.makedirs(os.path.join(self.root, 'raw'), exist_ok=True)
            if self.verbosity > 0:
                print(f'Downloading AlloBench.csv from GitHub...')
            download_url(self.CSV_URL, os.path.join(self.root, 'raw'), verbosity=self.verbosity)
        return local_path

    @staticmethod
    def _parse_allosteric_residues(residue_str):
        """Parse allosteric_site_residue column.

        Format is a Python list literal of strings like:
            "['B-THR-7', 'B-ILE-12', 'A-ASP-8']"

        Each element is CHAIN-RESNAME-RESNUMBER.

        Returns
        -------
        list of dict
            [{'chain': 'B', 'residue_number': 7, 'residue_name': 'THR'}, ...]
        """
        if not residue_str or residue_str.strip() in ('', '[]', 'nan'):
            return []
        try:
            items = ast.literal_eval(residue_str)
        except (ValueError, SyntaxError):
            return []

        residues = []
        for item in items:
            # Expected format: "CHAIN-RESNAME-RESNUMBER"
            parts = item.strip().strip("'\"").split('-')
            if len(parts) == 3:
                chain, resname, resnum_str = parts
                try:
                    resnum = int(resnum_str)
                    residues.append({
                        'chain': chain.strip(),
                        'residue_number': resnum,
                        'residue_name': resname.strip(),
                    })
                except ValueError:
                    pass
            elif len(parts) == 2:
                # No chain: "RESNAME-RESNUMBER"
                resname, resnum_str = parts
                try:
                    resnum = int(resnum_str)
                    residues.append({
                        'chain': '',
                        'residue_number': resnum,
                        'residue_name': resname.strip(),
                    })
                except ValueError:
                    pass
        return residues

    @staticmethod
    def _parse_active_residues(residue_str):
        """Parse active_site_residue column.

        Format is a Python list literal of integers (1-indexed sequence positions):
            "[63, 94, 99, 166, 167, 188]"

        Returns
        -------
        list of dict
            [{'chain': '', 'residue_number': 63}, ...]
        """
        if not residue_str or residue_str.strip() in ('', '[]', 'nan'):
            return []
        try:
            items = ast.literal_eval(residue_str)
        except (ValueError, SyntaxError):
            return []

        residues = []
        for item in items:
            try:
                residues.append({
                    'chain': '',
                    'residue_number': int(item),
                })
            except (ValueError, TypeError):
                pass
        return residues

    def _load_csv(self):
        """Load AlloBench.csv and build per-PDB annotation dictionaries.

        Multiple rows can share the same PDB ID (one row per modulator).
        We merge allosteric residues across modulators and keep metadata.
        """
        csv_path = self._resolve_csv_path()
        if not os.path.exists(csv_path):
            error(f'AlloBench CSV not found at {csv_path}', verbosity=self.verbosity)
            return

        self.annotations = {}

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                pdb_id = row.get('allosteric_pdb', '').strip().upper()
                if not pdb_id or len(pdb_id) != 4:
                    continue

                allosteric_residues = self._parse_allosteric_residues(
                    row.get('allosteric_site_residue', ''))
                active_residues = self._parse_active_residues(
                    row.get('active_site_residue', ''))

                modulator_info = {
                    'modulator_name': row.get('modulator_name', '').strip(),
                    'modulator_alias': row.get('modulator_alias', '').strip(),
                    'modulator_chain': row.get('modulator_chain', '').strip(),
                    'modulator_class': row.get('modulator_class', '').strip(),
                    'modulator_feature': row.get('modulator_feature', '').strip(),
                    'function': row.get('function', '').strip(),
                    'position': row.get('position', '').strip(),
                }

                site_overlap_str = row.get('site_overlap', '').strip()
                site_overlap = site_overlap_str.lower() == 'yes'

                if pdb_id not in self.annotations:
                    self.annotations[pdb_id] = {
                        'target_gene': row.get('target_gene', '').strip(),
                        'organism': row.get('organism', '').strip(),
                        'uniprot_id': row.get('pdb_uniprot', '').strip(),
                        'site_overlap': site_overlap,
                        'allosteric_sites': [],       # list of lists (one per modulator)
                        'allosteric_sites_info': [],   # one dict per modulator
                        'active_residues': active_residues,
                    }

                self.annotations[pdb_id]['allosteric_sites'].append(allosteric_residues)
                self.annotations[pdb_id]['allosteric_sites_info'].append(modulator_info)
                # Update overlap if any row says yes
                if site_overlap:
                    self.annotations[pdb_id]['site_overlap'] = True

        if self.verbosity > 0:
            print(f'Loaded AlloBench annotations for {len(self.annotations)} PDB structures')

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def download(self):
        """Download PDB files for all AlloBench entries."""
        self._load_csv()
        if not self.annotations:
            error('No AlloBench annotations loaded.', verbosity=self.verbosity)
            return
        self.from_list = sorted(self.annotations.keys())
        super().download()

    def add_protein_attributes(self, protein):
        """Add allosteric and active site annotations to the protein dict."""
        pdb_id = protein['protein']['ID'].upper()

        # Ensure annotations are loaded
        if not self.annotations:
            self._load_csv()

        if pdb_id not in self.annotations:
            # No annotation for this PDB
            protein['sites'] = {
                'allosteric': [],
                'allosteric_info': [],
                'active': [],
                'active_info': [],
            }
            n_res = len(protein['residue']['residue_number'])
            protein['residue']['allosteric_site'] = [0] * n_res
            protein['residue']['active_site'] = [0] * n_res
            protein['residue']['functional_site'] = [0] * n_res
            if 'atom' in protein:
                n_atom = len(protein['atom']['residue_number'])
                protein['atom']['allosteric_site'] = [0] * n_atom
                protein['atom']['active_site'] = [0] * n_atom
                protein['atom']['functional_site'] = [0] * n_atom
            protein['protein']['site_overlap'] = False
            protein['protein']['uniprot_id'] = ''
            protein['protein']['organism'] = ''
            protein['protein']['target_gene'] = ''
            return protein

        annot = self.annotations[pdb_id]

        # --- Protein-level metadata ---
        protein['protein']['site_overlap'] = annot['site_overlap']
        protein['protein']['uniprot_id'] = annot['uniprot_id']
        protein['protein']['organism'] = annot['organism']
        protein['protein']['target_gene'] = annot['target_gene']

        # --- Residue/atom arrays ---
        residue_numbers = np.array(protein['residue']['residue_number'])
        if 'chain_id' in protein['residue']:
            chain_ids = protein['residue']['chain_id']
        else:
            chain_ids = [None] * len(residue_numbers)

        # Build atom-level arrays
        if 'atom' in protein:
            atom_residue_numbers = np.array(protein['atom']['residue_number'])
            if 'chain_id' in protein['atom']:
                atom_chain_ids = protein['atom']['chain_id']
            else:
                atom_chain_ids = [None] * len(atom_residue_numbers)

        # --- Helper to match residue dicts to protein residue indices ---
        def _match_residue(res_dict, res_nums, ch_ids):
            """Return array of matching indices."""
            chain = res_dict.get('chain', '')
            resnum = res_dict['residue_number']
            if not chain:
                return np.where(res_nums == resnum)[0]
            else:
                chain_upper = chain.strip().upper()
                chain_match = np.array([
                    (c.strip().upper() if c else '') == chain_upper for c in ch_ids
                ])
                return np.where(chain_match & (res_nums == resnum))[0]

        # --- Allosteric sites ---
        allosteric_mask = np.zeros(len(residue_numbers), dtype=int)
        matched_allosteric_sites = []

        for site_residues in annot['allosteric_sites']:
            matched = []
            for res_dict in site_residues:
                idxs = _match_residue(res_dict, residue_numbers, chain_ids)
                for idx in idxs:
                    allosteric_mask[idx] = 1
                    actual_chain = chain_ids[idx] if chain_ids[idx] is not None else ''
                    actual_chain = str(actual_chain) if actual_chain else ''
                    matched.append({
                        'chain': actual_chain,
                        'residue_number': int(residue_numbers[idx]),
                    })
            if matched:
                matched_allosteric_sites.append(matched)

        # --- Active sites ---
        # Active site residues in AlloBench are 1-indexed sequence positions
        # (no chain specified). We match them to the actual residue_number in the PDB.
        active_mask = np.zeros(len(residue_numbers), dtype=int)
        matched_active = []

        for res_dict in annot['active_residues']:
            idxs = _match_residue(res_dict, residue_numbers, chain_ids)
            for idx in idxs:
                active_mask[idx] = 1
                actual_chain = chain_ids[idx] if chain_ids[idx] is not None else ''
                actual_chain = str(actual_chain) if actual_chain else ''
                matched_active.append({
                    'chain': actual_chain,
                    'residue_number': int(residue_numbers[idx]),
                })

        # --- Combined functional site mask ---
        functional_mask = np.clip(allosteric_mask + active_mask, 0, 1)

        # --- Sites structure ---
        protein['sites'] = {
            'allosteric': matched_allosteric_sites,
            'allosteric_info': annot['allosteric_sites_info'],
            'active': [matched_active] if matched_active else [],
            'active_info': [{'type': 'active_site'}] if matched_active else [],
        }

        protein['residue']['allosteric_site'] = allosteric_mask.tolist()
        protein['residue']['active_site'] = active_mask.tolist()
        protein['residue']['functional_site'] = functional_mask.tolist()

        # --- Atom-level annotations ---
        if 'atom' in protein:
            atom_allosteric_mask = np.zeros(len(atom_residue_numbers), dtype=int)
            atom_active_mask = np.zeros(len(atom_residue_numbers), dtype=int)

            for site_residues in annot['allosteric_sites']:
                for res_dict in site_residues:
                    idxs = _match_residue(res_dict, atom_residue_numbers, atom_chain_ids)
                    if len(idxs) > 0:
                        atom_allosteric_mask[idxs] = 1

            for res_dict in annot['active_residues']:
                idxs = _match_residue(res_dict, atom_residue_numbers, atom_chain_ids)
                if len(idxs) > 0:
                    atom_active_mask[idxs] = 1

            atom_functional_mask = np.clip(atom_allosteric_mask + atom_active_mask, 0, 1)

            protein['atom']['allosteric_site'] = atom_allosteric_mask.tolist()
            protein['atom']['active_site'] = atom_active_mask.tolist()
            protein['atom']['functional_site'] = atom_functional_mask.tolist()

        return protein

    def describe(self):
        """Produce dataset statistics."""
        desc = super().describe()
        desc['property'] = 'Allosteric and Active Site Residues (AlloBench)'
        desc['type'] = 'Binary (residue-level)'
        return desc
