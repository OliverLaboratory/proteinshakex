# -*- coding: utf-8 -*-
"""
Multi-conformation dataset class for proteins with multiple structures (e.g. NMR ensembles).
"""
import os
import numpy as np
import freesasa
from biopandas.pdb import PandasPdb
from joblib import Parallel, delayed

from proteinshake.datasets.dataset import Dataset, AA_THREE_TO_ONE
from proteinshake.utils import write_avro, progressbar


class MultiConfDataset(Dataset):
    """Dataset where each protein entry has multiple conformations.

    Extends the base Dataset by keeping all MODEL entries from PDB files
    instead of only the first one. Coordinates are stored as lists of lists,
    one per conformation, while sequence-level attributes (residue types,
    residue numbers, etc.) are shared across conformations.

    The protein dictionary has the following structure for coordinates::

        protein['residue']['x']  ->  [[x1, x2, ...], [x1', x2', ...], ...]
        protein['atom']['x']     ->  [[x1, x2, ...], [x1', x2', ...], ...]

    And a new field for the number of conformations::

        protein['protein']['num_conformations']  ->  int

    All other fields remain the same as the base Dataset.
    """

    def pdb2df(self, path):
        """Parses a PDB file into a list of DataFrames, one per model.

        Unlike the base class which only returns the first model, this
        returns all models in the PDB file.

        Parameters
        ----------
        path : str
            Path to PDB file.

        Returns
        -------
        list of DataFrame
            One DataFrame per model in the PDB file.
        """
        with open(path, 'r') as file:
            lines = file.read().split('\n')

        # Split lines into models
        models = []
        current_model = []
        has_model_records = any(line.startswith('MODEL') for line in lines)

        if not has_model_records:
            # Single model, no MODEL/ENDMDL records
            models.append(lines)
        else:
            header_lines = []
            for line in lines:
                if line.startswith('MODEL'):
                    current_model = list(header_lines)
                elif line.startswith('ENDMDL'):
                    models.append(current_model)
                    current_model = []
                elif not models and not current_model:
                    # Lines before any MODEL record (header)
                    header_lines.append(line)
                else:
                    current_model.append(line)

        IONS = ['ZN', 'MG']
        dfs = []
        for model_lines in models:
            df = PandasPdb().read_pdb_from_list(model_lines).df['ATOM']
            if len(df) == 0:
                continue
            df = df.loc[~df['residue_name'].isin(IONS)]
            df['residue_name'] = df['residue_name'].map(
                lambda x: AA_THREE_TO_ONE[x] if x in AA_THREE_TO_ONE else None
            )
            df = df.rename(columns={
                'atom_name': 'atom_type',
                'residue_name': 'residue_type',
                'x_coord': 'x',
                'y_coord': 'y',
                'z_coord': 'z',
            })
            df = df.sort_values(by=['chain_id', 'residue_number', 'atom_number'])
            dfs.append(df)

        return dfs

    def parse_pdb(self, path):
        """Parses a PDB file with multiple models into a protein dict.

        Coordinates are stored as nested lists (one list per conformation).
        Sequence-level attributes are taken from the first model.

        Parameters
        ----------
        path : str
            Path to PDB file.

        Returns
        -------
        dict or None
            A protein dict, or None if invalid.
        """
        pdbid = self.get_id_from_filename(os.path.basename(path))
        if pdbid in self.exclude_ids:
            return None

        atom_dfs = self.pdb2df(path)
        if not atom_dfs:
            return None

        # Validate using the first model
        if not self.validate(atom_dfs[0]):
            return None

        # Use first model for shared attributes
        first_atom_df = atom_dfs[0]
        first_residue_df = first_atom_df[first_atom_df['atom_type'] == 'CA']

        # Compute SASA from first model
        structure = freesasa.Structure(path)
        result = freesasa.calc(structure)
        residue_result = result.residueAreas()

        atom_sasa, residue_sasa, residue_rsa = [], [], []
        for i in first_atom_df['atom_number']:
            try:
                assert not np.isnan(result.atomArea(i)), "nan sasa"
                atom_sasa.append(result.atomArea(i))
            except:
                atom_sasa.append(-1)
        for i, chain in zip(first_residue_df['residue_number'], first_residue_df['chain_id']):
            try:
                assert not np.isnan(residue_result[chain][str(i)].total), "nan sasa"
                assert not np.isnan(residue_result[chain][str(i)].relativeTotal), "nan sasa"
                residue_sasa.append(residue_result[chain][str(i)].total)
                residue_rsa.append(residue_result[chain][str(i)].relativeTotal)
            except:
                residue_sasa.append(-1)
                residue_rsa.append(-1)

        # Collect coordinates from all conformations
        residue_x, residue_y, residue_z = [], [], []
        atom_x, atom_y, atom_z = [], [], []

        for atom_df in atom_dfs:
            residue_df = atom_df[atom_df['atom_type'] == 'CA']
            residue_x.append(residue_df['x'].tolist())
            residue_y.append(residue_df['y'].tolist())
            residue_z.append(residue_df['z'].tolist())
            atom_x.append(atom_df['x'].tolist())
            atom_y.append(atom_df['y'].tolist())
            atom_z.append(atom_df['z'].tolist())

        protein = {
            'protein': {
                'ID': pdbid,
                'sequence': ''.join(first_residue_df['residue_type']),
                'num_conformations': len(atom_dfs),
            },
            'residue': {
                'residue_number': first_residue_df['residue_number'].tolist(),
                'residue_type': first_residue_df['residue_type'].tolist(),
                'x': residue_x,
                'y': residue_y,
                'z': residue_z,
                'SASA': residue_sasa,
                'RSA': residue_rsa,
            },
            'atom': {
                'atom_number': first_atom_df['atom_number'].tolist(),
                'atom_type': first_atom_df['atom_type'].tolist(),
                'residue_number': first_atom_df['residue_number'].tolist(),
                'residue_type': first_atom_df['residue_type'].tolist(),
                'x': atom_x,
                'y': atom_y,
                'z': atom_z,
                'SASA': atom_sasa,
            },
        }

        if not self.only_single_chain:
            protein['residue']['chain_id'] = first_residue_df['chain_id'].tolist()
            protein['atom']['chain_id'] = first_atom_df['chain_id'].tolist()

        protein = self.add_protein_attributes(protein)
        return protein
