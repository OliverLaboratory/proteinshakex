# -*- coding: utf-8 -*-
"""
Dataset for proteins with catalytic site residue annotations from M-CSA (Mechanism and Catalytic Site Atlas).
"""
import os
import requests
import json

import numpy as np

from proteinshake.datasets import RCSBDataset
from proteinshake.utils import progressbar, warning, error


class MCSADataset(RCSBDataset):
    """ Proteins with annotated catalytic site residues from M-CSA (Mechanism and Catalytic Site Atlas).

    This dataset includes PDB structures with catalytic site residue annotations fetched from the
    M-CSA database (https://www.ebi.ac.uk/thornton-srv/m-csa/). The annotations specify which residues
    are part of catalytic sites based on experimental evidence.

    .. admonition:: Please cite

        Ribeiro, A. J. M., et al. "Mechanism and Catalytic Site Atlas (M-CSA): a database of enzyme
        catalytic mechanisms and active sites." Nucleic acids research 46.1 (2018): D618-D623.

    .. admonition:: Source

        Raw data was obtained from `M-CSA API <https://www.ebi.ac.uk/thornton-srv/m-csa/api/>`_,
        originally licensed under `CC-BY-4.0 <https://creativecommons.org/licenses/by/4.0/>`_.

    .. list-table:: Dataset stats
       :widths: 100
       :header-rows: 1

       * - # proteins
       * - ~1000+ (all proteins in M-CSA database when fetch_all_pdb_ids=True)

    .. list-table:: Annotations
        :widths: 25 35 45
        :header-rows: 1

        * - Attribute
          - Key
          - Sample value
        * - Catalytic site residues (binary)
          - :code:`protein['residue']['catalytic_site']`
          - :code:`[0, 0, 1, 0, 1, 0, ...]`
        * - M-CSA sites (list of residue lists)
          - :code:`protein['sites']['csa']`
          - :code:`[[{'chain': 'A', 'residue_number': 7}, ...], [...]]`
        * - M-CSA sites info (metadata)
          - :code:`protein['sites']['csa_info']`
          - :code:`[{'mcsa_id': 1, 'type': 'M-CSA-1', 'roles': [...]}, ...]`
        * - M-CSA site IDs (per residue)
          - :code:`protein['residue']['csa']`
          - :code:`['', '', '1', '', ['1', '2'], ...]`
        * - Catalytic site atoms (binary)
          - :code:`protein['atom']['catalytic_site']`
          - :code:`[0, 0, 1, 0, 1, 0, ...]`
        * - Catalytic site residue numbers
          - :code:`protein['protein']['catalytic_site_residues']`
          - :code:`[{'chain': 'A', 'residue_number': 7}, ...]`
        * - Catalytic site annotation string
          - :code:`protein['protein']['catalytic_site_annotation']`
          - :code:`'A:7,A:70,A:178'`
        * - M-CSA mechanism ID
          - :code:`protein['protein']['mcsa_id']`
          - :code:`1`
        * - Catalytic site roles
          - :code:`protein['protein']['catalytic_site_roles']`
          - :code:`['activator', 'proton acceptor', ...]`

    Parameters
    ----------
    pdb_ids: list, optional
        List of PDB IDs to include in the dataset. If None and fetch_all_pdb_ids is False,
        will fetch annotations for all PDB files found in the raw/files directory.
    fetch_all_pdb_ids: bool, default False
        If True, automatically fetches all PDB IDs from the M-CSA entries API (or local file)
        and downloads the corresponding PDB structures. This will download all proteins in the M-CSA database.
    annotation_file: str, default 'data/csa.json'
        Path to local M-CSA annotations JSON file. If the file exists, annotations will be
        loaded from it instead of the API, which is much faster. Falls back to API if file not found.
    mcsa_api_url: str, default 'https://www.ebi.ac.uk/thornton-srv/m-csa/api/residues/'
        Base URL for the M-CSA API residues endpoint (used as fallback if annotation_file not found).
    mcsa_entries_api_url: str, default 'https://www.ebi.ac.uk/thornton-srv/m-csa/api/entries/'
        Base URL for the M-CSA API entries endpoint (used as fallback if annotation_file not found).
    """

    description = 'Catalytic Site Residues (M-CSA)'

    def __init__(self, pdb_ids=None, fetch_all_pdb_ids=False, annotation_file='data/csa.json', mcsa_api_url='https://www.ebi.ac.uk/thornton-srv/m-csa/api/residues/', mcsa_entries_api_url='https://www.ebi.ac.uk/thornton-srv/m-csa/api/entries/', **kwargs):
        """
        Args:
            pdb_ids: List of PDB IDs to include. If None, will automatically load all PDB IDs
                from annotation_file if it exists, or use all PDB files in raw/files.
            fetch_all_pdb_ids: If True, fetches all PDB IDs from M-CSA entries API.
                If annotation_file exists, this is automatically set to True to use all PDBs from the file.
            annotation_file: Path to local M-CSA annotations JSON file (default: 'data/csa.json').
                If the file exists, annotations will be loaded from it and all PDB IDs in the file
                will be automatically included.
            mcsa_api_url: Base URL for M-CSA API residues endpoint (used as fallback if annotation_file not found).
            mcsa_entries_api_url: Base URL for M-CSA API entries endpoint.
        """
        self.pdb_ids = pdb_ids
        self.fetch_all_pdb_ids = fetch_all_pdb_ids
        self.annotation_file = annotation_file
        self.mcsa_api_url = mcsa_api_url
        self.mcsa_entries_api_url = mcsa_entries_api_url
        self.annotations = {}
        self.mcsa_data = {}  # Store full M-CSA data for each protein
        self._cached_pdb_annotations = None  # Cache for loaded annotations from file
        
        # If annotation_file is provided and exists, automatically fetch all PDB IDs from it
        # unless pdb_ids is explicitly provided
        if pdb_ids is None and not fetch_all_pdb_ids:
            # Check if annotation_file exists
            annotation_path = annotation_file
            if not os.path.isabs(annotation_path):
                # Try relative to current directory first
                if not os.path.exists(annotation_path):
                    # Try relative to root if provided in kwargs
                    if 'root' in kwargs and kwargs['root']:
                        root_path = os.path.join(kwargs['root'], annotation_path)
                        if os.path.exists(root_path):
                            annotation_path = root_path
            
            if os.path.exists(annotation_path):
                # File exists, automatically fetch all PDB IDs from it
                self.fetch_all_pdb_ids = True
        
        # Initialize with empty query and from_list - we'll set from_list in download()
        super().__init__(query=[], from_list=None, only_single_chain=False, **kwargs)

    def get_id_from_filename(self, filename):
        return os.path.basename(filename)[:4].upper()

    def fetch_all_pdb_ids_from_mcsa(self):
        """Fetch all PDB IDs from local file or M-CSA entries API.
        
        Returns
        -------
        set
            Set of unique PDB IDs (4-character codes) found in M-CSA.
        """
        # Try loading from local file first
        pdb_annotations = self.load_annotations_from_file()
        
        if pdb_annotations:
            # Extract PDB IDs from local file
            pdb_ids = set(pdb_annotations.keys())
            if self.verbosity > 0:
                print(f'✓ Found {len(pdb_ids)} unique PDB IDs in local M-CSA annotation file')
            return pdb_ids
        
        # Fallback to API
        pdb_ids = set()
        url = f'{self.mcsa_entries_api_url}?format=json'
        
        if self.verbosity > 0:
            print('Fetching all PDB IDs from M-CSA entries API...')
        
        page_count = 0
        while url:
            try:
                response = requests.get(url, timeout=30)
                if response.status_code != 200:
                    warning(f'M-CSA entries API request failed: HTTP {response.status_code}', verbosity=self.verbosity)
                    break
                
                data = response.json()
                
                # Extract PDB IDs from all entries in this page
                for entry in data.get('results', []):
                    if 'residues' in entry:
                        for residue in entry['residues']:
                            if 'residue_chains' in residue:
                                for chain_info in residue['residue_chains']:
                                    pdb_id = chain_info.get('pdb_id', '').upper()
                                    if pdb_id and len(pdb_id) == 4:
                                        pdb_ids.add(pdb_id)
                
                # Get next page URL
                url = data.get('next')
                page_count += 1
                
                if self.verbosity > 0:
                    print(f'  Processed page {page_count}, found {len(pdb_ids)} unique PDB IDs so far...', end='\r')
                
            except requests.exceptions.RequestException as e:
                warning(f'Failed to fetch M-CSA entries: {str(e)}', verbosity=self.verbosity)
                break
            except Exception as e:
                warning(f'Error parsing M-CSA entries: {str(e)}', verbosity=self.verbosity)
                break
        
        if self.verbosity > 0:
            print(f'\n✓ Found {len(pdb_ids)} unique PDB IDs in M-CSA database')
        
        return pdb_ids

    def fetch_mcsa_annotation(self, pdb_id):
        """Fetch catalytic site annotations from M-CSA API for a given PDB ID.
        
        Parameters
        ----------
        pdb_id: str
            PDB ID (4-character code, e.g., '1B73')
            
        Returns
        -------
        tuple or None
            (annotation_string, mcsa_data) tuple or None if not found.
            annotation_string: "A:7,A:70,B:178" format
            mcsa_data: Full M-CSA API response data
        """
        try:
            pdb_id = pdb_id.upper()
            url = f'{self.mcsa_api_url}?format=json&pdb_id={pdb_id}'
            response = requests.get(url, timeout=10)
            
            if response.status_code != 200:
                if self.verbosity > 1:
                    warning(f'M-CSA API request failed for {pdb_id}: HTTP {response.status_code}', verbosity=self.verbosity)
                return None
            
            data = response.json()
            if not data or len(data) == 0:
                return None
            
            # Extract residues from M-CSA response
            residues = []
            mcsa_ids = []
            all_roles = []
            
            for entry in data:
                mcsa_id = entry.get('mcsa_id', None)
                if mcsa_id:
                    mcsa_ids.append(mcsa_id)
                
                roles_summary = entry.get('roles_summary', '')
                if roles_summary:
                    all_roles.extend([r.strip() for r in roles_summary.split(',')])
                
                if 'residue_chains' in entry:
                    for chain_info in entry['residue_chains']:
                        chain = chain_info.get('chain_name', '')
                        resid = chain_info.get('resid', None) or chain_info.get('auth_resid', None)
                        if resid is not None:
                            residues.append((chain, int(resid)))
            
            if not residues:
                return None
            
            # Convert to annotation string format: "A:7,A:70,B:178" or "7,70,178" if single chain
            chains = set([c for c, _ in residues if c])
            if len(chains) == 1 and chains.pop():
                # Single chain, use format "7,70,178"
                annotation_str = ','.join([str(r) for _, r in residues])
            else:
                # Multiple chains or no chain info, use format "A:7,A:70,B:178"
                annotation_str = ','.join([f'{c}:{r}' if c else str(r) for c, r in residues])
            
            # Store additional metadata
            metadata = {
                'mcsa_ids': list(set(mcsa_ids)),
                'roles': list(set(all_roles))
            }
            
            return (annotation_str, data, metadata)
                
        except requests.exceptions.RequestException as e:
            if self.verbosity > 1:
                warning(f'Failed to fetch M-CSA annotation for {pdb_id}: {str(e)}', verbosity=self.verbosity)
            return None
        except Exception as e:
            if self.verbosity > 1:
                warning(f'Error parsing M-CSA annotation for {pdb_id}: {str(e)}', verbosity=self.verbosity)
            return None

    def parse_annotation(self, annotation_str):
        """Parse catalytic site residue annotation string.
        
        Supports format: "A:7,A:70,B:178" or "7,70,178"
        
        Parameters
        ----------
        annotation_str: str
            Annotation string to parse.
            
        Returns
        -------
        list
            List of (chain, residue_number) tuples. If no chain is specified, chain is None.
        """
        if not annotation_str or annotation_str.strip() == '':
            return []
        
        residues = []
        parts = [p.strip() for p in annotation_str.split(',')]
        
        for part in parts:
            if not part:
                continue
            
            # Check for chain:residue format (e.g., "A:7")
            if ':' in part:
                try:
                    chain, residue = part.split(':', 1)
                    chain = chain.strip()
                    residue = residue.strip()
                    residues.append((chain, int(residue)))
                except ValueError:
                    warning(f'Could not parse chain:residue: {part}', verbosity=self.verbosity)
            else:
                # Simple residue number (no chain specified)
                try:
                    residues.append((None, int(part)))
                except ValueError:
                    warning(f'Could not parse residue number: {part}', verbosity=self.verbosity)
        
        return residues

    def load_annotations_from_file(self):
        """Load annotations from local JSON file.
        
        Returns
        -------
        dict
            Dictionary mapping PDB ID to list of M-CSA entries for that protein.
        """
        # Use cached annotations if available
        if self._cached_pdb_annotations is not None:
            return self._cached_pdb_annotations
        
        # Resolve path - if relative, check relative to current directory first
        annotation_path = self.annotation_file
        if not os.path.isabs(annotation_path) and not os.path.exists(annotation_path):
            # Try relative to root directory if provided
            if hasattr(self, 'root') and self.root:
                root_path = os.path.join(self.root, annotation_path)
                if os.path.exists(root_path):
                    annotation_path = root_path
        
        if not os.path.exists(annotation_path):
            if self.verbosity > 0:
                warning(f'M-CSA annotation file not found: {self.annotation_file}. Will use API instead.', verbosity=self.verbosity)
            self._cached_pdb_annotations = {}
            return {}
        
        if self.verbosity > 0:
            print(f'Loading M-CSA annotations from {annotation_path}...')
        
        try:
            with open(annotation_path, 'r') as f:
                all_entries = json.load(f)
        except Exception as e:
            error(f'Failed to load M-CSA annotation file: {str(e)}', verbosity=self.verbosity)
            self._cached_pdb_annotations = {}
            return {}
        
        # Group entries by PDB ID
        pdb_annotations = {}
        for entry in all_entries:
            if 'residue_chains' in entry and len(entry['residue_chains']) > 0:
                pdb_id = entry['residue_chains'][0].get('pdb_id', '').upper()
                if pdb_id and len(pdb_id) == 4:
                    if pdb_id not in pdb_annotations:
                        pdb_annotations[pdb_id] = []
                    pdb_annotations[pdb_id].append(entry)
        
        if self.verbosity > 0:
            print(f'✓ Loaded annotations for {len(pdb_annotations)} PDB structures')
        
        # Cache the result
        self._cached_pdb_annotations = pdb_annotations
        
        return pdb_annotations

    def process_mcsa_entries(self, entries):
        """Process M-CSA entries (same format as API response) into annotation format.
        
        Parameters
        ----------
        entries: list
            List of M-CSA entry dictionaries.
            
        Returns
        -------
        tuple or None
            (annotation_string, mcsa_data, metadata) tuple or None if no residues found.
        """
        if not entries:
            return None
        
        # Extract residues from entries
        residues = []
        mcsa_ids = []
        all_roles = []
        
        for entry in entries:
            mcsa_id = entry.get('mcsa_id', None)
            if mcsa_id:
                mcsa_ids.append(mcsa_id)
            
            roles_summary = entry.get('roles_summary', '')
            if roles_summary:
                all_roles.extend([r.strip() for r in roles_summary.split(',')])
            
            if 'residue_chains' in entry:
                for chain_info in entry['residue_chains']:
                    chain = chain_info.get('chain_name', '')
                    resid = chain_info.get('resid', None) or chain_info.get('auth_resid', None)
                    if resid is not None:
                        residues.append((chain, int(resid)))
        
        if not residues:
            return None
        
        # Convert to annotation string format: "A:7,A:70,B:178" or "7,70,178" if single chain
        chains = set([c for c, _ in residues if c])
        if len(chains) == 1 and chains.pop():
            # Single chain, use format "7,70,178"
            annotation_str = ','.join([str(r) for _, r in residues])
        else:
            # Multiple chains or no chain info, use format "A:7,A:70,B:178"
            annotation_str = ','.join([f'{c}:{r}' if c else str(r) for c, r in residues])
        
        # Store additional metadata
        metadata = {
            'mcsa_ids': list(set(mcsa_ids)),
            'roles': list(set(all_roles))
        }
        
        return (annotation_str, entries, metadata)
    
    def process_mcsa_entries_with_sites(self, entries):
        """Process M-CSA entries and track which mcsa_id each residue belongs to.
        
        Parameters
        ----------
        entries: list
            List of M-CSA entry dictionaries.
            
        Returns
        -------
        list
            List of (chain, residue_number, mcsa_id) tuples.
        """
        if not entries:
            return []
        
        residues_with_sites = []
        
        for entry in entries:
            mcsa_id = entry.get('mcsa_id', None)
            if mcsa_id is None:
                continue
            
            # Convert mcsa_id to string for consistency
            mcsa_id_str = str(mcsa_id)
            
            if 'residue_chains' in entry:
                for chain_info in entry['residue_chains']:
                    chain = chain_info.get('chain_name', '')
                    resid = chain_info.get('resid', None) or chain_info.get('auth_resid', None)
                    if resid is not None:
                        residues_with_sites.append((chain, int(resid), mcsa_id_str))
        
        return residues_with_sites
    
    def parse_mcsa_sites(self, entries):
        """Parse M-CSA entries and organize sites by mcsa_id.
        
        Parameters
        ----------
        entries: list
            List of M-CSA entry dictionaries.
            
        Returns
        -------
        tuple
            (sites_list, sites_info_list) where:
            - sites_list: List of residue lists, each containing dictionaries with 'chain' and 'residue_number'
            - sites_info_list: List of site metadata dictionaries (mcsa_id, roles, etc.), indexed by site index
        """
        if not entries:
            return [], []
        
        # Group entries by mcsa_id
        sites_by_mcsa_id = {}
        for entry in entries:
            mcsa_id = entry.get('mcsa_id', None)
            if mcsa_id is None:
                continue
            
            mcsa_id_str = str(mcsa_id)
            if mcsa_id_str not in sites_by_mcsa_id:
                sites_by_mcsa_id[mcsa_id_str] = {
                    'residues': [],
                    'roles': [],
                    'main_annotation': entry.get('main_annotation', ''),
                    'roles_summary': entry.get('roles_summary', '')
                }
            
            # Collect residues for this mcsa_id
            if 'residue_chains' in entry:
                for chain_info in entry['residue_chains']:
                    chain = chain_info.get('chain_name', '')
                    resid = chain_info.get('resid', None) or chain_info.get('auth_resid', None)
                    if resid is not None:
                        sites_by_mcsa_id[mcsa_id_str]['residues'].append({
                            'chain': chain if chain else '',
                            'residue_number': int(resid)
                        })
            
            # Collect roles
            roles_summary = entry.get('roles_summary', '')
            if roles_summary:
                roles = [r.strip() for r in roles_summary.split(',')]
                sites_by_mcsa_id[mcsa_id_str]['roles'].extend(roles)
        
        # Convert to lists (ordered by mcsa_id)
        sites_list = []
        sites_info = []
        for mcsa_id_str in sorted(sites_by_mcsa_id.keys(), key=lambda x: int(x) if x.isdigit() else 0):
            site_data = sites_by_mcsa_id[mcsa_id_str]
            if site_data['residues']:
                sites_list.append(site_data['residues'])
                sites_info.append({
                    'mcsa_id': int(mcsa_id_str),
                    'type': f'M-CSA-{mcsa_id_str}',
                    'roles': list(set(site_data['roles'])),
                    'main_annotation': site_data['main_annotation']
                })
        
        return sites_list, sites_info

    def load_annotations(self):
        """Load annotations from local file or M-CSA API for all PDB files."""
        # Try loading from local file first
        pdb_annotations = self.load_annotations_from_file()
        
        if pdb_annotations:
            # Use local file annotations
            pdb_files = self.get_raw_files()
            pdb_ids = [self.get_id_from_filename(f) for f in pdb_files]
            
            if not pdb_ids:
                if self.verbosity > 0:
                    warning('No PDB files found. Please download PDB files first or provide pdb_ids.', verbosity=self.verbosity)
                return
            
            if self.verbosity > 0:
                print(f'Processing M-CSA annotations for {len(pdb_ids)} PDB structures...')
            
            for pdb_id in progressbar(pdb_ids, desc='Processing M-CSA annotations', verbosity=self.verbosity):
                if pdb_id in pdb_annotations:
                    entries = pdb_annotations[pdb_id]
                    result = self.process_mcsa_entries(entries)
                    if result:
                        annotation_str, mcsa_data, metadata = result
                        self.annotations[pdb_id] = annotation_str
                        self.mcsa_data[pdb_id] = {
                            'data': mcsa_data,
                            'mcsa_ids': metadata['mcsa_ids'],
                            'roles': metadata['roles']
                        }
        else:
            # Fallback to API
            pdb_files = self.get_raw_files()
            pdb_ids = [self.get_id_from_filename(f) for f in pdb_files]
            
            if not pdb_ids:
                if self.verbosity > 0:
                    warning('No PDB files found. Please download PDB files first or provide pdb_ids.', verbosity=self.verbosity)
                return
            
            if self.verbosity > 0:
                print(f'Fetching M-CSA annotations from API for {len(pdb_ids)} PDB structures...')
            
            for pdb_id in progressbar(pdb_ids, desc='Fetching M-CSA annotations', verbosity=self.verbosity):
                result = self.fetch_mcsa_annotation(pdb_id)
                if result:
                    annotation_str, mcsa_data, metadata = result
                    self.annotations[pdb_id] = annotation_str
                    self.mcsa_data[pdb_id] = {
                        'data': mcsa_data,
                        'mcsa_ids': metadata['mcsa_ids'],
                        'roles': metadata['roles']
                    }

    def download(self):
        """Download PDB files from RCSB for PDB IDs with M-CSA annotations."""
        # Determine which PDB IDs to download
        if self.fetch_all_pdb_ids:
            # Fetch all PDB IDs from local file or M-CSA entries API
            pdb_ids_set = self.fetch_all_pdb_ids_from_mcsa()
            pdb_ids_to_download = sorted(list(pdb_ids_set))
            if not pdb_ids_to_download:
                if self.verbosity > 0:
                    warning('No PDB IDs found in M-CSA. Please check annotation_file or API connection.', verbosity=self.verbosity)
                return
            # Update self.pdb_ids for later use
            self.pdb_ids = pdb_ids_to_download
        elif self.pdb_ids:
            # If PDB IDs are specified, download them
            pdb_ids_to_download = [pdb_id.upper() for pdb_id in self.pdb_ids]
        else:
            # Try to load PDB IDs from annotation file if it exists
            pdb_annotations = self.load_annotations_from_file()
            if pdb_annotations:
                # Use all PDB IDs from the annotation file
                pdb_ids_to_download = sorted(list(pdb_annotations.keys()))
                if self.verbosity > 0:
                    print(f'Automatically using {len(pdb_ids_to_download)} PDB IDs from annotation file')
            else:
                # No PDB IDs specified and no annotation file
                if self.verbosity > 0:
                    warning('No PDB IDs specified. Please provide pdb_ids parameter, set fetch_all_pdb_ids=True, or download PDB files manually.', verbosity=self.verbosity)
                return

        if not pdb_ids_to_download:
            if self.verbosity > 0:
                warning('No PDB IDs to download.', verbosity=self.verbosity)
            return

        # Use RCSBDataset's download with from_list
        self.from_list = pdb_ids_to_download
        super().download()

        # After downloading, load annotations
        self.load_annotations()

    def add_protein_attributes(self, protein):
        """Add catalytic site residue annotations to the protein dictionary."""
        pdb_id = protein['protein']['ID'].upper()
        
        if pdb_id not in self.annotations:
            # If annotations not loaded yet, try to load them
            if not self.annotations:
                self.load_annotations()
            
            # If still not found, try loading from file or fetching on-demand
            if pdb_id not in self.annotations:
                # Try loading from local file first
                pdb_annotations = self.load_annotations_from_file()
                if pdb_id in pdb_annotations:
                    entries = pdb_annotations[pdb_id]
                    result = self.process_mcsa_entries(entries)
                    if result:
                        annotation_str, mcsa_data, metadata = result
                        self.annotations[pdb_id] = annotation_str
                        self.mcsa_data[pdb_id] = {
                            'data': mcsa_data,
                            'mcsa_ids': metadata['mcsa_ids'],
                            'roles': metadata['roles']
                        }
                else:
                    # Fallback to API
                    result = self.fetch_mcsa_annotation(pdb_id)
                    if result:
                        annotation_str, mcsa_data, metadata = result
                        self.annotations[pdb_id] = annotation_str
                        self.mcsa_data[pdb_id] = {
                            'data': mcsa_data,
                            'mcsa_ids': metadata['mcsa_ids'],
                            'roles': metadata['roles']
                        }
            
            # If still no annotation found, mark all residues as non-catalytic
            if pdb_id not in self.annotations:
                protein['sites'] = {
                    'csa': [],
                    'csa_info': []
                }
                protein['residue']['catalytic_site'] = [0] * len(protein['residue']['residue_number'])
                protein['residue']['csa'] = [''] * len(protein['residue']['residue_number'])
                if 'atom' in protein:
                    protein['atom']['catalytic_site'] = [0] * len(protein['atom']['residue_number'])
                protein['protein']['catalytic_site_residues'] = []
                protein['protein']['catalytic_site_annotation'] = ''
                protein['protein']['mcsa_id'] = None
                protein['protein']['catalytic_site_roles'] = []
                return protein
        
        annotation_str = self.annotations[pdb_id]
        parsed_residues = self.parse_annotation(annotation_str)
        
        # Get M-CSA metadata and entries
        mcsa_metadata = self.mcsa_data.get(pdb_id, {})
        mcsa_ids = mcsa_metadata.get('mcsa_ids', [])
        roles = mcsa_metadata.get('roles', [])
        mcsa_entries = mcsa_metadata.get('data', [])
        
        # Parse sites structure
        sites_list, sites_info = self.parse_mcsa_sites(mcsa_entries)
        
        # Process entries with site information (for residue-level annotations)
        parsed_residues_with_sites = self.process_mcsa_entries_with_sites(mcsa_entries)
        
        # Get residue numbers and chain IDs from the protein
        residue_numbers = np.array(protein['residue']['residue_number'])
        if 'chain_id' in protein['residue']:
            chain_ids = np.array(protein['residue']['chain_id'])
            # Normalize empty strings to None for consistency
            chain_ids = np.array([c if c and c.strip() else None for c in chain_ids])
        else:
            chain_ids = np.array([None] * len(residue_numbers))
        
        # Match sites residues to protein residues and create sites structure
        matched_sites_list = []
        for site_residues in sites_list:
            matched_residues = []
            for res_dict in site_residues:
                res_chain = res_dict.get('chain', '')
                res_num = res_dict.get('residue_number')
                
                # Find matching residues in the protein
                if not res_chain or res_chain == '':
                    # Match by residue number only
                    matches = np.where(residue_numbers == res_num)[0]
                else:
                    # Match by both chain and residue number
                    res_chain = res_chain.strip().upper()
                    chain_match = np.array([(c.strip().upper() if c else '') == res_chain for c in chain_ids])
                    matches = np.where(chain_match & (residue_numbers == res_num))[0]
                
                # Add matched residues (use actual chain from protein)
                for match_idx in matches:
                    actual_chain = chain_ids[match_idx] if chain_ids[match_idx] is not None else ''
                    # Convert to regular string (not numpy string)
                    actual_chain = str(actual_chain) if actual_chain else ''
                    actual_residue = int(residue_numbers[match_idx])
                    matched_residues.append({
                        'chain': actual_chain,
                        'residue_number': actual_residue
                    })
            
            if matched_residues:
                matched_sites_list.append(matched_residues)
        
        # Create sites structure
        protein['sites'] = {
            'csa': matched_sites_list,
            'csa_info': sites_info
        }
        
        # Create binary mask for catalytic sites at residue level (for backward compatibility)
        catalytic_site_mask = np.zeros(len(residue_numbers), dtype=int)
        catalytic_site_residue_list = []
        
        # Dictionary to map residue index to list of mcsa_ids
        residue_index_to_mcsa_ids = {}
        
        # Process each annotated residue and find matching residues in the protein
        for chain, residue_num, mcsa_id in parsed_residues_with_sites:
            # Find matching residues in the protein
            if chain is None or chain == '':
                # Match by residue number only (assumes single chain or first matching residue)
                matches = np.where(residue_numbers == residue_num)[0]
            else:
                # Match by both chain and residue number
                # Normalize chain for comparison
                chain = chain.strip().upper()
                chain_match = np.array([(c.strip().upper() if c else '') == chain for c in chain_ids])
                matches = np.where(chain_match & (residue_numbers == residue_num))[0]
            
            # Add mcsa_id to all matching residues
            for match_idx in matches:
                if match_idx not in residue_index_to_mcsa_ids:
                    residue_index_to_mcsa_ids[match_idx] = []
                if mcsa_id not in residue_index_to_mcsa_ids[match_idx]:
                    residue_index_to_mcsa_ids[match_idx].append(mcsa_id)
                catalytic_site_mask[match_idx] = 1
        
        # Create csa array (for backward compatibility)
        csa_array = []
        for i in range(len(residue_numbers)):
            if i in residue_index_to_mcsa_ids:
                mcsa_id_list = residue_index_to_mcsa_ids[i]
                # Single site -> string, multiple sites -> list
                if len(mcsa_id_list) == 1:
                    csa_array.append(mcsa_id_list[0])
                else:
                    csa_array.append(mcsa_id_list)
            else:
                csa_array.append('')
        
        # Build catalytic_site_residue_list (unique residues)
        seen_residues = set()
        for chain, residue_num in parsed_residues:
            if chain is None or chain == '':
                matches = np.where(residue_numbers == residue_num)[0]
            else:
                chain = chain.strip().upper()
                chain_match = np.array([(c.strip().upper() if c else '') == chain for c in chain_ids])
                matches = np.where(chain_match & (residue_numbers == residue_num))[0]
            
            for match_idx in matches:
                actual_chain = chain_ids[match_idx] if chain_ids[match_idx] is not None else ''
                actual_residue = int(residue_numbers[match_idx])
                residue_key = (actual_chain, actual_residue)
                
                if residue_key not in seen_residues:
                    seen_residues.add(residue_key)
                    catalytic_site_residue_list.append({
                        'chain': actual_chain,
                        'residue_number': actual_residue
                    })
        
        # Keep residue-level annotations for backward compatibility
        protein['residue']['catalytic_site'] = catalytic_site_mask.tolist()
        protein['residue']['csa'] = csa_array
        protein['protein']['catalytic_site_residues'] = catalytic_site_residue_list
        protein['protein']['catalytic_site_annotation'] = annotation_str
        protein['protein']['mcsa_id'] = mcsa_ids[0] if mcsa_ids else None
        protein['protein']['catalytic_site_roles'] = roles
        
        # Also add atom-level annotations
        if 'atom' in protein:
            atom_residue_numbers = np.array(protein['atom']['residue_number'])
            if 'chain_id' in protein['atom']:
                atom_chain_ids = np.array(protein['atom']['chain_id'])
                atom_chain_ids = np.array([c if c and c.strip() else None for c in atom_chain_ids])
            else:
                atom_chain_ids = np.array([None] * len(atom_residue_numbers))
            
            catalytic_site_atom_mask = np.zeros(len(atom_residue_numbers), dtype=int)
            
            for chain, residue_num in parsed_residues:
                if chain is None or chain == '':
                    matches = np.where(atom_residue_numbers == residue_num)[0]
                else:
                    chain = chain.strip().upper()
                    chain_match = np.array([(c.strip().upper() if c else '') == chain for c in atom_chain_ids])
                    matches = np.where(chain_match & (atom_residue_numbers == residue_num))[0]
                
                if len(matches) > 0:
                    catalytic_site_atom_mask[matches] = 1
            
            protein['atom']['catalytic_site'] = catalytic_site_atom_mask.tolist()
        
        return protein

    def describe(self):
        """Produce dataset statistics."""
        desc = super().describe()
        desc['property'] = "Catalytic Site Residues (M-CSA)"
        desc['type'] = 'Binary (residue-level)'
        return desc
