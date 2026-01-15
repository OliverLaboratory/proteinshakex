# -*- coding: utf-8 -*-
"""
Dataset for proteins with functional site residue annotations from Table III.
"""
import os
import csv
import re

import numpy as np

from proteinshake.datasets import RCSBDataset
from proteinshake.utils import progressbar, warning, error


class FunctionalSiteDataset(RCSBDataset):
    """ Proteins with annotated functional site residues from Table III.

    This dataset includes PDB structures with functional site residue annotations.
    The annotations specify which residues are part of functional sites (e.g., catalytic sites,
    binding sites, etc.). Annotations are loaded from a CSV/TSV file.

    .. list-table:: Dataset stats
       :widths: 100
       :header-rows: 1

       * - # proteins
       * - Varies based on input file

    .. list-table:: Annotations
        :widths: 25 35 45
        :header-rows: 1

        * - Attribute
          - Key
          - Sample value
        * - Functional sites (list of residue lists)
          - :code:`protein['sites']['functional_sites']`
          - :code:`[[{'chain': 'A', 'residue_number': 45}, ...], [...]]`
        * - Functional sites info (metadata)
          - :code:`protein['sites']['functional_sites_info']`
          - :code:`[{'type': 'Cat site'}, {'type': 'BS'}]`
        * - Functional site residues (binary)
          - :code:`protein['residue']['functional_site']`
          - :code:`[0, 0, 1, 0, 1, 0, ...]`
        * - Functional site names (per residue)
          - :code:`protein['residue']['functional_sites']`
          - :code:`['', '', 'Cat site', '', ['Cat site', 'BS'], ...]`
        * - Functional site atoms (binary)
          - :code:`protein['atom']['functional_site']`
          - :code:`[0, 0, 1, 0, 1, 0, ...]`
        * - Functional site residue numbers
          - :code:`protein['protein']['functional_site_residues']`
          - :code:`[{'chain': 'A', 'residue_number': 45}, ...]`
        * - Functional site annotation string
          - :code:`protein['protein']['functional_site_annotation']`
          - :code:`'Cat site:45,67,89; BS:90,91'`

    Parameters
    ----------
    annotation_file: str, optional
        Path to a CSV/TSV file containing PDB IDs in the first column and functional site
        residue annotations in the second column. The file can have a header row.
        The annotation format can be:
        - Comma-separated residue numbers: "45,67,89"
        - Chain:residue format: "A:45,A:67,B:89"
        - Residue ranges: "45-50,67,89"
    delimiter: str, default ','
        Delimiter used in the annotation file (comma for CSV, tab for TSV).
    """

    description = 'Functional Site Residues'

    def __init__(self, annotation_file=None, delimiter=',', **kwargs):
        """
        Args:
            annotation_file: Path to file with PDB IDs and annotations. If None,
                will look for default file in root directory.
            delimiter: Delimiter for the annotation file.
        """
        if annotation_file is None:
            # Default to looking for the file in the root directory
            annotation_file = f'{kwargs.get("root", "data")}/raw/functional_sites.csv'
        
        self.annotation_file = annotation_file
        self.delimiter = delimiter
        self.annotations = {}
        # Initialize with empty query and from_list - we'll set from_list in download()
        super().__init__(query=[], from_list=None, only_single_chain=False, **kwargs)
        
        # Load annotations even if download() wasn't called (e.g., if done.txt exists)
        # This ensures annotations are available for add_protein_attributes()
        if not self.annotations:
            self.load_annotations()

    def get_id_from_filename(self, filename):
        return os.path.basename(filename)[:4].upper()

    def parse_annotation(self, annotation_str):
        """Parse functional site residue annotation string.
        
        Supports multiple formats:
        - Semicolon-separated functional sites: "BS:42,43,45; interface:27,30,31"
        - Comma-separated numbers: "45,67,89"
        - Chain:residue format: "A:45,A:67,B:89"
        - Residue ranges: "45-50,67,89"
        - Domain format: "DomA2:94,168-170 & DomA3:215,250-255"
        - Mixed formats
        
        Parameters
        ----------
        annotation_str: str
            Annotation string to parse. Can contain multiple functional sites separated by semicolons.
            
        Returns
        -------
        list
            List of (chain, residue_number) tuples. If no chain is specified, chain is None.
        """
        if not annotation_str or annotation_str.strip() == '':
            return []
        
        residues = []
        
        # First, split by semicolon to handle multiple functional sites
        # e.g., "Hem BS:42,43,45; AB interface:27,30,31"
        functional_sites = [s.strip() for s in annotation_str.split(';')]
        
        for site_str in functional_sites:
            if not site_str:
                continue
            
            # Extract the residue list part (after the site type, e.g., "BS:", "Cat site:", etc.)
            # Find where the actual residue list starts (after "BS:", "Cat site:", "interface:", etc.)
            residue_part = site_str
            
            # Remove site type prefixes like "BS:", "Cat site:", "interface:", etc.
            # But keep domain info like "DomA2:"
            residue_part = re.sub(r'^[^:]*?(?:BS|site|interface|helix|loop|Antenna)\s*:\s*', '', residue_part, flags=re.IGNORECASE)
            residue_part = re.sub(r'^\s*[A-Za-z\s]+\s*:\s*', '', residue_part)  # Remove any remaining prefix
            
            # Handle domain format: "DomA2:94,168-170 & DomA3:215,250-255"
            if '&' in residue_part:
                # Split by & and process each domain part
                domain_parts = [p.strip() for p in residue_part.split('&')]
                for domain_part in domain_parts:
                    # Extract domain and residues: "DomA2:94,168-170"
                    domain_match = re.match(r'Dom([A-Z0-9]+)\s*:\s*(.+)', domain_part, re.IGNORECASE)
                    if domain_match:
                        domain = domain_match.group(1)
                        domain_residues = domain_match.group(2)
                        # Parse the residues for this domain
                        domain_res_list = self._parse_residue_list(domain_residues, default_chain=None)
                        residues.extend(domain_res_list)
                    else:
                        # No domain, just parse as regular residues
                        residues.extend(self._parse_residue_list(domain_part))
            else:
                # Regular residue list
                residues.extend(self._parse_residue_list(residue_part))
        
        return residues
    
    def parse_annotation_with_sites(self, annotation_str):
        """Parse functional site residue annotation string and track which site each residue belongs to.
        
        Parameters
        ----------
        annotation_str: str
            Annotation string to parse. Can contain multiple functional sites separated by semicolons.
            
        Returns
        -------
        list
            List of (chain, residue_number, site_name) tuples. If no chain is specified, chain is None.
        """
        if not annotation_str or annotation_str.strip() == '':
            return []
        
        residues_with_sites = []
        
        # First, split by semicolon to handle multiple functional sites
        # e.g., "Cat site:DomC1:50,201; GSP BS:DomC1:48-55"
        functional_sites = [s.strip() for s in annotation_str.split(';')]
        
        for site_str in functional_sites:
            if not site_str:
                continue
            
            # Extract site name (everything before the first colon, but handle domain format)
            # Examples: "Cat site:DomC1:50" -> site_name = "Cat site"
            #           "GSP BS:DomC1:48-55" -> site_name = "GSP BS"
            #           "BS:42,43,45" -> site_name = "BS"
            
            # Find the site name - it's everything before the first colon that's not a domain
            # Look for pattern like "SiteName:Dom..." or "SiteName:residues"
            site_name_match = re.match(r'^([^:]+?)(?:\s*:\s*Dom|:)', site_str)
            if site_name_match:
                site_name = site_name_match.group(1).strip()
            else:
                # Try to extract site name before first colon
                colon_idx = site_str.find(':')
                if colon_idx > 0:
                    site_name = site_str[:colon_idx].strip()
                else:
                    # No site name specified, use default
                    site_name = 'functional site'
            
            # Keep the site name as-is (don't remove parts of it)
            site_name = site_name.strip()
            
            # Extract the residue list part
            residue_part = site_str
            
            # Remove site type prefixes like "BS:", "Cat site:", "interface:", etc.
            # But keep domain info like "DomA2:"
            residue_part = re.sub(r'^[^:]*?(?:BS|site|interface|helix|loop|Antenna)\s*:\s*', '', residue_part, flags=re.IGNORECASE)
            residue_part = re.sub(r'^\s*[A-Za-z\s]+\s*:\s*', '', residue_part)  # Remove any remaining prefix
            
            # Handle domain format: "DomA2:94,168-170 & DomA3:215,250-255" or "DomC1:48-55"
            if '&' in residue_part:
                # Split by & and process each domain part
                domain_parts = [p.strip() for p in residue_part.split('&')]
                for domain_part in domain_parts:
                    # Extract domain and residues: "DomA2:94,168-170"
                    domain_match = re.match(r'Dom([A-Z0-9]+)\s*:\s*(.+)', domain_part, re.IGNORECASE)
                    if domain_match:
                        domain = domain_match.group(1)
                        domain_residues = domain_match.group(2)
                        # Parse the residues for this domain
                        domain_res_list = self._parse_residue_list(domain_residues, default_chain=None)
                        residues_with_sites.extend([(chain, resid, site_name) for chain, resid in domain_res_list])
                    else:
                        # No domain, just parse as regular residues
                        res_list = self._parse_residue_list(domain_part)
                        residues_with_sites.extend([(chain, resid, site_name) for chain, resid in res_list])
            else:
                # Check if it's a domain format without &
                domain_match = re.match(r'Dom([A-Z0-9]+)\s*:\s*(.+)', residue_part, re.IGNORECASE)
                if domain_match:
                    domain = domain_match.group(1)
                    domain_residues = domain_match.group(2)
                    # Parse the residues for this domain
                    domain_res_list = self._parse_residue_list(domain_residues, default_chain=None)
                    residues_with_sites.extend([(chain, resid, site_name) for chain, resid in domain_res_list])
                else:
                    # Regular residue list
                    res_list = self._parse_residue_list(residue_part)
                    residues_with_sites.extend([(chain, resid, site_name) for chain, resid in res_list])
        
        return residues_with_sites
    
    def parse_sites(self, annotation_str):
        """Parse functional site annotation string and organize sites by index.
        
        Parameters
        ----------
        annotation_str: str
            Annotation string to parse. Can contain multiple functional sites separated by semicolons.
            
        Returns
        -------
        tuple
            (sites_list, sites_info_list) where:
            - sites_list: List of residue lists, each containing dictionaries with 'chain' and 'residue_number'
            - sites_info_list: List of site metadata dictionaries (type, etc.), indexed by site index
        """
        if not annotation_str or annotation_str.strip() == '':
            return [], []
        
        sites_list = []
        sites_info = []
        
        # Split by semicolon to handle multiple functional sites
        functional_sites = [s.strip() for s in annotation_str.split(';')]
        
        for site_idx, site_str in enumerate(functional_sites):
            if not site_str:
                continue
            
            # Extract site name/type
            site_name_match = re.match(r'^([^:]+?)(?:\s*:\s*Dom|:)', site_str)
            if site_name_match:
                site_name = site_name_match.group(1).strip()
            else:
                colon_idx = site_str.find(':')
                if colon_idx > 0:
                    site_name = site_str[:colon_idx].strip()
                else:
                    site_name = 'functional site'
            
            site_name = site_name.strip()
            
            # Extract the residue list part
            residue_part = site_str
            residue_part = re.sub(r'^[^:]*?(?:BS|site|interface|helix|loop|Antenna)\s*:\s*', '', residue_part, flags=re.IGNORECASE)
            residue_part = re.sub(r'^\s*[A-Za-z\s]+\s*:\s*', '', residue_part)
            
            # Parse residues for this site
            site_residues = []
            
            # Handle domain format
            if '&' in residue_part:
                domain_parts = [p.strip() for p in residue_part.split('&')]
                for domain_part in domain_parts:
                    domain_match = re.match(r'Dom([A-Z0-9]+)\s*:\s*(.+)', domain_part, re.IGNORECASE)
                    if domain_match:
                        domain = domain_match.group(1)
                        domain_residues = domain_match.group(2)
                        domain_res_list = self._parse_residue_list(domain_residues, default_chain=None)
                        site_residues.extend([{'chain': chain if chain else '', 'residue_number': resid} for chain, resid in domain_res_list])
                    else:
                        res_list = self._parse_residue_list(domain_part)
                        site_residues.extend([{'chain': chain if chain else '', 'residue_number': resid} for chain, resid in res_list])
            else:
                domain_match = re.match(r'Dom([A-Z0-9]+)\s*:\s*(.+)', residue_part, re.IGNORECASE)
                if domain_match:
                    domain = domain_match.group(1)
                    domain_residues = domain_match.group(2)
                    domain_res_list = self._parse_residue_list(domain_residues, default_chain=None)
                    site_residues.extend([{'chain': chain if chain else '', 'residue_number': resid} for chain, resid in domain_res_list])
                else:
                    res_list = self._parse_residue_list(residue_part)
                    site_residues.extend([{'chain': chain if chain else '', 'residue_number': resid} for chain, resid in res_list])
            
            if site_residues:
                sites_list.append(site_residues)
                sites_info.append({'type': site_name})
        
        return sites_list, sites_info
    
    def _parse_residue_list(self, residue_str, default_chain=None):
        """Parse a list of residues from a string.
        
        Parameters
        ----------
        residue_str: str
            String containing residues (e.g., "42,43,45" or "A:45,67" or "45-50")
        default_chain: str or None
            Default chain to use if not specified in the residue string.
            
        Returns
        -------
        list
            List of (chain, residue_number) tuples.
        """
        if not residue_str or not residue_str.strip():
            return []
        
        residues = []
        # Split by comma
        parts = [p.strip() for p in residue_str.split(',')]
        
        for part in parts:
            if not part:
                continue
            
            # Check for range format (e.g., "45-50")
            if '-' in part and ':' not in part:
                try:
                    start, end = map(int, part.split('-'))
                    residues.extend([(default_chain, r) for r in range(start, end + 1)])
                except ValueError:
                    warning(f'Could not parse range: {part}', verbosity=self.verbosity)
                continue
            
            # Check for chain:residue format (e.g., "A:45")
            if ':' in part and not part.startswith('Dom'):
                try:
                    chain, residue = part.split(':', 1)
                    chain = chain.strip()
                    residue = residue.strip()
                    
                    # Check if residue is a range
                    if '-' in residue:
                        start, end = map(int, residue.split('-'))
                        residues.extend([(chain, r) for r in range(start, end + 1)])
                    else:
                        residues.append((chain, int(residue)))
                except ValueError:
                    warning(f'Could not parse chain:residue: {part}', verbosity=self.verbosity)
                continue
            
            # Simple residue number
            try:
                residues.append((default_chain, int(part)))
            except ValueError:
                warning(f'Could not parse residue number: {part}', verbosity=self.verbosity)
        
        return residues

    def load_annotations(self):
        """Load annotations from the annotation file."""
        if not os.path.exists(self.annotation_file):
            error(f'Annotation file not found: {self.annotation_file}. Please provide a CSV/TSV file with PDB IDs and functional site residue annotations.', verbosity=self.verbosity)
            return
        
        self.annotations = {}
        import re
        
        with open(self.annotation_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=self.delimiter)
            first_row = next(reader, None)
            
            if not first_row:
                return
            
            # Determine column indices based on header
            header_lower = [col.lower().strip() for col in first_row]
            
            # Try to find PDB ID column and annotation column
            pdb_col_idx = None
            annotation_col_idx = None
            
            # Check for standard format: "Protein (ID)" in first column, annotations in third column
            if 'protein' in header_lower[0] and 'id' in header_lower[0]:
                pdb_col_idx = 0
                # Look for "functional site" column
                for i, col in enumerate(header_lower):
                    if 'functional' in col and 'site' in col:
                        annotation_col_idx = i
                        break
                # If not found, try second or third column
                if annotation_col_idx is None:
                    annotation_col_idx = 2 if len(first_row) > 2 else 1
            else:
                # Fallback: assume first column is PDB ID, second is annotation
                pdb_col_idx = 0
                annotation_col_idx = 1
            
            # Process data rows
            for row in reader:
                if len(row) <= max(pdb_col_idx, annotation_col_idx):
                    continue
                
                # Extract PDB ID from first column
                pdb_field = row[pdb_col_idx].strip()
                
                # Check if it's in format "Protein Name (1abc A)" or just "1ABC"
                pdb_match = re.search(r'\(([a-z0-9]{4})\s+[A-Za-z]\)', pdb_field, re.IGNORECASE)
                if pdb_match:
                    pdb_id = pdb_match.group(1).upper()
                else:
                    # Assume it's just the PDB ID
                    pdb_id = pdb_field.upper()
                    # Remove any non-alphanumeric characters and take first 4 chars
                    pdb_id = re.sub(r'[^A-Z0-9]', '', pdb_id)[:4]
                
                # Get annotation from the annotation column
                annotation = row[annotation_col_idx].strip()
                
                # Remove quotes if present
                annotation = annotation.strip('"').strip("'")
                
                if pdb_id and len(pdb_id) == 4 and annotation:
                    self.annotations[pdb_id] = annotation
            import re
            
            with open(self.annotation_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter=self.delimiter)
                first_row = next(reader, None)
                
                if first_row:
                    # Determine column indices based on header
                    header_lower = [col.lower().strip() for col in first_row]
                    
                    # Try to find PDB ID column and annotation column
                    pdb_col_idx = None
                    annotation_col_idx = None
                    
                    # Check for standard format: "Protein (ID)" in first column, annotations in third column
                    if 'protein' in header_lower[0] and 'id' in header_lower[0]:
                        pdb_col_idx = 0
                        # Look for "functional site" column
                        for i, col in enumerate(header_lower):
                            if 'functional' in col and 'site' in col:
                                annotation_col_idx = i
                                break
                        # If not found, try second or third column
                        if annotation_col_idx is None:
                            annotation_col_idx = 2 if len(first_row) > 2 else 1
                    else:
                        # Fallback: assume first column is PDB ID, second is annotation
                        pdb_col_idx = 0
                        annotation_col_idx = 1
                    
                    # Process data rows
                    for row in reader:
                        if len(row) <= max(pdb_col_idx, annotation_col_idx):
                            continue
                        
                        # Extract PDB ID from first column
                        pdb_field = row[pdb_col_idx].strip()
                        
                        # Check if it's in format "Protein Name (1abc A)" or just "1ABC"
                        pdb_match = re.search(r'\(([a-z0-9]{4})\s+[A-Za-z]\)', pdb_field, re.IGNORECASE)
                        if pdb_match:
                            pdb_id = pdb_match.group(1).upper()
                        else:
                            # Assume it's just the PDB ID
                            pdb_id = pdb_field.upper()
                            # Remove any non-alphanumeric characters and take first 4 chars
                            pdb_id = re.sub(r'[^A-Z0-9]', '', pdb_id)[:4]
                        
                        # Get annotation from the annotation column
                        annotation = row[annotation_col_idx].strip()
                        
                        # Remove quotes if present
                        annotation = annotation.strip('"').strip("'")
                        
                        if pdb_id and len(pdb_id) == 4 and annotation:
                            self.annotations[pdb_id] = annotation

    def download(self):
        """Download PDB files from RCSB for the PDB IDs in the annotation file."""
        # Load annotations first
        self.load_annotations()
        
        if not self.annotations:
            error('No annotations loaded. Please check the annotation file.', verbosity=self.verbosity)
            return
        
        # Get PDB IDs from annotations
        pdb_ids = list(self.annotations.keys())
        
        # Use RCSBDataset's download with from_list
        self.from_list = pdb_ids
        super().download()

    def add_protein_attributes(self, protein):
        """Add functional site residue annotations to the protein dictionary."""
        pdb_id = protein['protein']['ID'].upper()
        
        if pdb_id not in self.annotations:
            # If annotations not loaded yet, try to load them
            if not self.annotations:
                self.load_annotations()
            
            # If still no annotation found, mark all residues as non-functional
            if pdb_id not in self.annotations:
                protein['sites'] = {
                    'functional_sites': [],
                    'functional_sites_info': []
                }
                protein['residue']['functional_site'] = [0] * len(protein['residue']['residue_number'])
                protein['residue']['functional_sites'] = [''] * len(protein['residue']['residue_number'])
                if 'atom' in protein:
                    protein['atom']['functional_site'] = [0] * len(protein['atom']['residue_number'])
                protein['protein']['functional_site_residues'] = []
                protein['protein']['functional_site_annotation'] = ''
                return protein
        
        annotation_str = self.annotations[pdb_id]
        parsed_residues = self.parse_annotation(annotation_str)
        parsed_residues_with_sites = self.parse_annotation_with_sites(annotation_str)
        
        # Parse sites structure
        sites_list, sites_info = self.parse_sites(annotation_str)
        
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
        # sites_info is a list where index corresponds to site index
        protein['sites'] = {
            'functional_sites': matched_sites_list,
            'functional_sites_info': sites_info
        }
        
        # Create binary mask for functional sites at residue level (for backward compatibility)
        functional_site_mask = np.zeros(len(residue_numbers), dtype=int)
        functional_site_residue_list = []
        
        # Dictionary to map residue index to list of site names
        residue_index_to_sites = {}
        
        # Process each annotated residue and find matching residues in the protein
        for chain, residue_num, site_name in parsed_residues_with_sites:
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
            
            # Add site name to all matching residues
            for match_idx in matches:
                if match_idx not in residue_index_to_sites:
                    residue_index_to_sites[match_idx] = []
                if site_name not in residue_index_to_sites[match_idx]:
                    residue_index_to_sites[match_idx].append(site_name)
                functional_site_mask[match_idx] = 1
        
        # Create functional_sites array (for backward compatibility)
        functional_sites_array = []
        for i in range(len(residue_numbers)):
            if i in residue_index_to_sites:
                site_names = residue_index_to_sites[i]
                # Single site -> string, multiple sites -> list
                if len(site_names) == 1:
                    functional_sites_array.append(site_names[0])
                else:
                    functional_sites_array.append(site_names)
            else:
                functional_sites_array.append('')
        
        # Build functional_site_residue_list (unique residues)
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
                    functional_site_residue_list.append({
                        'chain': actual_chain,
                        'residue_number': actual_residue
                    })
        
        # Keep residue-level annotations for backward compatibility
        protein['residue']['functional_site'] = functional_site_mask.tolist()
        protein['residue']['functional_sites'] = functional_sites_array
        protein['protein']['functional_site_residues'] = functional_site_residue_list
        protein['protein']['functional_site_annotation'] = annotation_str
        
        # Also add atom-level annotations
        if 'atom' in protein:
            atom_residue_numbers = np.array(protein['atom']['residue_number'])
            if 'chain_id' in protein['atom']:
                atom_chain_ids = np.array(protein['atom']['chain_id'])
                atom_chain_ids = np.array([c if c and c.strip() else None for c in atom_chain_ids])
            else:
                atom_chain_ids = np.array([None] * len(atom_residue_numbers))
            
            functional_site_atom_mask = np.zeros(len(atom_residue_numbers), dtype=int)
            
            for chain, residue_num in parsed_residues:
                if chain is None or chain == '':
                    matches = np.where(atom_residue_numbers == residue_num)[0]
                else:
                    chain = chain.strip().upper()
                    chain_match = np.array([(c.strip().upper() if c else '') == chain for c in atom_chain_ids])
                    matches = np.where(chain_match & (atom_residue_numbers == residue_num))[0]
                
                if len(matches) > 0:
                    functional_site_atom_mask[matches] = 1
            
            protein['atom']['functional_site'] = functional_site_atom_mask.tolist()
        
        return protein

    def describe(self):
        """Produce dataset statistics."""
        desc = super().describe()
        desc['property'] = "Functional Site Residues"
        desc['type'] = 'Binary (residue-level)'
        return desc
