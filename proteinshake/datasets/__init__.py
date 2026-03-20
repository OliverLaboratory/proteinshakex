from .dataset import Dataset
from .rcsb import RCSBDataset
from .enzyme_commission import EnzymeCommissionDataset
from .gene_ontology import GeneOntologyDataset
from .protein_protein_interface import ProteinProteinInterfaceDataset
from .protein_ligand_interface import ProteinLigandInterfaceDataset
from .protein_family import ProteinFamilyDataset
from .tm_align import TMAlignDataset
from .alphafold import AlphaFoldDataset
from .scop import SCOPDataset
from .protein_ligand_decoys import ProteinLigandDecoysDataset
from .functional_site import FunctionalSiteDataset
from .mcsa import MCSADataset
from .multi_conf import MultiConfDataset
from .misato_protein_ligand_interface import MisatoProteinLigandDataset
from .unicorne import UNICORNEDataset
from .dance import DANCEDataset
from .cfold import CFOLDDataset
from .allobench import AlloBenchDataset
from .afdb_enzyme_commission import AFDBEnzymeCommissionDataset

__all__ = [
    'Dataset',
    'RCSBDataset',
    'AlphaFoldDataset',
    'GeneOntologyDataset',
    'EnzymeCommissionDataset',
    'ProteinFamilyDataset',
    'ProteinProteinInterfaceDataset',
    'ProteinLigandInterfaceDataset',
    'SCOPDataset',
    'TMAlignDataset',
    'ProteinLigandDecoysDataset',
    'FunctionalSiteDataset',
    'MCSADataset',
    'MisatoProteinLigandDataset',
    'MultiConfDataset',
    'UNICORNEDataset',
    'DANCEDataset',
    'CFOLDDataset',
    'AlloBenchDataset',
    'AFDBEnzymeCommissionDataset',
    ]

classes = __all__
