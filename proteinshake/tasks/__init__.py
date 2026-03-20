from .task import Task
from .dummy import DummyModel
from .enzyme_class import EnzymeClassTask
from .pfam_task import ProteinFamilyTask
from .ligand_affinity import LigandAffinityTask
from .binding_site_detection import BindingSiteDetectionTask
from .structure_similarity import StructureSimilarityTask
from .protein_protein_interface import ProteinProteinInterfaceTask
from .structure_search import StructureSearchTask
from .structural_class import StructuralClassTask
from .gene_ontology import GeneOntologyTask
from .virtual_screen import VirtualScreenTask
from .ensemble_prediction import EnsemblePredictionTask, UNICORNETask, DANCETask, CFOLDTask
from .allosteric_site_detection import AllostericSiteDetectionTask, ActiveSiteDetectionTask
from .afdb_enzyme_class import AFDBEnzymeClassTask
from .functional_site_task import AlloBenchFunctionalSiteTask

classes = ['Task',
           'GeneOntologyTask',
           'EnzymeClassTask',
           'ProteinFamilyTask',
           'LigandAffinityTask',
           'BindingSiteDetectionTask',
           'ProteinProteinInterfaceTask',
           'StructuralClassTask',
           'StructureSimilarityTask',
           'StructureSearchTask',
           'VirtualScreenTask',
           'EnsemblePredictionTask',
           'UNICORNETask',
           'DANCETask',
           'CFOLDTask',
           'AllostericSiteDetectionTask',
           'ActiveSiteDetectionTask',
           'AFDBEnzymeClassTask',
           'AlloBenchFunctionalSiteTask',
           ]

__all__ = classes
