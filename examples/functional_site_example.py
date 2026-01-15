"""
Example usage of FunctionalSiteDataset and MCSADataset.

This example shows how to use both datasets separately:
1. FunctionalSiteDataset - for functional site annotations from CSV files
2. MCSADataset - for catalytic site annotations from M-CSA API

The datasets automatically handle downloading and processing when use_precomputed=False.
They check the root directory for existing processed data and use it if available.
"""

from proteinshake.datasets import FunctionalSiteDataset, MCSADataset

# ============================================================================
# Example 1: FunctionalSiteDataset (from CSV file)
# ============================================================================
print("=== Example 1: FunctionalSiteDataset ===")

dataset_functional = FunctionalSiteDataset(
    annotation_file='table_III.csv',  # Path to your CSV file
    root='roots/fsite',  # Root directory for the dataset
    use_precomputed=True,  # Will automatically download and process if needed
    n_jobs=4  # Number of parallel jobs for downloading
)

# Access the proteins (dataset automatically handles download/processing)
proteins_functional = dataset_functional.proteins(resolution='residue')
prot = next(proteins_functional)
print(len(prot['sites']['functional_sites_info']))
print(len(prot['sites']['functional_sites']))

csa = MCSADataset(root='roots/csa', use_precomputed=True).proteins(resolution='residue')
prot = next(csa)
print(len(prot['sites']['csa_info']))
print(len(prot['sites']['csa']))
