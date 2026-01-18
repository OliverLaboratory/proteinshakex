"""
Test script for MisatoProteinLigandDataset.
Loads from precomputed and iterates through a few examples.
"""

from proteinshake.datasets import MisatoProteinLigandDataset

print("=" * 60)
print("Testing MisatoProteinLigandDataset")
print("=" * 60)

# Create dataset with use_precomputed=True
dataset = MisatoProteinLigandDataset(
    root='roots/misato',
    use_precomputed=True,
    verbosity=2
)

print(f"\n✅ Dataset created: {dataset.name}")
print(f"   Root: {dataset.root}")

# Get proteins iterator
proteins = dataset.proteins(resolution='residue')

# Iterate through a few examples
print("\n" + "=" * 60)
print("Iterating through first 5 examples:")
print("=" * 60)

for i, protein in enumerate(proteins):
    if i >= 5:
        break
    
    pdb_id = protein['protein']['ID']
    frame_id = protein['protein'].get('frame_id', 'N/A')
    sequence = protein['protein']['sequence']
    num_residues = len(protein['residue']['residue_number'])
    
    # Binding site info
    binding_site = protein['residue']['binding_site']
    num_binding_site_residues = sum(binding_site)
    
    # Binding affinity info
    kd = protein['protein'].get('kd')
    neglog_aff = protein['protein'].get('neglog_aff')
    resolution = protein['protein'].get('resolution')
    ligand_id = protein['protein'].get('ligand_id')
    
    print(f"\n--- Example {i+1} ---")
    print(f"PDB ID: {pdb_id}")
    print(f"Frame ID: {frame_id}")
    print(f"Sequence length: {len(sequence)} residues")
    print(f"Number of residues: {num_residues}")
    print(f"Binding site residues: {num_binding_site_residues} / {num_residues}")
    
    if kd is not None:
        print(f"Kd: {kd}")
    if neglog_aff is not None:
        print(f"-log(affinity): {neglog_aff}")
    if resolution is not None:
        print(f"Resolution: {resolution} Å")
    if ligand_id is not None:
        print(f"Ligand ID: {ligand_id}")
    
    # Show first few coordinates
    if len(protein['residue']['x']) > 0:
        print(f"First residue coordinates: ({protein['residue']['x'][0]:.2f}, "
              f"{protein['residue']['y'][0]:.2f}, {protein['residue']['z'][0]:.2f})")
    
    # Show binding site residues
    if num_binding_site_residues > 0:
        binding_residues = [protein['residue']['residue_number'][j] 
                           for j, is_site in enumerate(binding_site) if is_site]
        print(f"Binding site residue numbers: {binding_residues[:10]}{'...' if len(binding_residues) > 10 else ''}")

print("\n" + "=" * 60)
print("✅ Test completed successfully!")
print("=" * 60)
