"""
Example usage of MisatoProteinLigandDataset.
Loads the dataset and prints information about a few proteins.
"""

from proteinshake.datasets import MisatoProteinLigandDataset

print("=" * 70)
print("MisatoProteinLigandDataset Example")
print("=" * 70)

# Create dataset with use_precomputed=True
dataset = MisatoProteinLigandDataset(
    root='roots/misato',
    use_precomputed=True,
    verbosity=2
)

print(f"\n✅ Dataset: {dataset.name}")
print(f"   Description: {dataset.description}")

# Get proteins iterator
proteins = dataset.proteins(resolution='residue')

# Print information about first 5 unique proteins (different PDB IDs)
print("\n" + "=" * 70)
print("First 5 Unique Proteins (different PDB IDs):")
print("=" * 70)

seen_pdb_ids = set()
protein_count = 0

for i, protein in enumerate(proteins):
    pdb_id = protein['protein']['ID']
    
    # Skip if we've already seen this PDB ID
    if pdb_id in seen_pdb_ids:
        continue
    
    # Track this PDB ID and increment count
    seen_pdb_ids.add(pdb_id)
    protein_count += 1
    
    if protein_count > 5:
        break
    
    pdb_id = protein['protein']['ID']
    frame_id = protein['protein'].get('frame_id', 'N/A')
    sequence = protein['protein']['sequence']
    num_residues = len(protein['residue']['residue_number'])
    
    # Binding site info
    binding_site = protein['residue']['binding_site']
    num_binding_site_residues = sum(binding_site)
    binding_site_percentage = (num_binding_site_residues / num_residues * 100) if num_residues > 0 else 0
    
    # Binding affinity info
    kd = protein['protein'].get('kd')
    neglog_aff = protein['protein'].get('neglog_aff')
    resolution = protein['protein'].get('resolution')
    year = protein['protein'].get('year')
    ligand_id = protein['protein'].get('ligand_id')
    
    print(f"\n--- Protein {protein_count} (PDB: {pdb_id}) ---")
    print(f"PDB ID: {pdb_id}")
    print(f"Frame ID: {frame_id}")
    print(f"Sequence: {sequence[:50]}..." if len(sequence) > 50 else f"Sequence: {sequence}")
    print(f"Sequence length: {len(sequence)} residues")
    print(f"Number of residues: {num_residues}")
    print(f"Binding site residues: {num_binding_site_residues} / {num_residues} ({binding_site_percentage:.1f}%)")
    
    if kd is not None:
        print(f"Kd: {kd}")
    if neglog_aff is not None:
        print(f"-log(affinity): {neglog_aff:.2f}")
    if resolution is not None:
        print(f"Resolution: {resolution} Å")
    if year is not None:
        print(f"Year: {year}")
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
        print(f"Binding site residue numbers: {binding_residues[:15]}{'...' if len(binding_residues) > 15 else ''}")
    
    # Show residue types at binding site
    if num_binding_site_residues > 0:
        binding_residue_types = [protein['residue']['residue_type'][j] 
                                for j, is_site in enumerate(binding_site) if is_site]
        print(f"Binding site residue types: {''.join(binding_residue_types[:30])}{'...' if len(binding_residue_types) > 30 else ''}")

print("\n" + "=" * 70)
print("✅ Example completed!")
print("=" * 70)
