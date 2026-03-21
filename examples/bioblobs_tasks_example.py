"""End-to-end example: load ProteinShake tasks, train, and evaluate.

Demonstrates all new task classes:
  1. AFDBEnzymeClassTask — EC classification on 233k AlphaFold structures
  2. AllostericSiteDetectionTask — allosteric site residues (AlloBench)
  3. ActiveSiteDetectionTask — active/catalytic site residues (AlloBench)
  4. AlloBenchFunctionalSiteTask — combined functional sites (AlloBench)

Each task uses ProteinShakeLoader backed by ProteinStore for O(1) random
access and multi-worker DataLoader support.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# 1. AFDB Enzyme Classification (protein-level, 233k structures)
# ---------------------------------------------------------------------------

def train_ec_classifier():
    """Train a simple EC classifier on AFDB predicted structures."""
    from proteinshake.tasks import AFDBEnzymeClassTask

    # Load task — builds ProteinStore on first run, instant after that
    task = AFDBEnzymeClassTask(
        ec_level=2,                    # EC level 3 (~200+ classes)
        root='data/afdb_ec',           # downloads from Zenodo if not present
        use_precomputed=True,
    )

    print(f"AFDB EC Task: {task.size} proteins, {task.num_classes} classes")
    print(f"Token map sample: {dict(list(task.token_map.items())[:5])}")

    # Get loader with a featurizer transform
    def featurize_protein(protein):
        """Convert protein dict to a tensor dict for training."""
        coords = np.array(list(zip(
            protein['residue']['x'],
            protein['residue']['y'],
            protein['residue']['z'],
        )), dtype=np.float32)
        label = task.target(protein)
        return {
            'coords': torch.tensor(coords),
            'label': torch.tensor(label, dtype=torch.long),
            'id': protein['protein']['ID'],
        }

    loader = task.loader(resolution='residue', transform=featurize_protein)

    # Split: use first 80% train, next 10% val, last 10% test
    n = len(loader)
    train_idx = list(range(int(n * 0.8)))
    val_idx = list(range(int(n * 0.8), int(n * 0.9)))
    test_idx = list(range(int(n * 0.9), n))

    train_loader = loader.subset(train_idx)
    val_loader = loader.subset(val_idx)
    test_loader = loader.subset(test_idx)

    print(f"Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")

    # Simple model: mean-pool coordinates → MLP → EC class
    class SimpleECClassifier(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(3, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes),
            )

        def forward(self, coords):
            # Mean pool over residues
            pooled = coords.mean(dim=0, keepdim=True)  # [1, 3]
            return self.mlp(pooled)

    model = SimpleECClassifier(task.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Train for a few steps (demo only)
    model.train()
    for i, sample in enumerate(train_loader):
        if i >= 10:
            break
        logits = model(sample['coords'])
        loss = criterion(logits, sample['label'].unsqueeze(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 5 == 0:
            print(f"  Step {i}: loss={loss.item():.4f}")

    # Evaluate on a few test samples
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for i, sample in enumerate(test_loader):
            if i >= 50:
                break
            logits = model(sample['coords'])
            y_true.append(sample['label'].item())
            y_pred.append(logits.argmax(dim=-1).item())

    result = task.evaluate(y_true, y_pred)
    print(f"  EC Eval (50 samples): {result}")
    return task, model


# ---------------------------------------------------------------------------
# 2. AlloBench Residue-Level Tasks
# ---------------------------------------------------------------------------

def train_allosteric_detector():
    """Train a residue-level allosteric site detector on AlloBench."""
    from proteinshake.tasks import AllostericSiteDetectionTask

    task = AllostericSiteDetectionTask(
        root='data/allobench',
        split='random',
    )

    print(f"\nAllosteric Task: {task.size} proteins")
    print(f"  Train/Val/Test: {len(task.train_index)}/{len(task.val_index)}/{len(task.test_index)}")

    # Access proteins directly
    train_proteins = [task.proteins[i] for i in task.train_index[:10]]

    # Compute stats
    total_residues = 0
    positive_residues = 0
    for p in train_proteins:
        labels = p['residue']['allosteric_site']
        total_residues += len(labels)
        positive_residues += sum(labels)

    print(f"  Train sample: {total_residues} residues, {positive_residues} allosteric ({positive_residues/total_residues*100:.1f}%)")

    # Dummy evaluation
    dummy = task.dummy_output()
    result = task.evaluate(task.test_targets, dummy)
    print(f"  Dummy eval: {result}")
    return task


def train_functional_site_detector():
    """Train a combined functional site detector (allosteric + active)."""
    from proteinshake.tasks import AlloBenchFunctionalSiteTask

    task = AlloBenchFunctionalSiteTask(
        root='data/allobench',
        split='random',
    )

    print(f"\nFunctional Site Task: {task.size} proteins")
    print(f"  Train/Val/Test: {len(task.train_index)}/{len(task.val_index)}/{len(task.test_index)}")

    # Show a sample protein's annotations
    p = task.proteins[0]
    allosteric = sum(p['residue'].get('allosteric_site', []))
    active = sum(p['residue'].get('active_site', []))
    functional = sum(p['residue']['functional_site'])
    n = len(p['residue']['functional_site'])
    print(f"  Sample [{p['protein']['ID']}]: {n} residues")
    print(f"    Allosteric: {allosteric}, Active: {active}, Functional (union): {functional}")

    # Dummy evaluation
    dummy = task.dummy_output()
    result = task.evaluate(task.test_targets, dummy)
    print(f"  Dummy eval: {result}")
    return task


# ---------------------------------------------------------------------------
# 3. Full pipeline with ProteinShakeLoader + DataLoader
# ---------------------------------------------------------------------------

def dataloader_example():
    """Show how to use ProteinShakeLoader with PyTorch DataLoader."""
    from proteinshake.datasets import AFDBEnzymeCommissionDataset
    from proteinshake.tasks import AFDBEnzymeClassTask

    task = AFDBEnzymeClassTask(
        ec_level=0,
        root='data/afdb_ec',
        use_precomputed=True,
    )

    def collate_proteins(batch):
        """Custom collate: pad coordinates to max length in batch."""
        max_len = max(len(p['residue']['x']) for p in batch)
        coords = torch.zeros(len(batch), max_len, 3)
        labels = torch.zeros(len(batch), dtype=torch.long)
        masks = torch.zeros(len(batch), max_len, dtype=torch.bool)

        for i, p in enumerate(batch):
            n = len(p['residue']['x'])
            coords[i, :n, 0] = torch.tensor(p['residue']['x'])
            coords[i, :n, 1] = torch.tensor(p['residue']['y'])
            coords[i, :n, 2] = torch.tensor(p['residue']['z'])
            masks[i, :n] = True
            labels[i] = task.target(p)

        return {'coords': coords, 'mask': masks, 'label': labels}

    # Create DataLoader with multiple workers
    loader = task.loader(resolution='residue')
    train = loader.subset(list(range(1000)))  # first 1000 for demo

    dl = DataLoader(
        train,
        batch_size=16,
        num_workers=2,
        shuffle=True,
        collate_fn=collate_proteins,
    )

    print(f"\nDataLoader example: {len(train)} proteins, batch_size=16, num_workers=2")

    for i, batch in enumerate(dl):
        if i >= 3:
            break
        print(f"  Batch {i}: coords={batch['coords'].shape}, "
              f"labels={batch['label'].tolist()[:5]}...")

    return dl


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 70)
    print("ProteinShake Tasks — End-to-End Example")
    print("=" * 70)

    # EC classification
    print("\n--- AFDB EC Classification ---")
    ec_task, ec_model = train_ec_classifier()

    # AlloBench residue tasks
    print("\n--- AlloBench Residue Tasks ---")
    allo_task = train_allosteric_detector()
    func_task = train_functional_site_detector()

    # DataLoader integration
    print("\n--- DataLoader Integration ---")
    dl = dataloader_example()

    print("\n" + "=" * 70)
    print("Done! All tasks demonstrated successfully.")
    print("=" * 70)
