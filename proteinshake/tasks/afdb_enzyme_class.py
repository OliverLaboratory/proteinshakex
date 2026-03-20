# -*- coding: utf-8 -*-
"""
Task for enzyme class prediction using AlphaFold DB predicted structures.
"""
from sklearn import metrics
from functools import cached_property
import numpy as np

from proteinshake.datasets import AFDBEnzymeCommissionDataset
from proteinshake.tasks import Task


class AFDBEnzymeClassTask(Task):
    """Predict the enzyme commission class from AlphaFold DB predicted structures.

    Uses the AFDBEnzymeCommissionDataset (~233k Swiss-Prot reviewed enzymes)
    for training with significantly more data than the PDB-based
    EnzymeClassTask (~15k).

    The EC hierarchy has 4 levels. Set ``ec_level`` to choose which level
    to predict (0-indexed):
      - Level 0: ~7 classes (reaction type)
      - Level 1: ~70 classes (substrate class)
      - Level 2: ~200+ classes (substrate specificity)
      - Level 3: ~1000+ classes (specific enzyme)

    This is a protein-level multi-class prediction.

    .. admonition:: Task Summary

        * **Input:** one protein (AlphaFold predicted structure)
        * **Output:** enzyme class label
        * **Evaluation:** Accuracy, Precision, Recall

    Parameters
    ----------
    ec_level : int, default 0
        EC hierarchy level (0-indexed). 0 = most general, 3 = most specific.
    """

    DatasetClass = AFDBEnzymeCommissionDataset

    type = 'Multiclass Classification'
    input = 'Protein'
    output = 'Enzyme Commission (AFDB)'

    def __init__(self, ec_level=0, *args, **kwargs):
        self.ec_level = ec_level
        super().__init__(*args, **kwargs)

    @cached_property
    def token_map(self):
        labels = set()
        for p in self.proteins:
            ec = p['protein']['EC']
            parts = ec.split('.')
            if len(parts) > self.ec_level:
                labels.add(parts[self.ec_level])
        return {label: i for i, label in enumerate(sorted(labels))}

    def dummy_output(self):
        import random
        tokens = list(self.token_map.values())
        return [random.choice(tokens) for _ in range(len(self.test_targets))]

    @property
    def num_classes(self):
        return len(self.token_map)

    @property
    def task_type(self):
        return ('protein', 'multi_class')

    @property
    def task_in(self):
        return 'protein'

    @property
    def task_out(self):
        return 'multi_class'

    @property
    def target_dim(self):
        return len(self.token_map)

    @property
    def num_features(self):
        return 20

    def target(self, protein):
        ec = protein['protein']['EC']
        parts = ec.split('.')
        if len(parts) > self.ec_level:
            label = parts[self.ec_level]
            if label in self.token_map:
                return self.token_map[label]
        return -1  # unknown

    @property
    def default_metric(self):
        return 'accuracy'

    def evaluate(self, y_true, y_pred):
        y_true = np.array(y_true, dtype=int)
        y_pred = np.array(y_pred, dtype=int)
        return {
            'precision': metrics.precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall': metrics.recall_score(y_true, y_pred, average='macro', zero_division=0),
            'accuracy': metrics.accuracy_score(y_true, y_pred),
        }
