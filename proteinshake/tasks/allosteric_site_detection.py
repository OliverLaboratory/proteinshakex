# -*- coding: utf-8 -*-
"""
Task for allosteric site detection using the AlloBench dataset.
"""
from sklearn import metrics

from proteinshake.datasets import AlloBenchDataset
from proteinshake.tasks import Task


class AllostericSiteDetectionTask(Task):
    """Identify allosteric site residues in a protein structure.

    Given a protein structure, predict which residues belong to an allosteric
    binding site. This is a binary residue-level classification task.

    .. admonition:: Task Summary

        * **Input:** one protein structure
        * **Output:** binary label for each residue (1 = allosteric site, 0 = not)
        * **Evaluation:** Matthews Correlation Coefficient

    """

    DatasetClass = AlloBenchDataset

    type = 'Binary Classification'
    input = 'Residue'
    output = 'Allosteric Site Residues'

    @property
    def num_classes(self):
        return 2

    @property
    def task_in(self):
        return ('residue')

    @property
    def task_type(self):
        return ('residue', 'binary')

    @property
    def task_out(self):
        return ('binary')

    @property
    def target_dim(self):
        return (1)

    def dummy_output(self):
        import random
        return [random.randint(0, 1) for _ in self.test_targets]

    def target(self, protein):
        return protein['residue']['allosteric_site']

    def compute_targets(self):
        self.train_targets = [p for i in self.train_index for p in self.target(self.proteins[i])]
        self.val_targets = [p for i in self.val_index for p in self.target(self.proteins[i])]
        self.test_targets = [p for i in self.test_index for p in self.target(self.proteins[i])]

    @property
    def default_metric(self):
        return 'mcc'

    def evaluate(self, y_true, y_pred):
        return {
            'accuracy': metrics.accuracy_score(y_true, y_pred),
            'mcc': metrics.matthews_corrcoef(y_true, y_pred),
            'f1': metrics.f1_score(y_true, y_pred, zero_division=0),
            'precision': metrics.precision_score(y_true, y_pred, zero_division=0),
            'recall': metrics.recall_score(y_true, y_pred, zero_division=0),
        }


class ActiveSiteDetectionTask(Task):
    """Identify active/catalytic site residues in a protein structure.

    Given a protein structure from AlloBench, predict which residues belong to
    the active (catalytic) site. This is a binary residue-level classification task.

    .. admonition:: Task Summary

        * **Input:** one protein structure
        * **Output:** binary label for each residue (1 = active site, 0 = not)
        * **Evaluation:** Matthews Correlation Coefficient

    """

    DatasetClass = AlloBenchDataset

    type = 'Binary Classification'
    input = 'Residue'
    output = 'Active Site Residues'

    @property
    def num_classes(self):
        return 2

    @property
    def task_in(self):
        return ('residue')

    @property
    def task_type(self):
        return ('residue', 'binary')

    @property
    def task_out(self):
        return ('binary')

    @property
    def target_dim(self):
        return (1)

    def dummy_output(self):
        import random
        return [random.randint(0, 1) for _ in self.test_targets]

    def target(self, protein):
        return protein['residue']['active_site']

    def compute_targets(self):
        self.train_targets = [p for i in self.train_index for p in self.target(self.proteins[i])]
        self.val_targets = [p for i in self.val_index for p in self.target(self.proteins[i])]
        self.test_targets = [p for i in self.test_index for p in self.target(self.proteins[i])]

    @property
    def default_metric(self):
        return 'mcc'

    def evaluate(self, y_true, y_pred):
        return {
            'accuracy': metrics.accuracy_score(y_true, y_pred),
            'mcc': metrics.matthews_corrcoef(y_true, y_pred),
            'f1': metrics.f1_score(y_true, y_pred, zero_division=0),
            'precision': metrics.precision_score(y_true, y_pred, zero_division=0),
            'recall': metrics.recall_score(y_true, y_pred, zero_division=0),
        }
