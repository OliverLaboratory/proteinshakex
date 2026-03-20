# -*- coding: utf-8 -*-
"""
Task for functional site detection using the AlloBench dataset.

Combines allosteric and active site annotations into a single binary
residue-level classification: is this residue part of *any* functional site?
"""
from sklearn import metrics

from proteinshake.datasets import AlloBenchDataset
from proteinshake.tasks import Task


class AlloBenchFunctionalSiteTask(Task):
    """Identify functional site residues (allosteric or active) in a protein.

    Given a protein structure from AlloBench, predict which residues belong
    to any functional site (union of allosteric and active sites). This is
    useful for evaluating whether unsupervised partitioning methods recover
    biologically meaningful regions.

    .. admonition:: Task Summary

        * **Input:** one protein structure
        * **Output:** binary label per residue (1 = functional site, 0 = not)
        * **Evaluation:** Matthews Correlation Coefficient, F1, Precision, Recall

    """

    DatasetClass = AlloBenchDataset

    type = 'Binary Classification'
    input = 'Residue'
    output = 'Functional Site Residues (Allosteric + Active)'

    @property
    def num_classes(self):
        return 2

    @property
    def task_in(self):
        return 'residue'

    @property
    def task_type(self):
        return ('residue', 'binary')

    @property
    def task_out(self):
        return 'binary'

    @property
    def target_dim(self):
        return 1

    @property
    def num_features(self):
        return 20

    def dummy_output(self):
        import random
        return [random.randint(0, 1) for _ in self.test_targets]

    def target(self, protein):
        return protein['residue']['functional_site']

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
