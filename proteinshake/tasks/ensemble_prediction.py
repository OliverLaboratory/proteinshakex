# -*- coding: utf-8 -*-
"""
Ensemble prediction task.

Given a protein sequence, predict a set of 3D conformations.
Evaluation uses TM-score-based metrics: Coverage Rate (CR),
Discovery Rate (DR), Ensemble Utility (EU), and Earth Mover's
Distance (EMD).
"""
import os
import csv
import json

import numpy as np
from concurrent.futures import ThreadPoolExecutor

from proteinshake.tasks.task import Task
from proteinshake.utils import download_url


# --------------- Constants (from ConformationDiscoveryBenchmark) ---------------
HITS_TM_THRESHOLD = 0.9   # TM threshold for a GT to count as "discovered"
EMD_TM_FLOOR = 0.6        # TM below this is structurally meaningless
N_BOOTSTRAP_K = 50         # random permutations for Hits@K curves
TM_WORKERS = 8             # parallel workers for TM-score computation

# Benchmark preprocessing constants
GT_DEDUP_TM = 0.85        # cluster GT conformations above this TM as duplicates
MIN_GT = 2                 # min GT conformations after dedup to include ensemble
LENGTH_TOLERANCE = 0.50    # max fractional CA-count deviation from ref seq length


# --------------- TM-score computation ---------------

def _compute_tm_pair(ca, cb, seq):
    """Compute symmetric TM-score between two CA coordinate arrays.

    Parameters
    ----------
    ca, cb : np.ndarray
        (N, 3) and (M, 3) CA atom coordinates.
    seq : str
        Reference sequence (padded if needed).

    Returns
    -------
    float
        Symmetric TM-score (average of both normalization directions).
    """
    seq_a = seq[:ca.shape[0]] if len(seq) >= ca.shape[0] else seq + 'A' * (ca.shape[0] - len(seq))
    seq_b = seq[:cb.shape[0]] if len(seq) >= cb.shape[0] else seq + 'A' * (cb.shape[0] - len(seq))

    # Try C++ TMalign first
    try:
        from conformation_benchmark.evaluation.tmalign_exe import tm_align as _tm_align_exe
        tm1_ab, tm2_ab = _tm_align_exe(ca, cb, seq_a, seq_b)
        tm1_ba, tm2_ba = _tm_align_exe(cb, ca, seq_b, seq_a)
        tm_ab = (tm1_ab + tm2_ab) / 2
        tm_ba = (tm1_ba + tm2_ba) / 2
        return float((tm_ab + tm_ba) / 2)
    except (ImportError, RuntimeError, FileNotFoundError):
        pass

    # Fallback to tmtools
    try:
        from tmtools import tm_align
    except ImportError:
        raise ImportError(
            "TM-score computation requires either TMalign C++ executable "
            "or tmtools: pip install tmtools"
        )
    res_ab = tm_align(ca, cb, seq_a, seq_b)
    res_ba = tm_align(cb, ca, seq_b, seq_a)
    tm_ab = (res_ab.tm_norm_chain1 + res_ab.tm_norm_chain2) / 2
    tm_ba = (res_ba.tm_norm_chain1 + res_ba.tm_norm_chain2) / 2
    return float((tm_ab + tm_ba) / 2)


def compute_tm_matrix(conformations_a, conformations_b, sequence, n_workers=TM_WORKERS):
    """Compute pairwise TM-score matrix between two sets of conformations.

    Parameters
    ----------
    conformations_a : list of np.ndarray
        Ground truth CA coordinates, each (N_i, 3).
    conformations_b : list of np.ndarray
        Predicted CA coordinates, each (M_j, 3).
    sequence : str
        Reference protein sequence.
    n_workers : int
        Number of parallel workers.

    Returns
    -------
    np.ndarray
        (len_a, len_b) symmetric TM-score matrix.
    """
    n_a, n_b = len(conformations_a), len(conformations_b)
    mat = np.zeros((n_a, n_b))

    pairs = [(i, j) for i in range(n_a) for j in range(n_b)]

    def _worker(ij):
        i, j = ij
        return i, j, _compute_tm_pair(conformations_a[i], conformations_b[j], sequence)

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for i, j, score in pool.map(_worker, pairs):
            mat[i, j] = score

    return mat


# --------------- Metric functions ---------------

def discovery_rate(sim_xy, threshold=HITS_TM_THRESHOLD):
    """Fraction of GT conformations matched by at least one prediction.

    Parameters
    ----------
    sim_xy : np.ndarray
        (n_gt, n_pred) TM-score matrix.
    threshold : float
        TM threshold for a match.

    Returns
    -------
    float
        Discovery rate in [0, 1].
    """
    best_per_gt = sim_xy.max(axis=1)
    return float((best_per_gt >= threshold).sum() / len(best_per_gt))


def coverage_rate(sim_xy, threshold=HITS_TM_THRESHOLD):
    """Binary: 1.0 if all GT conformations are discovered, else 0.0."""
    return 1.0 if discovery_rate(sim_xy, threshold) >= 1.0 else 0.0


def ensemble_utility(sim_xy):
    """Mean of best TM-score per GT conformation.

    Parameters
    ----------
    sim_xy : np.ndarray
        (n_gt, n_pred) TM-score matrix.

    Returns
    -------
    float
        EU in [0, 1].
    """
    return float(sim_xy.max(axis=1).mean())


def earth_movers_distance(sim_xy, tm_floor=EMD_TM_FLOOR):
    """Earth Mover's Distance between GT and predicted distributions.

    Uses optimal transport on a cost matrix derived from TM-scores.
    TM values below ``tm_floor`` are clamped to 0 (structurally
    meaningless), and the remaining range is rescaled to [0, 1].

    Parameters
    ----------
    sim_xy : np.ndarray
        (n_gt, n_pred) TM-score matrix.
    tm_floor : float
        TM threshold below which similarity is 0.

    Returns
    -------
    float
        EMD value (lower is better).
    """
    import ot
    n_gt, n_pred = sim_xy.shape
    sim_clipped = np.clip((sim_xy - tm_floor) / (1.0 - tm_floor), 0.0, 1.0)
    cost = 1.0 - sim_clipped
    a = np.ones(n_gt) / n_gt
    b = np.ones(n_pred) / n_pred
    return float(ot.emd2(a, b, cost))


def hits_at_k(sim_xy, threshold=HITS_TM_THRESHOLD, n_bootstrap=N_BOOTSTRAP_K):
    """Bootstrap Hits@K curve (recall as a function of k predictions).

    Parameters
    ----------
    sim_xy : np.ndarray
        (n_gt, n_pred) TM-score matrix.
    threshold : float
        TM threshold for a match.
    n_bootstrap : int
        Number of random permutations to average over.

    Returns
    -------
    np.ndarray
        Hits@K curve of length n_pred.
    """
    n_gt, n_pred = sim_xy.shape
    rng = np.random.default_rng(42)
    curves = []
    for _ in range(n_bootstrap):
        perm = rng.permutation(n_pred)
        sim_perm = sim_xy[:, perm]
        cum_max = np.maximum.accumulate(sim_perm, axis=1)
        gt_covered = (cum_max >= threshold).sum(axis=0)
        curves.append(gt_covered / n_gt)
    return np.mean(curves, axis=0)


def precision_at_k(sim_xy, threshold=HITS_TM_THRESHOLD, n_bootstrap=N_BOOTSTRAP_K):
    """Bootstrap Precision@K curve (efficiency as a function of k).

    Parameters
    ----------
    sim_xy : np.ndarray
        (n_gt, n_pred) TM-score matrix.
    threshold : float
        TM threshold for a match.
    n_bootstrap : int
        Number of random permutations.

    Returns
    -------
    np.ndarray
        Precision@K curve of length n_pred.
    """
    n_gt, n_pred = sim_xy.shape
    rng = np.random.default_rng(42)
    curves = []
    for _ in range(n_bootstrap):
        perm = rng.permutation(n_pred)
        sim_perm = sim_xy[:, perm]
        cum_max = np.maximum.accumulate(sim_perm, axis=1)
        gt_covered = (cum_max >= threshold).sum(axis=0)
        curves.append(gt_covered / np.arange(1, n_pred + 1))
    return np.mean(curves, axis=0)


# --------------- Benchmark filtering ---------------

def _filter_by_length(conformations, ref_seq_len, tolerance=LENGTH_TOLERANCE):
    """Filter conformations by CA count relative to reference sequence length.

    Parameters
    ----------
    conformations : list of np.ndarray
        CA coordinate arrays, each (N_i, 3).
    ref_seq_len : int
        Reference sequence length.
    tolerance : float
        Max fractional deviation from ref_seq_len.

    Returns
    -------
    list of int
        Indices of conformations that pass the length filter.
    """
    kept = []
    for i, conf in enumerate(conformations):
        ca_count = conf.shape[0]
        if abs(ca_count - ref_seq_len) <= tolerance * ref_seq_len:
            kept.append(i)
    return kept


def _deduplicate_gt(sim_xx, dedup_tm=GT_DEDUP_TM):
    """Deduplicate GT conformations via hierarchical clustering on TM-scores.

    Clusters conformations at the given TM threshold using complete linkage,
    then selects the medoid (highest avg intra-cluster TM) from each cluster.

    Parameters
    ----------
    sim_xx : np.ndarray
        (n, n) GT self-similarity TM-score matrix.
    dedup_tm : float
        TM threshold above which conformations are considered duplicates.

    Returns
    -------
    list of int
        Indices of representative (medoid) conformations.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    n = sim_xx.shape[0]
    if n <= 1:
        return list(range(n))

    # Convert similarity to distance, ensure valid condensed form
    dist = 1.0 - sim_xx
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0.0, None)
    dist = (dist + dist.T) / 2  # symmetrize
    condensed = squareform(dist, checks=False)

    # Complete linkage clustering
    Z = linkage(condensed, method='complete')
    labels = fcluster(Z, t=1.0 - dedup_tm, criterion='distance')

    # Select medoid from each cluster
    kept = []
    for cluster_id in np.unique(labels):
        members = np.where(labels == cluster_id)[0]
        if len(members) == 1:
            kept.append(int(members[0]))
        else:
            # Medoid: member with highest average TM to other cluster members
            sub = sim_xx[np.ix_(members, members)]
            avg_tm = sub.mean(axis=1)
            medoid = members[avg_tm.argmax()]
            kept.append(int(medoid))

    return sorted(kept)


def benchmark_filter_ensemble(conformations, sequence, skip_dedup=False,
                              precomputed_sim_xx=None,
                              length_tolerance=LENGTH_TOLERANCE,
                              dedup_tm=GT_DEDUP_TM, min_gt=MIN_GT,
                              n_workers=TM_WORKERS):
    """Apply benchmark preprocessing to one ensemble's GT conformations.

    Steps:
    1. Filter by length (CA count within tolerance of ref seq length)
    2. Deduplicate via TM-score clustering (unless skip_dedup=True)
    3. Reject if fewer than min_gt conformations remain

    Parameters
    ----------
    conformations : list of np.ndarray
        GT CA coordinate arrays, each (N_i, 3).
    sequence : str
        Reference protein sequence.
    skip_dedup : bool
        If True, skip deduplication (e.g. for UNICORNE).
    precomputed_sim_xx : np.ndarray or None
        Precomputed GT self-similarity matrix. If None and dedup is
        needed, computes on the fly.
    length_tolerance : float
        Max fractional CA-count deviation from ref seq length.
    dedup_tm : float
        TM threshold for deduplication clustering.
    min_gt : int
        Minimum conformations required after filtering.
    n_workers : int
        Parallel workers for TM computation (only if no precomputed matrix).

    Returns
    -------
    list of int or None
        Indices of conformations that pass all filters, or None if
        the ensemble should be excluded.
    """
    ref_len = len(sequence)

    # Step 1: Length filter
    length_kept = _filter_by_length(conformations, ref_len, length_tolerance)
    if len(length_kept) < min_gt:
        return None

    # Step 2: Deduplication
    if skip_dedup or len(length_kept) <= 1:
        kept = length_kept
    elif precomputed_sim_xx is not None:
        # Subselect the precomputed matrix to length-filtered indices
        sim_xx = precomputed_sim_xx[np.ix_(length_kept, length_kept)]
        dedup_idx = _deduplicate_gt(sim_xx, dedup_tm)
        kept = [length_kept[i] for i in dedup_idx]
    else:
        # No precomputed matrix available — skip dedup, keep length-filtered
        kept = length_kept

    if len(kept) < min_gt:
        return None

    return kept


# --------------- Task class ---------------

class EnsemblePredictionTask(Task):
    """Task for predicting conformational ensembles from sequence.

    Input: protein sequence
    Output: set of predicted 3D structures (CA coordinates)

    Evaluation compares predicted ensembles against ground truth
    conformations using TM-score-based metrics.

    Supports any MultiConfDataset (UNICORNEDataset, DANCEDataset,
    CFOLDDataset, etc.).

    Supports ``split='benchmark'`` which reproduces the
    ConformationDiscoveryBenchmark preprocessing:

    1. Filter conformations by length (CA count within 50% of ref seq)
    2. Deduplicate GT via TM-score clustering at 0.85 (skipped for UNICORNE)
    3. Remove ensembles with < 2 conformations after filtering

    Usage::

        from proteinshake.tasks import UNICORNETask
        task = UNICORNETask(split='benchmark')  # only benchmark ensembles

        from proteinshake.tasks import DANCETask
        task = DANCETask(split='benchmark')

        # Or use split='none' for all ensembles
        task = UNICORNETask(split='none')

    Parameters
    ----------
    DatasetClass : class
        A MultiConfDataset subclass (e.g., UNICORNEDataset).
    root : str
        Data root directory.
    hits_threshold : float
        TM threshold for discovery/coverage metrics.
    emd_floor : float
        TM floor for EMD computation.
    n_workers : int
        Parallel workers for TM-score computation.
    **kwargs
        Passed to the dataset constructor.
    """

    type = 'Ensemble Prediction'
    input = 'sequence'
    output = 'ensemble of 3D structures'
    skip_dedup = False  # override in subclasses (True for UNICORNE)

    def __init__(self,
                 DatasetClass=None,
                 root='data',
                 split='none',
                 hits_threshold=HITS_TM_THRESHOLD,
                 emd_floor=EMD_TM_FLOOR,
                 n_workers=TM_WORKERS,
                 **kwargs):
        if DatasetClass is not None:
            self.DatasetClass = DatasetClass
        self.hits_threshold = hits_threshold
        self.emd_floor = emd_floor
        self.n_workers = n_workers
        # 'benchmark' is handled in compute_index; pass 'none' to base
        self._split_mode = split
        super().__init__(root=root, split='none', **kwargs)
        if split == 'benchmark':
            self.compute_benchmark_index()

    @property
    def task_type(self):
        return ('protein', 'ensemble_prediction')

    def target(self, protein):
        """Return ground truth conformations as list of (N, 3) arrays."""
        n_conf = protein['protein']['num_conformations']
        coords = []
        for c in range(n_conf):
            x = protein['residue']['x'][c]
            y = protein['residue']['y'][c]
            z = protein['residue']['z'][c]
            coords.append(np.array(list(zip(x, y, z)), dtype=np.float64))
        return coords

    def _load_precomputed_tm(self):
        """Download and load precomputed GT-GT TM-score matrices from Zenodo.

        Returns
        -------
        dict
            Mapping from ensemble ID -> (n, n) symmetric TM matrix, or
            empty dict if not available.
        """
        dataset_name = self.DatasetClass.__name__ if self.DatasetClass else self.dataset.name
        npz_name = f'{dataset_name}.tm_xx.npz'
        npz_path = os.path.join(self.root, npz_name)

        if not os.path.exists(npz_path):
            record_id = self.dataset.get_zenodo_record_id()
            url = f'https://zenodo.org/record/{record_id}/files/{npz_name}'
            try:
                download_url(url, self.root, verbosity=2)
            except Exception:
                print(f'Could not download precomputed TM matrices from {url}')
                return {}

        if not os.path.exists(npz_path):
            return {}

        data = np.load(npz_path, allow_pickle=False)
        return {k: data[k] for k in data.files}

    def compute_benchmark_index(self):
        """Filter ensembles using benchmark preprocessing logic.

        Reproduces the ConformationDiscoveryBenchmark filtering:
        1. Length filter (CA count within LENGTH_TOLERANCE of ref seq)
        2. GT deduplication via TM clustering at GT_DEDUP_TM
        3. Remove ensembles with < MIN_GT conformations

        Uses precomputed GT-GT TM matrices from Zenodo when available,
        falls back to on-the-fly computation otherwise.

        Sets ``self.benchmark_index`` (array of passing protein indices)
        and ``self.benchmark_kept_confs`` (dict: protein index -> list of
        kept conformation indices after filtering).
        """
        n_total = len(self.proteins)
        print(f'Computing benchmark split ({n_total} ensembles, '
              f'dedup={"off" if self.skip_dedup else "on"})...')

        # Load precomputed TM matrices
        tm_matrices = self._load_precomputed_tm() if not self.skip_dedup else {}
        if tm_matrices:
            print(f'  Loaded {len(tm_matrices)} precomputed GT-GT TM matrices')

        benchmark_index = []
        benchmark_kept_confs = {}
        n_computed = 0

        for i in range(n_total):
            protein = self.proteins[i]
            conformations = self.target(protein)
            sequence = protein['protein']['sequence']
            eid = protein['protein']['ID']

            # Use precomputed TM matrix if available and shape matches
            precomputed = tm_matrices.get(eid, None)
            if precomputed is not None and precomputed.shape[0] != len(conformations):
                precomputed = None  # shape mismatch, can't use
            if precomputed is None and not self.skip_dedup and len(conformations) > 1:
                n_computed += 1

            kept = benchmark_filter_ensemble(
                conformations, sequence,
                skip_dedup=self.skip_dedup,
                precomputed_sim_xx=precomputed,
                n_workers=self.n_workers,
            )
            if kept is not None:
                benchmark_index.append(i)
                benchmark_kept_confs[i] = kept

            if (i + 1) % 100 == 0 or i == n_total - 1:
                print(f'  [{i+1}/{n_total}] {len(benchmark_index)} passing')

        self.benchmark_index = np.array(benchmark_index)
        self.benchmark_kept_confs = benchmark_kept_confs
        if n_computed > 0:
            print(f'  {n_computed} ensembles had no matching precomputed TM matrix (dedup skipped for those)')
        print(f'Benchmark split: {len(benchmark_index)}/{n_total} ensembles pass')

    def benchmark_target(self, protein_idx):
        """Return only the benchmark-filtered conformations for a protein.

        Parameters
        ----------
        protein_idx : int
            Index into self.proteins (must be in self.benchmark_index).

        Returns
        -------
        list of np.ndarray
            Filtered GT CA coordinate arrays.
        """
        all_confs = self.target(self.proteins[protein_idx])
        kept = self.benchmark_kept_confs[protein_idx]
        return [all_confs[c] for c in kept]

    def evaluate_from_matrix(self, sim_xy):
        """Evaluate from a precomputed (n_gt, n_pred) TM-score matrix.

        Parameters
        ----------
        sim_xy : np.ndarray
            (n_gt, n_pred) pairwise TM-score matrix.

        Returns
        -------
        dict
            Metric name -> value.
        """
        return {
            'DR': discovery_rate(sim_xy, self.hits_threshold),
            'CR': coverage_rate(sim_xy, self.hits_threshold),
            'EU': ensemble_utility(sim_xy),
            'EMD': earth_movers_distance(sim_xy, self.emd_floor),
        }

    def evaluate_ensemble(self, gt_conformations, pred_conformations, sequence):
        """Evaluate predicted ensemble against ground truth.

        Computes the TM-score matrix internally, then evaluates.

        Parameters
        ----------
        gt_conformations : list of np.ndarray
            Ground truth CA coordinates, each (N_i, 3).
        pred_conformations : list of np.ndarray
            Predicted CA coordinates, each (M_j, 3).
        sequence : str
            Protein sequence.

        Returns
        -------
        dict
            Metric name -> value.
        """
        sim_xy = compute_tm_matrix(
            gt_conformations, pred_conformations, sequence,
            n_workers=self.n_workers,
        )
        return self.evaluate_from_matrix(sim_xy)

    def evaluate_ensemble_detailed(self, gt_conformations, pred_conformations, sequence):
        """Evaluate with full detail including curves.

        Parameters
        ----------
        gt_conformations : list of np.ndarray
            Ground truth CA coordinates.
        pred_conformations : list of np.ndarray
            Predicted CA coordinates.
        sequence : str
            Protein sequence.

        Returns
        -------
        dict
            Includes DR, CR, EU, EMD, hits_at_k curve, precision_at_k curve.
        """
        sim_xy = compute_tm_matrix(
            gt_conformations, pred_conformations, sequence,
            n_workers=self.n_workers,
        )
        result = self.evaluate_from_matrix(sim_xy)
        result['hits_at_k'] = hits_at_k(sim_xy, self.hits_threshold)
        result['precision_at_k'] = precision_at_k(sim_xy, self.hits_threshold)
        result['sim_xy'] = sim_xy
        return result

    def evaluate(self, y_true, y_pred):
        """Evaluate a batch of ensemble predictions.

        Parameters
        ----------
        y_true : list of dict
            Each dict has 'conformations' (list of (N,3) arrays) and 'sequence' (str).
        y_pred : list of list of np.ndarray
            Each element is a list of predicted (M,3) coordinate arrays.

        Returns
        -------
        dict
            Aggregated metrics (mean across all ensembles).
        """
        all_metrics = []
        for gt, pred in zip(y_true, y_pred):
            m = self.evaluate_ensemble(gt['conformations'], pred, gt['sequence'])
            all_metrics.append(m)

        if not all_metrics:
            return {}

        keys = all_metrics[0].keys()
        return {k: float(np.mean([m[k] for m in all_metrics])) for k in keys}


# --------------- Dataset-specific task classes ---------------

class UNICORNETask(EnsemblePredictionTask):
    """Ensemble prediction on UNICORNE (same-protein multi-conformation).

    648 proteins with 2-17 experimental conformations each.
    Conformations are sequence-aligned to the UniProt reference.

    With ``split='benchmark'``, skips GT deduplication (UNICORNE is
    pre-filtered) but still applies length filtering and MIN_GT.

    Usage::

        from proteinshake.tasks import UNICORNETask
        task = UNICORNETask(split='benchmark')
    """
    skip_dedup = True  # UNICORNE is pre-filtered, no dedup needed

    def __init__(self, root='data', **kwargs):
        from proteinshake.datasets import UNICORNEDataset
        super().__init__(DatasetClass=UNICORNEDataset, root=root, **kwargs)


class DANCETask(EnsemblePredictionTask):
    """Ensemble prediction on DANCE (structural homolog ensembles).

    4891 ensembles with 2-613 conformations each. Conformations can
    be sequence-divergent structural homologs.

    With ``split='benchmark'``, applies length filtering, GT
    deduplication (TM >= 0.85), and MIN_GT=2.

    Usage::

        from proteinshake.tasks import DANCETask
        task = DANCETask(split='benchmark')
    """
    def __init__(self, root='data', **kwargs):
        from proteinshake.datasets import DANCEDataset
        super().__init__(DatasetClass=DANCEDataset, root=root, **kwargs)


class CFOLDTask(EnsemblePredictionTask):
    """Ensemble prediction on CFOLD (binary conformational pairs).

    242 ensembles with exactly 2 conformations each, representing
    two distinct conformational states.

    With ``split='benchmark'``, applies length filtering, GT
    deduplication (TM >= 0.85), and MIN_GT=2.

    Usage::

        from proteinshake.tasks import CFOLDTask
        task = CFOLDTask(split='benchmark')
    """
    def __init__(self, root='data', **kwargs):
        from proteinshake.datasets import CFOLDDataset
        super().__init__(DatasetClass=CFOLDDataset, root=root, **kwargs)
