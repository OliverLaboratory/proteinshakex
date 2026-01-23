# -*- coding: utf-8 -*-
"""
MisatoProteinLigandDataset with Enhanced Progress Tracking

This dataset integrates:
- MISATO MD trajectories (CA-only extraction, multi-frame ensembles)
- PDBbind binding metadata
- Binding-site labels from precomputed pockets

It produces an Avro dataset that ProteinShake can consume for graph-based learning.

Behavior:
- If use_precomputed=True: the class uses a baked-in Sandbox Zenodo URL (or env override)
  to download a single local Avro at {root}/{name}.residue.avro and NEVER uses the parent
  class’ precomputed link. No need to pass a URL in main.
"""

import os
import re
import time
import pickle
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import h5py
from tqdm import tqdm
from fastavro import reader

from proteinshake.datasets import Dataset
from proteinshake.utils import extract_tar, download_url, warning, Generator


# --- Dictionaries for residue/atom mappings ---
AA_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H",
    "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "SEC": "U", "PYL": "O",
    "HIE": "H", "HSD": "H", "HSE": "H", "HSP": "H", "MSE": "M", "ASX": "B", "GLX": "Z", "CSO": "C",
    "SEP": "S", "TPO": "T", "PTR": "Y",
}

ATOMIC_NUM_TO_ELEM = {
    1: 'H', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si',
    15: 'P', 16: 'S', 17: 'Cl', 19: 'K', 20: 'Ca', 34: 'Se', 35: 'Br', 53: 'I'
}

EXCLUDED_RESIDUES = {"MOL", "HOH", "WAT", "ACE", "NME", "SO4", "PO4", "NA", "CL", "K",
                     "CA", "MG", "ZN", "F", "BR", "I"}


def _load_mappings(mapdir: Path):
    """
    Load pickled mapping dictionaries:
      - residue_map: atom index → residue name
      - type_map: atom type index → type string
      - name_map: (residue, index, type) → atom name
    """
    res = pickle.load(open(mapdir / 'atoms_residue_map.pickle?download=1', 'rb'))
    typ = pickle.load(open(mapdir / 'atoms_type_map.pickle?download=1', 'rb'))
    nam = pickle.load(open(mapdir / 'atoms_name_map_for_pdb.pickle?download=1', 'rb'))
    return res, typ, nam


def _h5_entries(f: h5py.File, struct: str, frame: int):
    """
    Extract arrays for a given protein structure and frame.
    Returns: (traj, atoms_type, atoms_number, atoms_residue)
    """
    traj = f[f'{struct}/trajectory_coordinates'][frame]
    atoms_type = f[f'{struct}/atoms_type'][:]
    atoms_number = f[f'{struct}/atoms_number'][:]
    atoms_residue = f[f'{struct}/atoms_residue'][:]
    return traj, atoms_type, atoms_number, atoms_residue


def _atom_name(i, atoms_number, residue_atom_index, residue_name, type_string, name_map):
    """Infer atom name from type and residue; fall back to element+index if lookup fails."""
    try:
        if residue_name == 'MOL':
            return ATOMIC_NUM_TO_ELEM.get(atoms_number[i], 'C') + str(residue_atom_index)
        else:
            return name_map[(residue_name, residue_atom_index - 1, type_string)]
    except Exception:
        return ATOMIC_NUM_TO_ELEM.get(atoms_number[i], 'C') + str(residue_atom_index)


def _update_residue_indices(residue_number, i, type_string, atoms_type,
                            atoms_residue, residue_name, residue_atom_index,
                            residue_map, type_map):
    """Handle transitions between residues when iterating atoms."""
    if i < len(atoms_type) - 1:
        next_type = type_map[atoms_type[i + 1]]
        if (type_string[0] == 'O' and next_type[0] == 'N') or residue_map[atoms_residue[i + 1]] == 'MOL':
            if not ((residue_name == 'GLN' and residue_atom_index in [12, 14]) or
                    (residue_name == 'ASN' and residue_atom_index in [9, 11])):
                residue_number += 1
                residue_atom_index = 0
    return residue_number, residue_atom_index


def _count_ca_atoms_frame(coords, atoms_type, atoms_number, atoms_residue,
                          type_map, residue_map, name_map):
    """Extract all Cα atoms for a single frame. Returns list of tuples."""
    ca_atoms = []
    residue_number = 1
    residue_atom_index = 0
    for i in range(len(atoms_type)):
        residue_type_idx = atoms_residue[i]
        residue_name = residue_map[residue_type_idx]
        if residue_name in EXCLUDED_RESIDUES:
            continue
        residue_atom_index += 1
        type_string = type_map[atoms_type[i]]
        atom_name = _atom_name(i, atoms_number, residue_atom_index, residue_name, type_string, name_map)
        if atom_name == 'CA':
            x, y, z = coords[i]
            elem = ATOMIC_NUM_TO_ELEM.get(atoms_number[i], 'C')
            ca_atoms.append((residue_number, residue_name, x, y, z, elem))
        residue_number, residue_atom_index = _update_residue_indices(
            residue_number, i, type_string, atoms_type, atoms_residue,
            residue_name, residue_atom_index, residue_map, type_map
        )
    return ca_atoms


def _export_one_struct(struct: str, md_path: Path, mapdir: Path,
                       out_dir: Path, num_frames: int) -> str:
    """Export one protein’s CA-only trajectory into a stacked PDB file."""
    residue_map, type_map, name_map = _load_mappings(mapdir)
    all_frames_ca = []
    with h5py.File(md_path, 'r', locking=False) as f:
        available = f[f'{struct}/trajectory_coordinates'].shape[0]
        n = min(num_frames, available)
        for frame in range(n):
            coords, atoms_type, atoms_number, atoms_residue = _h5_entries(f, struct, frame)
            ca_atoms = _count_ca_atoms_frame(coords, atoms_type, atoms_number,
                                             atoms_residue, type_map, residue_map, name_map)
            for (resno, resname, x, y, z, elem) in ca_atoms:
                all_frames_ca.append((frame, resno, resname, x, y, z, elem))

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{struct}_CA_stacked_{n}_frames.pdb"
    with path.open('w') as of:
        current_frame = -1
        atom_number = 1
        for (frame, resno, resname, x, y, z, elem) in all_frames_ca:
            if frame != current_frame:
                if current_frame >= 0:
                    of.write("TER\n")
                current_frame = frame
            line = (
                f"ATOM  {atom_number:5d} {'CA':<4} {resname:<4}{resno:5d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {elem:<2}\n"
            )
            of.write(line)
            atom_number += 1
        of.write("END\n")
    return str(path)


def _parse_stacked_ca_pdb_to_frames(pdb_path: Path, max_frames: Optional[int] = None):
    """Parse a stacked PDB back into per-frame residue lists."""
    frames, current = [], []
    with pdb_path.open('r') as f:
        for line in f:
            rec = line[:6].strip().upper()
            if rec in ('ATOM', 'HETATM'):
                atom_name = line[12:16].strip()
                if atom_name != 'CA':
                    continue
                resname = line[17:20].strip()
                try:
                    resseq = int(line[22:26].strip())
                except ValueError:
                    parts = line.split()
                    resseq = next((int(p) for p in parts if p.isdigit()), None)
                    if resseq is None:
                        continue
                try:
                    x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                except ValueError:
                    parts = line.split()
                    floats = [p for p in parts if re.fullmatch(r"[+\-]?\d+(?:\.\d+)?", p)]
                    if len(floats) >= 3:
                        x, y, z = map(float, floats[-3:])
                    else:
                        continue
                current.append((resseq, resname, x, y, z))
            elif rec == 'TER':
                if current:
                    frames.append(current)
                    current = []
            elif rec == 'END':
                if current:
                    frames.append(current)
                    current = []
                break

    if not frames:
        raise ValueError(f"No frames parsed from {pdb_path.name}")

    if max_frames is not None and len(frames) > max_frames:
        frames = frames[:max_frames]

    n0 = len(frames[0])
    for i, fr in enumerate(frames):
        if len(fr) != n0:
            raise ValueError(f"Frame {i} residue count {len(fr)} != {n0}")

    return frames


class MisatoProteinLigandDataset(Dataset):
    """
    ProteinShake-compatible dataset class for:
      - MISATO protein ensembles
      - PDBbind binding affinity labels
      - Binding-site annotations

    IMPORTANT:
    - We BYPASS the parent precomputed mechanism. When self._use_precomputed=True,
      we only use a local Avro (copied or downloaded from PRECOMPUTED_URL / env).
    """

    description = "MISATO CA-only (100 frames) + PDBbind binding metadata + binding_site"

    PRECOMPUTED_URL = "https://sandbox.zenodo.org/records/429595/files/MisatoProteinLigandDataset.residue.avro?download=1"

    # ---- convenience: always coerce to Path when doing path math ----
    @property
    def root_path(self) -> Path:
        return Path(self.root)

    def __init__(self,
                 root: str = "data",
                 mapdir: Optional[str] = None,
                 index_path: Optional[str] = None,
                 index_other_path: Optional[str] = None,
                 pocket_root: Optional[str] = None,
                 num_frames: int = 100,
                 max_pdbs: Optional[int] = None,
                 pdb_out_dir: Optional[str] = None,
                 precomputed_avro: Optional[str] = None,  
                 use_precomputed: bool = False,
                 n_jobs: int = 8,
                 verbosity: int = 2,
                 **kwargs):
        """
        Initialize dataset with paths, configs, and precomputed options.
        NOTE: parent may overwrite self.root -> string; use self.root_path for Path ops.
        """
        # keep parent-compatible root; use self.root_path for Path ops
        self.root = root

        # core configs
        self.num_frames = int(num_frames)
        self._max_pdbs = max_pdbs
        self.precomputed_avro = Path(precomputed_avro) if precomputed_avro else None
        self._use_precomputed = bool(use_precomputed)

        # resolve URL internally (env beats class constant)
        self.precomputed_url = os.getenv("MISATO_PRECOMPUTED_URL") or self.PRECOMPUTED_URL

        # path-like fields derived from root
        self.mapdir = Path(mapdir) if mapdir else self.root_path / "raw" / "files"
        index_source = index_path or index_other_path
        # prefer a local plain-text index (if checked out) before falling back to download
        local_index = self.root_path.parent / "PDBbind_v2020_plain_text_index 2" / "index" / "INDEX_general_PL_data.2020"
        if not index_source and local_index.exists():
            index_source = local_index
        self.index_path = Path(index_source) if index_source else self.root_path / "raw" / "files" / "INDEX_general_PL_data.2020?download=1"
        self.pocket_root = Path(pocket_root) if pocket_root else self.root_path / "raw" / "files" / "pockets"
        self.md_path = self.root_path / "raw" / "files" / "MD.hdf5"
        self.pdb_out_dir = Path(pdb_out_dir) if pdb_out_dir else (self.root_path / "raw" / "updated_residue_data")

        base_pocket_root = self.pocket_root
        base_files_root = base_pocket_root.parent
        self._pocket_roots = []
        for p in [
            base_pocket_root,
            base_files_root,
            base_files_root / "refined-set",
            base_files_root / "v2020-other-PL",
        ]:
            if p not in self._pocket_roots:
                self._pocket_roots.append(p)

        # call base Dataset init with use_precomputed=False to avoid parent fetching its own link
        super().__init__(root=root,
                         use_precomputed=False,
                         n_jobs=n_jobs,
                         verbosity=verbosity,
                         **kwargs)

        # if using precomputed Avro, ensure it exists locally
        if self._use_precomputed:
            self._ensure_precomputed_avro()

    @property
    def name(self):
        """Dataset name used for Avro file naming."""
        return "MisatoProteinLigandDataset"

    @property
    def limit(self):
        """Limit number of PDBs to parse."""
        return self._max_pdbs

    # ---------- PRECOMPUTED HANDLING (LOCAL OR SANDBOX ZENODO) ----------

    def _ensure_precomputed_avro(self):
        """
        Ensure {root}/{name}.residue.avro exists.
        Priority: local file (precomputed_avro) > URL (precomputed_url).
        Robust to Zenodo saving as '...avro?download=1'.
        """
        dst = self.root_path / f"{self.name}.residue.avro"
        if dst.exists():
            return

        dst.parent.mkdir(parents=True, exist_ok=True)

        # 1) Local file provided
        if self.precomputed_avro and self.precomputed_avro.exists():
            shutil.copyfile(self.precomputed_avro, dst)
            return

        # 2) Remote URL baked into the class (or env)
        if not self.precomputed_url:
            raise FileNotFoundError(
                "use_precomputed=True but no PRECOMPUTED_URL set (or MISATO_PRECOMPUTED_URL env)."
            )

        # Download; older proteinshake.utils.download_url may not accept 'filename'
        try:
            download_url(self.precomputed_url, str(dst.parent), verbosity=self.verbosity, filename=dst.name)
        except TypeError:
            download_url(self.precomputed_url, str(dst.parent), verbosity=self.verbosity)

        if dst.exists():
            return

        base = os.path.basename(self.precomputed_url)      
        no_q = base.split('?', 1)[0]                        
        for p in [
            dst.parent / base,
            dst.parent / no_q,
            dst.parent / f"{self.name}.residue.avro?download=1",
        ]:
            if p.exists():
                if p != dst:
                    shutil.move(str(p), str(dst))
                return

        # last resort: pick newest *.avro
        avros = sorted(dst.parent.glob("*.avro"), key=lambda p: p.stat().st_mtime, reverse=True)
        if avros:
            shutil.move(str(avros[0]), str(dst))
            return

        raise FileNotFoundError(f"Downloaded Avro but couldn't place it at {dst}.")

    # -------------------------- LABELS / DOWNLOAD --------------------------

    def affinity_parse(self, s):
        """Parse affinity strings like 'Kd=5.6nM' into structured dict."""
        operator = "".join(re.findall(r"[=|<|>|~]", s))
        measures = ['Kd', 'Ki', 'IC50']
        measure = next((m for m in measures if s.startswith(m)), None)
        value = float(re.search(r"\d+[.,]?\d*", s).group())
        unit = re.search(r"[m|u|n|f|p]M", s).group()
        return {'operator': operator, 'measure': measure, 'value': value, 'unit': unit}

    def parse_pdbbind_PL_index(self, index_path):
        """Parse PDBbind index file into dict of metadata per protein ID."""
        data = {}
        with open(index_path, 'r') as ind_file:
            for line in ind_file:
                if line.startswith("#"):
                    continue
                pre, post = line.split("//")
                pdbid, res, date, neglog, kd = pre.split()
                kd = self.affinity_parse(kd)
                lig_id = post.split("(")[1].rstrip(")")
                if lig_id.endswith('-mer'):
                    continue
                try:
                    resolution = float(res)
                except ValueError:
                    resolution = None
                data[pdbid] = {
                    'resolution': resolution,
                    'date': int(date),
                    'kd': kd,
                    'neglog_aff': float(neglog),
                    'ligand_id': lig_id
                }
        return data

    def download_labels(self):
        """Download PDBbind refined/other sets, general PL index (for affinities), and mappings from Zenodo sandbox."""
        refined_url = 'https://sandbox.zenodo.org/records/429595/files/PDBbind_v2020_refined.tar.gz?download=1'
        other_url = 'https://sandbox.zenodo.org/records/429595/files/PDBbind_v2020_other_PL.tar.gz?download=1'

        def _ensure_tar(tar_path: Path, url: str, min_mb: int):
            if tar_path.exists() and tar_path.stat().st_size >= min_mb * 1024 * 1024:
                return
            download_url(url, str(tar_path.parent))

        refined_tar = self.root_path / 'raw' / 'PDBbind_v2020_refined.tar.gz?download=1'
        other_tar = self.root_path / 'raw' / 'PDBbind_v2020_other_PL.tar.gz?download=1'
        _ensure_tar(refined_tar, refined_url, min_mb=600)
        _ensure_tar(other_tar, other_url, min_mb=200)

        if not self.index_path.exists():
            download_url(
                'https://sandbox.zenodo.org/records/429595/files/INDEX_general_PL_data.2020?download=1',
                str(self.root_path / 'raw' / 'files')
            )
        download_url(
            'https://sandbox.zenodo.org/records/429595/files/atoms_name_map_for_pdb.pickle?download=1',
            str(self.root_path / 'raw' / 'files')
        )
        download_url(
            'https://sandbox.zenodo.org/records/429595/files/atoms_residue_map.pickle?download=1',
            str(self.root_path / 'raw' / 'files')
        )
        download_url(
            'https://sandbox.zenodo.org/records/429595/files/atoms_type_map.pickle?download=1',
            str(self.root_path / 'raw' / 'files')
        )
        def _extract_with_retry(tar_path: Path, url: str):
            try:
                extract_tar(
                    str(tar_path),
                    str(self.root_path / 'raw' / 'files'),
                    extract_members=True,
                    strip=1
                )
            except EOFError:
                warning(f"Corrupted archive detected for {tar_path.name}; re-downloading.")
                if tar_path.exists():
                    tar_path.unlink()
                download_url(url, str(tar_path.parent))
                extract_tar(
                    str(tar_path),
                    str(self.root_path / 'raw' / 'files'),
                    extract_members=True,
                    strip=1
                )

        _extract_with_retry(refined_tar, refined_url)
        _extract_with_retry(other_tar, other_url)

        # general_PL index already contains refined + other entries plus affinities
        self.index_data = self.parse_pdbbind_PL_index(self.index_path)

    def download(self):
        """
        Ensure MD.hdf5 and overlap proteins exist.
        In precomputed mode, just ensure the precomputed Avro is present.
        """
        if self._use_precomputed:
            self._ensure_precomputed_avro()
            print(f"✅ Using precomputed Avro at {self.root_path / f'{self.name}.residue.avro'}")
            return

        if not hasattr(self, "index_data"):
            self.download_labels()
        # prefer a user-provided MD.hdf5 if present
        alt_md_env = os.getenv("MISATO_MD_PATH")
        if not self.md_path.exists() and alt_md_env:
            alt_md = Path(alt_md_env)
            if alt_md.exists():
                self.md_path = alt_md
        if not self.md_path.exists():
            # also check common sibling location (raw/MD.hdf5) before downloading
            alt_md = self.root_path / "raw" / "MD.hdf5"
            if alt_md.exists():
                self.md_path = alt_md
        if not self.md_path.exists():
            self.md_path.parent.mkdir(parents=True, exist_ok=True)
            md_url = "https://zenodo.org/records/7711953/files/MD.hdf5?download=1"
            print(f"⬇️  Downloading MISATO MD.hdf5 → {self.md_path}")
            download_url(md_url, str(self.md_path.parent), verbosity=self.verbosity)
        else:
            print(f"✅ Using existing MD.hdf5: {self.md_path}")

        existing = list(self.pdb_out_dir.glob("*_CA_stacked_*_frames.pdb"))
        if existing:
            print(f"✅ Found {len(existing)} stacked PDB(s) in {self.pdb_out_dir}. Skipping export.")
            return

        idx_ids = {k.lower() for k in self.index_data.keys()}
        print("🔎 Scanning MISATO MD for labeled proteins...")
        with h5py.File(self.md_path, 'r', locking=False) as f:
            available = [pid for pid in sorted(f.keys()) if 'trajectory_coordinates' in f[pid]]

        labeled = [pid for pid in available if pid.lower() in idx_ids]
        if self.limit:
            labeled = labeled[:self.limit]

        print(f"✅ Found {len(labeled)} labeled proteins present in MD.hdf5")
        if not labeled:
            warning("No labeled proteins present in both PDBbind (general_PL) and MISATO MD.")
            return

        print(f"🚀 Exporting CA-only stacked PDBs to: {self.pdb_out_dir}")
        for pid in tqdm(labeled, desc="Export PDBs"):
            try:
                _export_one_struct(pid.upper(), self.md_path, self.mapdir, self.pdb_out_dir, self.num_frames)
            except Exception as e:
                warning(f"Failed exporting {pid}: {e}", verbosity=self.verbosity)

    # ---------------------------- PARSE → AVRO ----------------------------

    def get_raw_files(self) -> List[str]:
        """Return list of available raw PDB files (respecting max_pdbs limit)."""
        files = sorted(str(p) for p in self.pdb_out_dir.glob("*.pdb"))
        return files[:self.limit] if self.limit else files

    def get_id_from_filename(self, filename: str) -> str:
        """Extract PDB ID (first 4 chars of filename stem)."""
        return Path(filename).stem[:4].upper()

    def _load_binding_site_mask(self, pdbid: str, residue_numbers: List[int]) -> List[int]:
        """Load binary mask for binding-site residues based on precomputed pockets."""
        if not self._pocket_roots:
            return [0] * len(residue_numbers)
        pid_low = pdbid.lower()
        candidates = []
        for root in self._pocket_roots:
            candidates.extend([
                root / pid_low / f"{pid_low}_pocket.pdb",
                root / pid_low / f"{pdbid}_pocket.pdb",
                root / pdbid / f"{pid_low}_pocket.pdb",
                root / pdbid / f"{pdbid}_pocket.pdb",
            ])
        pocket_path = next((p for p in candidates if p.exists()), None)
        if pocket_path is None:
            return [0] * len(residue_numbers)
        pocket_df = self.pdb2df(str(pocket_path))
        col_atom = 'atom_type' if 'atom_type' in pocket_df.columns else (
            'atom_name' if 'atom_name' in pocket_df.columns else None
        )
        if col_atom is None or 'residue_number' not in pocket_df.columns:
            return [0] * len(residue_numbers)
        ca_mask = (pocket_df[col_atom].astype(str).str.upper() == 'CA')
        pocket_res = np.array(pocket_df.loc[ca_mask, 'residue_number'].tolist(), dtype=int)
        protein_res = np.array(residue_numbers, dtype=int).reshape(-1, 1)
        is_site_res = np.zeros_like(protein_res)
        if pocket_res.size > 0:
            is_site_res[(pocket_res == protein_res).sum(axis=1).nonzero()] = 1
        return list(map(int, is_site_res.squeeze().tolist()))

    def parse(self):
        """
        Parse raw PDBs into Avro format with schema.
        In precomputed mode, we skip parsing after ensuring Avro exists.
        """
        out_path = self.root_path / f"{self.name}.residue.avro"

        if self._use_precomputed:
            self._ensure_precomputed_avro()
            print(f"✅ Using precomputed Avro (no parse): {out_path}")
            return

        if out_path.exists():
            print(f"✅ Avro already exists: {out_path}")
            return

        # ensure raw pdbs exist
        paths = self.get_raw_files()
        if not paths:
            print("⚠️  No PDB files found, running download()...")
            self.download()
            paths = self.get_raw_files()

        if not paths:
            print("❌ Still no PDB files after download; aborting parse.")
            return

        if not hasattr(self, "index_data"):
            self.index_data = self.parse_pdbbind_PL_index(self.index_path)

        stats = {"proteins": 0, "frames": 0, "errors": 0}
        t0 = time.time()
        print(f"🔄 Starting to parse {len(paths)} PDB files...")

        SCHEMA = {
            "type": "record",
            "name": "ProteinRecord",
            "fields": [
                {"name": "protein", "type": {
                    "type": "record", "name": "ProteinInfo",
                    "fields": [
                        {"name": "ID", "type": "string"},
                        {"name": "sequence", "type": "string"},
                        {"name": "kd", "type": ["null", "double"]},
                        {"name": "neglog_aff", "type": ["null", "double"]},
                        {"name": "resolution", "type": ["null", "double"]},
                        {"name": "year", "type": ["null", "int"]},
                        {"name": "ligand_id", "type": ["null", "string"]}
                    ]
                }},
                {"name": "residue", "type": {
                    "type": "record", "name": "ResidueInfo",
                    "fields": [
                        {"name": "residue_number", "type": {"type": "array", "items": "int"}},
                        {"name": "residue_type", "type": {"type": "array", "items": "string"}},
                        {"name": "x", "type": {"type": "array", "items": {"type": "array", "items": "double"}}},
                        {"name": "y", "type": {"type": "array", "items": {"type": "array", "items": "double"}}},
                        {"name": "z", "type": {"type": "array", "items": {"type": "array", "items": "double"}}},
                        {"name": "binding_site", "type": {"type": "array", "items": "int"}}
                    ]
                }}
            ]
        }

        def record_iter():
            for p in tqdm(paths, desc="Parsing PDBs"):
                try:
                    rec = self.parse_pdb(p)
                except Exception as e:
                    stats["errors"] += 1
                    print(f"❌ Error parsing {os.path.basename(p)}: {e}")
                    continue
                if rec is None:
                    continue
                stats["proteins"] += 1
                frames_in_protein = len(rec['residue']['x'])
                stats["frames"] += frames_in_protein
                if stats["proteins"] % 50 == 0:
                    print(f"  📊 Parsed {stats['proteins']} proteins, {stats['frames']} total frames")
                yield rec

        try:
            from fastavro import writer, parse_schema
            parsed = parse_schema(SCHEMA)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("wb") as f:
                writer(f, parsed, record_iter())
            dt = time.time() - t0
            if stats["proteins"] == 0:
                print("⚠️  No valid proteins parsed; nothing written.")
                if out_path.exists():
                    out_path.unlink()
                return
            print(f"✅ Parsed {stats['proteins']} proteins with {stats['frames']} total frames")
            print(f"🎉 Wrote {stats['proteins']} proteins → {out_path} in {dt:.1f}s")
            print(f"📈 Total frames available: {stats['frames']}")
        except Exception as e_fast:
            raise RuntimeError(f"Failed to write Avro: {e_fast}")

    def parse_pdb(self, path: str) -> Dict[str, Any]:
        """Parse one stacked PDB into a structured dict."""
        pdb_path = Path(path)
        pdbid = self.get_id_from_filename(pdb_path.name)
        frames = _parse_stacked_ca_pdb_to_frames(pdb_path, max_frames=self.num_frames)
        resnos = [r[0] for r in frames[0]]
        resnames3 = [r[1] for r in frames[0]]
        resnames1 = [AA_THREE_TO_ONE.get(x, 'X') for x in resnames3]
        X = [[r[2] for r in f] for f in frames]
        Y = [[r[3] for r in f] for f in frames]
        Z = [[r[4] for r in f] for f in frames]
        protein = {
            'protein': {
                'ID': pdbid,
                'sequence': ''.join(resnames1),
                'kd': None,
                'neglog_aff': None,
                'resolution': None,
                'year': None,
                'ligand_id': None,
            },
            'residue': {
                'residue_number': resnos,
                'residue_type': resnames1,
                'x': X, 'y': Y, 'z': Z,
                'binding_site': self._load_binding_site_mask(pdbid, resnos),
            },
        }
        key = pdbid.lower()
        if hasattr(self, "index_data") and key in self.index_data:
            bind = self.index_data[key]
            protein['protein']['kd'] = float(bind['kd']['value']) if bind['kd'] and 'value' in bind['kd'] else None
            protein['protein']['neglog_aff'] = float(bind['neglog_aff']) if bind.get('neglog_aff') is not None else None
            protein['protein']['resolution'] = float(bind['resolution']) if bind.get('resolution') is not None else None
            protein['protein']['year'] = int(bind['date']) if bind.get('date') is not None else None
            protein['protein']['ligand_id'] = bind.get('ligand_id')
        return protein

    def read_avro(self, path: Path):
        """Use fastavro to iterate over Avro file records."""
        with path.open("rb") as f:
            yield from reader(f)

    def proteins(self, resolution="residue", count_frames: bool = True):
        """
        Expand proteins from Avro into per-frame records.
        In precomputed mode, load our local Avro (no parent Zenodo link).
        """
        out_path = self.root_path / f"{self.name}.{resolution}.avro"
        canonical = self.root_path / f"{self.name}.residue.avro"

        if self._use_precomputed:
            # Ensure local precomputed avro exists (download/copy if needed)
            self._ensure_precomputed_avro()
            base_avro = canonical
        else:
            # Non-precomputed path: ensure we parsed locally
            if not out_path.exists() and not canonical.exists():
                # Trigger a local parse to create the canonical residue avro
                self.parse()
            base_avro = canonical if canonical.exists() else out_path

        if not base_avro.exists():
            raise FileNotFoundError(f"No Avro found at {base_avro}. "
                                    f"{'Set PRECOMPUTED_URL or env MISATO_PRECOMPUTED_URL' if self._use_precomputed else 'Run parse() first.'}")

        total_frames = None
        if count_frames:
            total_proteins = 0
            total_frames = 0
            for prot in self.read_avro(base_avro):
                total_proteins += 1
                total_frames += len(prot['residue']['x'])
            print(f"🔄 Expanding {total_proteins} proteins into {total_frames} frames...")
        else:
            print("🔄 Expanding proteins into frames (counting disabled)...")

        def _iter():
            frame_count = 0
            for prot in self.read_avro(base_avro):
                frames = len(prot['residue']['x'])
                # Ensure binding_site exists (for back-compat Avros)
                if 'binding_site' not in prot['residue']:
                    prot['residue']['binding_site'] = self._load_binding_site_mask(
                        prot['protein']['ID'],
                        prot['residue']['residue_number']
                    )
                for f in range(frames):
                    frame_count += 1
                    if frame_count % 500 == 0:
                        if total_frames:
                            print(f"  📊 Expanded {frame_count}/{total_frames} frames "
                                  f"({frame_count/total_frames*100:.1f}%)")
                        else:
                            print(f"  📊 Expanded {frame_count} frames...")
                    yield {
                        "protein": {
                            **prot["protein"],
                            "frame_id": f
                        },
                        "residue": {
                            "residue_number": prot["residue"]["residue_number"],
                            "residue_type": prot["residue"]["residue_type"],
                            "x": prot["residue"]["x"][f],
                            "y": prot["residue"]["y"][f],
                            "z": prot["residue"]["z"][f],
                            "binding_site": prot["residue"]["binding_site"],
                        },
                    }
            print(f"✅ Finished expanding all {total_frames} frames")

        return Generator(_iter(), length=total_frames)


if __name__ == "__main__":
    ds = MisatoProteinLigandDataset(
        root="data",
        max_pdbs=None,
        use_precomputed=True,   # set False to regenerate locally
        n_jobs=8,
        verbosity=2,
    )
    ds.parse()
    print("Dataset ready.")
