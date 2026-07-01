"""
Channel catalogs + tolerant name->index mapper.

Concepts
--------
1) CHANNELS_BY_DATASET
   - What channels exist in your *processed dataset files*.
   - Use these to decide/validate dataset-side selection and derive `in_chans`.

2) MONTAGE_BY_DATASET
   - The MODEL montage (names+order) you want the encoder to use per dataset.
   - The Lightning module’s 1x1 Conv adapts from `in_chans` -> `len(montage)`.

Utilities
---------
- make_channel_indices(available, wanted, strict=True):
    robust mapping from header names in files -> indices for slicing.

- DEFAULT_EEGPT58:
    convenient default MODEL montage (your 58-ch pretrain).

Example
-------
from channels import CHANNELS_BY_DATASET, MONTAGE_BY_DATASET, make_channel_indices, DEFAULT_EEGPT58

available = ["Fp1","Fp2","F7","F3","Fz","F4","F8","T7","C3","Cz","C4","T8","P7","P3","Pz","P4","P8","O1","O2"]
wanted_ds   = CHANNELS_BY_DATASET["BCIC2a"]       # dataset side (19)
wanted_model= MONTAGE_BY_DATASET["BCIC2a"]     # model side (19)
idxs = make_channel_indices(available, wanted_ds)   # indices into `available`
"""

from __future__ import annotations
from typing import Iterable, List, Dict

# Your default pretrain montage (58-ch); handy fallback
DEFAULT_EEGPT58 = [
    'FP1','FPZ','FP2',
    'AF7','AF3','AF4','AF8',
    'F7','F5','F3','F1','FZ','F2','F4','F6','F8',
    'FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8',
    'T7','C5','C3','C1','CZ','C2','C4','C6','T8',
    'TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8',
    'P7','P5','P3','P1','PZ','P2','P4','P6','P8',
    'PO7','PO5','PO3','POZ','PO4','PO6','PO8',
    'O1','OZ','O2',
]

# MUST match pretrain/preprocess 19-ch exactly (same names + same order)
DEFAULT_CH_19 = [
    "FP1", "FP2",
    "F3", "F4",
    "C3", "C4",
    "P3", "P4",
    "O1", "O2",
    "F7", "F8",
    "T3", "T4",
    "T5", "T6",
    "FZ", "CZ", "PZ",]

DEFAULT_CH_19_EEGPT = [
    "FP1","FP2","F7","F3","FZ","F4","F8",
    "T7","C3","CZ","C4","T8",
    "P7","P3","PZ","P4","P8",
    "O1","O2",
]

# 16-ch option (for TUEG experiments, etc.) this 16 channel list is not exactly like CBraMod channel list just random 16 channels for extracting channel len to pass into the conv channel adapter
DEFAULT_CH_16: Sequence[str] = (
    "FP1","FP2","F7","F3","FZ","F4","F8","T7","C3","CZ","C4","T8","P7","P3","PZ","P4"
)
# ---------------------------------------------------------------------
# DATASET-SIDE CHANNEL SETS (what your processed data actually contains)
# ---------------------------------------------------------------------

# PhysioP300 (original 62-ch list kept for reference)
PHYSIOP300_62 = [
    'FP1','FPZ','FP2',
    'AF3','AF4',
    'F7','F5','F3','F1','FZ','F2','F4','F6','F8',
    'FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8',
    'T7','C5','C3','C1','CZ','C2','C4','C6','T8',
    'TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8',
    'P7','P5','P3','P1','PZ','P2','P4','P6','P8',
    'PO7','PO3','POZ','PO4','PO8',
    'O1','OZ','O2',
]
# Your processed PhysioP300 uses 58 channels (drop 4). Adjust if needed.
_PHYSIOP300_DROP4 = {'PO7','PO8','AF3','AF4'}
PHYSIOP300_58 = [ch for ch in PHYSIOP300_62 if ch not in _PHYSIOP300_DROP4]

# TODO:BCIC2a different from common 19-ch 10–20?
BCIC2a_22 = [
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22'
]
BCICIV_2A_CHANNELS_EEGPT = [
    "FZ",
    "FC3", "FC1", "FCZ", "FC2", "FC4",
    "C5", "C3", "C1", "CZ", "C2", "C4", "C6",
    "CP3", "CP1", "CPZ", "CP2", "CP4",
    "P1", "PZ", "P2",
    "POZ",
]



BCIC2a_19 = DEFAULT_CH_19
# BCIC2b (3 channels)
BCIC2b_3 = ['C3','CZ','C4']
BCIC2b_MONTAGE = [ 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6' ]  # 7-ch model montage

# Kaggle ERN
# If you work with the full header, keep the 56-ch list here (optional).
KAGGLE_ERN_56 = "Fp1,Fp2,AF7,AF3,AF4,AF8,F7,F5,F3,F1,Fz,F2,F4,F6,F8,FT7,FC5,FC3,FC1,FCz,FC2,FC4,FC6,FT8,T7,C5,C3,C1,Cz,C2,C4,C6,T8,TP7,CP5,CP3,CP1,CPz,CP2,CP4,CP6,TP8,P7,P5,P3,P1,Pz,P2,P4,P6,P8,PO7,POz,PO8,O1,O2".split(",")
KAGGLE_ERN_56 = [x.upper() for x in KAGGLE_ERN_56]
# Common 19-ch subset aligned with 10–20
KAGGLE_ERN_19 = DEFAULT_CH_19

# SleepEDF (pairs; if your tensors hold separate refs, use SLEEPEDF_4)
SLEEPEDF_PAIRS = ['Fpz-Cz','Pz-Oz']
SLEEPEDF_4 = ['FPZ','CZ','PZ','OZ']
SLEEPEDF_EEGPT_DATASET = ['F3', 'F4']
SLEEPEDF_CBRAMOD = [
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22'
]

# TUAB / TUEV typically mapped to 19-ch 10–20 on the processed side
TUAB_19 = DEFAULT_CH_19
TUEV_19 = DEFAULT_CH_19
TUEV_16 = [
    "AF7", "FT7", "TP7", "PO7",
    "AF8", "FT8", "TP8", "PO8",
    "AF3", "FC3", "CP3", "PO3",
    "AF4", "FC4", "CP4", "PO4",
]

TUEV_EEGPT = [
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23'
]

TUAB_EEGPT = [
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23'
]

FACED_32 = [ '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32' ]
# random channel names to have 64 channel names to calculate in_chans
PHYSIOMI_64 = [
    "Fp1","Fpz","Fp2",
    "AF7","AF3","AFz","AF4","AF8",
    "F7","F5","F3","F1","Fz","F2","F4","F6","F8",
    "FT7","FC5","FC3","FC1","FCz","FC2","FC4","FC6","FT8",
    "T7","C5","C3","C1","Cz","C2","C4","C6","T8",
    "TP7","CP5","CP3","CP1","CPz","CP2","CP4","CP6","TP8",
    "P7","P5","P3","P1","Pz","P2","P4","P6","P8",
    "PO7","PO5","PO3","POz","PO4","PO6","PO8",
    "O1","Oz","O2","Iz",
]


# Public registry of *dataset-side* channels
CHANNELS_BY_DATASET: Dict[str, List[str]] = {
    # canonical task keys
    "BCIC2a": BCIC2a_22,
    "BCIC2b": BCIC2b_3,
    "KaggleERN": KAGGLE_ERN_19,   # use 19 here if you slice down during IO
    "PhysioP300": PHYSIOP300_58,
    "SleepEDF_4": SLEEPEDF_4,
    "TUAB": TUAB_19,
    "TUEV": TUEV_19,
    "TUEV_16": TUEV_16,
    "TUEV_EEGPT": TUEV_EEGPT,
    "TUAB_EEGPT": TUAB_EEGPT,
    "FACED": FACED_32,
    # optional alternates / explicit names
    "KaggleERN_56": KAGGLE_ERN_56,
    "KaggleERN_19": KAGGLE_ERN_19,
    "BCIC2a_19": BCIC2a_19,
    "BCIC2b_3": BCIC2b_3,
    "PhysioP300_58": PHYSIOP300_58,
    "PhysioP300_62": PHYSIOP300_62,
    "SleepEDF_pairs": SLEEPEDF_PAIRS,
    "SLEEPEDF": SLEEPEDF_EEGPT_DATASET,
    "PhysioMI": PHYSIOMI_64
}

# ---------------------------------------------------------------------
# MODEL-SIDE MONTAGES (what the encoder expects for channel embeddings)
# ---------------------------------------------------------------------

TUEV_EEGPT_MONTAGE = ['FP1','FPZ', 'FP2',
        'F7', 'F3', 'FZ', 'F4', 'F8',
        'T7', 'C3', 'CZ', 'C4', 'T8',
        'P7', 'P3', 'PZ', 'P4', 'P8',
                'O1', 'O2' ]

TUAB_EEGPT_MONTAGE = [      
               'FP1', 'FP2',
        'F7', 'F3', 'FZ', 'F4', 'F8',
        'T7', 'C3', 'CZ', 'C4', 'T8',
        'P7', 'P3', 'PZ', 'P4', 'P8',
                'O1', 'O2' ]

SLEEPEDF_EEGPT_MONTAGE = ['F3', 'F4', 'C3', 'C4', 'P3','P4', 'FPZ', 'FZ', 'CZ', 'CPZ', 'PZ', 'POZ', 'OZ' ]

BCIC2b_MONTAGE = ['C3','CZ','C4']

# Per-dataset model montage (use-channel names)
MONTAGE_BY_DATASET: Dict[str, List[str]] = {
    "BCIC2a": DEFAULT_CH_19,         # model uses 19-ch
    # "BCIC2a_EEGPT": DEFAULT_CH_19_EEGPT,
    "BCICIV2a_22_CHANNELS_EEGPT": BCICIV_2A_CHANNELS_EEGPT,
    "BCIC2b": BCIC2b_MONTAGE,          # model uses 3-ch
    "KaggleERN": DEFAULT_CH_19,      # common choice: reuse 19-ch
    "KaggleERN_EEGPT": DEFAULT_CH_19_EEGPT,
    "PhysioP300": PHYSIOP300_58, # model uses 58-ch
    "TUAB": TUAB_19,             # 19-ch
    "TUEV": TUEV_19,             # 19-ch
    "TUEV_EEGPT": TUEV_EEGPT_MONTAGE,
    "TUAB_EEGPT": TUAB_EEGPT_MONTAGE,
    "TUEV_16": TUEV_16,
    "SLEEPEDF": DEFAULT_CH_19,
    "SLEEPEDF_EEGPT": SLEEPEDF_EEGPT_MONTAGE,
    "PhysioMI": DEFAULT_CH_19,
    "PhysioMI_EEGPT": DEFAULT_CH_19_EEGPT,
    "FACED": DEFAULT_CH_19,
    "FACED_EEGPT": DEFAULT_CH_19_EEGPT
}



# ---------------------------------------------------------------------
# Robust name normalizer + mapper
# ---------------------------------------------------------------------

# Legacy alias mapping (normalize old names to modern 10–20)
_ALIAS = {
    "t3": "t7", "t4": "t8", "t5": "p7", "t6": "p8",  # legacy -> modern
    # keep canonical lowercases for common midline labels
    "fpz": "fpz", "poz": "poz", "fcz": "fcz", "cpz": "cpz",
}

def _norm(n: str) -> str:
    """
    Normalize a channel label:
      - lowercase
      - strip spaces and dots
      - strip trailing montage suffixes (-ref, -le, -re, -avg, -aver)
      - keep inner dashes (for pair labels) until comparison stage
    """
    s = n.strip().lower().replace(" ", "").replace(".", "")
    for suf in ("-ref", "-le", "-re", "-aver", "-avg"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s

def _alias(n: str) -> str:
    return _ALIAS.get(n, n)

def make_channel_indices(
    available: Iterable[str],
    wanted: Iterable[str],
    strict: bool = True
) -> List[int]:
    """
    Map `wanted` channel names to indices in `available`.

    Tolerant matching:
      • case/whitespace/dot-insensitive
      • strips EDF suffixes (-Ref, -LE, -RE, -AVG, -AVER)
      • legacy aliases (T3/T4/T5/T6 → T7/T8/P7/P8)
      • pair labels 'Fpz-Cz': try exact pair first, else fall back to first electrode ('Fpz').

    Returns indices into `available` in the order of `wanted`.
    Raises KeyError if a wanted channel is missing and `strict=True`.
    """
    avail_raw = list(available)

    # Build normalized lookup table
    avail_norm_to_idx: Dict[str, int] = {}
    for i, a in enumerate(avail_raw):
        na = _norm(a)
        # pair key as-is (e.g., 'fpz-cz') and single-electrode key (dashes removed)
        avail_norm_to_idx.setdefault(na, i)
        avail_norm_to_idx.setdefault(_alias(na.replace("-", "")), i)

    idxs: List[int] = []
    for w in wanted:
        w_norm = _norm(w)

        # 1) exact pair present?
        if "-" in w_norm and w_norm in avail_norm_to_idx:
            idxs.append(avail_norm_to_idx[w_norm])
            continue

        # 2) fallback to first electrode of the pair
        if "-" in w_norm:
            first = _alias(w_norm.split("-")[0]).replace("-", "")
            if first in avail_norm_to_idx:
                idxs.append(avail_norm_to_idx[first])
                continue

        # 3) single-electrode lookup
        key = _alias(w_norm.replace("-", ""))
        if key in avail_norm_to_idx:
            idxs.append(avail_norm_to_idx[key])
            continue

        # 4) defensive case variants
        if key.upper() in avail_norm_to_idx:
            idxs.append(avail_norm_to_idx[key.upper()]); continue
        if key.lower() in avail_norm_to_idx:
            idxs.append(avail_norm_to_idx[key.lower()]); continue

        if strict:
            raise KeyError(f"Channel '{w}' not found in available header: {available}")
        # non-strict: skip

    return idxs

__all__ = [
    "CHANNELS_BY_DATASET",
    "MONTAGE_BY_DATASET",
    "DEFAULT_EEGPT58",
    "make_channel_indices",
]
