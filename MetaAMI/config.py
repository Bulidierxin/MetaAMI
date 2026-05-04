from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


@dataclass
class MetaAMIConfig:
    """Configuration for the MetaAMI pipeline.

    The default workflow follows the project design:
    preprocessing -> 10 random projections to 40 dimensions -> 8 baseline models
    -> stacking comparison for top8/top6/top4/top4-pairs -> select highest AUC.
    """

    data_path: str | Path
    label_path: str | Path | None = None
    output_dir: str | Path = "metaami_results"

    id_col: str = "hadm_id"
    label_col: str = "label"
    positive_label: int = 1

    rp_dim: int = 40
    n_rp: int = 10
    n_splits: int = 10
    n_repeats: int = 1
    random_state: int = 42
    n_jobs: int = -1

    models: Sequence[str] = field(default_factory=lambda: (
        "lr", "svm", "rf", "xgb", "mlp", "ae_mlp", "ot_mlp", "transformer"
    ))
    meta_model: str = "svm_rbf"  # svm_rbf or logistic
    threshold: float | None = 0.14
    threshold_strategy: str = "fixed"  # fixed, f1, youden

    top_k_groups: Sequence[int] = field(default_factory=lambda: (8, 6, 4))
    compare_top4_pairs: bool = True

    # lightweight neural defaults; increase for final experiments
    epochs: int = 40
    batch_size: int = 128
    patience: int = 8
    learning_rate: float = 1e-3

    def outdir(self) -> Path:
        return Path(self.output_dir)
