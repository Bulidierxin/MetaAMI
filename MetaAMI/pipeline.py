from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from .config import MetaAMIConfig
from .metrics import binary_metrics, choose_threshold, summarize_metrics
from .models import fit_predict_proba, make_meta_model
from .preprocessing import Preprocessor, load_ami_data
from .random_projection import make_random_projection


def _rank_models(baseline_summary: pd.DataFrame) -> list[str]:
    base = baseline_summary[baseline_summary["fold"] != "mean±std"].copy()
    return (
        base.groupby("model", as_index=False)["auc"]
        .mean()
        .sort_values("auc", ascending=False)["model"]
        .tolist()
    )


def _build_groups(ranked_models: list[str], cfg: MetaAMIConfig) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for k in cfg.top_k_groups:
        kk = min(k, len(ranked_models))
        if kk > 0:
            groups[f"top{kk}"] = ranked_models[:kk]
    if cfg.compare_top4_pairs:
        top4 = ranked_models[: min(4, len(ranked_models))]
        for a, b in combinations(top4, 2):
            groups[f"pair_{a}_{b}"] = [a, b]
    return groups


def run_metaami(cfg: MetaAMIConfig) -> dict[str, Path]:
    """Run the complete MetaAMI pipeline and save all outputs.

    Outputs:
    - baseline_oof_predictions.csv: OOF probability for every model and RP replicate.
    - baseline_metrics.csv: per-fold/per-RP baseline metrics with mean±std row.
    - stacking_metrics.csv: top8/top6/top4/top4-pair stacking metrics.
    - stacking_predictions_<best_group>.csv: OOF predictions from the best stacking group.
    - best_result.csv: one-row summary of the best AUC model/group.
    """
    outdir = cfg.outdir()
    outdir.mkdir(parents=True, exist_ok=True)

    X_df, y_ser, ids_ser = load_ami_data(cfg.data_path, cfg.label_path, cfg.id_col, cfg.label_col)
    y = y_ser.astype(int).to_numpy()
    ids = ids_ser.to_numpy()

    cv = RepeatedStratifiedKFold(
        n_splits=cfg.n_splits,
        n_repeats=cfg.n_repeats,
        random_state=cfg.random_state,
    )

    baseline_rows: list[dict] = []
    baseline_pred_df = pd.DataFrame({"hadm_id": ids, "y_true": y})
    fold_ids = np.empty(len(y), dtype=int)

    print("[MetaAMI] Running baseline models on random projections...")
    for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(X_df, y), start=1):
        fold_ids[va_idx] = fold_idx
        X_tr_df, X_va_df = X_df.iloc[tr_idx], X_df.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        prep = Preprocessor().fit(X_tr_df)
        X_tr_scaled = prep.transform(X_tr_df)
        X_va_scaled = prep.transform(X_va_df)

        for rp_idx in range(1, cfg.n_rp + 1):
            seed = cfg.random_state + rp_idx * 1000 + fold_idx
            X_tr_rp, X_va_rp, _ = make_random_projection(
                X_tr_scaled, X_va_scaled, n_components=cfg.rp_dim, seed=seed
            )
            for model_name in cfg.models:
                col = f"{model_name}_rp{rp_idx:02d}_prob"
                try:
                    y_prob = fit_predict_proba(
                        model_name, X_tr_rp, y_tr, X_va_rp,
                        random_state=seed, n_jobs=cfg.n_jobs,
                        epochs=cfg.epochs, batch_size=cfg.batch_size,
                        lr=cfg.learning_rate, patience=cfg.patience,
                    )
                except ImportError as exc:
                    print(f"[WARN] Skipping {model_name}: {exc}")
                    continue

                baseline_pred_df.loc[va_idx, col] = y_prob
                th = choose_threshold(
                    y_va, y_prob,
                    strategy=cfg.threshold_strategy,
                    fixed_threshold=0.14 if cfg.threshold is None else cfg.threshold,
                )
                m = binary_metrics(y_va, y_prob, threshold=th)
                m.update({"fold": fold_idx, "rp": rp_idx, "model": model_name})
                baseline_rows.append(m)
                print(f"  fold={fold_idx:03d} rp={rp_idx:02d} model={model_name:<11} AUC={m['auc']:.3f}")

    baseline_pred_df.insert(0, "fold", fold_ids)
    baseline_metrics = summarize_metrics(baseline_rows)
    baseline_pred_path = outdir / "baseline_oof_predictions.csv"
    baseline_metrics_path = outdir / "baseline_metrics.csv"
    baseline_pred_df.to_csv(baseline_pred_path, index=False)
    baseline_metrics.to_csv(baseline_metrics_path, index=False)

    ranked = _rank_models(baseline_metrics)
    groups = _build_groups(ranked, cfg)

    print("[MetaAMI] Running stacking comparison...")
    stacking_rows: list[dict] = []
    best_auc = -np.inf
    best_group = None
    best_pred = None

    folds = sorted(np.unique(fold_ids))
    for group_name, model_names in groups.items():
        prob_cols = [c for c in baseline_pred_df.columns if any(c.startswith(f"{m}_rp") for m in model_names)]
        if not prob_cols:
            continue
        all_prob = np.zeros(len(y), dtype=float)
        all_pred = np.zeros(len(y), dtype=int)
        all_th = np.zeros(len(y), dtype=float)
        for fold in folds:
            train_mask = fold_ids != fold
            test_mask = fold_ids == fold
            X_train = baseline_pred_df.loc[train_mask, prob_cols].to_numpy(dtype=float)
            X_test = baseline_pred_df.loc[test_mask, prob_cols].to_numpy(dtype=float)
            y_train = y[train_mask]
            y_test = y[test_mask]

            meta = make_meta_model(cfg.meta_model, random_state=cfg.random_state)
            meta.fit(X_train, y_train)
            y_prob_tr = meta.predict_proba(X_train)[:, 1]
            th = choose_threshold(
                y_train, y_prob_tr,
                strategy=cfg.threshold_strategy,
                fixed_threshold=0.14 if cfg.threshold is None else cfg.threshold,
            )
            y_prob = meta.predict_proba(X_test)[:, 1]
            all_prob[test_mask] = y_prob
            all_pred[test_mask] = (y_prob >= th).astype(int)
            all_th[test_mask] = th
            m = binary_metrics(y_test, y_prob, threshold=th)
            m.update({"fold": int(fold), "group": group_name, "models": "+".join(model_names), "n_features": len(prob_cols)})
            stacking_rows.append(m)

        group_auc = binary_metrics(y, all_prob, threshold=float(np.nanmedian(all_th))) ["auc"]
        if group_auc > best_auc:
            best_auc = group_auc
            best_group = group_name
            best_pred = pd.DataFrame({
                "fold": fold_ids,
                "hadm_id": ids,
                "y_true": y,
                "y_prob": all_prob,
                "y_pred": all_pred,
                "t_star": all_th,
                "group": group_name,
                "models": "+".join(model_names),
            })
        print(f"  group={group_name:<25} models={'+'.join(model_names)} OOF_AUC={group_auc:.3f}")

    stacking_metrics = summarize_metrics(stacking_rows)
    stacking_metrics_path = outdir / "stacking_metrics.csv"
    stacking_metrics.to_csv(stacking_metrics_path, index=False)

    best_pred_path = outdir / f"stacking_predictions_{best_group}.csv"
    if best_pred is not None:
        best_pred.to_csv(best_pred_path, index=False)

    # best result: choose by mean fold AUC in stacking_metrics
    st = pd.DataFrame(stacking_rows)
    best_summary = (
        st.groupby(["group", "models", "n_features"], as_index=False)["auc"]
        .mean()
        .sort_values("auc", ascending=False)
        .head(1)
    )
    best_result_path = outdir / "best_result.csv"
    best_summary.to_csv(best_result_path, index=False)

    return {
        "baseline_predictions": baseline_pred_path,
        "baseline_metrics": baseline_metrics_path,
        "stacking_metrics": stacking_metrics_path,
        "best_predictions": best_pred_path,
        "best_result": best_result_path,
    }
