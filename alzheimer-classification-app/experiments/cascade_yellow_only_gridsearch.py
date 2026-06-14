"""Cascade experiment where Stage 2 is trained only on the yellow zone.

Design:
1. External repeated holdout over seeds.
2. Stage 1 probabilities in train are generated with LOO-CV.
3. Thresholds are learned on those out-of-fold probabilities.
4. Stage 2 hyperparameter C is tuned only inside yellow-train via stratified CV.
5. Final Stage 2 is fit only on yellow-train and applied only to yellow-test.
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "alzheimer-classification-app" / "data"

df_metab = pd.read_excel(DATA_DIR / "residuales_grupos3.xlsx")
df_clin = pd.read_excel(BASE_DIR / "Segundo_Archivo_clean.xlsx")

y = (df_metab["Group"] == "AD").astype(int).values
ids = df_metab["ID"].values

FEATURES_S1 = [
    "DOPA",
    "Cer(d18:1/20:0)",
    "lysoPC.a.C18:2",
    "PC.aa.C40:4",
    "DHEAS",
    "Arg",
    "HexCer(d18:1/26:1)",
]

df_metab["DHEAS/lysoPC"] = df_metab["DHEAS"] / (df_metab["lysoPC.a.C18:2"] + 1e-10)
df_metab["DOPA*DHEAS"] = df_metab["DOPA"] * df_metab["DHEAS"]
df_metab["PC*DOPA"] = df_metab["PC.aa.C40:4"] * df_metab["DOPA"]
df_metab["DHEAS/Cer20"] = df_metab["DHEAS"] / (df_metab["Cer(d18:1/20:0)"] + 1e-10)
FEATURES_S1 += ["DHEAS/lysoPC", "DOPA*DHEAS", "PC*DOPA", "DHEAS/Cer20"]

X = df_metab[FEATURES_S1].values

df_merged = df_metab[["ID"]].merge(
    df_clin[["ID", "MMSE", "APOE", "Depression", "Cardiovascular disorder", "Age [y]"]], on="ID"
)
mmse = df_merged["MMSE"].values.astype(float)
apoe = df_merged["APOE"].map({"e3/e3": 0, "e3/e4": 1, "e4/e4": 2, "e2/e3": 0, "e2/e4": 1}).fillna(0).values
dep = (df_merged["Depression"] == "Si").astype(int).values
cardio = (df_merged["Cardiovascular disorder"] == "Si").astype(int).values
age = df_merged["Age [y]"].values.astype(float)

C_S1 = 0.1
CW_S1 = "balanced"
CW_S2 = {0: 1, 1: 2}


def winsorize_train(X_data: np.ndarray) -> np.ndarray:
    X_out = X_data.copy()
    for j in range(X_out.shape[1]):
        lo, hi = np.percentile(X_out[:, j], [5, 95])
        X_out[:, j] = np.clip(X_out[:, j], lo, hi)
    return X_out


def find_thresholds(
    probs: np.ndarray,
    labels: np.ndarray,
    max_err_green: int = 0,
    max_err_red: int = 0,
) -> tuple[float | None, float | None]:
    probs_sorted = np.sort(np.unique(probs))
    candidates = np.concatenate([[0.0], probs_sorted, [1.0]])
    best_t_lo, best_t_hi, best_coverage = None, None, -1

    for t_lo in candidates:
        err_green = ((probs < t_lo) & (labels == 1)).sum()
        if err_green > max_err_green:
            continue
        for t_hi in candidates:
            if t_hi <= t_lo:
                continue
            err_red = ((probs > t_hi) & (labels == 0)).sum()
            if err_red > max_err_red:
                continue
            coverage = ((probs < t_lo) | (probs > t_hi)).sum()
            if coverage > best_coverage:
                best_coverage = coverage
                best_t_lo, best_t_hi = t_lo, t_hi
    return best_t_lo, best_t_hi


def build_X2(indices: np.ndarray, p_s1_values: np.ndarray) -> np.ndarray:
    return np.column_stack([p_s1_values, mmse[indices], apoe[indices], dep[indices], cardio[indices], age[indices]])


def stage1_oof_train_probs(X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    p_s1_train = np.zeros(len(y_train))
    loo = LeaveOneOut()
    for tr, te in loo.split(X_train, y_train):
        X_tr_w = winsorize_train(X_train[tr])
        sc = StandardScaler().fit(X_tr_w)
        lr1 = LogisticRegression(C=C_S1, class_weight=CW_S1, max_iter=2000, random_state=42)
        lr1.fit(sc.transform(X_tr_w), y_train[tr])
        p_s1_train[te] = lr1.predict_proba(sc.transform(X_train[te]))[:, 1]
    return p_s1_train


def choose_stage2_cv_splits(y_yellow: np.ndarray, max_splits: int = 5) -> int:
    minority = int(min((y_yellow == 0).sum(), (y_yellow == 1).sum()))
    if minority < 2:
        return 0
    return min(max_splits, minority)


def tune_c_yellow_only(
    train_indices: np.ndarray,
    y_train: np.ndarray,
    yellow_mask: np.ndarray,
    p_s1_train_oof: np.ndarray,
    c_grid: np.ndarray,
    class_weight: dict[int, int],
    random_state: int = 42,
) -> tuple[float | None, list[dict[str, float]], int]:
    yellow_indices = train_indices[yellow_mask]
    y_yellow = y_train[yellow_mask]
    n_splits = choose_stage2_cv_splits(y_yellow)
    if n_splits == 0:
        return None, [], 0

    X2_yellow = build_X2(yellow_indices, p_s1_train_oof[yellow_mask])
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_rows: list[dict[str, float]] = []

    for c_value in c_grid:
        fold_bas = []
        fold_errs = []
        fold_sens = []
        fold_specs = []

        for tr, va in cv.split(X2_yellow, y_yellow):
            lr2 = LogisticRegression(C=float(c_value), class_weight=class_weight, max_iter=2000, random_state=42)
            lr2.fit(X2_yellow[tr], y_yellow[tr])
            pred = (lr2.predict_proba(X2_yellow[va])[:, 1] >= 0.5).astype(int)

            tp = ((pred == 1) & (y_yellow[va] == 1)).sum()
            tn = ((pred == 0) & (y_yellow[va] == 0)).sum()
            fp = ((pred == 1) & (y_yellow[va] == 0)).sum()
            fn = ((pred == 0) & (y_yellow[va] == 1)).sum()
            sens = tp / max(tp + fn, 1)
            spec = tn / max(tn + fp, 1)

            fold_bas.append((sens + spec) / 2)
            fold_errs.append(fn + fp)
            fold_sens.append(sens)
            fold_specs.append(spec)

        cv_rows.append(
            {
                "C": float(c_value),
                "mean_BA": float(np.mean(fold_bas)),
                "std_BA": float(np.std(fold_bas)),
                "mean_errors": float(np.mean(fold_errs)),
                "mean_sens": float(np.mean(fold_sens)),
                "mean_spec": float(np.mean(fold_specs)),
            }
        )

    best_row = sorted(cv_rows, key=lambda row: (-row["mean_BA"], row["mean_errors"], row["C"]))[0]
    return float(best_row["C"]), cv_rows, n_splits


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    return {
        "BA": (sens + spec) / 2,
        "Sens": sens,
        "Spec": spec,
        "FN": float(fn),
        "FP": float(fp),
        "Errors": float(fn + fp),
    }


def run_cascade_yellow_only(seed: int, c_grid: np.ndarray, holdout_pct: float = 0.2) -> dict[str, float] | None:
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(indices, test_size=holdout_pct, stratify=y, random_state=seed)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    p_s1_train_oof = stage1_oof_train_probs(X_train, y_train)
    t_lo, t_hi = find_thresholds(p_s1_train_oof, y_train, max_err_green=0, max_err_red=0)
    if t_lo is None:
        return None

    green_train = p_s1_train_oof < t_lo
    red_train = p_s1_train_oof > t_hi
    yellow_train = ~green_train & ~red_train

    chosen_c, cv_rows, n_splits = tune_c_yellow_only(
        train_idx,
        y_train,
        yellow_train,
        p_s1_train_oof,
        c_grid,
        CW_S2,
    )
    if chosen_c is None:
        return None

    # Fit final Stage 1 on full train
    X_train_w = winsorize_train(X_train)
    sc = StandardScaler().fit(X_train_w)
    lr1_final = LogisticRegression(C=C_S1, class_weight=CW_S1, max_iter=2000, random_state=42)
    lr1_final.fit(sc.transform(X_train_w), y_train)

    p_s1_train_full = lr1_final.predict_proba(sc.transform(X_train_w))[:, 1]
    p_s1_test = lr1_final.predict_proba(sc.transform(X_test))[:, 1]

    pred_s1_test = (p_s1_test >= 0.5).astype(int)
    ba_s1_test = balanced_accuracy_score(y_test, pred_s1_test)

    green_test = p_s1_test < t_lo
    red_test = p_s1_test > t_hi
    yellow_test = ~green_test & ~red_test

    # Fit Stage 2 only on yellow train, using final Stage 1 probabilities restricted to yellow-train
    X2_train_yellow = build_X2(train_idx[yellow_train], p_s1_train_full[yellow_train])
    y2_train_yellow = y_train[yellow_train]
    lr2_final = LogisticRegression(C=chosen_c, class_weight=CW_S2, max_iter=2000, random_state=42)
    lr2_final.fit(X2_train_yellow, y2_train_yellow)

    pred_test = np.full(len(y_test), -1)
    pred_test[green_test] = 0
    pred_test[red_test] = 1
    p_s2_test = np.full(len(y_test), np.nan)
    if yellow_test.sum() > 0:
        X2_test_yellow = build_X2(test_idx[yellow_test], p_s1_test[yellow_test])
        p_s2_test[yellow_test] = lr2_final.predict_proba(X2_test_yellow)[:, 1]
        pred_test[yellow_test] = (p_s2_test[yellow_test] >= 0.5).astype(int)

    test_metrics = calc_metrics(y_test, pred_test)

    return {
        "seed": float(seed),
        "chosen_C": float(chosen_c),
        "cv_splits": float(n_splits),
        "t_lo": float(t_lo),
        "t_hi": float(t_hi),
        "n_yellow_train": float(yellow_train.sum()),
        "n_yellow_test": float(yellow_test.sum()),
        "n_yellow_train_nc": float((y_train[yellow_train] == 0).sum()),
        "n_yellow_train_ad": float((y_train[yellow_train] == 1).sum()),
        "ba_s1_test": float(ba_s1_test),
        "test_BA": float(test_metrics["BA"]),
        "test_Sens": float(test_metrics["Sens"]),
        "test_Spec": float(test_metrics["Spec"]),
        "test_Errors": float(test_metrics["Errors"]),
        "test_FN": float(test_metrics["FN"]),
        "test_FP": float(test_metrics["FP"]),
        "cascade_improvement": float(test_metrics["BA"] - ba_s1_test),
        "cv_grid": cv_rows,
    }


def summarize(results: list[dict[str, float]], c_grid: np.ndarray, n_requested_seeds: int) -> dict[str, object]:
    numeric_df = pd.DataFrame([{k: v for k, v in row.items() if k != "cv_grid"} for row in results])
    c_counts = Counter(float(row["chosen_C"]) for row in results)

    per_c = []
    for c_value in c_grid:
        seed_rows = []
        for row in results:
            for cv_row in row["cv_grid"]:
                if abs(cv_row["C"] - float(c_value)) < 1e-12:
                    seed_rows.append(cv_row)
                    break
        if seed_rows:
            per_c.append(
                {
                    "C": float(c_value),
                    "mean_inner_BA": float(np.mean([row["mean_BA"] for row in seed_rows])),
                    "mean_inner_errors": float(np.mean([row["mean_errors"] for row in seed_rows])),
                }
            )

    return {
        "n_valid_seeds": int(len(results)),
        "n_requested_seeds": int(n_requested_seeds),
        "summary": {
            "test_BA_mean": float(numeric_df["test_BA"].mean()),
            "test_BA_std": float(numeric_df["test_BA"].std()),
            "test_Sens_mean": float(numeric_df["test_Sens"].mean()),
            "test_Spec_mean": float(numeric_df["test_Spec"].mean()),
            "test_Errors_mean": float(numeric_df["test_Errors"].mean()),
            "test_FN_mean": float(numeric_df["test_FN"].mean()),
            "test_FP_mean": float(numeric_df["test_FP"].mean()),
            "stage1_BA_mean": float(numeric_df["ba_s1_test"].mean()),
            "improvement_mean": float(numeric_df["cascade_improvement"].mean()),
            "yellow_train_mean": float(numeric_df["n_yellow_train"].mean()),
            "yellow_test_mean": float(numeric_df["n_yellow_test"].mean()),
            "t_lo_mean": float(numeric_df["t_lo"].mean()),
            "t_hi_mean": float(numeric_df["t_hi"].mean()),
        },
        "chosen_C_distribution": {str(key): int(val) for key, val in sorted(c_counts.items())},
        "grid_coverage": per_c,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Yellow-only Stage 2 cascade experiment")
    parser.add_argument("--n-seeds", type=int, default=100)
    parser.add_argument("--holdout-pct", type=float, default=0.2)
    parser.add_argument("--grid-min-exp", type=float, default=-3.0)
    parser.add_argument("--grid-max-exp", type=float, default=3.0)
    parser.add_argument("--grid-points", type=int, default=17)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    c_grid = np.logspace(args.grid_min_exp, args.grid_max_exp, args.grid_points)

    print("=" * 80)
    print("  CASCADE YELLOW-ONLY EXPERIMENT")
    print("=" * 80)
    print(f"Seeds externas: {args.n_seeds}")
    print(f"Holdout externo: {args.holdout_pct:.2f}")
    print(f"Grid C Stage 2: {args.grid_points} puntos en [1e{args.grid_min_exp:.0f}, 1e{args.grid_max_exp:.0f}]")
    print("Stage 2 entrenado SOLO con amarilla-train")
    print()

    results = []
    for seed in range(args.n_seeds):
        row = run_cascade_yellow_only(seed=seed, c_grid=c_grid, holdout_pct=args.holdout_pct)
        if row is not None:
            results.append(row)
        if (seed + 1) % 10 == 0 or seed == args.n_seeds - 1:
            print(f"  Completadas {seed + 1}/{args.n_seeds} seeds...")

    summary = summarize(results, c_grid, args.n_seeds)
    print()
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nResultados guardados en: {args.output_json}")


if __name__ == "__main__":
    main()
