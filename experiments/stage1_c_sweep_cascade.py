"""Sweep of Stage 1 regularization inside the current cascade system.

Purpose:
1. Evaluate how C_S1 changes the learned thresholds t_lo / t_hi.
2. Measure how the train/test zone distribution (green/yellow/red) shifts.
3. Quantify the effect on full-system BA, sensitivity, specificity and errors.
4. Keep Stage 2 fixed to the current system (train on all train, C_S2=10).

This is an exploratory outer-seed sweep, not a nested model-selection procedure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import LeaveOneOut, train_test_split
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "alzheimer-classification-app" / "data"
OUT_DIR = BASE_DIR / "docs" / "review_revision_assets"
OUT_DIR.mkdir(parents=True, exist_ok=True)


df_metab = pd.read_excel(DATA_DIR / "residuales_grupos3.xlsx")
df_clin = pd.read_excel(BASE_DIR / "Segundo_Archivo_clean.xlsx")

y = (df_metab["Group"] == "AD").astype(int).values
indices = np.arange(len(y))

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
    df_clin[["ID", "MMSE", "APOE", "Depression", "Cardiovascular disorder", "Age [y]"]],
    on="ID",
)
mmse = df_merged["MMSE"].values.astype(float)
apoe = df_merged["APOE"].map({"e3/e3": 0, "e3/e4": 1, "e4/e4": 2, "e2/e3": 0, "e2/e4": 1}).fillna(0).values
dep = (df_merged["Depression"] == "Si").astype(int).values
cardio = (df_merged["Cardiovascular disorder"] == "Si").astype(int).values
age = df_merged["Age [y]"].values.astype(float)


CW_S1 = "balanced"
C_S2 = 10.0
CW_S2 = {0: 1, 1: 2}
DEFAULT_STAGE1_GRID = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1 C sweep over current cascade")
    parser.add_argument("--n-seeds", type=int, default=100)
    parser.add_argument("--holdout-pct", type=float, default=0.2)
    parser.add_argument("--stage1-grid", type=float, nargs="*", default=DEFAULT_STAGE1_GRID)
    return parser.parse_args()


def winsorize_train(X_data: np.ndarray) -> np.ndarray:
    X_out = X_data.copy()
    for j in range(X_out.shape[1]):
        lo, hi = np.percentile(X_out[:, j], [5, 95])
        X_out[:, j] = np.clip(X_out[:, j], lo, hi)
    return X_out


def find_thresholds(probs: np.ndarray, labels: np.ndarray, max_err_green: int = 0, max_err_red: int = 0):
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


def build_X2(idx: np.ndarray, p_s1_values: np.ndarray) -> np.ndarray:
    return np.column_stack([p_s1_values, mmse[idx], apoe[idx], dep[idx], cardio[idx], age[idx]])


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


def stage1_oof_train_probs(X_train: np.ndarray, y_train: np.ndarray, c_s1: float) -> np.ndarray:
    p_s1_train = np.zeros(len(y_train))
    for tr, te in LeaveOneOut().split(X_train, y_train):
        X_tr_w = winsorize_train(X_train[tr])
        scaler = StandardScaler().fit(X_tr_w)
        lr1 = LogisticRegression(C=c_s1, class_weight=CW_S1, max_iter=2000, random_state=42)
        lr1.fit(scaler.transform(X_tr_w), y_train[tr])
        p_s1_train[te] = lr1.predict_proba(scaler.transform(X_train[te]))[:, 1]
    return p_s1_train


def run_seed(c_s1: float, seed: int, holdout_pct: float) -> dict[str, float] | None:
    train_idx, test_idx = train_test_split(indices, test_size=holdout_pct, stratify=y, random_state=seed)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    p_s1_train_oof = stage1_oof_train_probs(X_train, y_train, c_s1)
    t_lo, t_hi = find_thresholds(p_s1_train_oof, y_train, 0, 0)
    if t_lo is None:
        return None

    green_train = p_s1_train_oof < t_lo
    red_train = p_s1_train_oof > t_hi
    yellow_train = ~green_train & ~red_train

    X_train_w = winsorize_train(X_train)
    scaler = StandardScaler().fit(X_train_w)
    lr1_final = LogisticRegression(C=c_s1, class_weight=CW_S1, max_iter=2000, random_state=42)
    lr1_final.fit(scaler.transform(X_train_w), y_train)
    p_s1_train_full = lr1_final.predict_proba(scaler.transform(X_train_w))[:, 1]
    p_s1_test = lr1_final.predict_proba(scaler.transform(X_test))[:, 1]

    pred_s1_test = (p_s1_test >= 0.5).astype(int)
    stage1_metrics = calc_metrics(y_test, pred_s1_test)

    green_test = p_s1_test < t_lo
    red_test = p_s1_test > t_hi
    yellow_test = ~green_test & ~red_test

    direct_green_err_test = int(((green_test) & (y_test == 1)).sum())
    direct_red_err_test = int(((red_test) & (y_test == 0)).sum())

    X2_train_full = build_X2(train_idx, p_s1_train_full)
    lr2 = LogisticRegression(C=C_S2, class_weight=CW_S2, max_iter=2000, random_state=42)
    lr2.fit(X2_train_full, y_train)

    pred_test = np.full(len(y_test), -1)
    pred_test[green_test] = 0
    pred_test[red_test] = 1
    if yellow_test.sum() > 0:
        X2_test = build_X2(test_idx[yellow_test], p_s1_test[yellow_test])
        pred_test[yellow_test] = (lr2.predict_proba(X2_test)[:, 1] >= 0.5).astype(int)

    metrics = calc_metrics(y_test, pred_test)
    direct_coverage_test = float((green_test.sum() + red_test.sum()) / len(y_test))

    return {
        "C_S1": float(c_s1),
        "seed": float(seed),
        "t_lo": float(t_lo),
        "t_hi": float(t_hi),
        "zone_width": float(t_hi - t_lo),
        "n_green_train": float(green_train.sum()),
        "n_yellow_train": float(yellow_train.sum()),
        "n_red_train": float(red_train.sum()),
        "n_green_test": float(green_test.sum()),
        "n_yellow_test": float(yellow_test.sum()),
        "n_red_test": float(red_test.sum()),
        "direct_coverage_test": direct_coverage_test,
        "direct_green_err_test": float(direct_green_err_test),
        "direct_red_err_test": float(direct_red_err_test),
        "stage1_BA_test": float(stage1_metrics["BA"]),
        "cascade_BA_test": float(metrics["BA"]),
        "cascade_Sens_test": float(metrics["Sens"]),
        "cascade_Spec_test": float(metrics["Spec"]),
        "cascade_Errors_test": float(metrics["Errors"]),
        "cascade_FN_test": float(metrics["FN"]),
        "cascade_FP_test": float(metrics["FP"]),
        "cascade_improvement": float(metrics["BA"] - stage1_metrics["BA"]),
    }


def summarize(df: pd.DataFrame) -> dict[str, object]:
    metrics = [
        "stage1_BA_test",
        "cascade_BA_test",
        "cascade_Sens_test",
        "cascade_Spec_test",
        "cascade_Errors_test",
        "cascade_FN_test",
        "cascade_FP_test",
        "cascade_improvement",
        "t_lo",
        "t_hi",
        "zone_width",
        "n_green_test",
        "n_yellow_test",
        "n_red_test",
        "direct_coverage_test",
        "direct_green_err_test",
        "direct_red_err_test",
    ]
    out: dict[str, object] = {}
    for c_s1, group in df.groupby("C_S1"):
        out[str(c_s1)] = {m: {"mean": float(group[m].mean()), "std": float(group[m].std())} for m in metrics}
        out[str(c_s1)]["n_valid_seeds"] = int(len(group))

    best_c_by_ba = None
    best_c_by_coverage = None
    if len(df) > 0:
        best_c_by_ba = float(df.groupby("C_S1")["cascade_BA_test"].mean().sort_values(ascending=False).index[0])
        best_c_by_coverage = float(
            df.groupby("C_S1")["direct_coverage_test"].mean().sort_values(ascending=False).index[0]
        )

    return {
        "per_C": out,
        "best_C_by_mean_BA": best_c_by_ba,
        "best_C_by_direct_coverage": best_c_by_coverage,
    }


def main() -> None:
    args = parse_args()
    rows = []
    for c_idx, c_s1 in enumerate(args.stage1_grid, start=1):
        print(f"\n[C sweep {c_idx}/{len(args.stage1_grid)}] Evaluando C_S1={c_s1:g}")
        for seed in range(args.n_seeds):
            row = run_seed(c_s1, seed, args.holdout_pct)
            if row is not None:
                rows.append(row)
            if (seed + 1) % 20 == 0 or seed == args.n_seeds - 1:
                print(f"  completadas {seed + 1}/{args.n_seeds} seeds para C_S1={c_s1:g}")

    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "stage1_c_sweep_cascade_results.csv"
    json_path = OUT_DIR / "stage1_c_sweep_cascade_summary.json"
    df.to_csv(csv_path, index=False)

    summary = summarize(df)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 88)
    print("STAGE 1 C SWEEP INSIDE CURRENT CASCADE")
    print("=" * 88)
    for c_s1, group in df.groupby("C_S1"):
        print(f"\nC_S1={c_s1:g}")
        print(f"  BA cascada     = {group['cascade_BA_test'].mean():.3f} ± {group['cascade_BA_test'].std():.3f}")
        print(f"  Sens           = {group['cascade_Sens_test'].mean():.3f} ± {group['cascade_Sens_test'].std():.3f}")
        print(f"  Spec           = {group['cascade_Spec_test'].mean():.3f} ± {group['cascade_Spec_test'].std():.3f}")
        print(
            f"  Errores        = {group['cascade_Errors_test'].mean():.2f} ± {group['cascade_Errors_test'].std():.2f}"
        )
        print(
            f"  Cobertura dir. = {group['direct_coverage_test'].mean():.3f} ± {group['direct_coverage_test'].std():.3f}"
        )
        print(
            f"  Verde/Roja/Amarilla test = {group['n_green_test'].mean():.2f} / {group['n_red_test'].mean():.2f} / {group['n_yellow_test'].mean():.2f}"
        )
        print(f"  t_lo / t_hi    = {group['t_lo'].mean():.3f} / {group['t_hi'].mean():.3f}")
        print(
            f"  Err directos verde / roja = {group['direct_green_err_test'].mean():.2f} / {group['direct_red_err_test'].mean():.2f}"
        )

    print(f"\nMejor C_S1 por BA media: {summary['best_C_by_mean_BA']}")
    print(f"Mejor C_S1 por cobertura directa media: {summary['best_C_by_direct_coverage']}")
    print(f"\nCSV guardado en: {csv_path}")
    print(f"Resumen guardado en: {json_path}")


if __name__ == "__main__":
    main()
