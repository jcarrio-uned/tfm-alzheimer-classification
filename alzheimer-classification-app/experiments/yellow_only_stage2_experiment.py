"""Comparación entre la cascada actual y una cascada estricta con Stage 2 entrenado solo en amarillo.

Objetivo:
1. Mantener el mismo Stage 1 y el mismo aprendizaje de umbrales.
2. Comparar dos variantes de Stage 2 sobre los mismos outer splits:
   - baseline actual: Stage 2 entrenado con todo el train, C fijo=10
   - yellow-only: Stage 2 entrenado solo con amarilla_train,
                  con C seleccionado por GridSearchCV dentro de amarilla_train
3. Resumir BA, sensibilidad, especificidad, errores y distribución de C óptimos.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, LeaveOneOut, RepeatedStratifiedKFold, train_test_split
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


C_S1 = 0.1
CW_S1 = "balanced"
CW_S2 = {0: 1, 1: 2}
C_GRID = np.logspace(-2, 2, 9)
N_SEEDS = 100


def winsorize_train(X_data: np.ndarray) -> np.ndarray:
    X_out = X_data.copy()
    for j in range(X_out.shape[1]):
        lo, hi = np.percentile(X_out[:, j], [5, 95])
        X_out[:, j] = np.clip(X_out[:, j], lo, hi)
    return X_out


def find_thresholds(P_s1_arr: np.ndarray, y_arr: np.ndarray, max_err_green: int = 0, max_err_red: int = 0):
    probs_sorted = np.sort(np.unique(P_s1_arr))
    candidates = np.concatenate([[0.0], probs_sorted, [1.0]])
    best_t_lo, best_t_hi, best_coverage = None, None, -1

    for t_lo in candidates:
        err_green = ((P_s1_arr < t_lo) & (y_arr == 1)).sum()
        if err_green > max_err_green:
            continue
        for t_hi in candidates:
            if t_hi <= t_lo:
                continue
            err_red = ((P_s1_arr > t_hi) & (y_arr == 0)).sum()
            if err_red > max_err_red:
                continue
            coverage = ((P_s1_arr < t_lo) | (P_s1_arr > t_hi)).sum()
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


def get_stage1_train_oof(X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    p_s1_train = np.zeros(len(y_train))
    for tr, te in LeaveOneOut().split(X_train, y_train):
        X_tr_w = winsorize_train(X_train[tr])
        scaler = StandardScaler().fit(X_tr_w)
        lr1 = LogisticRegression(C=C_S1, class_weight=CW_S1, max_iter=2000, random_state=42)
        lr1.fit(scaler.transform(X_tr_w), y_train[tr])
        p_s1_train[te] = lr1.predict_proba(scaler.transform(X_train[te]))[:, 1]
    return p_s1_train


def fit_stage1_final(X_train: np.ndarray, y_train: np.ndarray):
    X_train_w = winsorize_train(X_train)
    scaler = StandardScaler().fit(X_train_w)
    lr1 = LogisticRegression(C=C_S1, class_weight=CW_S1, max_iter=2000, random_state=42)
    lr1.fit(scaler.transform(X_train_w), y_train)
    return scaler, lr1, X_train_w


def choose_c_yellow_only(
    train_idx: np.ndarray, yellow_mask: np.ndarray, p_s1_train_oof: np.ndarray, y_train: np.ndarray
):
    yellow_idx_local = np.where(yellow_mask)[0]
    y_yellow = y_train[yellow_idx_local]

    if len(y_yellow) < 4:
        return None, {"reason": "too_few_yellow"}

    cls_counts = np.bincount(y_yellow, minlength=2)
    minority = int(cls_counts.min())
    if minority < 2:
        return None, {"reason": "single_class_yellow", "counts": cls_counts.tolist()}

    n_splits = min(5, minority)
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=3, random_state=42)

    X2_yellow = build_X2(train_idx[yellow_idx_local], p_s1_train_oof[yellow_idx_local])
    lr = LogisticRegression(class_weight=CW_S2, max_iter=2000, random_state=42)
    gs = GridSearchCV(
        estimator=lr,
        param_grid={"C": C_GRID},
        scoring="balanced_accuracy",
        cv=cv,
        refit=True,
        n_jobs=-1,
    )
    gs.fit(X2_yellow, y_yellow)

    return float(gs.best_params_["C"]), {
        "best_score": float(gs.best_score_),
        "n_splits": n_splits,
        "minority": minority,
        "yellow_n": int(len(y_yellow)),
        "counts": cls_counts.tolist(),
    }


def run_seed(seed: int):
    train_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=y, random_state=seed)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Stage 1 OOF in train -> thresholds
    p_s1_train_oof = get_stage1_train_oof(X_train, y_train)
    t_lo, t_hi = find_thresholds(p_s1_train_oof, y_train, 0, 0)
    if t_lo is None:
        return None

    yellow_train = (p_s1_train_oof >= t_lo) & (p_s1_train_oof <= t_hi)
    green_train = p_s1_train_oof < t_lo
    red_train = p_s1_train_oof > t_hi

    scaler, lr1_final, X_train_w = fit_stage1_final(X_train, y_train)
    p_s1_train_full = lr1_final.predict_proba(scaler.transform(X_train_w))[:, 1]
    p_s1_test = lr1_final.predict_proba(scaler.transform(X_test))[:, 1]

    green_test = p_s1_test < t_lo
    red_test = p_s1_test > t_hi
    yellow_test = ~green_test & ~red_test

    # Baseline current Stage 2: train on all train, use only on yellow test.
    X2_train_full = build_X2(train_idx, p_s1_train_full)
    lr2_current = LogisticRegression(C=10.0, class_weight=CW_S2, max_iter=2000, random_state=42)
    lr2_current.fit(X2_train_full, y_train)

    pred_current = np.full(len(y_test), -1)
    pred_current[green_test] = 0
    pred_current[red_test] = 1
    if yellow_test.sum() > 0:
        X2_test_yellow = build_X2(test_idx[yellow_test], p_s1_test[yellow_test])
        pred_current[yellow_test] = (lr2_current.predict_proba(X2_test_yellow)[:, 1] >= 0.5).astype(int)

    metrics_current = calc_metrics(y_test, pred_current)

    # Yellow-only Stage 2 with inner C selection on yellow_train.
    c_best, c_info = choose_c_yellow_only(train_idx, yellow_train, p_s1_train_oof, y_train)
    if c_best is None:
        return {
            "seed": seed,
            "valid": False,
            "reason": c_info,
            "n_yellow_train": int(yellow_train.sum()),
            "n_yellow_test": int(yellow_test.sum()),
        }

    yellow_train_idx = np.where(yellow_train)[0]
    X2_yellow_train = build_X2(train_idx[yellow_train_idx], p_s1_train_oof[yellow_train_idx])
    y_yellow_train = y_train[yellow_train_idx]
    lr2_yellow = LogisticRegression(C=c_best, class_weight=CW_S2, max_iter=2000, random_state=42)
    lr2_yellow.fit(X2_yellow_train, y_yellow_train)

    pred_yellow = np.full(len(y_test), -1)
    pred_yellow[green_test] = 0
    pred_yellow[red_test] = 1
    if yellow_test.sum() > 0:
        X2_test_yellow = build_X2(test_idx[yellow_test], p_s1_test[yellow_test])
        pred_yellow[yellow_test] = (lr2_yellow.predict_proba(X2_test_yellow)[:, 1] >= 0.5).astype(int)

    metrics_yellow = calc_metrics(y_test, pred_yellow)

    cls_counts_yellow = np.bincount(y_train[yellow_train], minlength=2)

    return {
        "seed": seed,
        "valid": True,
        "t_lo": float(t_lo),
        "t_hi": float(t_hi),
        "n_green_train": int(green_train.sum()),
        "n_red_train": int(red_train.sum()),
        "n_yellow_train": int(yellow_train.sum()),
        "n_green_test": int(green_test.sum()),
        "n_red_test": int(red_test.sum()),
        "n_yellow_test": int(yellow_test.sum()),
        "yellow_train_nc": int(cls_counts_yellow[0]),
        "yellow_train_ad": int(cls_counts_yellow[1]),
        "yellow_train_minority": int(cls_counts_yellow.min()),
        "c_best_yellow": c_best,
        "c_best_cv_ba": c_info["best_score"],
        "c_cv_splits": c_info["n_splits"],
        "current_test_BA": metrics_current["BA"],
        "current_test_Sens": metrics_current["Sens"],
        "current_test_Spec": metrics_current["Spec"],
        "current_test_Errors": metrics_current["Errors"],
        "current_test_FN": metrics_current["FN"],
        "current_test_FP": metrics_current["FP"],
        "yellow_test_BA": metrics_yellow["BA"],
        "yellow_test_Sens": metrics_yellow["Sens"],
        "yellow_test_Spec": metrics_yellow["Spec"],
        "yellow_test_Errors": metrics_yellow["Errors"],
        "yellow_test_FN": metrics_yellow["FN"],
        "yellow_test_FP": metrics_yellow["FP"],
        "delta_BA": metrics_yellow["BA"] - metrics_current["BA"],
        "delta_Errors": metrics_yellow["Errors"] - metrics_current["Errors"],
    }


def summarize(df: pd.DataFrame) -> dict:
    metrics = [
        "current_test_BA",
        "current_test_Sens",
        "current_test_Spec",
        "current_test_Errors",
        "yellow_test_BA",
        "yellow_test_Sens",
        "yellow_test_Spec",
        "yellow_test_Errors",
        "delta_BA",
        "delta_Errors",
        "n_yellow_train",
        "yellow_train_minority",
        "n_yellow_test",
    ]
    out = {m: {"mean": float(df[m].mean()), "std": float(df[m].std())} for m in metrics}
    c_counts = Counter(df["c_best_yellow"].round(10).tolist())
    out["c_distribution"] = {str(k): int(v) for k, v in sorted(c_counts.items(), key=lambda kv: kv[0])}
    out["best_c_mode"] = float(df["c_best_yellow"].mode().iloc[0])
    out["best_c_median"] = float(df["c_best_yellow"].median())
    out["yellow_better_count"] = int((df["delta_BA"] > 0).sum())
    out["yellow_not_worse_count"] = int((df["delta_BA"] >= 0).sum())
    out["valid_seeds"] = int(len(df))
    return out


def main():
    results = []
    for i, seed in enumerate(range(N_SEEDS), start=1):
        r = run_seed(seed)
        if r is not None:
            results.append(r)
        if i % 10 == 0:
            print(f"  completados {i}/{N_SEEDS} seeds")

    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "yellow_only_stage2_results.csv", index=False)

    valid_df = df[df["valid"]].copy()
    summary = summarize(valid_df)
    summary_path = OUT_DIR / "yellow_only_stage2_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 88)
    print("YELLOW-ONLY STAGE 2 vs STAGE 2 ACTUAL")
    print("=" * 88)
    print(f"Seeds válidas: {len(valid_df)}/{N_SEEDS}")
    print()
    print("Baseline actual (S2 entrenado con todo el train, C=10):")
    print(f"  BA      = {valid_df['current_test_BA'].mean():.3f} ± {valid_df['current_test_BA'].std():.3f}")
    print(f"  Sens    = {valid_df['current_test_Sens'].mean():.3f} ± {valid_df['current_test_Sens'].std():.3f}")
    print(f"  Spec    = {valid_df['current_test_Spec'].mean():.3f} ± {valid_df['current_test_Spec'].std():.3f}")
    print(f"  Errors  = {valid_df['current_test_Errors'].mean():.2f} ± {valid_df['current_test_Errors'].std():.2f}")
    print()
    print("Yellow-only Stage 2 (S2 entrenado solo con amarilla_train, C por inner CV):")
    print(f"  BA      = {valid_df['yellow_test_BA'].mean():.3f} ± {valid_df['yellow_test_BA'].std():.3f}")
    print(f"  Sens    = {valid_df['yellow_test_Sens'].mean():.3f} ± {valid_df['yellow_test_Sens'].std():.3f}")
    print(f"  Spec    = {valid_df['yellow_test_Spec'].mean():.3f} ± {valid_df['yellow_test_Spec'].std():.3f}")
    print(f"  Errors  = {valid_df['yellow_test_Errors'].mean():.2f} ± {valid_df['yellow_test_Errors'].std():.2f}")
    print()
    print("Diferencia yellow-only - actual:")
    print(f"  ΔBA     = {valid_df['delta_BA'].mean():+.3f} ± {valid_df['delta_BA'].std():.3f}")
    print(f"  ΔErrors = {valid_df['delta_Errors'].mean():+.2f} ± {valid_df['delta_Errors'].std():.2f}")
    print(f"  Mejora estricta de BA en {int((valid_df['delta_BA'] > 0).sum())}/{len(valid_df)} seeds")
    print(f"  No empeora BA en {int((valid_df['delta_BA'] >= 0).sum())}/{len(valid_df)} seeds")
    print()
    print("Cobertura de amarilla_train:")
    print(f"  N amarilla train = {valid_df['n_yellow_train'].mean():.1f} ± {valid_df['n_yellow_train'].std():.1f}")
    print(
        f"  Minoría en amarilla_train = {valid_df['yellow_train_minority'].mean():.1f} ± {valid_df['yellow_train_minority'].std():.1f}"
    )
    print()
    print("Distribución de C óptimos (yellow-only):")
    for c_val, count in sorted(Counter(valid_df["c_best_yellow"].round(10).tolist()).items(), key=lambda kv: kv[0]):
        print(f"  C={c_val:g}: {count} seeds")
    print(f"  Modo:   {valid_df['c_best_yellow'].mode().iloc[0]:g}")
    print(f"  Mediana:{valid_df['c_best_yellow'].median():g}")
    print()
    print(f"CSV guardado en: {OUT_DIR / 'yellow_only_stage2_results.csv'}")
    print(f"Resumen guardado en: {summary_path}")


if __name__ == "__main__":
    main()
