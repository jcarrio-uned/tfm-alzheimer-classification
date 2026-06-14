"""Experimento: Stacking con segundo modelo en Stage 1.

Evalúa si añadir un segundo modelo complementario a LogReg en Stage 1
(creando un stacking real) mejora la performance del sistema cascada.

Hipótesis: un modelo no-lineal puede identificar pacientes dudosos
diferentes a los de LogReg, y combinar ambas "opiniones" en Stage 2
puede resolver mejor la zona amarilla.

Candidatos evaluados:
- SVM (RBF): captura no-linealidades suaves
- KNN: sensible a estructura local del espacio
- Naive Bayes: asunciones independientes, complementario a LR
- Random Forest (ligero): captura interacciones
- HistGradientBoosting (ligero): boosting con early stopping
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import LeaveOneOut, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# ============================================================================
# DATOS
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "alzheimer-classification-app" / "data"

df_metab = pd.read_excel(DATA_DIR / "residuales_grupos3.xlsx")
df_clin = pd.read_excel(BASE_DIR / "Segundo_Archivo_clean.xlsx")

y = (df_metab["Group"] == "AD").astype(int).values
ids = df_metab["ID"].values
n = len(y)

FEATURES_S1 = [
    "DOPA", "Cer(d18:1/20:0)", "lysoPC.a.C18:2", "PC.aa.C40:4", "DHEAS",
    "Arg", "HexCer(d18:1/26:1)",
]
df_metab["DHEAS/lysoPC"] = df_metab["DHEAS"] / (df_metab["lysoPC.a.C18:2"] + 1e-10)
df_metab["DOPA*DHEAS"] = df_metab["DOPA"] * df_metab["DHEAS"]
df_metab["PC*DOPA"] = df_metab["PC.aa.C40:4"] * df_metab["DOPA"]
df_metab["DHEAS/Cer20"] = df_metab["DHEAS"] / (df_metab["Cer(d18:1/20:0)"] + 1e-10)
FEATURES_S1 += ["DHEAS/lysoPC", "DOPA*DHEAS", "PC*DOPA", "DHEAS/Cer20"]

X = df_metab[FEATURES_S1].values

# Clinical vars
df_merged = df_metab[["ID"]].merge(
    df_clin[["ID", "MMSE", "APOE", "Depression", "Cardiovascular disorder", "Age [y]"]], on="ID"
)
mmse = df_merged["MMSE"].values.astype(float)
apoe = df_merged["APOE"].map({"e3/e3": 0, "e3/e4": 1, "e4/e4": 2, "e2/e3": 0, "e2/e4": 1}).fillna(0).values
dep = (df_merged["Depression"] == "Si").astype(int).values
cardio = (df_merged["Cardiovascular disorder"] == "Si").astype(int).values
age = df_merged["Age [y]"].values.astype(float)

# Parameters
C_S1 = 0.1
CW_S1 = "balanced"
C_S2 = 10.0
CW_S2 = {0: 1, 1: 2}


def winsorize_train(X_data):
    """Winsoriza al 5/95."""
    X_out = X_data.copy()
    for j in range(X_out.shape[1]):
        lo, hi = np.percentile(X_out[:, j], [5, 95])
        X_out[:, j] = np.clip(X_out[:, j], lo, hi)
    return X_out


def find_thresholds(P, y_arr, max_err_g=0, max_err_r=0):
    """Busca umbrales óptimos."""
    probs = np.sort(np.unique(P))
    cands = np.concatenate([[0.0], probs, [1.0]])
    best_tlo, best_thi, best_cov = None, None, -1
    for tl in cands:
        if ((P < tl) & (y_arr == 1)).sum() > max_err_g:
            continue
        for th in cands:
            if th <= tl:
                continue
            if ((P > th) & (y_arr == 0)).sum() > max_err_r:
                continue
            cov = ((P < tl) | (P > th)).sum()
            if cov > best_cov:
                best_cov = cov
                best_tlo, best_thi = tl, th
    return best_tlo, best_thi


def build_X2(indices, P_s1_lr, P_s1_model2=None):
    """Construye matriz Stage 2. Si P_s1_model2 se provee, incluye ambas probabilidades."""
    cols = [P_s1_lr, mmse[indices], apoe[indices], dep[indices], cardio[indices], age[indices]]
    if P_s1_model2 is not None:
        cols.insert(1, P_s1_model2)  # P(AD) del segundo modelo justo después de P_LR
    return np.column_stack(cols)


# ============================================================================
# MODELOS CANDIDATOS
# ============================================================================

CANDIDATES = {
    "SVM-RBF (C=1, gamma=scale)": SVC(C=1, kernel="rbf", gamma="scale", probability=True, random_state=42),
    "SVM-RBF (C=0.5, gamma=0.05)": SVC(C=0.5, kernel="rbf", gamma=0.05, probability=True, random_state=42),
    "KNN (k=7, distance)": KNeighborsClassifier(n_neighbors=7, weights="distance", metric="euclidean"),
    "KNN (k=11, uniform)": KNeighborsClassifier(n_neighbors=11, weights="uniform"),
    "NaiveBayes": GaussianNB(),
    "RF (n=50, depth=3)": RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42, class_weight="balanced"),
    "RF (n=100, depth=2)": RandomForestClassifier(n_estimators=100, max_depth=2, random_state=42, class_weight="balanced"),
    "HistGB (lr=0.05, depth=2)": HistGradientBoostingClassifier(
        learning_rate=0.05, max_depth=2, max_iter=100, min_samples_leaf=10, random_state=42
    ),
    "HistGB (lr=0.03, depth=3)": HistGradientBoostingClassifier(
        learning_rate=0.03, max_depth=3, max_iter=80, min_samples_leaf=15, random_state=42
    ),
}


# ============================================================================
# FASE 1: SELECCIÓN DEL MEJOR MODELO COMPLEMENTARIO
# ============================================================================

print("=" * 80)
print("  FASE 1: SELECCIÓN DEL MEJOR MODELO COMPLEMENTARIO A LOGREG")
print("=" * 80)
print()
print("Para cada candidato, evaluamos el stacking completo (LOO) con seed=42:")
print("  Stage 1: LOO → P_LR + P_modelo2")
print("  Umbrales: basados en P_LR (como en cascada actual)")
print("  Stage 2: LOO → LogReg(P_LR, P_modelo2, MMSE, APOE, Dep, Cardio, Age)")
print()

seed = 42
indices_all = np.arange(n)
train_idx, test_idx = train_test_split(indices_all, test_size=0.2, stratify=y, random_state=seed)
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
n_train = len(y_train)

# Baseline: cascada actual (solo LR)
print("--- BASELINE: Cascada actual (solo LR en S1) ---")
P_s1_lr_train = np.zeros(n_train)
for tr, te in LeaveOneOut().split(X_train, y_train):
    X_w = winsorize_train(X_train[tr])
    sc = StandardScaler().fit(X_w)
    lr = LogisticRegression(C=C_S1, class_weight=CW_S1, max_iter=2000, random_state=42)
    lr.fit(sc.transform(X_w), y_train[tr])
    P_s1_lr_train[te] = lr.predict_proba(sc.transform(X_train[te]))[:, 1]

t_lo, t_hi = find_thresholds(P_s1_lr_train, y_train)
verde = P_s1_lr_train < t_lo
roja = P_s1_lr_train > t_hi
amarilla = ~verde & ~roja

# Stage 2 baseline (solo P_LR + clinical)
P_s2_base = np.zeros(n_train)
for tr, te in LeaveOneOut().split(X_train, y_train):
    X_w = winsorize_train(X_train[tr])
    sc = StandardScaler().fit(X_w)
    lr1 = LogisticRegression(C=C_S1, class_weight=CW_S1, max_iter=2000, random_state=42)
    lr1.fit(sc.transform(X_w), y_train[tr])
    P_lr_tr = lr1.predict_proba(sc.transform(X_w))[:, 1]
    P_lr_te = lr1.predict_proba(sc.transform(X_train[te]))[:, 1]

    X2_tr = build_X2(train_idx[tr], P_lr_tr)
    X2_te = build_X2(train_idx[te], P_lr_te)
    lr2 = LogisticRegression(C=C_S2, class_weight=CW_S2, max_iter=2000, random_state=42)
    lr2.fit(X2_tr, y_train[tr])
    P_s2_base[te] = lr2.predict_proba(X2_te)[:, 1]

pred_base = np.full(n_train, -1)
pred_base[verde] = 0
pred_base[roja] = 1
pred_base[amarilla] = (P_s2_base[amarilla] >= 0.5).astype(int)

fn_base = ((pred_base == 0) & (y_train == 1)).sum()
fp_base = ((pred_base == 1) & (y_train == 0)).sum()
ba_base = balanced_accuracy_score(y_train, pred_base)
print(f"  Errores: {fn_base + fp_base} (FN={fn_base}, FP={fp_base}) | BA={ba_base:.4f}")
print()

# Evaluar cada candidato
print(f"{'Modelo':<35} {'Err':>4} {'FN':>4} {'FP':>4} {'BA':>7} {'ΔErr':>5} {'ΔBA':>7} {'Disagree%':>10}")
print("-" * 85)

results_phase1 = []

for name, model2 in CANDIDATES.items():
    # LOO para obtener P_model2 en train
    P_m2_train = np.zeros(n_train)
    P_lr_train_inner = np.zeros(n_train)

    for tr, te in LeaveOneOut().split(X_train, y_train):
        X_w = winsorize_train(X_train[tr])
        sc = StandardScaler().fit(X_w)

        # LR
        lr1 = LogisticRegression(C=C_S1, class_weight=CW_S1, max_iter=2000, random_state=42)
        lr1.fit(sc.transform(X_w), y_train[tr])
        P_lr_train_inner[te] = lr1.predict_proba(sc.transform(X_train[te]))[:, 1]

        # Model 2
        model2_clone = model2.__class__(**model2.get_params())
        model2_clone.fit(sc.transform(X_w), y_train[tr])
        P_m2_train[te] = model2_clone.predict_proba(sc.transform(X_train[te]))[:, 1]

    # Disagreement: % de veces que LR y model2 dan predicción diferente (t=0.5)
    pred_lr_simple = (P_lr_train_inner >= 0.5).astype(int)
    pred_m2_simple = (P_m2_train >= 0.5).astype(int)
    disagree_pct = (pred_lr_simple != pred_m2_simple).mean() * 100

    # Stage 2 con ambas probabilidades
    P_s2_stack = np.zeros(n_train)
    for tr, te in LeaveOneOut().split(X_train, y_train):
        X_w = winsorize_train(X_train[tr])
        sc = StandardScaler().fit(X_w)

        lr1 = LogisticRegression(C=C_S1, class_weight=CW_S1, max_iter=2000, random_state=42)
        lr1.fit(sc.transform(X_w), y_train[tr])
        P_lr_tr = lr1.predict_proba(sc.transform(X_w))[:, 1]
        P_lr_te = lr1.predict_proba(sc.transform(X_train[te]))[:, 1]

        model2_clone = model2.__class__(**model2.get_params())
        model2_clone.fit(sc.transform(X_w), y_train[tr])
        P_m2_tr = model2_clone.predict_proba(sc.transform(X_w))[:, 1]
        P_m2_te = model2_clone.predict_proba(sc.transform(X_train[te]))[:, 1]

        X2_tr = build_X2(train_idx[tr], P_lr_tr, P_m2_tr)
        X2_te = build_X2(train_idx[te], P_lr_te, P_m2_te)

        lr2 = LogisticRegression(C=C_S2, class_weight=CW_S2, max_iter=2000, random_state=42)
        lr2.fit(X2_tr, y_train[tr])
        P_s2_stack[te] = lr2.predict_proba(X2_te)[:, 1]

    # Predicciones con umbrales (mismos que baseline)
    pred_stack = np.full(n_train, -1)
    pred_stack[verde] = 0
    pred_stack[roja] = 1
    pred_stack[amarilla] = (P_s2_stack[amarilla] >= 0.5).astype(int)

    fn_s = ((pred_stack == 0) & (y_train == 1)).sum()
    fp_s = ((pred_stack == 1) & (y_train == 0)).sum()
    ba_s = balanced_accuracy_score(y_train, pred_stack)
    delta_err = (fn_s + fp_s) - (fn_base + fp_base)
    delta_ba = ba_s - ba_base

    results_phase1.append({
        "model": name, "errors": fn_s + fp_s, "FN": fn_s, "FP": fp_s,
        "BA": ba_s, "delta_err": delta_err, "delta_ba": delta_ba,
        "disagree": disagree_pct, "model_obj": model2,
    })

    marker = " ← MEJORA" if delta_err < 0 else ""
    print(f"  {name:<33} {fn_s+fp_s:>4} {fn_s:>4} {fp_s:>4} {ba_s:>7.4f} {delta_err:>+5} {delta_ba:>+7.4f} {disagree_pct:>9.1f}%{marker}")

print()
print(f"  BASELINE (solo LR):              {fn_base+fp_base:>4} {fn_base:>4} {fp_base:>4} {ba_base:>7.4f}")
print()

# Seleccionar los mejores
results_phase1.sort(key=lambda x: (x["errors"], -x["BA"]))
best = results_phase1[0]
print(f"  MEJOR: {best['model']} → {best['errors']} errores, BA={best['BA']:.4f} (Δerr={best['delta_err']:+d})")

# ============================================================================
# FASE 2: HIPERPARÁMETRO TUNING DEL MEJOR MODELO
# ============================================================================

print()
print("=" * 80)
print("  FASE 2: TUNING DEL MEJOR MODELO COMPLEMENTARIO")
print("=" * 80)
print()

# Determinar tipo de mejor modelo y hacer grid search
best_model_name = best["model"]

if "SVM" in best_model_name:
    param_grid = [
        {"C": c, "gamma": g, "kernel": "rbf"}
        for c in [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
        for g in [0.01, 0.03, 0.05, 0.1, "scale"]
    ]
    model_class = SVC
    extra_params = {"probability": True, "random_state": 42}
elif "KNN" in best_model_name:
    param_grid = [
        {"n_neighbors": k, "weights": w, "metric": m}
        for k in [3, 5, 7, 9, 11, 13, 15]
        for w in ["uniform", "distance"]
        for m in ["euclidean", "manhattan"]
    ]
    model_class = KNeighborsClassifier
    extra_params = {}
elif "NaiveBayes" in best_model_name:
    param_grid = [{"var_smoothing": vs} for vs in np.logspace(-12, -6, 10)]
    model_class = GaussianNB
    extra_params = {}
elif "RF" in best_model_name:
    param_grid = [
        {"n_estimators": ne, "max_depth": md, "min_samples_leaf": ml}
        for ne in [30, 50, 100, 150]
        for md in [2, 3, 4]
        for ml in [3, 5, 8]
    ]
    model_class = RandomForestClassifier
    extra_params = {"random_state": 42, "class_weight": "balanced"}
elif "HistGB" in best_model_name:
    param_grid = [
        {"learning_rate": lr, "max_depth": md, "max_iter": mi, "min_samples_leaf": ml}
        for lr in [0.01, 0.03, 0.05, 0.08, 0.1]
        for md in [2, 3, 4]
        for mi in [50, 80, 100, 150]
        for ml in [5, 10, 15, 20]
    ]
    model_class = HistGradientBoostingClassifier
    extra_params = {"random_state": 42}
else:
    param_grid = []
    model_class = None
    extra_params = {}

print(f"  Modelo base: {best_model_name}")
print(f"  Grid size: {len(param_grid)} configuraciones")
print()

# Evaluar cada configuración con LOO stacking completo
# (esto es costoso, así que limitamos si hay muchas)
if len(param_grid) > 60:
    # Random subsample of grid
    rng = np.random.default_rng(42)
    param_grid = [param_grid[i] for i in rng.choice(len(param_grid), 60, replace=False)]
    print(f"  (Subsampled a 60 configuraciones)")

print(f"  Evaluando {len(param_grid)} configuraciones (LOO stacking)...")
print()

tuning_results = []
for pi, params in enumerate(param_grid):
    if (pi + 1) % 10 == 0:
        print(f"    {pi + 1}/{len(param_grid)}...")

    model2 = model_class(**params, **extra_params)

    # LOO completo del stacking
    P_s2_tune = np.zeros(n_train)
    for tr, te in LeaveOneOut().split(X_train, y_train):
        X_w = winsorize_train(X_train[tr])
        sc = StandardScaler().fit(X_w)

        lr1 = LogisticRegression(C=C_S1, class_weight=CW_S1, max_iter=2000, random_state=42)
        lr1.fit(sc.transform(X_w), y_train[tr])
        P_lr_tr = lr1.predict_proba(sc.transform(X_w))[:, 1]
        P_lr_te = lr1.predict_proba(sc.transform(X_train[te]))[:, 1]

        try:
            model2_fit = model_class(**params, **extra_params)
            model2_fit.fit(sc.transform(X_w), y_train[tr])
            P_m2_tr = model2_fit.predict_proba(sc.transform(X_w))[:, 1]
            P_m2_te = model2_fit.predict_proba(sc.transform(X_train[te]))[:, 1]
        except Exception:
            P_m2_tr = np.full(len(tr), 0.5)
            P_m2_te = np.array([0.5])

        X2_tr = build_X2(train_idx[tr], P_lr_tr, P_m2_tr)
        X2_te = build_X2(train_idx[te], P_lr_te, P_m2_te)

        lr2 = LogisticRegression(C=C_S2, class_weight=CW_S2, max_iter=2000, random_state=42)
        lr2.fit(X2_tr, y_train[tr])
        P_s2_tune[te] = lr2.predict_proba(X2_te)[:, 1]

    pred_tune = np.full(n_train, -1)
    pred_tune[verde] = 0
    pred_tune[roja] = 1
    pred_tune[amarilla] = (P_s2_tune[amarilla] >= 0.5).astype(int)

    fn_t = ((pred_tune == 0) & (y_train == 1)).sum()
    fp_t = ((pred_tune == 1) & (y_train == 0)).sum()
    ba_t = balanced_accuracy_score(y_train, pred_tune)

    tuning_results.append({"params": params, "errors": fn_t + fp_t, "FN": fn_t, "FP": fp_t, "BA": ba_t})

# Mejor configuración
tuning_results.sort(key=lambda x: (x["errors"], -x["BA"]))
best_tuned = tuning_results[0]

print()
print(f"  MEJOR CONFIG: {best_tuned['params']}")
print(f"  Errores: {best_tuned['errors']} (FN={best_tuned['FN']}, FP={best_tuned['FP']}) | BA={best_tuned['BA']:.4f}")
print(f"  vs Baseline: Δerr={best_tuned['errors'] - (fn_base + fp_base):+d}, ΔBA={best_tuned['BA'] - ba_base:+.4f}")
print()

# Top 5
print("  Top 5 configuraciones:")
for i, r in enumerate(tuning_results[:5]):
    print(f"    #{i+1}: err={r['errors']}, BA={r['BA']:.4f} | {r['params']}")

# ============================================================================
# FASE 3: EVALUACIÓN MULTI-SEED (ESTABILIDAD)
# ============================================================================

print()
print("=" * 80)
print("  FASE 3: ESTABILIDAD MULTI-SEED (30 seeds)")
print("=" * 80)
print()

best_params = best_tuned["params"]
N_SEEDS = 30

results_multiseed = {"Cascada (solo LR)": [], "Stacking (LR + model2)": []}

for seed_i in range(N_SEEDS):
    idx = np.arange(n)
    tr_i, te_i = train_test_split(idx, test_size=0.2, stratify=y, random_state=seed_i)
    X_tr, X_te = X[tr_i], X[te_i]
    y_tr, y_te = y[tr_i], y[te_i]
    n_tr = len(y_tr)

    # Stage 1 LOO (LR + model2)
    P_lr_loo = np.zeros(n_tr)
    P_m2_loo = np.zeros(n_tr)
    for tr, te in LeaveOneOut().split(X_tr, y_tr):
        X_w = winsorize_train(X_tr[tr])
        sc = StandardScaler().fit(X_w)

        lr1 = LogisticRegression(C=C_S1, class_weight=CW_S1, max_iter=2000, random_state=42)
        lr1.fit(sc.transform(X_w), y_tr[tr])
        P_lr_loo[te] = lr1.predict_proba(sc.transform(X_tr[te]))[:, 1]

        m2 = model_class(**best_params, **extra_params)
        m2.fit(sc.transform(X_w), y_tr[tr])
        P_m2_loo[te] = m2.predict_proba(sc.transform(X_tr[te]))[:, 1]

    # Thresholds (based on LR as in current cascade)
    tl, th = find_thresholds(P_lr_loo, y_tr)
    if tl is None:
        continue

    verde_s = P_lr_loo < tl
    roja_s = P_lr_loo > th
    amarilla_s = ~verde_s & ~roja_s

    # === CASCADA BASELINE ===
    P_s2_base_s = np.zeros(n_tr)
    for tr, te in LeaveOneOut().split(X_tr, y_tr):
        X_w = winsorize_train(X_tr[tr])
        sc = StandardScaler().fit(X_w)
        lr1 = LogisticRegression(C=C_S1, class_weight=CW_S1, max_iter=2000, random_state=42)
        lr1.fit(sc.transform(X_w), y_tr[tr])
        P_lr_tr = lr1.predict_proba(sc.transform(X_w))[:, 1]
        P_lr_te = lr1.predict_proba(sc.transform(X_tr[te]))[:, 1]
        X2_tr = build_X2(tr_i[tr], P_lr_tr)
        X2_te = build_X2(tr_i[te], P_lr_te)
        lr2 = LogisticRegression(C=C_S2, class_weight=CW_S2, max_iter=2000, random_state=42)
        lr2.fit(X2_tr, y_tr[tr])
        P_s2_base_s[te] = lr2.predict_proba(X2_te)[:, 1]

    pred_base_s = np.full(n_tr, -1)
    pred_base_s[verde_s] = 0
    pred_base_s[roja_s] = 1
    pred_base_s[amarilla_s] = (P_s2_base_s[amarilla_s] >= 0.5).astype(int)

    # Test baseline
    X_w_full = winsorize_train(X_tr)
    sc_f = StandardScaler().fit(X_w_full)
    lr1_f = LogisticRegression(C=C_S1, class_weight=CW_S1, max_iter=2000, random_state=42)
    lr1_f.fit(sc_f.transform(X_w_full), y_tr)
    P_lr_test = lr1_f.predict_proba(sc_f.transform(X_te))[:, 1]
    P_lr_tr_full = lr1_f.predict_proba(sc_f.transform(X_w_full))[:, 1]
    X2_tr_full = build_X2(tr_i, P_lr_tr_full)
    lr2_f = LogisticRegression(C=C_S2, class_weight=CW_S2, max_iter=2000, random_state=42)
    lr2_f.fit(X2_tr_full, y_tr)

    pred_te_base = np.full(len(y_te), -1)
    verde_te = P_lr_test < tl
    roja_te = P_lr_test > th
    amarilla_te = ~verde_te & ~roja_te
    pred_te_base[verde_te] = 0
    pred_te_base[roja_te] = 1
    if amarilla_te.sum() > 0:
        X2_te_am = build_X2(te_i[amarilla_te], P_lr_test[amarilla_te])
        pred_te_base[amarilla_te] = (lr2_f.predict_proba(X2_te_am)[:, 1] >= 0.5).astype(int)

    ba_te_base = balanced_accuracy_score(y_te, pred_te_base)
    results_multiseed["Cascada (solo LR)"].append(ba_te_base)

    # === STACKING ===
    P_s2_stack_s = np.zeros(n_tr)
    for tr, te in LeaveOneOut().split(X_tr, y_tr):
        X_w = winsorize_train(X_tr[tr])
        sc = StandardScaler().fit(X_w)

        lr1 = LogisticRegression(C=C_S1, class_weight=CW_S1, max_iter=2000, random_state=42)
        lr1.fit(sc.transform(X_w), y_tr[tr])
        P_lr_tr = lr1.predict_proba(sc.transform(X_w))[:, 1]
        P_lr_te = lr1.predict_proba(sc.transform(X_tr[te]))[:, 1]

        m2 = model_class(**best_params, **extra_params)
        m2.fit(sc.transform(X_w), y_tr[tr])
        P_m2_tr = m2.predict_proba(sc.transform(X_w))[:, 1]
        P_m2_te = m2.predict_proba(sc.transform(X_tr[te]))[:, 1]

        X2_tr = build_X2(tr_i[tr], P_lr_tr, P_m2_tr)
        X2_te = build_X2(tr_i[te], P_lr_te, P_m2_te)
        lr2 = LogisticRegression(C=C_S2, class_weight=CW_S2, max_iter=2000, random_state=42)
        lr2.fit(X2_tr, y_tr[tr])
        P_s2_stack_s[te] = lr2.predict_proba(X2_te)[:, 1]

    # Test stacking
    m2_f = model_class(**best_params, **extra_params)
    m2_f.fit(sc_f.transform(X_w_full), y_tr)
    P_m2_test = m2_f.predict_proba(sc_f.transform(X_te))[:, 1]
    P_m2_tr_full = m2_f.predict_proba(sc_f.transform(X_w_full))[:, 1]

    X2_tr_stack = build_X2(tr_i, P_lr_tr_full, P_m2_tr_full)
    lr2_stack = LogisticRegression(C=C_S2, class_weight=CW_S2, max_iter=2000, random_state=42)
    lr2_stack.fit(X2_tr_stack, y_tr)

    pred_te_stack = np.full(len(y_te), -1)
    pred_te_stack[verde_te] = 0
    pred_te_stack[roja_te] = 1
    if amarilla_te.sum() > 0:
        X2_te_stack = build_X2(te_i[amarilla_te], P_lr_test[amarilla_te], P_m2_test[amarilla_te])
        pred_te_stack[amarilla_te] = (lr2_stack.predict_proba(X2_te_stack)[:, 1] >= 0.5).astype(int)

    ba_te_stack = balanced_accuracy_score(y_te, pred_te_stack)
    results_multiseed["Stacking (LR + model2)"].append(ba_te_stack)

    if (seed_i + 1) % 10 == 0:
        print(f"    Completadas {seed_i + 1}/{N_SEEDS} seeds...")

print()
print("  RESULTADOS MULTI-SEED:")
print(f"  {'Modelo':<25} {'BA media':>10} {'BA std':>10} {'BA min':>10} {'BA max':>10}")
print(f"  {'-'*65}")
for name, vals in results_multiseed.items():
    v = np.array(vals)
    print(f"  {name:<25} {v.mean():>10.4f} {v.std():>10.4f} {v.min():>10.4f} {v.max():>10.4f}")

# Paired comparison
ba_base_arr = np.array(results_multiseed["Cascada (solo LR)"])
ba_stack_arr = np.array(results_multiseed["Stacking (LR + model2)"])
diff = ba_stack_arr - ba_base_arr
print()
print(f"  Diferencia paired (Stacking - Baseline):")
print(f"    Media: {diff.mean():+.4f} ({diff.mean()*100:+.2f} pp)")
print(f"    Std:   {diff.std():.4f}")
print(f"    % seeds donde stacking > baseline: {(diff > 0).mean()*100:.0f}%")
print(f"    % seeds donde stacking = baseline: {(diff == 0).mean()*100:.0f}%")
print(f"    % seeds donde stacking < baseline: {(diff < 0).mean()*100:.0f}%")

# ============================================================================
# CONCLUSIONES
# ============================================================================

print()
print("=" * 80)
print("  CONCLUSIONES")
print("=" * 80)
print()

improvement = diff.mean()
pct_better = (diff > 0).mean() * 100

if improvement > 0.01 and pct_better > 60:
    print(f"  ✅ El stacking con {best_model_name} MEJORA de forma consistente:")
    print(f"     +{improvement*100:.2f} pp de BA, mejora en {pct_better:.0f}% de seeds")
    print(f"     Config óptima: {best_params}")
    print()
    print("  RECOMENDACIÓN: Incorporar al sistema cascada como feature adicional en Stage 2.")
elif improvement > 0.005 and pct_better > 50:
    print(f"  🟡 El stacking aporta mejora MARGINAL ({improvement*100:+.2f} pp, {pct_better:.0f}% seeds)")
    print(f"     La complejidad adicional puede no justificarse.")
    print()
    print("  RECOMENDACIÓN: Opcional. Mencionar como exploración en el TFM.")
else:
    print(f"  ❌ El stacking NO mejora de forma significativa ({improvement*100:+.2f} pp, {pct_better:.0f}% seeds)")
    print(f"     El problema es suficientemente lineal para que LR capture toda la señal.")
    print()
    print("  RECOMENDACIÓN: Mantener cascada actual (solo LR). Es más simple e interpretable.")
    print("  La señal del segundo modelo ya está capturada implícitamente por LR en Stage 1.")

print()
print("=" * 80)
print("FIN DEL EXPERIMENTO")
print("=" * 80)
