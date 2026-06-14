"""Experimento: Blending ponderado LR + Boosting para clasificación AD/NC.

Evalúa si un ensemble simple (0.7*LR + 0.3*Boosting) mejora la Balanced Accuracy
frente a Logistic Regression sola, en un dataset de 148 muestras con señal
predominantemente lineal.

Metodología:
- RepeatedStratifiedKFold (10 folds × 10 repeticiones) para estabilidad
- Out-of-fold predictions (sin data leakage)
- Threshold tuning interno (dentro de cada fold) para escenario B
- Winsorización solo en train de cada fold
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================================
# 1. CARGA Y PREPARACIÓN DE DATOS
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "alzheimer-classification-app" / "data"

# Cargar datos metabolómicos
df_metab = pd.read_excel(DATA_DIR / "residuales_grupos3.xlsx")

# Cargar datos clínicos
df_clin = pd.read_excel(BASE_DIR / "Segundo_Archivo_clean.xlsx")

# Target
y = (df_metab["Group"] == "AD").astype(int).values  # 1=AD, 0=NC
ids = df_metab["ID"].values

# --- Features metabolómicos (20 residuos) ---
metab_cols = [c for c in df_metab.columns if c not in ["ID", "Group", "Age [y]", "Sex", "Fasting time [h]"]]
X_metab = df_metab[metab_cols].values

# --- Features clínicos ---
df_clin_merged = df_metab[["ID"]].merge(df_clin[["ID", "APOE", "Depression", "Cardiovascular disorder", "Age [y]", "Fasting time [h]"]], on="ID")

# APOE: ordinal encoding (0=bajo riesgo, 1=intermedio, 2=alto)
apoe_map = {"e3/e3": 0, "e2/e3": 0, "e3/e4": 1, "e2/e4": 1, "e4/e4": 2}
apoe = df_clin_merged["APOE"].map(apoe_map).fillna(0).values.reshape(-1, 1)

# Binarias
dep = (df_clin_merged["Depression"] == "Si").astype(int).values.reshape(-1, 1)
cardio = (df_clin_merged["Cardiovascular disorder"] == "Si").astype(int).values.reshape(-1, 1)

# Continuas
age = df_clin_merged["Age [y]"].values.reshape(-1, 1)
fasting = df_clin_merged["Fasting time [h]"].values.reshape(-1, 1)

# Matriz combinada: metabolitos + clínicos
X_full = np.hstack([X_metab, apoe, dep, cardio, age, fasting])
feature_names = metab_cols + ["APOE", "Depression", "Cardiovascular", "Age", "Fasting_h"]

print(f"Dataset: {X_full.shape[0]} muestras, {X_full.shape[1]} features")
print(f"Clases: AD={y.sum()}, NC={(1-y).sum()}")
print(f"Features: {len(metab_cols)} metabolitos + 5 clínicos = {X_full.shape[1]}")
print()

# ============================================================================
# 2. FUNCIONES DE PREPROCESAMIENTO Y EVALUACIÓN
# ============================================================================


def winsorize_train(X_train, percentile=0.05):
    """Winsoriza train al 5%/95%. Devuelve datos clipped y bounds."""
    lower = np.percentile(X_train, percentile * 100, axis=0)
    upper = np.percentile(X_train, (1 - percentile) * 100, axis=0)
    X_w = X_train.copy()
    for j in range(X_w.shape[1]):
        X_w[:, j] = np.clip(X_w[:, j], lower[j], upper[j])
    return X_w, lower, upper


def preprocess_fold(X_train, X_test, winsorize_pct=0.05):
    """Preprocesado por fold: winsorizar train, escalar ambos.

    Test NO se winsoriza (metodología TFM).
    """
    # Solo winsorizar los metabolitos (primeras 20 columnas)
    n_metab = 20
    X_tr = X_train.copy()
    X_te = X_test.copy()

    # Winsorizar metabolitos del train
    X_tr[:, :n_metab], lower, upper = winsorize_train(X_tr[:, :n_metab], winsorize_pct)

    # Escalar todo (fit en train, transform en test)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    return X_tr, X_te


def find_optimal_threshold(y_true, proba, metric="ba"):
    """Encuentra threshold óptimo buscando maximizar BA o recall con spec mínimo."""
    thresholds = np.linspace(0.2, 0.8, 61)
    best_score = -1
    best_t = 0.5

    for t in thresholds:
        preds = (proba >= t).astype(int)
        if metric == "ba":
            score = balanced_accuracy_score(y_true, preds)
        elif metric == "recall_constrained":
            # Maximizar recall con specificity >= 0.6
            recall = recall_score(y_true, preds, zero_division=0)
            tn = ((preds == 0) & (y_true == 0)).sum()
            neg = (y_true == 0).sum()
            spec = tn / max(neg, 1)
            score = recall if spec >= 0.6 else -1
        if score > best_score:
            best_score = score
            best_t = t

    return best_t


def compute_metrics(y_true, y_pred, y_proba):
    """Calcula métricas de clasificación."""
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()

    recall = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    ba = (recall + spec) / 2

    roc = roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else np.nan
    pr_auc = average_precision_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else np.nan

    return {"BA": ba, "Recall": recall, "Specificity": spec, "ROC_AUC": roc, "PR_AUC": pr_auc}


# ============================================================================
# 3. CONFIGURACIÓN DE MODELOS
# ============================================================================

# LR: baseline lineal con regularización moderada y balanceo
lr_params = {
    "C": 0.1,
    "class_weight": "balanced",
    "max_iter": 2000,
    "solver": "lbfgs",
    "random_state": 42,
}

# XGBoost: boosting LIGERO para capturar interacciones suaves
xgb_params = {
    "n_estimators": 100,
    "max_depth": 2,            # Muy poco profundo → solo interacciones de orden 2
    "learning_rate": 0.05,     # Lento para no sobreajustar
    "subsample": 0.8,          # Subsampling para regularización
    "colsample_bytree": 0.8,   # No usar todas las features en cada árbol
    "min_child_weight": 5,     # Evitar hojas con pocas muestras
    "reg_alpha": 0.1,          # L1 regularización
    "reg_lambda": 1.0,         # L2 regularización
    "scale_pos_weight": 59 / 89,  # Compensar desbalance (NC/AD ratio)
    "eval_metric": "logloss",
    "random_state": 42,
    "verbosity": 0,
}

# Blending weights a evaluar
BLENDING_CONFIGS = {
    "Blend 0.7/0.3": (0.7, 0.3),   # Foco principal
    "Blend 0.8/0.2": (0.8, 0.2),   # Más conservador
    "Blend 0.6/0.4": (0.6, 0.4),   # Más boosting
}

# ============================================================================
# 4. EVALUACIÓN CON REPEATED STRATIFIED K-FOLD
# ============================================================================

N_SPLITS = 10
N_REPEATS = 10
RANDOM_STATE = 42

cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)

# Almacenar resultados
results = {name: [] for name in ["LR", "XGBoost"] + list(BLENDING_CONFIGS.keys())}
results_threshold = {name: [] for name in results}

# OOF predictions (para la primera repetición completa, análisis posterior)
oof_proba_lr = np.zeros(len(y))
oof_proba_xgb = np.zeros(len(y))
oof_counts = np.zeros(len(y))

print("=" * 70)
print("EJECUTANDO EVALUACIÓN")
print(f"CV: {N_SPLITS}-fold × {N_REPEATS} repeticiones = {N_SPLITS * N_REPEATS} folds")
print("=" * 70)

for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_full, y)):
    X_train, X_test = X_full[train_idx], X_full[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # --- Preprocesado (sin leakage) ---
    X_tr_proc, X_te_proc = preprocess_fold(X_train, X_test)

    # --- Entrenar LR ---
    lr = LogisticRegression(**lr_params)
    lr.fit(X_tr_proc, y_train)
    proba_lr = lr.predict_proba(X_te_proc)[:, 1]

    # --- Entrenar XGBoost ---
    # Para XGB: usar early stopping con validación interna
    # Split interno del train para early stopping (15%)
    n_tr = len(y_train)
    n_val_internal = max(int(n_tr * 0.15), 10)

    # Stratified split interno
    cv_internal = StratifiedKFold(n_splits=6, shuffle=True, random_state=RANDOM_STATE + fold_idx)
    tr_int, val_int = next(cv_internal.split(X_tr_proc, y_train))

    xgb_model = XGBClassifier(**xgb_params)
    xgb_model.fit(
        X_tr_proc[tr_int], y_train[tr_int],
        eval_set=[(X_tr_proc[val_int], y_train[val_int])],
        verbose=False,
    )
    proba_xgb = xgb_model.predict_proba(X_te_proc)[:, 1]

    # --- Guardar OOF de primera repetición ---
    if fold_idx < N_SPLITS:
        oof_proba_lr[test_idx] = proba_lr
        oof_proba_xgb[test_idx] = proba_xgb
        oof_counts[test_idx] += 1

    # === ESCENARIO A: Threshold fijo (0.5) ===
    pred_lr = (proba_lr >= 0.5).astype(int)
    pred_xgb = (proba_xgb >= 0.5).astype(int)

    metrics_lr = compute_metrics(y_test, pred_lr, proba_lr)
    metrics_xgb = compute_metrics(y_test, pred_xgb, proba_xgb)

    results["LR"].append(metrics_lr)
    results["XGBoost"].append(metrics_xgb)

    for name, (w_lr, w_xgb) in BLENDING_CONFIGS.items():
        proba_blend = w_lr * proba_lr + w_xgb * proba_xgb
        pred_blend = (proba_blend >= 0.5).astype(int)
        metrics_blend = compute_metrics(y_test, pred_blend, proba_blend)
        results[name].append(metrics_blend)

    # === ESCENARIO B: Threshold optimizado (interno, sin leakage) ===
    # Usar probabilidades del train para encontrar threshold óptimo
    proba_lr_train = lr.predict_proba(X_tr_proc)[:, 1]

    # Re-entrenar XGB en todo el train (sin early stopping split) para threshold
    xgb_full = XGBClassifier(**xgb_params)
    xgb_full.fit(X_tr_proc, y_train, verbose=False)
    proba_xgb_train = xgb_full.predict_proba(X_tr_proc)[:, 1]
    proba_xgb_test_full = xgb_full.predict_proba(X_te_proc)[:, 1]

    # Threshold tuning en train (CV interno para evitar overfitting al threshold)
    # Usamos las predicciones del train con LOO-like approach via CV interno
    # Para simplificar: usamos OOF del train split interno como proxy
    proba_lr_val = lr.predict_proba(X_tr_proc[val_int])[:, 1]
    proba_xgb_val = xgb_model.predict_proba(X_tr_proc[val_int])[:, 1]

    t_lr = find_optimal_threshold(y_train[val_int], proba_lr_val, metric="ba")
    t_xgb = find_optimal_threshold(y_train[val_int], proba_xgb_val, metric="ba")

    pred_lr_t = (proba_lr >= t_lr).astype(int)
    pred_xgb_t = (proba_xgb_test_full >= t_xgb).astype(int)

    results_threshold["LR"].append(compute_metrics(y_test, pred_lr_t, proba_lr))
    results_threshold["XGBoost"].append(compute_metrics(y_test, pred_xgb_t, proba_xgb_test_full))

    for name, (w_lr, w_xgb) in BLENDING_CONFIGS.items():
        proba_blend_t = w_lr * proba_lr + w_xgb * proba_xgb_test_full
        proba_blend_val = w_lr * proba_lr_val + w_xgb * proba_xgb_val
        t_blend = find_optimal_threshold(y_train[val_int], proba_blend_val, metric="ba")
        pred_blend_t = (proba_blend_t >= t_blend).astype(int)
        results_threshold[name].append(compute_metrics(y_test, pred_blend_t, proba_blend_t))

    if (fold_idx + 1) % 20 == 0:
        print(f"  Completados {fold_idx + 1}/{N_SPLITS * N_REPEATS} folds...")

print("\n✓ Evaluación completada.")

# ============================================================================
# 5. RESULTADOS
# ============================================================================


def summarize_results(results_dict, scenario_name):
    """Genera tabla resumen con media ± std."""
    print(f"\n{'=' * 70}")
    print(f"  {scenario_name}")
    print(f"{'=' * 70}")
    print(f"{'Modelo':<18} {'BA':>12} {'Recall':>12} {'Spec':>12} {'ROC AUC':>12} {'PR AUC':>12}")
    print("-" * 80)

    summary_rows = []
    for name, fold_results in results_dict.items():
        df_r = pd.DataFrame(fold_results)
        row = {}
        for metric in ["BA", "Recall", "Specificity", "ROC_AUC", "PR_AUC"]:
            mean = df_r[metric].mean()
            std = df_r[metric].std()
            row[metric] = f"{mean:.3f}±{std:.3f}"
        summary_rows.append((name, row))

        print(f"{name:<18} {row['BA']:>12} {row['Recall']:>12} {row['Specificity']:>12} {row['ROC_AUC']:>12} {row['PR_AUC']:>12}")

    print()
    return summary_rows


# Escenario A
summary_a = summarize_results(results, "ESCENARIO A — Threshold fijo (0.5)")

# Escenario B
summary_b = summarize_results(results_threshold, "ESCENARIO B — Threshold optimizado (BA)")

# ============================================================================
# 6. ANÁLISIS COMPARATIVO DETALLADO
# ============================================================================

print("\n" + "=" * 70)
print("  ANÁLISIS COMPARATIVO")
print("=" * 70)

# Diferencias de BA entre modelos (paired, por fold)
ba_lr = [r["BA"] for r in results["LR"]]
ba_xgb = [r["BA"] for r in results["XGBoost"]]
ba_blend = [r["BA"] for r in results["Blend 0.7/0.3"]]

diff_blend_lr = np.array(ba_blend) - np.array(ba_lr)
diff_xgb_lr = np.array(ba_xgb) - np.array(ba_lr)

print(f"\n--- Diferencia de BA (Escenario A, threshold=0.5) ---")
print(f"  Blend(0.7/0.3) - LR:  media={diff_blend_lr.mean():+.4f}, std={diff_blend_lr.std():.4f}")
print(f"  XGBoost - LR:         media={diff_xgb_lr.mean():+.4f}, std={diff_xgb_lr.std():.4f}")
print(f"  % folds donde Blend > LR: {(diff_blend_lr > 0).mean()*100:.1f}%")
print(f"  % folds donde Blend = LR: {(diff_blend_lr == 0).mean()*100:.1f}%")
print(f"  % folds donde Blend < LR: {(diff_blend_lr < 0).mean()*100:.1f}%")

# Escenario B
ba_lr_t = [r["BA"] for r in results_threshold["LR"]]
ba_blend_t = [r["BA"] for r in results_threshold["Blend 0.7/0.3"]]
diff_blend_lr_t = np.array(ba_blend_t) - np.array(ba_lr_t)

print(f"\n--- Diferencia de BA (Escenario B, threshold optimizado) ---")
print(f"  Blend(0.7/0.3) - LR:  media={diff_blend_lr_t.mean():+.4f}, std={diff_blend_lr_t.std():.4f}")
print(f"  % folds donde Blend > LR: {(diff_blend_lr_t > 0).mean()*100:.1f}%")

# Recall comparison
recall_lr = [r["Recall"] for r in results["LR"]]
recall_blend = [r["Recall"] for r in results["Blend 0.7/0.3"]]
diff_recall = np.array(recall_blend) - np.array(recall_lr)
print(f"\n--- Diferencia de Recall (Escenario A) ---")
print(f"  Blend(0.7/0.3) - LR:  media={diff_recall.mean():+.4f}, std={diff_recall.std():.4f}")

# ROC AUC comparison (probability quality)
auc_lr = [r["ROC_AUC"] for r in results["LR"]]
auc_blend = [r["ROC_AUC"] for r in results["Blend 0.7/0.3"]]
diff_auc = np.array(auc_blend) - np.array(auc_lr)
print(f"\n--- Diferencia de ROC AUC (calidad probabilística) ---")
print(f"  Blend(0.7/0.3) - LR:  media={diff_auc.mean():+.4f}, std={diff_auc.std():.4f}")

# ============================================================================
# 7. ESTABILIDAD: VARIANZA ENTRE REPETICIONES
# ============================================================================

print(f"\n{'=' * 70}")
print("  ESTABILIDAD (varianza inter-repetición)")
print(f"{'=' * 70}")

# Agrupar por repetición
ba_by_rep = {"LR": [], "XGBoost": [], "Blend 0.7/0.3": []}
for rep in range(N_REPEATS):
    start = rep * N_SPLITS
    end = start + N_SPLITS
    for name in ba_by_rep:
        fold_metrics = [results[name][i]["BA"] for i in range(start, end)]
        ba_by_rep[name].append(np.mean(fold_metrics))

for name in ba_by_rep:
    vals = ba_by_rep[name]
    print(f"  {name:<18}: BA por rep = {np.mean(vals):.3f} ± {np.std(vals):.4f} (rango: {np.min(vals):.3f}–{np.max(vals):.3f})")

# ============================================================================
# 8. TABLA FINAL RESUMEN (FORMATO PAPER)
# ============================================================================

print(f"\n{'=' * 70}")
print("  TABLA RESUMEN PARA TFM")
print(f"{'=' * 70}")
print()
print("┌─────────────────────┬──────────────┬──────────────┬──────────────┬──────────────┐")
print("│ Modelo              │ BA           │ Recall (AD)  │ Specificity  │ ROC AUC      │")
print("├─────────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤")

for scenario_name, res_dict in [("A (t=0.5)", results), ("B (t opt)", results_threshold)]:
    for model_name in ["LR", "XGBoost", "Blend 0.7/0.3"]:
        df_r = pd.DataFrame(res_dict[model_name])
        ba_str = f"{df_r['BA'].mean():.3f}±{df_r['BA'].std():.3f}"
        rec_str = f"{df_r['Recall'].mean():.3f}±{df_r['Recall'].std():.3f}"
        spec_str = f"{df_r['Specificity'].mean():.3f}±{df_r['Specificity'].std():.3f}"
        auc_str = f"{df_r['ROC_AUC'].mean():.3f}±{df_r['ROC_AUC'].std():.3f}"
        label = f"{model_name} ({scenario_name})"
        print(f"│ {label:<19} │ {ba_str:<12} │ {rec_str:<12} │ {spec_str:<12} │ {auc_str:<12} │")
    print("├─────────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤")

print("└─────────────────────┴──────────────┴──────────────┴──────────────┴──────────────┘")

# ============================================================================
# 9. INTERPRETACIÓN Y CONCLUSIONES
# ============================================================================

print(f"\n{'=' * 70}")
print("  INTERPRETACIÓN Y CONCLUSIONES")
print(f"{'=' * 70}")

ba_lr_mean = np.mean(ba_lr)
ba_blend_mean = np.mean(ba_blend)
ba_xgb_mean = np.mean(ba_xgb)
improvement = ba_blend_mean - ba_lr_mean

print(f"""
1. ¿El blending aporta mejora real frente a LR sola?
   → Diferencia media en BA: {improvement:+.4f} ({improvement*100:+.2f} puntos porcentuales)
   → {'SÍ, mejora modesta pero consistente' if improvement > 0.005 and (diff_blend_lr > 0).mean() > 0.55 else 'MARGINAL/NO SIGNIFICATIVA' if abs(improvement) < 0.005 else 'Mejora pero poco consistente'}

2. ¿La mejora es consistente o marginal/inestable?
   → El blend supera a LR en {(diff_blend_lr > 0).mean()*100:.0f}% de los folds
   → Std de la diferencia: {diff_blend_lr.std():.4f}
   → {'Consistente' if (diff_blend_lr > 0).mean() > 0.6 and diff_blend_lr.std() < 0.05 else 'Inestable/marginal'}

3. ¿Mejora especialmente el recall de la clase positiva (AD)?
   → Diferencia media en Recall: {diff_recall.mean():+.4f}
   → {'Sí, mejora recall' if diff_recall.mean() > 0.005 else 'No mejora significativamente recall'}

4. ¿El boosting por sí solo empeora la robustez respecto a LR?
   → BA XGBoost: {ba_xgb_mean:.3f} vs LR: {ba_lr_mean:.3f} (diff: {ba_xgb_mean - ba_lr_mean:+.4f})
   → Std XGBoost: {np.std(ba_xgb):.4f} vs LR: {np.std(ba_lr):.4f}
   → {'Sí, XGBoost es menos robusto' if np.std(ba_xgb) > np.std(ba_lr) * 1.2 else 'Robustez comparable'}

5. ¿Tiene sentido metodológico este ensemble?
   → Con señal predominantemente lineal y N=148, el blending actúa como regularización suave.
   → El boosting ligero (depth=2) solo puede capturar interacciones de orden 2.
   → {'Razonable: captura no-linealidades sin comprometer la base lineal' if improvement > 0 else 'Poco justificado: no aporta sobre la base lineal'}

6. ¿Recomendación para el TFM?
""")

if improvement > 0.01 and (diff_blend_lr > 0).mean() > 0.6:
    print("   → RECOMENDADO: El blending aporta mejora consistente.")
    print("     Incluir como modelo complementario con justificación de que captura")
    print("     interacciones suaves que LR no puede modelar.")
elif improvement > 0.005:
    print("   → OPCIONAL: Mejora marginal. Se puede mencionar como exploración,")
    print("     pero LR + threshold tuning probablemente sea suficiente y más interpretable.")
else:
    print("   → NO RECOMENDADO: El blending no aporta mejora sustancial.")
    print("     Mejor quedarse con LR + threshold tuning, que es más simple,")
    print("     interpretable y tiene rendimiento equivalente.")
    print()
    print("   Alternativas a considerar:")
    print("   • LR con threshold optimizado (ya implementado en el sistema cascada)")
    print("   • LR calibrada (Platt/isotónica) para mejorar probabilidades")
    print("   • Stacking con meta-modelo lineal (poco probable que mejore con N=148)")

print()
print("=" * 70)
print("FIN DEL EXPERIMENTO")
print("=" * 70)
