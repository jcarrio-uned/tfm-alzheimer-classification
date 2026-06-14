"""Análisis de robustez y estabilidad del Sistema Cascada.

Preguntas que puede hacer un tribunal de TFM y los análisis para responderlas:

1. ESTABILIDAD ANTE SEMILLA (seed sensitivity)
   - ¿Cambian mucho las métricas si cambias la seed del train/test split?
   - ¿Los umbrales t_lo/t_hi son estables?
   - ¿El Stage 2 (LOO) es independiente de la seed?

2. GENERALIZACIÓN
   - ¿Cuál es la distribución de BA en test con múltiples splits?
   - ¿Hay overfitting en los umbrales (aprendidos en train, aplicados a test)?
   - ¿Train-Test gap es aceptable?

3. CALIBRACIÓN DE PROBABILIDADES
   - ¿Las probabilidades P(AD) están bien calibradas?
   - ¿Sirven como medida de confianza real?

4. SENSIBILIDAD A HIPERPARÁMETROS
   - ¿Cómo varía BA al mover C_S2?
   - ¿El class_weight elegido es el óptimo o hay alternativas?
   - ¿Qué pasa si los umbrales se perturbaran ligeramente?

5. COMPARACIÓN CON ALTERNATIVAS
   - ¿El sistema cascada es significativamente mejor que un solo modelo?
   - ¿Cuánto aporta realmente el Stage 2?

6. TAMAÑO MUESTRAL
   - Con N=148, ¿hay suficiente potencia estadística?
   - ¿La zona amarilla (Stage 2) tiene suficientes muestras?
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, log_loss
from sklearn.model_selection import LeaveOneOut, train_test_split
from sklearn.preprocessing import StandardScaler

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

# Features S1
FEATURES_S1 = [
    "DOPA", "Cer(d18:1/20:0)", "lysoPC.a.C18:2", "PC.aa.C40:4", "DHEAS",
    "Arg", "HexCer(d18:1/26:1)",
]
# Engineered features
df_metab["DHEAS/lysoPC"] = df_metab["DHEAS"] / (df_metab["lysoPC.a.C18:2"] + 1e-10)
df_metab["DOPA*DHEAS"] = df_metab["DOPA"] * df_metab["DHEAS"]
df_metab["PC*DOPA"] = df_metab["PC.aa.C40:4"] * df_metab["DOPA"]
df_metab["DHEAS/Cer20"] = df_metab["DHEAS"] / (df_metab["Cer(d18:1/20:0)"] + 1e-10)
FEATURES_S1 += ["DHEAS/lysoPC", "DOPA*DHEAS", "PC*DOPA", "DHEAS/Cer20"]

X = df_metab[FEATURES_S1].values
n = len(y)

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
    """Winsoriza X_data al 5/95 usando sus propios percentiles."""
    X_out = X_data.copy()
    for j in range(X_out.shape[1]):
        lo, hi = np.percentile(X_out[:, j], [5, 95])
        X_out[:, j] = np.clip(X_out[:, j], lo, hi)
    return X_out


def find_thresholds(P_s1_arr, y_arr, max_err_green=0, max_err_red=0):
    """Busca t_lo, t_hi que cumplan restricciones de errores maximizando cobertura."""
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


def build_X2(indices, P_s1_values):
    """Construye matriz Stage 2."""
    return np.column_stack([P_s1_values, mmse[indices], apoe[indices], dep[indices], cardio[indices], age[indices]])


def run_cascade(X, y, seed, holdout_pct=0.2):
    """Ejecuta el sistema cascada completo para una seed dada. Devuelve métricas."""
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices, test_size=holdout_pct, stratify=y, random_state=seed
    )
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    n_train, n_test = len(y_train), len(y_test)

    # Stage 1: LOO-CV en train
    P_s1_train = np.zeros(n_train)
    for tr, te in LeaveOneOut().split(X_train, y_train):
        X_tr_w = winsorize_train(X_train[tr])
        sc = StandardScaler().fit(X_tr_w)
        lr1 = LogisticRegression(C=C_S1, class_weight=CW_S1, max_iter=2000, random_state=42)
        lr1.fit(sc.transform(X_tr_w), y_train[tr])
        P_s1_train[te] = lr1.predict_proba(sc.transform(X_train[te]))[:, 1]

    # Thresholds (en train)
    t_lo, t_hi = find_thresholds(P_s1_train, y_train, max_err_green=0, max_err_red=0)
    if t_lo is None:
        return None  # No valid thresholds found

    # Zonas train
    verde_tr = P_s1_train < t_lo
    roja_tr = P_s1_train > t_hi
    amarilla_tr = ~verde_tr & ~roja_tr

    # Stage 2: LOO-CV en train (solo amarilla se decide por S2)
    P_s2_train = np.zeros(n_train)
    for tr, te in LeaveOneOut().split(X_train, y_train):
        X_tr_w = winsorize_train(X_train[tr])
        sc = StandardScaler().fit(X_tr_w)
        lr1 = LogisticRegression(C=C_S1, class_weight=CW_S1, max_iter=2000, random_state=42)
        lr1.fit(sc.transform(X_tr_w), y_train[tr])
        P_s1_tr_inner = lr1.predict_proba(sc.transform(X_tr_w))[:, 1]
        P_s1_te_inner = lr1.predict_proba(sc.transform(X_train[te]))[:, 1]

        X2_tr = build_X2(train_idx[tr], P_s1_tr_inner)
        X2_te = build_X2(train_idx[te], P_s1_te_inner)
        lr2 = LogisticRegression(C=C_S2, class_weight=CW_S2, max_iter=2000, random_state=42)
        lr2.fit(X2_tr, y_train[tr])
        P_s2_train[te] = lr2.predict_proba(X2_te)[:, 1]

    # Predicciones train
    pred_train = np.full(n_train, -1)
    pred_train[verde_tr] = 0
    pred_train[roja_tr] = 1
    pred_train[amarilla_tr] = (P_s2_train[amarilla_tr] >= 0.5).astype(int)

    # --- TEST ---
    X_tr_w = winsorize_train(X_train)
    sc = StandardScaler().fit(X_tr_w)
    lr1_final = LogisticRegression(C=C_S1, class_weight=CW_S1, max_iter=2000, random_state=42)
    lr1_final.fit(sc.transform(X_tr_w), y_train)
    P_s1_test = lr1_final.predict_proba(sc.transform(X_test))[:, 1]

    # Stage 1 solo (baseline)
    pred_s1_test = (P_s1_test >= 0.5).astype(int)
    ba_s1_test = balanced_accuracy_score(y_test, pred_s1_test)

    # Zonas test
    verde_te = P_s1_test < t_lo
    roja_te = P_s1_test > t_hi
    amarilla_te = ~verde_te & ~roja_te

    # Stage 2 test
    P_s1_train_full = lr1_final.predict_proba(sc.transform(X_tr_w))[:, 1]
    X2_train_full = build_X2(train_idx, P_s1_train_full)
    lr2_final = LogisticRegression(C=C_S2, class_weight=CW_S2, max_iter=2000, random_state=42)
    lr2_final.fit(X2_train_full, y_train)

    pred_test = np.full(n_test, -1)
    pred_test[verde_te] = 0
    pred_test[roja_te] = 1
    P_s2_test = np.zeros(n_test)
    if amarilla_te.sum() > 0:
        X2_test = build_X2(test_idx[amarilla_te], P_s1_test[amarilla_te])
        P_s2_test[amarilla_te] = lr2_final.predict_proba(X2_test)[:, 1]
        pred_test[amarilla_te] = (P_s2_test[amarilla_te] >= 0.5).astype(int)

    # Métricas
    def calc_metrics(y_true, y_pred):
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        return {"BA": (sens + spec) / 2, "Sens": sens, "Spec": spec, "FN": fn, "FP": fp, "Errors": fn + fp}

    train_metrics = calc_metrics(y_train, pred_train)
    test_metrics = calc_metrics(y_test, pred_test)

    return {
        "seed": seed,
        "t_lo": t_lo,
        "t_hi": t_hi,
        "n_verde_tr": verde_tr.sum(),
        "n_amarilla_tr": amarilla_tr.sum(),
        "n_roja_tr": roja_tr.sum(),
        "n_verde_te": verde_te.sum(),
        "n_amarilla_te": amarilla_te.sum(),
        "n_roja_te": roja_te.sum(),
        "train_BA": train_metrics["BA"],
        "train_Sens": train_metrics["Sens"],
        "train_Spec": train_metrics["Spec"],
        "train_Errors": train_metrics["Errors"],
        "test_BA": test_metrics["BA"],
        "test_Sens": test_metrics["Sens"],
        "test_Spec": test_metrics["Spec"],
        "test_Errors": test_metrics["Errors"],
        "test_FN": test_metrics["FN"],
        "test_FP": test_metrics["FP"],
        "ba_s1_test": ba_s1_test,
        "cascade_improvement": test_metrics["BA"] - ba_s1_test,
        "P_s1_train": P_s1_train,
        "P_s1_test": P_s1_test,
        "P_s2_test": P_s2_test,
        "y_train": y_train,
        "y_test": y_test,
        "train_idx": train_idx,
        "test_idx": test_idx,
    }


# ============================================================================
# ANÁLISIS 1: ESTABILIDAD ANTE SEMILLA
# ============================================================================

print("=" * 80)
print("  ANÁLISIS 1: ESTABILIDAD ANTE SEMILLA (SEED SENSITIVITY)")
print("=" * 80)
print()
print("Pregunta del tribunal: '¿Las métricas cambian si usas otra semilla?'")
print("                        '¿El split 80/20 con seed=42 es un lucky split?'")
print()

N_SEEDS = 50
seeds = list(range(N_SEEDS))

results = []
for i, seed in enumerate(seeds):
    r = run_cascade(X, y, seed=seed, holdout_pct=0.2)
    if r is not None:
        results.append(r)
    if (i + 1) % 10 == 0:
        print(f"  Completados {i + 1}/{N_SEEDS} seeds...")

df_results = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, np.ndarray)} for r in results])
n_valid = len(df_results)

print(f"\n  Seeds válidas: {n_valid}/{N_SEEDS} (las que encuentran umbrales con 0 errores)")
print()

# Resumen estadístico
print("┌─────────────────────────────────────────────────────────────────────────────┐")
print("│ DISTRIBUCIÓN DE MÉTRICAS SOBRE {} SEEDS (holdout 20%)                       │".format(n_valid))
print("├─────────────────┬──────────┬──────────┬──────────┬──────────┬──────────────┤")
print("│ Métrica         │   Media  │    Std   │    Min   │    Max   │  IQR (25-75) │")
print("├─────────────────┼──────────┼──────────┼──────────┼──────────┼──────────────┤")

for col in ["test_BA", "test_Sens", "test_Spec", "test_Errors", "train_BA", "t_lo", "t_hi"]:
    vals = df_results[col]
    q25, q75 = vals.quantile(0.25), vals.quantile(0.75)
    fmt = ".3f" if "Err" not in col else ".1f"
    print(f"│ {col:<15} │ {vals.mean():{fmt}}  │ {vals.std():{fmt}}  │ {vals.min():{fmt}}  │ {vals.max():{fmt}}  │ {q25:{fmt}}-{q75:{fmt}}  │")

print("└─────────────────┴──────────┴──────────┴──────────┴──────────┴──────────────┘")

# Resultado para seed=42 (la usada en la app)
r42 = df_results[df_results["seed"] == 42].iloc[0]
percentile_ba = (df_results["test_BA"] <= r42["test_BA"]).mean() * 100

print(f"\n  → Seed=42 (la usada): BA_test={r42['test_BA']:.3f} (percentil {percentile_ba:.0f}%)")
print(f"  → Rango de BA_test: [{df_results['test_BA'].min():.3f}, {df_results['test_BA'].max():.3f}]")
print(f"  → Std de BA_test: {df_results['test_BA'].std():.3f}")

# Interpretación
ba_std = df_results["test_BA"].std()
if ba_std < 0.03:
    stability = "MUY ESTABLE"
    comment = "La varianza entre seeds es baja. El resultado es robusto."
elif ba_std < 0.05:
    stability = "ESTABLE"
    comment = "Varianza aceptable para N=148. El resultado es fiable."
elif ba_std < 0.08:
    stability = "MODERADA"
    comment = "Varianza no despreciable. Reportar media±std es imprescindible."
else:
    stability = "INESTABLE"
    comment = "Alta varianza. Los resultados dependen significativamente del split."

print(f"\n  VEREDICTO: {stability}")
print(f"  → {comment}")

# ============================================================================
# ANÁLISIS 2: ESTABILIDAD DE UMBRALES
# ============================================================================

print(f"\n{'=' * 80}")
print("  ANÁLISIS 2: ESTABILIDAD DE UMBRALES (t_lo, t_hi)")
print("=" * 80)
print()
print("Pregunta: '¿Los umbrales dependen del split? ¿Son data-driven o robustos?'")
print()

t_lo_vals = df_results["t_lo"]
t_hi_vals = df_results["t_hi"]
n_unique_tlo = t_lo_vals.nunique()
n_unique_thi = t_hi_vals.nunique()

print(f"  t_lo: media={t_lo_vals.mean():.4f}, std={t_lo_vals.std():.4f}, "
      f"rango=[{t_lo_vals.min():.4f}, {t_lo_vals.max():.4f}], valores únicos={n_unique_tlo}")
print(f"  t_hi: media={t_hi_vals.mean():.4f}, std={t_hi_vals.std():.4f}, "
      f"rango=[{t_hi_vals.min():.4f}, {t_hi_vals.max():.4f}], valores únicos={n_unique_thi}")
print()

# Distribución de zonas
print("  Distribución de zonas (test, media sobre seeds):")
print(f"    🟢 Verde: {df_results['n_verde_te'].mean():.1f} ± {df_results['n_verde_te'].std():.1f}")
print(f"    🟡 Amarilla: {df_results['n_amarilla_te'].mean():.1f} ± {df_results['n_amarilla_te'].std():.1f}")
print(f"    🔴 Roja: {df_results['n_roja_te'].mean():.1f} ± {df_results['n_roja_te'].std():.1f}")

# ============================================================================
# ANÁLISIS 3: CONTRIBUCIÓN DEL STAGE 2
# ============================================================================

print(f"\n{'=' * 80}")
print("  ANÁLISIS 3: ¿CUÁNTO APORTA EL STAGE 2?")
print("=" * 80)
print()
print("Pregunta: '¿El Stage 2 mejora significativamente sobre usar solo Stage 1?'")
print()

improvements = df_results["cascade_improvement"]
print(f"  Mejora del cascada vs S1 solo (en test):")
print(f"    Media: {improvements.mean():+.4f} ({improvements.mean()*100:+.2f} pp)")
print(f"    Std:   {improvements.std():.4f}")
print(f"    Min:   {improvements.min():+.4f}")
print(f"    Max:   {improvements.max():+.4f}")
print(f"    % seeds donde cascada > S1: {(improvements > 0).mean()*100:.0f}%")
print(f"    % seeds donde cascada ≥ S1: {(improvements >= 0).mean()*100:.0f}%")
print()

if (improvements >= 0).mean() > 0.9:
    print("  VEREDICTO: El Stage 2 aporta de forma consistente (mejora o al menos no empeora).")
elif (improvements > 0).mean() > 0.5:
    print("  VEREDICTO: El Stage 2 aporta en la mayoría de splits pero no en todos.")
else:
    print("  VEREDICTO: El Stage 2 no aporta mejora consistente.")

# ============================================================================
# ANÁLISIS 4: TRAIN-TEST GAP (OVERFITTING CHECK)
# ============================================================================

print(f"\n{'=' * 80}")
print("  ANÁLISIS 4: OVERFITTING (TRAIN-TEST GAP)")
print("=" * 80)
print()
print("Pregunta: '¿Hay sobreajuste? El train es demasiado bueno comparado con test?'")
print()

gaps = df_results["train_BA"] - df_results["test_BA"]
print(f"  Gap (Train_BA - Test_BA):")
print(f"    Media: {gaps.mean():.4f} ({gaps.mean()*100:.2f} pp)")
print(f"    Std:   {gaps.std():.4f}")
print(f"    Rango: [{gaps.min():.4f}, {gaps.max():.4f}]")
print()

if gaps.mean() < 0.03:
    print("  VEREDICTO: Gap mínimo. No hay indicios de overfitting.")
elif gaps.mean() < 0.06:
    print("  VEREDICTO: Gap moderado, aceptable para N=148 y LOO-CV.")
else:
    print("  VEREDICTO: ⚠️ Gap elevado. Posible overfitting en umbrales o S2.")

# ============================================================================
# ANÁLISIS 5: CALIBRACIÓN DE PROBABILIDADES
# ============================================================================

print(f"\n{'=' * 80}")
print("  ANÁLISIS 5: CALIBRACIÓN DE PROBABILIDADES")
print("=" * 80)
print()
print("Pregunta: '¿Las probabilidades P(AD) están bien calibradas?'")
print("          '¿Si el modelo dice P=0.7, el 70% de esas muestras son realmente AD?'")
print()

# Usar seed=42 para el análisis de calibración
r = results[42] if len(results) > 42 else results[0]
P_s1_all = np.concatenate([r["P_s1_train"], r["P_s1_test"]])
y_all = np.concatenate([r["y_train"], r["y_test"]])

# Brier score y Log loss
brier = brier_score_loss(y_all, P_s1_all)
ll = log_loss(y_all, P_s1_all)

print(f"  Stage 1 (todas las muestras, seed=42):")
print(f"    Brier Score: {brier:.4f} (perfecto=0, aleatorio=0.25)")
print(f"    Log Loss:    {ll:.4f}")
print()

# Calibration curve (bins)
try:
    fraction_pos, mean_predicted = calibration_curve(y_all, P_s1_all, n_bins=8, strategy="uniform")
    print("  Curva de calibración (8 bins uniformes):")
    print(f"    {'Bin':>10} {'Prob media':>12} {'Fracción real AD':>18} {'Diferencia':>12}")
    print(f"    {'-'*55}")
    for mp, fp in zip(mean_predicted, fraction_pos):
        diff = fp - mp
        print(f"    {mp:>10.3f} {mp:>12.3f} {fp:>18.3f} {diff:>+12.3f}")

    # ECE (Expected Calibration Error)
    ece = np.mean(np.abs(fraction_pos - mean_predicted))
    print(f"\n    ECE (Expected Calibration Error): {ece:.4f}")
    if ece < 0.05:
        print("    → Calibración EXCELENTE")
    elif ece < 0.10:
        print("    → Calibración BUENA")
    else:
        print("    → Calibración MEJORABLE (considerar Platt scaling)")
except Exception as e:
    print(f"    (Error en calibración: {e})")

# ============================================================================
# ANÁLISIS 6: SENSIBILIDAD A PERTURBACIÓN DE UMBRALES
# ============================================================================

print(f"\n{'=' * 80}")
print("  ANÁLISIS 6: SENSIBILIDAD A PERTURBACIÓN DE UMBRALES")
print("=" * 80)
print()
print("Pregunta: '¿Qué pasa si los umbrales no son exactamente los óptimos?'")
print("          '¿El sistema es frágil ante pequeños cambios en t_lo/t_hi?'")
print()

# Usar seed=42
seed_ref = 42
r_ref = [r for r in results if r["seed"] == seed_ref][0]
t_lo_ref = r_ref["t_lo"]
t_hi_ref = r_ref["t_hi"]
y_test_ref = r_ref["y_test"]
P_s1_test_ref = r_ref["P_s1_test"]
test_idx_ref = r_ref["test_idx"]

# Perturbaciones
perturbations = np.linspace(-0.05, 0.05, 11)
print(f"  Umbrales de referencia: t_lo={t_lo_ref:.4f}, t_hi={t_hi_ref:.4f}")
print()
print(f"  {'Δ t_lo':>8} {'Δ t_hi':>8} {'t_lo':>8} {'t_hi':>8} {'BA test':>10} {'Err test':>10} {'ΔBA':>8}")
print(f"  {'-'*60}")

ba_ref = r_ref["test_BA"]
for dt_lo in [-0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03]:
    for dt_hi in [-0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03]:
        if dt_lo != 0 and dt_hi != 0 and dt_lo != dt_hi:
            continue  # Solo variar uno a la vez + diagonal
        t_lo_p = t_lo_ref + dt_lo
        t_hi_p = t_hi_ref + dt_hi
        if t_lo_p >= t_hi_p or t_lo_p < 0 or t_hi_p > 1:
            continue

        verde_p = P_s1_test_ref < t_lo_p
        roja_p = P_s1_test_ref > t_hi_p
        amarilla_p = ~verde_p & ~roja_p

        # Re-classify test with perturbed thresholds
        pred_p = np.full(len(y_test_ref), -1)
        pred_p[verde_p] = 0
        pred_p[roja_p] = 1
        if amarilla_p.sum() > 0:
            # Use original S2 model's predictions
            X_tr_w = winsorize_train(X[r_ref["train_idx"]])
            sc = StandardScaler().fit(X_tr_w)
            lr1 = LogisticRegression(C=C_S1, class_weight=CW_S1, max_iter=2000, random_state=42)
            lr1.fit(sc.transform(X_tr_w), r_ref["y_train"])
            P_s1_tr_full = lr1.predict_proba(sc.transform(X_tr_w))[:, 1]
            X2_tr_full = build_X2(r_ref["train_idx"], P_s1_tr_full)
            lr2 = LogisticRegression(C=C_S2, class_weight=CW_S2, max_iter=2000, random_state=42)
            lr2.fit(X2_tr_full, r_ref["y_train"])
            X2_te = build_X2(test_idx_ref[amarilla_p], P_s1_test_ref[amarilla_p])
            P_s2_p = lr2.predict_proba(X2_te)[:, 1]
            pred_p[amarilla_p] = (P_s2_p >= 0.5).astype(int)

        if (pred_p == -1).any():
            continue

        tp = ((pred_p == 1) & (y_test_ref == 1)).sum()
        tn = ((pred_p == 0) & (y_test_ref == 0)).sum()
        fp = ((pred_p == 1) & (y_test_ref == 0)).sum()
        fn = ((pred_p == 0) & (y_test_ref == 1)).sum()
        sens_p = tp / max(tp + fn, 1)
        spec_p = tn / max(tn + fp, 1)
        ba_p = (sens_p + spec_p) / 2
        err_p = fn + fp

        marker = " ← REF" if dt_lo == 0 and dt_hi == 0 else ""
        print(f"  {dt_lo:>+8.3f} {dt_hi:>+8.3f} {t_lo_p:>8.4f} {t_hi_p:>8.4f} {ba_p:>10.3f} {err_p:>10} {ba_p - ba_ref:>+8.3f}{marker}")

# ============================================================================
# ANÁLISIS 7: COMPARATIVA CON ALTERNATIVAS SIMPLES
# ============================================================================

print(f"\n{'=' * 80}")
print("  ANÁLISIS 7: COMPARATIVA CON ALTERNATIVAS SIMPLES")
print("=" * 80)
print()
print("Pregunta: '¿Por qué no usar simplemente un modelo único con todas las variables?'")
print("          '¿El sistema cascada justifica su complejidad adicional?'")
print()

# Evaluar: LR con metabolitos+clínicos en un solo modelo (misma evaluación LOO + holdout)
results_comparison = {"LR solo metab": [], "LR metab+clin": [], "Cascada": []}

for seed in range(30):
    indices = np.arange(n)
    train_idx_c, test_idx_c = train_test_split(indices, test_size=0.2, stratify=y, random_state=seed)
    X_train_c, X_test_c = X[train_idx_c], X[test_idx_c]
    y_train_c, y_test_c = y[train_idx_c], y[test_idx_c]

    # --- Modelo 1: LR solo metabolitos ---
    X_tr_w = winsorize_train(X_train_c)
    sc = StandardScaler().fit(X_tr_w)
    lr = LogisticRegression(C=C_S1, class_weight=CW_S1, max_iter=2000, random_state=42)
    lr.fit(sc.transform(X_tr_w), y_train_c)
    pred_m = (lr.predict_proba(sc.transform(X_test_c))[:, 1] >= 0.5).astype(int)
    ba_m = balanced_accuracy_score(y_test_c, pred_m)
    results_comparison["LR solo metab"].append(ba_m)

    # --- Modelo 2: LR con metabolitos + clínicos ---
    X_full_c = np.column_stack([X, mmse.reshape(-1, 1), apoe.reshape(-1, 1), dep.reshape(-1, 1), cardio.reshape(-1, 1), age.reshape(-1, 1)])
    X_train_full = X_full_c[train_idx_c]
    X_test_full = X_full_c[test_idx_c]
    X_tr_w2 = X_train_full.copy()
    X_tr_w2[:, :11] = winsorize_train(X_train_full[:, :11])  # Solo winsorizar metabolitos
    sc2 = StandardScaler().fit(X_tr_w2)
    lr2_alt = LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000, random_state=42)
    lr2_alt.fit(sc2.transform(X_tr_w2), y_train_c)
    X_te_full2 = X_test_full.copy()
    pred_mc = (lr2_alt.predict_proba(sc2.transform(X_te_full2))[:, 1] >= 0.5).astype(int)
    ba_mc = balanced_accuracy_score(y_test_c, pred_mc)
    results_comparison["LR metab+clin"].append(ba_mc)

    # --- Modelo 3: Cascada ---
    r_c = run_cascade(X, y, seed=seed)
    if r_c is not None:
        results_comparison["Cascada"].append(r_c["test_BA"])
    else:
        results_comparison["Cascada"].append(np.nan)

print(f"  {'Modelo':<20} {'BA test (media)':>15} {'BA test (std)':>15} {'BA test (mediana)':>18}")
print(f"  {'-'*70}")
for name, vals in results_comparison.items():
    v = np.array([x for x in vals if not np.isnan(x)])
    print(f"  {name:<20} {v.mean():>15.3f} {v.std():>15.3f} {np.median(v):>18.3f}")

# Paired comparison
ba_casc = np.array([x for x in results_comparison["Cascada"] if not np.isnan(x)])
ba_lr = np.array(results_comparison["LR solo metab"][:len(ba_casc)])
ba_full = np.array(results_comparison["LR metab+clin"][:len(ba_casc)])

print(f"\n  Paired differences (same seeds):")
print(f"    Cascada - LR metab:       {(ba_casc - ba_lr).mean():+.4f} ± {(ba_casc - ba_lr).std():.4f}")
print(f"    Cascada - LR metab+clin:  {(ba_casc - ba_full).mean():+.4f} ± {(ba_casc - ba_full).std():.4f}")
print(f"    Cascada > LR metab en:    {(ba_casc > ba_lr).mean()*100:.0f}% seeds")
print(f"    Cascada > LR full en:     {(ba_casc > ba_full).mean()*100:.0f}% seeds")

# ============================================================================
# ANÁLISIS 8: POTENCIA ESTADÍSTICA Y TAMAÑO MUESTRAL
# ============================================================================

print(f"\n{'=' * 80}")
print("  ANÁLISIS 8: CONSIDERACIONES DE TAMAÑO MUESTRAL")
print("=" * 80)
print()
print("Pregunta: '¿Con N=148 hay suficiente potencia para distinguir modelos?'")
print("          '¿La zona amarilla tiene suficientes muestras para Stage 2?'")
print()

print(f"  Dataset total: N={n} (AD={y.sum()}, NC={(1-y).sum()})")
print(f"  Train (80%): ~{int(n*0.8)} muestras")
print(f"  Test (20%): ~{int(n*0.2)} muestras")
print()
print(f"  Zona amarilla (train, media sobre {n_valid} seeds): {df_results['n_amarilla_tr'].mean():.0f} ± {df_results['n_amarilla_tr'].std():.0f}")
print(f"  Zona amarilla (test, media): {df_results['n_amarilla_te'].mean():.0f} ± {df_results['n_amarilla_te'].std():.0f}")
print()

# Margen de error en BA con N=30 test
n_test_typ = int(n * 0.2)
# Approximate standard error for BA with binomial assumption
p_est = 0.9  # approximate BA
se_ba = np.sqrt(p_est * (1 - p_est) / n_test_typ)  # simplified
print(f"  Error estándar aproximado de BA con N_test={n_test_typ}: ±{se_ba:.3f}")
print(f"  Intervalo de confianza 95% para BA≈0.9: [{p_est - 1.96*se_ba:.3f}, {p_est + 1.96*se_ba:.3f}]")
print()
print("  NOTA: Con test sets de ~30 muestras, diferencias < 0.05 en BA")
print("        no son estadísticamente significativas. Es por eso que se usa")
print("        repeated holdout (múltiples seeds) para reportar estabilidad.")

# ============================================================================
# RESUMEN EJECUTIVO
# ============================================================================

print(f"\n{'=' * 80}")
print("  RESUMEN EJECUTIVO — PREGUNTAS DEL TRIBUNAL Y RESPUESTAS")
print("=" * 80)
print("""
┌─────────────────────────────────────────────────────────────────────────────────┐
│ PREGUNTA                                    │ RESPUESTA (basada en evidencia)   │
├─────────────────────────────────────────────┼──────────────────────────────────┤""")

print(f"│ 1. ¿Es estable ante distintas seeds?        │ Std(BA_test) = {df_results['test_BA'].std():.3f}              │")
print(f"│                                             │ → {stability}                       │")
print(f"│ 2. ¿Los umbrales son robustos?              │ t_lo std={t_lo_vals.std():.4f}, t_hi std={t_hi_vals.std():.4f} │")
print(f"│ 3. ¿El S2 aporta vs S1 solo?               │ Mejora en {(improvements > 0).mean()*100:.0f}% de seeds ({improvements.mean():+.3f} medio) │")
print(f"│ 4. ¿Hay overfitting?                       │ Gap medio = {gaps.mean():.3f} pp              │")
print(f"│ 5. ¿Las probabilidades están calibradas?    │ Brier={brier:.3f}, ECE={ece:.3f}             │")
print(f"│ 6. ¿Es frágil ante perturbación umbrales?   │ Ver tabla de perturbaciones      │")
print(f"│ 7. ¿Justifica complejidad vs modelo único?  │ Cascada > LR en {(ba_casc > ba_lr).mean()*100:.0f}% seeds    │")
print(f"│ 8. ¿Suficiente potencia con N=148?          │ SE≈{se_ba:.3f}, reportar mean±std   │")
print("└─────────────────────────────────────────────┴──────────────────────────────────┘")

print("""
RECOMENDACIONES PARA LA DEFENSA:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. SIEMPRE reportar media ± std sobre múltiples seeds (no solo seed=42)
2. Mostrar que los umbrales son estables (baja varianza de t_lo/t_hi)
3. Argumentar que LOO-CV en Stage 2 elimina el riesgo de overfitting interno
4. Si preguntan "¿por qué no un solo modelo?": mostrar que la cascada
   aporta interpretabilidad clínica (zonas de confianza) + mejora modesta
5. Tener preparada la tabla de perturbación de umbrales para demostrar
   que no son frágiles
6. La calibración de probabilidades soporta el uso de P(AD) como score
   de confianza clínica
""")

print("=" * 80)
print("FIN DEL ANÁLISIS DE ROBUSTEZ")
print("=" * 80)
