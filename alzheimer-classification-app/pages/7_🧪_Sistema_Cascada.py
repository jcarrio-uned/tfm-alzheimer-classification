"""Módulo 7: Sistema de Clasificación en Cascada (Interactivo).

Página interactiva para explorar el sistema Stage 1 + Stage 2:
- Búsqueda de umbrales según errores tolerados en zonas roja/verde
- Selección de variables del Stage 2
- Optimización de C (regularización)
- Configuración de class_weight (balanceo)
- Análisis de multicolinealidad automático
- Métricas y gráficos completos
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import (
    LeaveOneOut,
    RepeatedStratifiedKFold,
    StratifiedShuffleSplit,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app_utils import (
    DEFAULT_S1_FEATURES,
    find_clinical_data,
    initialize_session_state,
    load_data,
)

# ============================================================================
st.set_page_config(page_title="Sistema Cascada Interactivo", page_icon="🧪", layout="wide")
initialize_session_state()

st.title("🧪 Sistema de Clasificación en Cascada — Exploración Interactiva")
st.markdown("""
Explora el sistema de dos etapas ortogonales: **Stage 1** (metabolómico) determina umbrales
de confianza, y **Stage 2** (clínico-integrador) resuelve la zona de incertidumbre.
Todos los parámetros son configurables para experimentar.
""")

# ============================================================================
# Cargar datos
# ============================================================================

df_metab = load_data()

# Cargar datos clínicos
clinical_path = find_clinical_data()
if clinical_path is None:
    st.error("No se encuentra el archivo clínico 'Segundo_Archivo_clean.xlsx'")
    st.stop()

df_clin = pd.read_excel(clinical_path)
df_clin_sel = df_clin[["ID", "MMSE", "APOE", "Depression", "Cardiovascular disorder", "Age [y]"]].copy()
df_clin_sel = df_clin_sel.rename(columns={"Age [y]": "Age_clin"})
df = df_metab.merge(df_clin_sel, on="ID")

# Features Stage 1 — parametrizado desde el modelo entrenado en page 4
if st.session_state.get("selected_features"):
    FEATURES_S1 = st.session_state["selected_features"]
else:
    # Fallback: 11 features del pipeline por defecto
    FEATURES_S1 = list(DEFAULT_S1_FEATURES)

# Obtener C y class_weight del modelo entrenado
if "model_config" in st.session_state:
    _mc = st.session_state["model_config"]
    C_S1 = _mc.get("C", 0.1)
    CW_S1 = _mc.get("class_weight", "balanced")
else:
    C_S1 = 0.1
    CW_S1 = "balanced"

# Asegurar que las features engineered existen en df
for f in FEATURES_S1:
    if f not in df.columns:
        # Intentar crear
        if f == "DHEAS/lysoPC" and "DHEAS" in df.columns and "lysoPC.a.C18:2" in df.columns:
            df["DHEAS/lysoPC"] = df["DHEAS"] / (df["lysoPC.a.C18:2"] + 1e-10)
        elif f == "DOPA*DHEAS" and "DOPA" in df.columns and "DHEAS" in df.columns:
            df["DOPA*DHEAS"] = df["DOPA"] * df["DHEAS"]
        elif f == "PC*DOPA" and "PC.aa.C40:4" in df.columns and "DOPA" in df.columns:
            df["PC*DOPA"] = df["PC.aa.C40:4"] * df["DOPA"]
        elif f == "DHEAS/Cer20" and "DHEAS" in df.columns and "Cer(d18:1/20:0)" in df.columns:
            df["DHEAS/Cer20"] = df["DHEAS"] / (df["Cer(d18:1/20:0)"] + 1e-10)

# Verificar que todas las features existen
missing_s1 = [f for f in FEATURES_S1 if f not in df.columns]
if missing_s1:
    st.error(f"❌ Features de Stage 1 no disponibles: {missing_s1}")
    st.info("Ve a **Selección de Variables** para configurar las features.")
    st.stop()

X = df[FEATURES_S1].values
y = (df["Group"] == "AD").astype(int).values
n = len(y)
ids = df["ID"].values

# Variables clínicas (todas disponibles)
age = df["Age_clin"].values.astype(float)
mmse = df["MMSE"].values.astype(float)
apoe_raw = df["APOE"]
apoe = apoe_raw.map({"e3/e3": 0, "e3/e4": 1, "e4/e4": 2, "e2/e3": 0, "e2/e4": 1}).fillna(0).values
dep = (df["Depression"] == "Si").astype(int).values
cardio = (df["Cardiovascular disorder"] == "Si").astype(int).values

CLINICAL_VARS = {
    "P(AD)_S1": None,  # Se calcula internamente
    "MMSE": mmse,
    "APOE (0/1/2)": apoe,
    "Depresión (0/1)": dep,
    "Cardiopatía (0/1)": cardio,
    "Edad": age,
}


# ============================================================================
# SECCIÓN 1: Stage 1 — Modelo Metabolómico
# ============================================================================

st.header("1️⃣ Stage 1: Modelo Metabolómico")

# Mostrar configuración del modelo
st.markdown(f"""
**Configuración Stage 1** (desde entrenamiento):
- Features: **{len(FEATURES_S1)}** variables
- Modelo: LogReg(C={C_S1}, class_weight='{CW_S1}')
- Winsorización: 5% (solo train)
""")

with st.expander("Ver features del Stage 1"):
    for i, f in enumerate(FEATURES_S1, 1):
        st.caption(f"{i}. {f}")

# --- Configuración de validación ---
st.subheader("⚙️ Configuración de Validación")

col_cv1, col_cv2, col_cv3 = st.columns(3)

with col_cv1:
    cv_method = st.selectbox(
        "Método de CV (train)",
        ["LOOCV", "RepeatedStratifiedKFold", "StratifiedShuffleSplit"],
        index=0,
        help="LOOCV: más conservador. RSKF/SSS: más rápido.",
    )

with col_cv2:
    if cv_method == "RepeatedStratifiedKFold":
        cv_n_folds = st.number_input("Folds", 3, 10, 5)
        cv_n_repeats = st.number_input("Repeticiones", 1, 20, 10)
    elif cv_method == "StratifiedShuffleSplit":
        cv_n_splits = st.number_input("N splits", 50, 500, 200, step=50)
        cv_test_size = st.slider("Test size (%)", 10, 40, 20, step=5)
    else:
        st.caption("LOO sobre train set")

with col_cv3:
    holdout_pct = st.slider("% Test holdout", 10, 40, 20, step=5, key="cascade_holdout")
    holdout_seed = st.number_input("Semilla split", 0, 999, 42, key="cascade_seed")


@st.cache_data
def run_stage1_cv(_X, _y, method, C, CW, holdout_pct, holdout_seed, **cv_params):
    """Ejecuta Stage 1 con el método de CV seleccionado. Siempre usa train/test split."""
    n_total = len(_y)

    # Train/test split (siempre)
    indices = np.arange(n_total)
    train_idx, test_idx = train_test_split(
        indices, test_size=holdout_pct / 100.0, stratify=_y, random_state=holdout_seed
    )
    X_train, X_test = _X[train_idx], _X[test_idx]
    y_train, y_test = _y[train_idx], _y[test_idx]

    # Obtener P(AD) para cada muestra de train mediante CV
    P_s1_train = np.zeros(len(y_train))

    if method == "LOOCV":
        for tr, te in LeaveOneOut().split(X_train, y_train):
            X_tr_w = _winsorize(X_train[tr])
            sc = StandardScaler()
            sc.fit(X_tr_w)
            lr = LogisticRegression(C=C, class_weight=CW, max_iter=2000, random_state=42)
            lr.fit(sc.transform(X_tr_w), y_train[tr])
            # Test fold: raw (sin winsorizar), solo escalado
            P_s1_train[te] = lr.predict_proba(sc.transform(X_train[te]))[:, 1]
    elif method == "RepeatedStratifiedKFold":
        cv = RepeatedStratifiedKFold(
            n_splits=cv_params.get("n_folds", 5),
            n_repeats=cv_params.get("n_repeats", 10),
            random_state=42,
        )
        counts = np.zeros(len(y_train))
        for tr, te in cv.split(X_train, y_train):
            X_tr_w = _winsorize(X_train[tr])
            sc = StandardScaler()
            sc.fit(X_tr_w)
            lr = LogisticRegression(C=C, class_weight=CW, max_iter=2000, random_state=42)
            lr.fit(sc.transform(X_tr_w), y_train[tr])
            P_s1_train[te] += lr.predict_proba(sc.transform(X_train[te]))[:, 1]
            counts[te] += 1
        P_s1_train /= counts
    else:  # StratifiedShuffleSplit
        cv = StratifiedShuffleSplit(
            n_splits=cv_params.get("n_splits", 200),
            test_size=cv_params.get("test_size", 0.2),
            random_state=42,
        )
        counts = np.zeros(len(y_train))
        for tr, te in cv.split(X_train, y_train):
            X_tr_w = _winsorize(X_train[tr])
            sc = StandardScaler()
            sc.fit(X_tr_w)
            lr = LogisticRegression(C=C, class_weight=CW, max_iter=2000, random_state=42)
            lr.fit(sc.transform(X_tr_w), y_train[tr])
            P_s1_train[te] += lr.predict_proba(sc.transform(X_train[te]))[:, 1]
            counts[te] += 1
        P_s1_train /= np.maximum(counts, 1)

    # Entrenar en todo train y predecir test (test NO winsorizado)
    X_tr_w = _winsorize(X_train)
    sc = StandardScaler()
    sc.fit(X_tr_w)
    lr = LogisticRegression(C=C, class_weight=CW, max_iter=2000, random_state=42)
    lr.fit(sc.transform(X_tr_w), y_train)
    P_s1_test = lr.predict_proba(sc.transform(X_test))[:, 1]

    return P_s1_train, P_s1_test, train_idx, test_idx, y_train, y_test


def _winsorize(X_data):
    """Winsoriza X_data usando sus propios percentiles 5/95. Solo para datos de TRAIN."""
    X_out = X_data.copy()
    for j in range(X_out.shape[1]):
        lo, hi = np.percentile(X_out[:, j], [5, 95])
        X_out[:, j] = np.clip(X_out[:, j], lo, hi)
    return X_out


# Build CV params
_cv_params = {}
if cv_method == "RepeatedStratifiedKFold":
    _cv_params = {"n_folds": cv_n_folds, "n_repeats": cv_n_repeats}
elif cv_method == "StratifiedShuffleSplit":
    _cv_params = {"n_splits": cv_n_splits, "test_size": cv_test_size / 100.0}

with st.spinner(f"Ejecutando Stage 1 ({cv_method})..."):
    P_s1_train, P_s1_test, train_idx, test_idx, y_train, y_test = run_stage1_cv(
        X,
        y,
        cv_method,
        C_S1,
        CW_S1,
        holdout_pct,
        holdout_seed,
        **_cv_params,
    )

# Usar P_s1 del TRAIN para buscar umbrales, del TEST para evaluar
P_s1 = P_s1_train  # Para búsqueda de umbrales (en train)

# Métricas Stage 1 en train (CV)
pred_s1_train = (P_s1_train >= 0.5).astype(int)
ba_s1_train = balanced_accuracy_score(y_train, pred_s1_train)
sens_s1_train = ((pred_s1_train == 1) & (y_train == 1)).sum() / max((y_train == 1).sum(), 1)
spec_s1_train = ((pred_s1_train == 0) & (y_train == 0)).sum() / max((y_train == 0).sum(), 1)

fn_s1 = ((pred_s1_train == 0) & (y_train == 1)).sum()
fp_s1 = ((pred_s1_train == 1) & (y_train == 0)).sum()

# Stage 1 en test
pred_s1_test = (P_s1_test >= 0.5).astype(int)
ba_s1_test = balanced_accuracy_score(y_test, pred_s1_test)

col1, col2, col3, col4 = st.columns(4)
col1.metric("BA S1 (Train CV)", f"{ba_s1_train:.3f}")
col2.metric("BA S1 (Test)", f"{ba_s1_test:.3f}")
col3.metric("Sens (Train)", f"{sens_s1_train:.3f}")
col4.metric("Spec (Train)", f"{spec_s1_train:.3f}")

# Override variables for downstream sections
n_train = len(y_train)
y_for_thresh = y_train
ids_train = ids[train_idx]

st.divider()

# ============================================================================
# SECCIÓN 2: Búsqueda Interactiva de Umbrales
# ============================================================================

st.header("2️⃣ Búsqueda de Umbrales por Errores Tolerados")
st.markdown("""
Configura cuántos errores estás dispuesto a aceptar en las zonas de alta confianza (verde y roja).
El sistema busca los umbrales óptimos que cumplan esas restricciones, **maximizando la cobertura**
(pacientes resueltos sin necesidad del Stage 2).
""")

col1, col2 = st.columns(2)
with col1:
    max_err_verde = st.number_input(
        "🟢 Máx errores zona verde (AD perdidos)",
        min_value=0,
        max_value=10,
        value=0,
        step=1,
        help="Pacientes AD clasificados como NC directo. 0 = certeza absoluta.",
    )
with col2:
    max_err_roja = st.number_input(
        "🔴 Máx errores zona roja (NC en roja)",
        min_value=0,
        max_value=10,
        value=0,
        step=1,
        help="Pacientes NC clasificados como AD directo. 0 = certeza absoluta.",
    )


@st.cache_data
def find_thresholds(P_s1_arr, y_arr, max_err_green, max_err_red):
    """Busca umbrales t_lo y t_hi que cumplan restricciones de errores.

    Maximiza la cobertura (pacientes fuera de zona amarilla).
    """
    # Generar candidatos ordenados
    probs_sorted = np.sort(np.unique(P_s1_arr))
    # Insertar bordes
    candidates = np.concatenate([[0.0], probs_sorted, [1.0]])

    best_t_lo = None
    best_t_hi = None
    best_coverage = -1

    for t_lo in candidates:
        # Errores en verde: pacientes AD con P < t_lo
        err_green = ((P_s1_arr < t_lo) & (y_arr == 1)).sum()
        if err_green > max_err_green:
            continue

        for t_hi in candidates:
            if t_hi <= t_lo:
                continue
            # Errores en roja: pacientes NC con P > t_hi
            err_red = ((P_s1_arr > t_hi) & (y_arr == 0)).sum()
            if err_red > max_err_red:
                continue

            # Cobertura: pacientes fuera de amarilla
            coverage = ((P_s1_arr < t_lo) | (P_s1_arr > t_hi)).sum()
            if coverage > best_coverage:
                best_coverage = coverage
                best_t_lo = t_lo
                best_t_hi = t_hi

    return best_t_lo, best_t_hi


t_lo, t_hi = find_thresholds(P_s1, y_for_thresh, max_err_verde, max_err_roja)

if t_lo is None or t_hi is None:
    st.error("No se encontraron umbrales que cumplan las restricciones. Aumenta los errores tolerados.")
    st.stop()

# Calcular zonas (en train, para visualización de umbrales)
verde = P_s1 < t_lo
roja = P_s1 > t_hi
amarilla = ~verde & ~roja

err_verde_real = ((verde) & (y_for_thresh == 1)).sum()
err_roja_real = ((roja) & (y_for_thresh == 0)).sum()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("t_lo (umbral inferior)", f"{t_lo:.4f}")
    st.caption(f"🟢 Verde: {verde.sum()} pacientes, {err_verde_real} errores")
with col2:
    st.metric("Zona solapamiento", f"{t_hi - t_lo:.4f}")
    st.caption(f"🟡 Amarilla: {amarilla.sum()} pacientes → Stage 2")
with col3:
    st.metric("t_hi (umbral superior)", f"{t_hi:.4f}")
    st.caption(f"🔴 Roja: {roja.sum()} pacientes, {err_roja_real} errores")

# Tabla de sweep completo
with st.expander("📊 Ver tabla de combinaciones de errores tolerados"):
    sweep_results = []
    for eg in range(6):
        for er in range(6):
            tl, th = find_thresholds(P_s1, y_for_thresh, eg, er)
            if tl is not None:
                v = (P_s1 < tl).sum()
                r = (P_s1 > th).sum()
                a = n_train - v - r
                sweep_results.append(
                    {
                        "Err verde": eg,
                        "Err roja": er,
                        "t_lo": f"{tl:.4f}",
                        "t_hi": f"{th:.4f}",
                        "N verde": v,
                        "N amarilla": a,
                        "N roja": r,
                        "Cobertura": f"{(v + r) / n_train * 100:.1f}%",
                    }
                )
    st.dataframe(pd.DataFrame(sweep_results), hide_index=True, use_container_width=True)

# Gráfico de distribuciones con umbrales
fig = go.Figure()
fig.add_trace(
    go.Histogram(
        x=P_s1[y_for_thresh == 0],
        name="NC",
        marker_color="rgba(39,174,96,0.6)",
        nbinsx=25,
        histnorm="probability density",
    )
)
fig.add_trace(
    go.Histogram(
        x=P_s1[y_for_thresh == 1],
        name="AD",
        marker_color="rgba(231,76,60,0.6)",
        nbinsx=25,
        histnorm="probability density",
    )
)
fig.add_vline(
    x=t_lo, line_dash="dash", line_color="green", annotation_text=f"t_lo={t_lo:.3f}", annotation_position="top left"
)
fig.add_vline(
    x=t_hi, line_dash="dash", line_color="red", annotation_text=f"t_hi={t_hi:.3f}", annotation_position="top right"
)
fig.add_vrect(
    x0=0,
    x1=t_lo,
    fillcolor="green",
    opacity=0.08,
    annotation_text=f"NC directo ({verde.sum()})",
    annotation_position="top left",
)
fig.add_vrect(
    x0=t_hi,
    x1=1,
    fillcolor="red",
    opacity=0.08,
    annotation_text=f"AD directo ({roja.sum()})",
    annotation_position="top right",
)
fig.add_vrect(
    x0=t_lo,
    x1=t_hi,
    fillcolor="yellow",
    opacity=0.05,
    annotation_text=f"→ Stage 2 ({amarilla.sum()})",
    annotation_position="top",
)
fig.update_layout(
    title="Distribución P(AD)_S1 con umbrales configurados",
    xaxis_title="P(AD) Stage 1",
    yaxis_title="Densidad",
    barmode="overlay",
    height=400,
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ============================================================================
# SECCIÓN 3: Stage 2 — Configuración Interactiva
# ============================================================================

st.header("3️⃣ Stage 2: Modelo Clínico-Integrador")
st.markdown("""
Configura las variables, la regularización y el balanceo del Stage 2.
Solo se aplica a pacientes en **zona amarilla**.
""")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Variables del Stage 2")
    selected_vars = st.multiselect(
        "Selecciona los regresores",
        options=list(CLINICAL_VARS.keys()),
        default=list(CLINICAL_VARS.keys()),
        help="P(AD)_S1 siempre se incluye como puente entre etapas.",
    )
    # Asegurar que P(AD)_S1 siempre está
    if "P(AD)_S1" not in selected_vars:
        selected_vars = ["P(AD)_S1", *selected_vars]
        st.info("P(AD)_S1 añadido automáticamente (es el puente entre etapas).")

with col_right:
    st.subheader("Hiperparámetros")

    # Regularización
    regularization_mode = st.radio(
        "Regularización (C)",
        ["Valor fijo", "Búsqueda automática"],
        horizontal=True,
        help="Fijo: usa el C seleccionado. Automática: prueba un rango y selecciona el mejor.",
    )

    if regularization_mode == "Valor fijo":
        C_value = st.select_slider(
            "C (inverso de regularización)",
            options=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0],
            value=10.0,
            help="Mayor C = menos regularización. C=10 es el óptimo validado.",
        )
    else:
        C_range = st.slider(
            "Rango de C a explorar (log10)",
            min_value=-2.0,
            max_value=2.0,
            value=(-1.0, 2.0),
            step=0.5,
            help="Se probarán valores en este rango logarítmico.",
        )

    # Balanceo
    balance_mode = st.selectbox(
        "class_weight (balanceo)",
        [
            "Ninguno {0:1, 1:1}",
            "Balanced (proporcional a N)",
            "Priorizar AD {0:1, 1:2}",
            "Muy agresivo {0:1, 1:3}",
            "Priorizar NC {0:2, 1:1}",
            "Personalizado",
        ],
        index=2,
        help="Controla el trade-off FN/FP. Priorizar AD = menos FN.",
    )

    if balance_mode == "Personalizado":
        col_w0, col_w1 = st.columns(2)
        with col_w0:
            w0 = st.number_input("Peso clase 0 (NC)", 1, 10, 1)
        with col_w1:
            w1 = st.number_input("Peso clase 1 (AD)", 1, 10, 2)
        class_weight = {0: w0, 1: w1}
    else:
        weight_map = {
            "Ninguno {0:1, 1:1}": {0: 1, 1: 1},
            "Balanced (proporcional a N)": "balanced",
            "Priorizar AD {0:1, 1:2}": {0: 1, 1: 2},
            "Muy agresivo {0:1, 1:3}": {0: 1, 1: 3},
            "Priorizar NC {0:2, 1:1}": {0: 2, 1: 1},
        }
        class_weight = weight_map[balance_mode]

st.divider()

# ============================================================================
# SECCIÓN 4: Ejecución del Pipeline
# ============================================================================

st.header("4️⃣ Ejecución del Sistema Completo")

st.info(
    f"📊 Umbrales se aprenden en train ({len(y_train)} muestras). Evaluación final en test ({len(y_test)} muestras)."
)

run_button = st.button("▶️ Ejecutar Pipeline", type="primary", use_container_width=True)


# --- Preparar columnas del Stage 2 (definido a nivel módulo para reutilizar) ---
def build_X2(indices, P_s1_values):
    """Construye la matriz X2 para los índices dados (indices relativos al df completo)."""
    cols = []
    for var in selected_vars:
        if var == "P(AD)_S1":
            cols.append(P_s1_values)
        elif var == "MMSE":
            cols.append(mmse[indices])
        elif var == "APOE (0/1/2)":
            cols.append(apoe[indices])
        elif var == "Depresión (0/1)":
            cols.append(dep[indices])
        elif var == "Cardiopatía (0/1)":
            cols.append(cardio[indices])
        elif var == "Edad":
            cols.append(age[indices])
    return np.column_stack(cols)


# C_value: si es modo fijo ya está definido; si es automático, usar fallback hasta que se ejecute pipeline
if regularization_mode == "Búsqueda automática" and "C_value" not in dir():
    C_value = 10.0  # Default hasta que el pipeline lo calcule


if run_button:
    # --- Búsqueda de C si es automática ---
    if regularization_mode == "Búsqueda automática":
        st.subheader("🔍 Búsqueda de C óptimo (en train)")
        C_candidates = np.logspace(C_range[0], C_range[1], 15)

        c_results = []
        progress_bar = st.progress(0, text="Buscando C óptimo...")

        X_train_s1 = X[train_idx]
        y_train_s1 = y[train_idx]

        for ci, C_test in enumerate(C_candidates):
            progress_bar.progress((ci + 1) / len(C_candidates), text=f"Probando C={C_test:.3f}...")
            P_s2_c = np.zeros(n_train)

            for tr, te in LeaveOneOut().split(X_train_s1, y_train_s1):
                # Stage 1
                X_tr_w = _winsorize(X_train_s1[tr])
                sc = StandardScaler()
                sc.fit(X_tr_w)
                lr1 = LogisticRegression(C=C_S1, class_weight=CW_S1, max_iter=2000, random_state=42)
                lr1.fit(sc.transform(X_tr_w), y_train_s1[tr])
                P_s1_tr = lr1.predict_proba(sc.transform(X_tr_w))[:, 1]
                P_s1_te_val = lr1.predict_proba(sc.transform(X_train_s1[te]))[:, 1]

                # Stage 2
                X2_tr = build_X2(train_idx[tr], P_s1_tr)
                X2_te = build_X2(train_idx[te], P_s1_te_val)
                lr2 = LogisticRegression(C=C_test, class_weight=class_weight, max_iter=2000, random_state=42)
                lr2.fit(X2_tr, y_train_s1[tr])
                P_s2_c[te] = lr2.predict_proba(X2_te)[:, 1]

            # Evaluar con umbrales (en train)
            pred_final_c = np.full(n_train, -1)
            pred_final_c[verde] = 0
            pred_final_c[roja] = 1
            pred_final_c[amarilla] = (P_s2_c[amarilla] >= 0.5).astype(int)

            fn_c = ((pred_final_c == 0) & (y_for_thresh == 1)).sum()
            fp_c = ((pred_final_c == 1) & (y_for_thresh == 0)).sum()
            sens_c = ((pred_final_c == 1) & (y_for_thresh == 1)).sum() / max((y_for_thresh == 1).sum(), 1)
            spec_c = ((pred_final_c == 0) & (y_for_thresh == 0)).sum() / max((y_for_thresh == 0).sum(), 1)
            ba_c = (sens_c + spec_c) / 2

            c_results.append(
                {
                    "C": C_test,
                    "Errores": fn_c + fp_c,
                    "FN": fn_c,
                    "FP": fp_c,
                    "BA": ba_c,
                    "Sens": sens_c,
                    "Spec": spec_c,
                }
            )

        progress_bar.empty()
        c_df = pd.DataFrame(c_results)

        # Mostrar resultado
        best_row = c_df.loc[c_df["Errores"].idxmin()]
        C_value = best_row["C"]

        col1, col2 = st.columns([1, 2])
        with col1:
            st.success(f"**C óptimo: {C_value:.4f}** ({int(best_row['Errores'])} errores)")
            st.dataframe(
                c_df.style.highlight_min(subset=["Errores"], color="#d4efdf"), hide_index=True, use_container_width=True
            )
        with col2:
            fig_c = go.Figure()
            fig_c.add_trace(
                go.Scatter(
                    x=np.log10(c_df["C"]),
                    y=c_df["Errores"],
                    mode="lines+markers",
                    name="Errores totales",
                    line={"width": 3, "color": "#e74c3c"},
                )
            )
            fig_c.add_trace(
                go.Scatter(
                    x=np.log10(c_df["C"]),
                    y=c_df["FN"],
                    mode="lines+markers",
                    name="FN",
                    line={"width": 2, "dash": "dash", "color": "#8e44ad"},
                )
            )
            fig_c.add_trace(
                go.Scatter(
                    x=np.log10(c_df["C"]),
                    y=c_df["FP"],
                    mode="lines+markers",
                    name="FP",
                    line={"width": 2, "dash": "dot", "color": "#3498db"},
                )
            )
            fig_c.add_vline(x=np.log10(C_value), line_dash="dash", annotation_text=f"C*={C_value:.2f}")
            fig_c.update_layout(
                title="Optimización de C — Stage 2", xaxis_title="log₁₀(C)", yaxis_title="Errores", height=350
            )
            st.plotly_chart(fig_c, use_container_width=True)

    # --- Pipeline final con C seleccionado ---
    st.subheader("📊 Resultados del Pipeline")

    X_train_s1 = X[train_idx]
    y_train_s1 = y[train_idx]

    P_s2_train = np.zeros(n_train)
    train_ba_s1_list = []
    train_ba_s2_list = []
    coefs_list = []

    progress = st.progress(0, text="Ejecutando LOO-CV en train...")
    for idx, (tr, te) in enumerate(LeaveOneOut().split(X_train_s1, y_train_s1)):
        progress.progress((idx + 1) / n_train, text=f"LOO iteración {idx + 1}/{n_train}")
        # Stage 1
        X_tr_w = _winsorize(X_train_s1[tr])
        sc = StandardScaler()
        sc.fit(X_tr_w)
        lr1 = LogisticRegression(C=C_S1, class_weight=CW_S1, max_iter=2000, random_state=42)
        lr1.fit(sc.transform(X_tr_w), y_train_s1[tr])
        P_s1_tr = lr1.predict_proba(sc.transform(X_tr_w))[:, 1]
        P_s1_te_val = lr1.predict_proba(sc.transform(X_train_s1[te]))[:, 1]

        # Train BA S1
        train_ba_s1_list.append(balanced_accuracy_score(y_train_s1[tr], (P_s1_tr >= 0.5).astype(int)))

        # Stage 2
        X2_tr = build_X2(train_idx[tr], P_s1_tr)
        X2_te = build_X2(train_idx[te], P_s1_te_val)
        lr2 = LogisticRegression(C=C_value, class_weight=class_weight, max_iter=2000, random_state=42)
        lr2.fit(X2_tr, y_train_s1[tr])
        P_s2_train[te] = lr2.predict_proba(X2_te)[:, 1]

        # Train BA S2
        train_ba_s2_list.append(
            balanced_accuracy_score(y_train_s1[tr], (lr2.predict_proba(X2_tr)[:, 1] >= 0.5).astype(int))
        )
        coefs_list.append(lr2.coef_[0])

    progress.empty()

    # --- Métricas en train (CV) ---
    pred_train = np.full(n_train, -1)
    pred_train[verde] = 0
    pred_train[roja] = 1
    pred_train[amarilla] = (P_s2_train[amarilla] >= 0.5).astype(int)

    tp_tr = ((pred_train == 1) & (y_for_thresh == 1)).sum()
    tn_tr = ((pred_train == 0) & (y_for_thresh == 0)).sum()
    fp_tr = ((pred_train == 1) & (y_for_thresh == 0)).sum()
    fn_tr = ((pred_train == 0) & (y_for_thresh == 1)).sum()
    sens_tr = tp_tr / max(tp_tr + fn_tr, 1)
    spec_tr = tn_tr / max(tn_tr + fp_tr, 1)
    ba_train_final = (sens_tr + spec_tr) / 2

    # --- Evaluación en TEST (test NO winsorizado, matching Evaluación page) ---
    X_train_w = _winsorize(X_train_s1)
    sc_final = StandardScaler()
    sc_final.fit(X_train_w)
    lr1_final = LogisticRegression(C=C_S1, class_weight=CW_S1, max_iter=2000, random_state=42)
    lr1_final.fit(sc_final.transform(X_train_w), y_train_s1)
    P_s1_train_full = lr1_final.predict_proba(sc_final.transform(X_train_w))[:, 1]
    # Test: raw (sin winsorizar), solo escalado
    P_s1_test_eval = lr1_final.predict_proba(sc_final.transform(X[test_idx]))[:, 1]

    # Clasificar test con umbrales
    n_test = len(y_test)
    verde_test = P_s1_test_eval < t_lo
    roja_test = P_s1_test_eval > t_hi
    amarilla_test = ~verde_test & ~roja_test

    # Stage 2 en test: entrenar S2 en todo train, predecir amarilla_test
    X2_train_full = build_X2(train_idx, P_s1_train_full)
    lr2_final = LogisticRegression(C=C_value, class_weight=class_weight, max_iter=2000, random_state=42)
    lr2_final.fit(X2_train_full, y_train_s1)

    pred_test = np.full(n_test, -1)
    pred_test[verde_test] = 0
    pred_test[roja_test] = 1
    P_s2_test_full = np.zeros(n_test)
    if amarilla_test.sum() > 0:
        X2_test = build_X2(test_idx[amarilla_test], P_s1_test_eval[amarilla_test])
        P_s2_test = lr2_final.predict_proba(X2_test)[:, 1]
        pred_test[amarilla_test] = (P_s2_test >= 0.5).astype(int)
        P_s2_test_full[amarilla_test] = P_s2_test

    tp_te = ((pred_test == 1) & (y_test == 1)).sum()
    tn_te = ((pred_test == 0) & (y_test == 0)).sum()
    fp_te = ((pred_test == 1) & (y_test == 0)).sum()
    fn_te = ((pred_test == 0) & (y_test == 1)).sum()
    sens_test = tp_te / max(tp_te + fn_te, 1)
    spec_test = tn_te / max(tn_te + fp_te, 1)
    ba_test_final = (sens_test + spec_test) / 2

    # --- Métricas globales (modelo unificado) ---
    st.success(
        f"✅ Sistema Cascada — Train CV: {fn_tr + fp_tr} errores (BA={ba_train_final:.3f}) | "
        f"**Test: {fn_te + fp_te} errores (BA={ba_test_final:.3f})**"
    )

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("BA (Train CV)", f"{ba_train_final:.3f}")
    col2.metric("BA (Test)", f"{ba_test_final:.3f}")
    col3.metric("Sens (Test)", f"{sens_test:.3f}")
    col4.metric("Spec (Test)", f"{spec_test:.3f}")
    col5.metric("Err Train", f"{fn_tr + fp_tr} / {n_train}")
    col6.metric("Err Test", f"{fn_te + fp_te} / {n_test}")

    # Zonas en test
    st.markdown("#### Distribución del test por zonas")
    zones_test_df = pd.DataFrame(
        {
            "Zona": ["🟢 Verde", "🟡 Amarilla → S2", "🔴 Roja"],
            "N": [verde_test.sum(), amarilla_test.sum(), roja_test.sum()],
            "Errores": [
                ((verde_test) & (y_test == 1)).sum(),
                ((pred_test[amarilla_test] != y_test[amarilla_test]).sum() if amarilla_test.sum() > 0 else 0),
                ((roja_test) & (y_test == 0)).sum(),
            ],
        }
    )
    st.dataframe(zones_test_df, hide_index=True, use_container_width=True)

    # Train vs Test gap
    train_ba_s1_mean = np.mean(train_ba_s1_list)
    train_ba_s2_mean = np.mean(train_ba_s2_list)

    st.markdown("#### Train vs Test (gap de sobreajuste)")
    gap_df = pd.DataFrame(
        {
            "Etapa": ["Stage 1", "Stage 2 (sistema completo)"],
            "Train BA": [f"{train_ba_s1_mean:.3f}", f"{train_ba_s2_mean:.3f}"],
            "CV BA": [f"{ba_s1_train:.3f}", f"{ba_train_final:.3f}"],
            "Gap": [f"{train_ba_s1_mean - ba_s1_train:.3f}", f"{train_ba_s2_mean - ba_train_final:.3f}"],
            "Interpretación": [
                "✅ Bajo" if train_ba_s1_mean - ba_s1_train < 0.05 else "⚠️ Alto",
                "✅ Bajo" if train_ba_s2_mean - ba_train_final < 0.05 else "⚠️ Alto",
            ],
        }
    )
    st.dataframe(gap_df, hide_index=True, use_container_width=True)

    st.divider()

    # ============================================================================
    # SECCIÓN 5: Análisis de Multicolinealidad
    # ============================================================================

    st.header("5️⃣ Análisis de Multicolinealidad")
    st.markdown("Calculado automáticamente sobre las variables seleccionadas del Stage 2.")

    # Construir X2 completo para análisis (usando train)
    X2_full = build_X2(train_idx, P_s1)

    # VIF

    X2_with_const = np.column_stack([np.ones(n_train), X2_full])
    corr_matrix = np.corrcoef(X2_full, rowvar=False)
    n_vars = X2_full.shape[1]

    vif_values = []
    for j in range(n_vars):
        # VIF = 1 / (1 - R²_j), donde R²_j es de regresión de x_j sobre el resto
        others = [k for k in range(n_vars) if k != j]
        if len(others) == 0:
            vif_values.append(1.0)
            continue
        X_others = np.column_stack([np.ones(n_train), X2_full[:, others]])
        try:
            beta = np.linalg.lstsq(X_others, X2_full[:, j], rcond=None)[0]
            y_pred = X_others @ beta
            ss_res = ((X2_full[:, j] - y_pred) ** 2).sum()
            ss_tot = ((X2_full[:, j] - X2_full[:, j].mean()) ** 2).sum()
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            vif_values.append(1 / (1 - r2) if r2 < 1 else 999)
        except Exception:
            vif_values.append(np.nan)

    # Condition number
    cond_number = np.linalg.cond(X2_with_const)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### VIF (Variance Inflation Factor)")
        vif_df = pd.DataFrame(
            {
                "Variable": selected_vars,
                "VIF": [f"{v:.2f}" for v in vif_values],
                "Estado": ["✅ OK" if v < 5 else ("⚠️ Moderado" if v < 10 else "❌ Severo") for v in vif_values],
            }
        )
        st.dataframe(vif_df, hide_index=True, use_container_width=True)
        st.metric("Número de condición (κ)", f"{cond_number:.2f}", help="< 30: OK. > 30: posibles problemas.")

        max_vif = max(vif_values)
        if max_vif < 5 and cond_number < 30:
            st.success("✅ Sin multicolinealidad. Los regresores son informativamente independientes.")
        elif max_vif < 10:
            st.warning("⚠️ Multicolinealidad moderada detectada. Considerar eliminar variables.")
        else:
            st.error("❌ Multicolinealidad severa. Eliminar variables correlacionadas.")

    with col2:
        st.markdown("#### Matriz de Correlaciones")
        fig_corr = go.Figure(
            data=go.Heatmap(
                z=corr_matrix,
                x=selected_vars,
                y=selected_vars,
                text=np.round(corr_matrix, 2),
                texttemplate="%{text}",
                colorscale="RdBu_r",
                zmid=0,
                zmin=-1,
                zmax=1,
            )
        )
        fig_corr.update_layout(height=400, title="Correlaciones entre regresores S2")
        st.plotly_chart(fig_corr, use_container_width=True)

    st.divider()

    # ============================================================================
    # SECCIÓN 6: Coeficientes y Estabilidad
    # ============================================================================

    st.header("6️⃣ Coeficientes del Stage 2")

    coefs_arr = np.array(coefs_list)  # shape: (148, n_vars)
    coef_mean = coefs_arr.mean(axis=0)
    coef_std = coefs_arr.std(axis=0)
    coef_cv = np.abs(coef_std / coef_mean) * 100  # CV%

    coef_df = pd.DataFrame(
        {
            "Variable": selected_vars,
            "Coef (media)": [f"{c:.3f}" for c in coef_mean],
            "Coef (std)": [f"{s:.3f}" for s in coef_std],
            "CV%": [f"{cv:.1f}%" for cv in coef_cv],
            "OR": [f"{np.exp(c):.2f}" for c in coef_mean],
            "Estabilidad": [
                "✅ Estable" if cv < 15 else ("⚠️ Variable" if cv < 30 else "❌ Inestable") for cv in coef_cv
            ],
        }
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.dataframe(coef_df, hide_index=True, use_container_width=True)

    with col2:
        # Pareto diagram
        abs_coef = np.abs(coef_mean)
        sorted_idx = np.argsort(abs_coef)[::-1]
        fig_pareto = go.Figure()
        colors = ["#e74c3c" if coef_mean[i] > 0 else "#3498db" for i in sorted_idx]
        fig_pareto.add_trace(
            go.Bar(
                x=[selected_vars[i] for i in sorted_idx],
                y=[abs_coef[i] for i in sorted_idx],
                marker_color=colors,
                text=[f"{coef_mean[i]:+.2f}" for i in sorted_idx],
                textposition="outside",
            )
        )
        fig_pareto.update_layout(
            title="Importancia de regresores (|coef|)",
            xaxis_title="Variable",
            yaxis_title="|Coeficiente|",
            height=350,
        )
        st.plotly_chart(fig_pareto, use_container_width=True)

    # Distribución de coeficientes LOO
    with st.expander(f"📈 Distribución de coeficientes en {n_train} modelos LOO"):
        n_coefs = len(selected_vars)
        rows = (n_coefs + 2) // 3
        fig_stab = make_subplots(rows=rows, cols=3, subplot_titles=selected_vars[:n_coefs])
        for i in range(n_coefs):
            row = i // 3 + 1
            col = i % 3 + 1
            fig_stab.add_trace(
                go.Histogram(x=coefs_arr[:, i], nbinsx=20, marker_color="steelblue", showlegend=False), row=row, col=col
            )
            fig_stab.add_vline(x=coef_mean[i], line_dash="dash", line_color="red", row=row, col=col)
        fig_stab.update_layout(height=250 * rows, title_text="Estabilidad de coeficientes (LOO-CV)")
        st.plotly_chart(fig_stab, use_container_width=True)

    st.divider()

    # ============================================================================
    # SECCIÓN 7: Matriz de Confusión y Análisis de Errores
    # ============================================================================

    st.header("7️⃣ Resultados Detallados")

    # --- Totales (Train + Test) ---
    tp_total = tp_tr + tp_te
    tn_total = tn_tr + tn_te
    fp_total = fp_tr + fp_te
    fn_total = fn_tr + fn_te
    n_total = n_train + n_test
    sens_total = tp_total / max(tp_total + fn_total, 1)
    spec_total = tn_total / max(tn_total + fp_total, 1)
    ba_total = (sens_total + spec_total) / 2

    # Zonas totales
    verde_total = verde.sum() + verde_test.sum()
    amarilla_total = amarilla.sum() + amarilla_test.sum()
    roja_total = roja.sum() + roja_test.sum()

    st.markdown(
        f"**Resultados globales (N={n_total})**: {fn_total + fp_total} errores — "
        f"BA={ba_total:.3f}, Sens={sens_total:.3f}, Spec={spec_total:.3f}"
    )

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.markdown("#### Matriz de Confusión (Total)")
        fig_cm = go.Figure(
            data=go.Heatmap(
                z=[[tn_total, fp_total], [fn_total, tp_total]],
                x=["Pred NC", "Pred AD"],
                y=["Real NC", "Real AD"],
                text=[[f"TN={tn_total}", f"FP={fp_total}"], [f"FN={fn_total}", f"TP={tp_total}"]],
                texttemplate="%{text}",
                textfont={"size": 18},
                colorscale=[[0, "#27ae60"], [0.5, "#f7dc6f"], [1, "#e74c3c"]],
                showscale=False,
            )
        )
        fig_cm.update_layout(height=300)
        st.plotly_chart(fig_cm, use_container_width=True)

    with col2:
        st.markdown("#### Confusión Train CV")
        fig_cm_tr = go.Figure(
            data=go.Heatmap(
                z=[[tn_tr, fp_tr], [fn_tr, tp_tr]],
                x=["Pred NC", "Pred AD"],
                y=["Real NC", "Real AD"],
                text=[[f"TN={tn_tr}", f"FP={fp_tr}"], [f"FN={fn_tr}", f"TP={tp_tr}"]],
                texttemplate="%{text}",
                textfont={"size": 14},
                colorscale=[[0, "#27ae60"], [0.5, "#f7dc6f"], [1, "#e74c3c"]],
                showscale=False,
            )
        )
        fig_cm_tr.update_layout(height=250)
        st.plotly_chart(fig_cm_tr, use_container_width=True)

    with col3:
        st.markdown("#### Confusión Test")
        fig_cm_te = go.Figure(
            data=go.Heatmap(
                z=[[tn_te, fp_te], [fn_te, tp_te]],
                x=["Pred NC", "Pred AD"],
                y=["Real NC", "Real AD"],
                text=[[f"TN={tn_te}", f"FP={fp_te}"], [f"FN={fn_te}", f"TP={tp_te}"]],
                texttemplate="%{text}",
                textfont={"size": 14},
                colorscale=[[0, "#27ae60"], [0.5, "#f7dc6f"], [1, "#e74c3c"]],
                showscale=False,
            )
        )
        fig_cm_te.update_layout(height=250)
        st.plotly_chart(fig_cm_te, use_container_width=True)

    # Distribución por zonas (total)
    st.markdown("#### Distribución por zonas (Total)")
    err_verde_test = (verde_test & (y_test == 1)).sum()
    err_roja_test = (roja_test & (y_test == 0)).sum()
    err_amarilla_train = (pred_train[amarilla] != y_for_thresh[amarilla]).sum()
    err_amarilla_test = (pred_test[amarilla_test] != y_test[amarilla_test]).sum() if amarilla_test.sum() > 0 else 0

    zones_df = pd.DataFrame(
        {
            "Zona": ["🟢 Verde (NC directo)", "🟡 Amarilla (→ S2)", "🔴 Roja (AD directo)", "**TOTAL**"],
            "N (Train)": [verde.sum(), amarilla.sum(), roja.sum(), n_train],
            "N (Test)": [verde_test.sum(), amarilla_test.sum(), roja_test.sum(), n_test],
            "N (Total)": [verde_total, amarilla_total, roja_total, n_total],
            "Err Train": [err_verde_real, err_amarilla_train, err_roja_real, fn_tr + fp_tr],
            "Err Test": [err_verde_test, err_amarilla_test, err_roja_test, fn_te + fp_te],
            "Err Total": [
                err_verde_real + err_verde_test,
                err_amarilla_train + err_amarilla_test,
                err_roja_real + err_roja_test,
                fn_total + fp_total,
            ],
        }
    )
    st.dataframe(zones_df, hide_index=True, use_container_width=True)

    # Resumen gráfico
    fig_pie = go.Figure(
        go.Pie(
            labels=[f"🟢 Verde ({verde_total})", f"🟡 Amarilla ({amarilla_total})", f"🔴 Roja ({roja_total})"],
            values=[verde_total, amarilla_total, roja_total],
            marker_colors=["#27ae60", "#f39c12", "#e74c3c"],
            hole=0.3,
        )
    )
    fig_pie.update_layout(height=250, margin={"t": 20, "b": 20})
    st.plotly_chart(fig_pie, use_container_width=True)

    # Análisis de errores (TODOS: train + test)
    st.markdown("#### Detalle de errores")

    # Errores en train
    fn_mask_tr = (pred_train == 0) & (y_for_thresh == 1)
    fp_mask_tr = (pred_train == 1) & (y_for_thresh == 0)
    # Errores en test
    fn_mask_te = (pred_test == 0) & (y_test == 1)
    fp_mask_te = (pred_test == 1) & (y_test == 0)

    total_fn = fn_mask_tr.sum() + fn_mask_te.sum()
    total_fp = fp_mask_tr.sum() + fp_mask_te.sum()

    if total_fn > 0:
        st.markdown(f"**Falsos Negativos ({total_fn})** — pacientes AD no detectados:")
        fn_data = []
        for i in np.where(fn_mask_tr)[0]:
            zona = "Verde" if verde[i] else "Amarilla"
            fn_data.append(
                {
                    "Set": "Train",
                    "ID": ids_train[i],
                    "P(AD)_S1": f"{P_s1[i]:.3f}",
                    "P(AD)_S2": f"{P_s2_train[i]:.3f}",
                    "MMSE": int(mmse[train_idx[i]]),
                    "APOE": apoe_raw.iloc[train_idx[i]],
                    "Depresión": "Sí" if dep[train_idx[i]] else "No",
                    "Cardiopatía": "Sí" if cardio[train_idx[i]] else "No",
                    "Edad": int(age[train_idx[i]]),
                    "Zona": zona,
                }
            )
        for i in np.where(fn_mask_te)[0]:
            zona = "Verde" if verde_test[i] else "Amarilla"
            fn_data.append(
                {
                    "Set": "Test",
                    "ID": ids[test_idx[i]],
                    "P(AD)_S1": f"{P_s1_test_eval[i]:.3f}",
                    "P(AD)_S2": f"{P_s2_test_full[i]:.3f}" if amarilla_test[i] else "—",
                    "MMSE": int(mmse[test_idx[i]]),
                    "APOE": apoe_raw.iloc[test_idx[i]],
                    "Depresión": "Sí" if dep[test_idx[i]] else "No",
                    "Cardiopatía": "Sí" if cardio[test_idx[i]] else "No",
                    "Edad": int(age[test_idx[i]]),
                    "Zona": zona,
                }
            )
        st.dataframe(pd.DataFrame(fn_data), hide_index=True, use_container_width=True)

    if total_fp > 0:
        st.markdown(f"**Falsos Positivos ({total_fp})** — pacientes NC clasificados como AD:")
        fp_data = []
        for i in np.where(fp_mask_tr)[0]:
            zona = "Roja" if roja[i] else "Amarilla"
            fp_data.append(
                {
                    "Set": "Train",
                    "ID": ids_train[i],
                    "P(AD)_S1": f"{P_s1[i]:.3f}",
                    "P(AD)_S2": f"{P_s2_train[i]:.3f}",
                    "MMSE": int(mmse[train_idx[i]]),
                    "APOE": apoe_raw.iloc[train_idx[i]],
                    "Depresión": "Sí" if dep[train_idx[i]] else "No",
                    "Cardiopatía": "Sí" if cardio[train_idx[i]] else "No",
                    "Edad": int(age[train_idx[i]]),
                    "Zona": zona,
                }
            )
        for i in np.where(fp_mask_te)[0]:
            zona = "Roja" if roja_test[i] else "Amarilla"
            fp_data.append(
                {
                    "Set": "Test",
                    "ID": ids[test_idx[i]],
                    "P(AD)_S1": f"{P_s1_test_eval[i]:.3f}",
                    "P(AD)_S2": f"{P_s2_test_full[i]:.3f}" if amarilla_test[i] else "—",
                    "MMSE": int(mmse[test_idx[i]]),
                    "APOE": apoe_raw.iloc[test_idx[i]],
                    "Depresión": "Sí" if dep[test_idx[i]] else "No",
                    "Cardiopatía": "Sí" if cardio[test_idx[i]] else "No",
                    "Edad": int(age[test_idx[i]]),
                    "Zona": zona,
                }
            )
        st.dataframe(pd.DataFrame(fp_data), hide_index=True, use_container_width=True)

    st.divider()

    # ============================================================================
    # SECCIÓN 8: Diagrama Resumen
    # ============================================================================

    st.header("8️⃣ Diagrama del Sistema")

    vars_str = ", ".join([v.split(" ")[0] for v in selected_vars])
    cw_str = str(class_weight) if isinstance(class_weight, dict) else class_weight

    st.markdown(f"""
```
                    ┌──────────────────────────────────────┐
                    │      STAGE 1: Modelo Metabolómico     │
                    │  {len(FEATURES_S1)} features, LogReg(C={C_S1}, {CW_S1})  │
                    └──────────────────┬───────────────────┘
                                       │
                                  P(AD)_S1
                                       │
          ┌────────────────────────────┼────────────────────────────┐
          │                            │                            │
          ▼                            ▼                            ▼
   P(AD) < {t_lo:.4f}          {t_lo:.4f} ≤ P ≤ {t_hi:.4f}          P(AD) > {t_hi:.4f}
  ┌──────────────┐         ┌───────────────────────────┐    ┌──────────────┐
  │ 🟢 NC DIRECTO │         │  STAGE 2: Clínico         │    │ 🔴 AD DIRECTO │
  │  {verde_total} pacientes  │         │  LogReg(C={C_value}, {cw_str})  │    │  {roja_total} pacientes  │
  │  {err_verde_real + err_verde_test} errores    │         │  Variables: {len(selected_vars)}          │    │  {err_roja_real + err_roja_test} errores    │
  └──────────────┘         │  [{vars_str}]             │    └──────────────┘
                            └────────────┬──────────────┘
                                         │
                                    P(AD)_S2 ≥ 0.5?
                                         │
                              ┌──────────┴───────────┐
                              │                      │
                              ▼                      ▼
                          NC ({(pred_train[amarilla] == 0).sum() + (pred_test[amarilla_test] == 0).sum()})              AD ({(pred_train[amarilla] == 1).sum() + (pred_test[amarilla_test] == 1).sum()})

  TOTAL: {fn_total + fp_total} errores ({fn_total} FN + {fp_total} FP) | BA = {ba_total:.3f}
  (Train: {fn_tr + fp_tr} err, Test: {fn_te + fp_te} err)
```
""")

    # ============================================================================
    # SECCIÓN 9: Comparativa de configuraciones
    # ============================================================================

    st.header("9️⃣ Comparativa rápida")
    st.markdown("Comparación del sistema cascada vs Stage 1 solo.")

    # S1 solo en test
    fn_s1_test = ((pred_s1_test == 0) & (y_test == 1)).sum()
    fp_s1_test = ((pred_s1_test == 1) & (y_test == 0)).sum()

    comp_data = pd.DataFrame(
        {
            "Configuración": [
                "Solo Stage 1 (t=0.5) — Train",
                "Solo Stage 1 (t=0.5) — Test",
                f"Cascada (C_S2={C_value}, {cw_str}) — Train",
                f"Cascada (C_S2={C_value}, {cw_str}) — Test",
                "**Cascada — TOTAL**",
            ],
            "N": [n_train, n_test, n_train, n_test, n_total],
            "Errores": [fn_s1 + fp_s1, fn_s1_test + fp_s1_test, fn_tr + fp_tr, fn_te + fp_te, fn_total + fp_total],
            "FN": [fn_s1, fn_s1_test, fn_tr, fn_te, fn_total],
            "FP": [fp_s1, fp_s1_test, fp_tr, fp_te, fp_total],
            "BA": [
                f"{ba_s1_train:.3f}",
                f"{ba_s1_test:.3f}",
                f"{ba_train_final:.3f}",
                f"{ba_test_final:.3f}",
                f"{ba_total:.3f}",
            ],
        }
    )
    st.dataframe(comp_data, hide_index=True, use_container_width=True)

else:
    st.info("Pulsa **▶️ Ejecutar Pipeline** para correr el sistema con la configuración seleccionada.")

# ============================================================================
# SECCIÓN 10: Análisis de Estabilidad Multi-Seed (independiente del pipeline)
# ============================================================================

st.divider()
st.header("🔟 Análisis de Estabilidad Multi-Seed")
st.markdown("""
Evalúa la robustez del sistema ejecutándolo con múltiples semillas de train/test split.
Esto demuestra que los resultados no dependen de un split afortunado.
""")

col_ms1, col_ms2 = st.columns(2)
with col_ms1:
    n_seeds = st.number_input("Número de seeds a evaluar", 5, 100, 20, step=5, key="n_seeds_stability")
with col_ms2:
    run_stability = st.button("🔄 Ejecutar análisis de estabilidad", type="secondary")

if run_stability:
    _placeholder = st.empty()
    _placeholder.info("Ejecutando análisis multi-seed...")
    ms_results = []

    for seed_i in range(n_seeds):
        _placeholder.progress((seed_i + 1) / n_seeds, text=f"Seed {seed_i + 1}/{n_seeds}...")

        # Train/test split con esta seed
        idx_all = np.arange(n)
        tr_idx_ms, te_idx_ms = train_test_split(idx_all, test_size=holdout_pct / 100.0, stratify=y, random_state=seed_i)
        X_tr_ms, X_te_ms = X[tr_idx_ms], X[te_idx_ms]
        y_tr_ms, y_te_ms = y[tr_idx_ms], y[te_idx_ms]
        n_tr_ms = len(y_tr_ms)

        # Stage 1: LOO-CV en train
        P_s1_tr_ms = np.zeros(n_tr_ms)
        for tr, te in LeaveOneOut().split(X_tr_ms, y_tr_ms):
            X_w = _winsorize(X_tr_ms[tr])
            sc_ms = StandardScaler().fit(X_w)
            lr1_ms = LogisticRegression(C=C_S1, class_weight=CW_S1, max_iter=2000, random_state=42)
            lr1_ms.fit(sc_ms.transform(X_w), y_tr_ms[tr])
            P_s1_tr_ms[te] = lr1_ms.predict_proba(sc_ms.transform(X_tr_ms[te]))[:, 1]

        # Umbrales (en train)
        probs_sorted_ms = np.sort(np.unique(P_s1_tr_ms))
        cands_ms = np.concatenate([[0.0], probs_sorted_ms, [1.0]])
        best_tlo_ms, best_thi_ms, best_cov_ms = None, None, -1
        for tl in cands_ms:
            if ((P_s1_tr_ms < tl) & (y_tr_ms == 1)).sum() > max_err_verde:
                continue
            for th in cands_ms:
                if th <= tl:
                    continue
                if ((P_s1_tr_ms > th) & (y_tr_ms == 0)).sum() > max_err_roja:
                    continue
                cov_ms = ((P_s1_tr_ms < tl) | (P_s1_tr_ms > th)).sum()
                if cov_ms > best_cov_ms:
                    best_cov_ms = cov_ms
                    best_tlo_ms, best_thi_ms = tl, th

        if best_tlo_ms is None:
            continue  # No valid thresholds for this seed

        # Zonas train
        verde_ms = P_s1_tr_ms < best_tlo_ms
        roja_ms = P_s1_tr_ms > best_thi_ms
        amarilla_ms = ~verde_ms & ~roja_ms

        # Stage 2: LOO en train
        P_s2_tr_ms = np.zeros(n_tr_ms)
        for tr, te in LeaveOneOut().split(X_tr_ms, y_tr_ms):
            X_w = _winsorize(X_tr_ms[tr])
            sc_ms = StandardScaler().fit(X_w)
            lr1_ms = LogisticRegression(C=C_S1, class_weight=CW_S1, max_iter=2000, random_state=42)
            lr1_ms.fit(sc_ms.transform(X_w), y_tr_ms[tr])
            P_s1_inner_tr = lr1_ms.predict_proba(sc_ms.transform(X_w))[:, 1]
            P_s1_inner_te = lr1_ms.predict_proba(sc_ms.transform(X_tr_ms[te]))[:, 1]

            X2_tr_ms = build_X2(tr_idx_ms[tr], P_s1_inner_tr)
            X2_te_ms = build_X2(tr_idx_ms[te], P_s1_inner_te)
            lr2_ms = LogisticRegression(C=C_value, class_weight=class_weight, max_iter=2000, random_state=42)
            lr2_ms.fit(X2_tr_ms, y_tr_ms[tr])
            P_s2_tr_ms[te] = lr2_ms.predict_proba(X2_te_ms)[:, 1]

        # Predicciones train
        pred_tr_ms = np.full(n_tr_ms, -1)
        pred_tr_ms[verde_ms] = 0
        pred_tr_ms[roja_ms] = 1
        pred_tr_ms[amarilla_ms] = (P_s2_tr_ms[amarilla_ms] >= 0.5).astype(int)

        tp_tr_ms = ((pred_tr_ms == 1) & (y_tr_ms == 1)).sum()
        tn_tr_ms = ((pred_tr_ms == 0) & (y_tr_ms == 0)).sum()
        fn_tr_ms = ((pred_tr_ms == 0) & (y_tr_ms == 1)).sum()
        fp_tr_ms = ((pred_tr_ms == 1) & (y_tr_ms == 0)).sum()
        sens_tr_ms = tp_tr_ms / max(tp_tr_ms + fn_tr_ms, 1)
        spec_tr_ms = tn_tr_ms / max(tn_tr_ms + fp_tr_ms, 1)
        ba_tr_ms = (sens_tr_ms + spec_tr_ms) / 2

        # Test evaluation
        X_w_full = _winsorize(X_tr_ms)
        sc_final_ms = StandardScaler().fit(X_w_full)
        lr1_final_ms = LogisticRegression(C=C_S1, class_weight=CW_S1, max_iter=2000, random_state=42)
        lr1_final_ms.fit(sc_final_ms.transform(X_w_full), y_tr_ms)
        P_s1_te_ms = lr1_final_ms.predict_proba(sc_final_ms.transform(X_te_ms))[:, 1]

        # Stage 1 solo test
        ba_s1_te_ms = balanced_accuracy_score(y_te_ms, (P_s1_te_ms >= 0.5).astype(int))

        # Zonas test
        verde_te_ms = P_s1_te_ms < best_tlo_ms
        roja_te_ms = P_s1_te_ms > best_thi_ms
        amarilla_te_ms = ~verde_te_ms & ~roja_te_ms

        # Stage 2 test
        P_s1_tr_full = lr1_final_ms.predict_proba(sc_final_ms.transform(X_w_full))[:, 1]
        X2_full_ms = build_X2(tr_idx_ms, P_s1_tr_full)
        lr2_final_ms = LogisticRegression(C=C_value, class_weight=class_weight, max_iter=2000, random_state=42)
        lr2_final_ms.fit(X2_full_ms, y_tr_ms)

        pred_te_ms = np.full(len(y_te_ms), -1)
        pred_te_ms[verde_te_ms] = 0
        pred_te_ms[roja_te_ms] = 1
        if amarilla_te_ms.sum() > 0:
            X2_te_ms_am = build_X2(te_idx_ms[amarilla_te_ms], P_s1_te_ms[amarilla_te_ms])
            P_s2_te_am = lr2_final_ms.predict_proba(X2_te_ms_am)[:, 1]
            pred_te_ms[amarilla_te_ms] = (P_s2_te_am >= 0.5).astype(int)

        tp_te_ms = ((pred_te_ms == 1) & (y_te_ms == 1)).sum()
        tn_te_ms = ((pred_te_ms == 0) & (y_te_ms == 0)).sum()
        fn_te_ms = ((pred_te_ms == 0) & (y_te_ms == 1)).sum()
        fp_te_ms = ((pred_te_ms == 1) & (y_te_ms == 0)).sum()
        sens_te_ms = tp_te_ms / max(tp_te_ms + fn_te_ms, 1)
        spec_te_ms = tn_te_ms / max(tn_te_ms + fp_te_ms, 1)
        ba_te_ms = (sens_te_ms + spec_te_ms) / 2

        ms_results.append(
            {
                "Seed": seed_i,
                "t_lo": best_tlo_ms,
                "t_hi": best_thi_ms,
                "Train_BA": ba_tr_ms,
                "Train_Err": fn_tr_ms + fp_tr_ms,
                "Test_BA": ba_te_ms,
                "Test_Sens": sens_te_ms,
                "Test_Spec": spec_te_ms,
                "Test_Err": fn_te_ms + fp_te_ms,
                "Test_FN": fn_te_ms,
                "Test_FP": fp_te_ms,
                "S1_Test_BA": ba_s1_te_ms,
                "Cascade_Improvement": ba_te_ms - ba_s1_te_ms,
                "N_verde_te": verde_te_ms.sum(),
                "N_amarilla_te": amarilla_te_ms.sum(),
                "N_roja_te": roja_te_ms.sum(),
            }
        )

    _placeholder.empty()

    # Guardar en session_state para persistir entre reruns
    if ms_results:
        st.session_state["ms_stability_results"] = pd.DataFrame(ms_results)
        st.session_state["ms_stability_n_seeds"] = n_seeds
    else:
        st.session_state["ms_stability_results"] = None
    st.rerun()

# --- Mostrar resultados guardados ---
if "ms_stability_results" in st.session_state and st.session_state["ms_stability_results"] is not None:
    df_ms = st.session_state["ms_stability_results"]
    n_valid = len(df_ms)
    _n_seeds_used = st.session_state.get("ms_stability_n_seeds", n_valid)

    st.success(f"✅ Análisis completado: {n_valid}/{_n_seeds_used} seeds válidas")

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    col_s1.metric("BA Test (media)", f"{df_ms['Test_BA'].mean():.3f}")
    col_s2.metric("BA Test (std)", f"{df_ms['Test_BA'].std():.3f}")
    col_s3.metric("Errores Test (media)", f"{df_ms['Test_Err'].mean():.1f}")
    col_s4.metric("Mejora vs S1 (media)", f"{df_ms['Cascade_Improvement'].mean():+.3f}")

    # Percentil de seed actual
    if holdout_seed in df_ms["Seed"].values:
        ba_current = df_ms[df_ms["Seed"] == holdout_seed]["Test_BA"].values[0]
        percentile = (df_ms["Test_BA"] <= ba_current).mean() * 100
        st.info(f"📌 Seed actual ({holdout_seed}): BA_test={ba_current:.3f} — percentil {percentile:.0f}%")

    # Tabla resumen
    st.markdown("#### Distribución de métricas")
    summary_metrics = ["Test_BA", "Test_Sens", "Test_Spec", "Test_Err", "Train_BA", "Cascade_Improvement"]
    summary_data = []
    for metric in summary_metrics:
        vals = df_ms[metric]
        summary_data.append(
            {
                "Métrica": metric,
                "Media": f"{vals.mean():.3f}",
                "Std": f"{vals.std():.3f}",
                "Min": f"{vals.min():.3f}",
                "Max": f"{vals.max():.3f}",
                "IQR": f"[{vals.quantile(0.25):.3f}, {vals.quantile(0.75):.3f}]",
            }
        )
    st.dataframe(pd.DataFrame(summary_data), hide_index=True, use_container_width=True)

    # Estabilidad de umbrales
    st.markdown("#### Estabilidad de umbrales")
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        st.metric("t_lo (media ± std)", f"{df_ms['t_lo'].mean():.4f} ± {df_ms['t_lo'].std():.4f}")
    with col_u2:
        st.metric("t_hi (media ± std)", f"{df_ms['t_hi'].mean():.4f} ± {df_ms['t_hi'].std():.4f}")

    # Train-Test gap
    gap = df_ms["Train_BA"] - df_ms["Test_BA"]
    st.markdown("#### Overfitting check")
    st.markdown(
        f"Gap medio (Train − Test): **{gap.mean():.3f}** ± {gap.std():.3f} → "
        f"{'✅ Sin overfitting' if gap.mean() < 0.03 else '⚠️ Gap moderado' if gap.mean() < 0.06 else '❌ Posible overfitting'}"
    )

    # Gráficos
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        fig_hist = go.Figure()
        fig_hist.add_trace(
            go.Histogram(
                x=df_ms["Test_BA"],
                nbinsx=12,
                name="BA Test",
                marker_color="steelblue",
                opacity=0.7,
            )
        )
        fig_hist.add_vline(
            x=df_ms["Test_BA"].mean(), line_dash="dash", annotation_text=f"Media={df_ms['Test_BA'].mean():.3f}"
        )
        if holdout_seed in df_ms["Seed"].values:
            fig_hist.add_vline(x=ba_current, line_dash="dot", line_color="red", annotation_text=f"Seed {holdout_seed}")
        fig_hist.update_layout(
            title="Distribución de BA (Test) sobre seeds",
            xaxis_title="Balanced Accuracy",
            yaxis_title="Frecuencia",
            height=350,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_g2:
        fig_scatter = go.Figure()
        fig_scatter.add_trace(
            go.Scatter(
                x=df_ms["S1_Test_BA"],
                y=df_ms["Test_BA"],
                mode="markers",
                marker={"size": 8, "color": "steelblue"},
                text=df_ms["Seed"],
                name="Seeds",
            )
        )
        # Línea diagonal (cascada = S1)
        min_v = min(df_ms["S1_Test_BA"].min(), df_ms["Test_BA"].min()) - 0.02
        max_v = max(df_ms["S1_Test_BA"].max(), df_ms["Test_BA"].max()) + 0.02
        fig_scatter.add_trace(
            go.Scatter(
                x=[min_v, max_v],
                y=[min_v, max_v],
                mode="lines",
                line={"dash": "dash", "color": "gray"},
                name="Cascada = S1",
                showlegend=True,
            )
        )
        fig_scatter.update_layout(
            title="Cascada vs S1 solo (por seed)",
            xaxis_title="BA S1 solo (Test)",
            yaxis_title="BA Cascada (Test)",
            height=350,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Aporte del Stage 2
    pct_better = (df_ms["Cascade_Improvement"] > 0).mean() * 100
    pct_equal = (df_ms["Cascade_Improvement"] == 0).mean() * 100
    pct_worse = (df_ms["Cascade_Improvement"] < 0).mean() * 100
    st.markdown(
        f"**Aporte del Stage 2**: Mejora en {pct_better:.0f}% de seeds, "
        f"igual en {pct_equal:.0f}%, peor en {pct_worse:.0f}%"
    )

    # Interpretación
    ba_std = df_ms["Test_BA"].std()
    if ba_std < 0.03:
        verdict = "🟢 MUY ESTABLE"
        explanation = "La varianza entre seeds es baja. El resultado es robusto."
    elif ba_std < 0.05:
        verdict = "🟢 ESTABLE"
        explanation = "Varianza aceptable para N=148. El resultado es fiable."
    elif ba_std < 0.08:
        verdict = "🟡 MODERADA"
        explanation = "Varianza esperable por el tamaño muestral (N_test≈30). Reportar media±std."
    else:
        verdict = "🔴 INESTABLE"
        explanation = "Alta varianza. Los resultados dependen del split."

    st.markdown(f"#### Veredicto: {verdict}")
    st.markdown(f"> {explanation}")

    # Tabla completa expandible
    with st.expander("📋 Ver resultados por seed"):
        st.dataframe(
            df_ms.style.format(
                {
                    "t_lo": "{:.4f}",
                    "t_hi": "{:.4f}",
                    "Train_BA": "{:.3f}",
                    "Test_BA": "{:.3f}",
                    "Test_Sens": "{:.3f}",
                    "Test_Spec": "{:.3f}",
                    "S1_Test_BA": "{:.3f}",
                    "Cascade_Improvement": "{:+.3f}",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )
