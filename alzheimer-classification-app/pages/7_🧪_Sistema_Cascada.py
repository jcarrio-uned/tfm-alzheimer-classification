"""
Módulo 7: Sistema de Clasificación en Cascada (Interactivo)
============================================================
Página interactiva para explorar el sistema Stage 1 + Stage 2:
- Búsqueda de umbrales según errores tolerados en zonas roja/verde
- Selección de variables del Stage 2
- Optimización de C (regularización)
- Configuración de class_weight (balanceo)
- Análisis de multicolinealidad automático
- Métricas y gráficos completos
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    LeaveOneOut,
    RepeatedStratifiedKFold,
    StratifiedShuffleSplit,
    train_test_split,
)
from sklearn.metrics import balanced_accuracy_score
from scipy.stats import pearsonr
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app_utils import load_data, initialize_session_state, WinsorizerTransformer

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
_candidates = [
    os.path.join(os.path.dirname(__file__), "..", "..", "Segundo_Archivo_clean.xlsx"),
    os.path.join(os.path.dirname(__file__), "..", "data", "Segundo_Archivo_clean.xlsx"),
    os.path.join(os.getcwd(), "Segundo_Archivo_clean.xlsx"),
]
clinical_path = None
for _p in _candidates:
    if os.path.exists(_p):
        clinical_path = _p
        break

if clinical_path is None:
    st.error("No se encuentra el archivo clínico 'Segundo_Archivo_clean.xlsx'")
    st.stop()

df_clin = pd.read_excel(clinical_path)
df_clin_sel = df_clin[["ID", "MMSE", "APOE", "Depression", "Cardiovascular disorder", "Age [y]"]].copy()
df_clin_sel.rename(columns={"Age [y]": "Age_clin"}, inplace=True)
df = df_metab.merge(df_clin_sel, on="ID")

# Features Stage 1 — parametrizado desde el modelo entrenado en page 4
if "selected_features" in st.session_state and st.session_state["selected_features"]:
    FEATURES_S1 = st.session_state["selected_features"]
else:
    # Fallback: 11 features del pipeline por defecto
    FEATURES_S1 = [
        "DOPA",
        "Cer(d18:1/20:0)",
        "lysoPC.a.C18:2",
        "PC.aa.C40:4",
        "DHEAS",
        "Arg",
        "HexCer(d18:1/26:1)",
        "DHEAS/lysoPC",
        "DOPA*DHEAS",
        "PC*DOPA",
        "DHEAS/Cer20",
    ]

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
    """
    Busca umbrales t_lo y t_hi que cumplan restricciones de errores
    y maximicen la cobertura (pacientes fuera de zona amarilla).
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
        selected_vars = ["P(AD)_S1"] + selected_vars
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

if run_button:
    # --- Preparar columnas del Stage 2 ---
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
                    line=dict(width=3, color="#e74c3c"),
                )
            )
            fig_c.add_trace(
                go.Scatter(
                    x=np.log10(c_df["C"]),
                    y=c_df["FN"],
                    mode="lines+markers",
                    name="FN",
                    line=dict(width=2, dash="dash", color="#8e44ad"),
                )
            )
            fig_c.add_trace(
                go.Scatter(
                    x=np.log10(c_df["C"]),
                    y=c_df["FP"],
                    mode="lines+markers",
                    name="FP",
                    line=dict(width=2, dash="dot", color="#3498db"),
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
    if amarilla_test.sum() > 0:
        X2_test = build_X2(test_idx[amarilla_test], P_s1_test_eval[amarilla_test])
        P_s2_test = lr2_final.predict_proba(X2_test)[:, 1]
        pred_test[amarilla_test] = (P_s2_test >= 0.5).astype(int)

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
    from numpy.linalg import inv

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

    st.header("7️⃣ Resultados Detallados (Train CV)")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Matriz de Confusión")
        fig_cm = go.Figure(
            data=go.Heatmap(
                z=[[tn_tr, fp_tr], [fn_tr, tp_tr]],
                x=["Pred NC", "Pred AD"],
                y=["Real NC", "Real AD"],
                text=[[f"TN={tn_tr}", f"FP={fp_tr}"], [f"FN={fn_tr}", f"TP={tp_tr}"]],
                texttemplate="%{text}",
                textfont={"size": 18},
                colorscale=[[0, "#27ae60"], [0.5, "#f7dc6f"], [1, "#e74c3c"]],
                showscale=False,
            )
        )
        fig_cm.update_layout(height=300)
        st.plotly_chart(fig_cm, use_container_width=True)

    with col2:
        st.markdown("#### Distribución por zonas")
        zones_df = pd.DataFrame(
            {
                "Zona": ["🟢 Verde (NC directo)", "🟡 Amarilla (→ S2)", "🔴 Roja (AD directo)"],
                "N pacientes": [verde.sum(), amarilla.sum(), roja.sum()],
                "AD reales": [
                    (verde & (y_for_thresh == 1)).sum(),
                    (amarilla & (y_for_thresh == 1)).sum(),
                    (roja & (y_for_thresh == 1)).sum(),
                ],
                "NC reales": [
                    (verde & (y_for_thresh == 0)).sum(),
                    (amarilla & (y_for_thresh == 0)).sum(),
                    (roja & (y_for_thresh == 0)).sum(),
                ],
                "Errores": [err_verde_real, (pred_train[amarilla] != y_for_thresh[amarilla]).sum(), err_roja_real],
            }
        )
        st.dataframe(zones_df, hide_index=True, use_container_width=True)

        # Resumen gráfico
        fig_pie = go.Figure(
            go.Pie(
                labels=[f"🟢 Verde ({verde.sum()})", f"🟡 Amarilla ({amarilla.sum()})", f"🔴 Roja ({roja.sum()})"],
                values=[verde.sum(), amarilla.sum(), roja.sum()],
                marker_colors=["#27ae60", "#f39c12", "#e74c3c"],
                hole=0.3,
            )
        )
        fig_pie.update_layout(height=250, margin=dict(t=20, b=20))
        st.plotly_chart(fig_pie, use_container_width=True)

    # Análisis de errores
    st.markdown("#### Detalle de errores")

    fn_mask = (pred_train == 0) & (y_for_thresh == 1)
    fp_mask = (pred_train == 1) & (y_for_thresh == 0)

    if fn_mask.sum() > 0:
        st.markdown(f"**Falsos Negativos ({fn_mask.sum()})** — pacientes AD no detectados:")
        fn_data = []
        for i in np.where(fn_mask)[0]:
            zona = "Verde" if verde[i] else "Amarilla"
            fn_data.append(
                {
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
        st.dataframe(pd.DataFrame(fn_data), hide_index=True, use_container_width=True)

    if fp_mask.sum() > 0:
        st.markdown(f"**Falsos Positivos ({fp_mask.sum()})** — pacientes NC clasificados como AD:")
        fp_data = []
        for i in np.where(fp_mask)[0]:
            zona = "Roja" if roja[i] else "Amarilla"
            fp_data.append(
                {
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
  │  {verde.sum()} pacientes  │         │  LogReg(C={C_value}, {cw_str})  │    │  {roja.sum()} pacientes  │
  │  {err_verde_real} errores    │         │  Variables: {len(selected_vars)}          │    │  {err_roja_real} errores    │
  └──────────────┘         │  [{vars_str}]             │    └──────────────┘
                            └────────────┬──────────────┘
                                         │
                                    P(AD)_S2 ≥ 0.5?
                                         │
                              ┌──────────┴───────────┐
                              │                      │
                              ▼                      ▼
                          NC ({(pred_train[amarilla] == 0).sum()})              AD ({(pred_train[amarilla] == 1).sum()})

  RESULTADO (Train CV): {fn_tr + fp_tr} errores ({fn_tr} FN + {fp_tr} FP) | BA = {ba_train_final:.3f}
```
""")

    # ============================================================================
    # SECCIÓN 9: Comparativa de configuraciones
    # ============================================================================

    st.header("9️⃣ Comparativa rápida")
    st.markdown("Comparación del sistema actual vs alternativas de referencia.")

    comp_data = pd.DataFrame(
        {
            "Configuración": [
                "Solo Stage 1 (t=0.5)",
                f"S1+S2 actual (C={C_value}, {cw_str})",
                f"Configuración actual + umbrales ({max_err_verde}/{max_err_roja} err)",
            ],
            "Errores": [fn_s1 + fp_s1, fn_tr + fp_tr, fn_tr + fp_tr],
            "FN": [fn_s1, fn_tr, fn_tr],
            "FP": [fp_s1, fp_tr, fp_tr],
            "BA": [f"{ba_s1_train:.3f}", f"{ba_train_final:.3f}", f"{ba_train_final:.3f}"],
            "Cobertura zonas": ["0%", "0% (todos por S2)", f"{(verde.sum() + roja.sum()) * 100 / n_train:.0f}%"],
        }
    )
    st.dataframe(comp_data, hide_index=True, use_container_width=True)

else:
    st.info("Pulsa **▶️ Ejecutar Pipeline** para correr el sistema con la configuración seleccionada.")
