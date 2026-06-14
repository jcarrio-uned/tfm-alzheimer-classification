"""Módulo 2: Selección de Variables.

Pipeline completo de selección: L1 Stability → Forward → Engineering → Cleanup.
Reproduce exactamente la metodología del TFM que obtiene 7 metabolitos + 4 ingenieriles = 11 features.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Añadir path al módulo principal
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app_utils import (
    ORIGINAL_METABS,
    apply_winsorization,
    initialize_session_state,
    load_data,
)

# ============================================================================
# Configuración de la página
# ============================================================================

st.set_page_config(page_title="Selección de Variables", page_icon="🔍", layout="wide")

initialize_session_state()

# ============================================================================
# Header
# ============================================================================

st.title("🔍 Selección de Variables")
st.markdown("""
Pipeline completo de selección de variables del TFM:
1. **L1 Stability Selection** — identificar el núcleo estable de metabolitos
2. **Forward Selection** — añadir variables sub-umbral que aporten señal
3. **Feature Engineering** — crear ratios/productos con justificación biológica
4. **Cleanup post-engineering** — eliminar redundancias creadas por las ingenieriles

Resultado: **7 metabolitos base + 4 features ingenieriles = 11 features**.
""")

# ============================================================================
# Cargar datos
# ============================================================================

with st.spinner("Cargando datos..."):
    df = load_data()
    metabolites = [m for m in ORIGINAL_METABS if m in df.columns]

y = (df["Group"] == "AD").astype(int).values

# ============================================================================
# Sidebar: Parámetros ajustables
# ============================================================================

st.sidebar.header("⚙️ Parámetros")

st.sidebar.subheader("L1 Stability")
C_l1 = st.sidebar.select_slider(
    "C (Lasso)",
    options=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
    value=0.1,
    help="Regularización para el modelo L1. Menor C = más agresivo (menos variables estables).",
)
stability_threshold = st.sidebar.slider(
    "Umbral estabilidad (%)",
    min_value=10,
    max_value=90,
    value=50,
    step=5,
    help="Frecuencia mínima de selección para considerar una variable estable (estándar: 50%).",
)

st.sidebar.subheader("Forward Selection")
delta_forward = st.sidebar.number_input(
    "ΔBA mínimo",
    min_value=0.001,
    max_value=0.05,
    value=0.003,
    step=0.001,
    format="%.3f",
    help="Solo se añade una variable si mejora BA en más que este umbral.",
)

st.sidebar.subheader("Evaluación")
C_eval = st.sidebar.select_slider(
    "C (modelo final)",
    options=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
    value=0.1,
    help="Regularización para la evaluación de subconjuntos.",
)
n_splits = st.sidebar.select_slider(
    "N splits (SSS)",
    options=[50, 100, 150, 200, 300],
    value=200,
    help="Número de splits StratifiedShuffleSplit.",
)
winsorize_pct = st.sidebar.select_slider(
    "Winsorización (%)",
    options=[0.0, 0.01, 0.025, 0.05, 0.10],
    value=0.05,
)


# ============================================================================
# Función de evaluación
# ============================================================================


def evaluate_feature_set(feature_arrays, y_data, C, n_splits_eval, return_auc=False):
    """Evalúa un conjunto de features (array numpy) con SSS."""
    X_subset = feature_arrays
    sss = StratifiedShuffleSplit(n_splits=n_splits_eval, test_size=0.2, random_state=42)
    model = LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=42)

    ba_test_scores, ba_train_scores, auc_scores = [], [], []

    for train_idx, test_idx in sss.split(X_subset, y_data):
        X_train, X_test = X_subset[train_idx].copy(), X_subset[test_idx].copy()
        y_train, y_test = y_data[train_idx], y_data[test_idx]

        if winsorize_pct > 0:
            X_train = apply_winsorization(X_train, winsorize_pct)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model.fit(X_train_s, y_train)
        ba_test_scores.append(balanced_accuracy_score(y_test, model.predict(X_test_s)))
        ba_train_scores.append(balanced_accuracy_score(y_train, model.predict(X_train_s)))
        if return_auc:
            auc_scores.append(roc_auc_score(y_test, model.predict_proba(X_test_s)[:, 1]))

    result = (np.mean(ba_test_scores), np.std(ba_test_scores), np.mean(ba_train_scores))
    if return_auc:
        result += (np.mean(auc_scores),)
    return result


def get_feature_matrix(feature_names, df_data, eng_dict):
    """Construye la matriz X a partir de nombres de features (base + ingenieriles)."""
    cols = []
    for f in feature_names:
        if f in df_data.columns:
            cols.append(df_data[f].values)
        elif f in eng_dict:
            cols.append(eng_dict[f])
        else:
            raise ValueError(f"Feature '{f}' not found")
    return np.column_stack(cols)


# ============================================================================
# Ejecución del pipeline
# ============================================================================

if st.button("▶️ Ejecutar Pipeline Completo", type="primary"):
    progress = st.progress(0, text="Iniciando pipeline...")

    # ==================================================================
    # PASO 1: L1 STABILITY SELECTION
    # ==================================================================
    st.header("1️⃣ L1 Stability Selection")
    st.caption(
        f"LogReg-L1 (C={C_l1}) en {n_splits} splits SSS. "
        f"Variables con frecuencia > {stability_threshold}% son estables."
    )

    X_all = df[metabolites].values
    sss_stab = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)

    selection_counts = np.zeros(len(metabolites))
    for split_i, (train_idx, _test_idx) in enumerate(sss_stab.split(X_all, y)):
        if split_i % 40 == 0:
            progress.progress(
                int(5 + 20 * split_i / n_splits),
                text=f"L1 Stability: split {split_i}/{n_splits}...",
            )

        X_train = X_all[train_idx].copy()
        if winsorize_pct > 0:
            X_train = apply_winsorization(X_train, winsorize_pct)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)

        model_l1 = LogisticRegression(
            C=C_l1,
            penalty="l1",
            solver="saga",
            class_weight="balanced",
            max_iter=5000,
            random_state=42,
        )
        model_l1.fit(X_train_s, y[train_idx])
        selection_counts += (np.abs(model_l1.coef_[0]) > 1e-10).astype(int)

    selection_freq = selection_counts / n_splits * 100

    # Tabla de frecuencias
    stab_results = []
    for feat, freq in zip(metabolites, selection_freq, strict=False):
        stab_results.append(
            {
                "Variable": feat,
                "Frecuencia (%)": freq,
                "Estable": "✅" if freq >= stability_threshold else "",
            }
        )
    stab_df = pd.DataFrame(stab_results).sort_values("Frecuencia (%)", ascending=False).reset_index(drop=True)

    col_table, col_chart = st.columns([1, 1])
    with col_table:
        st.dataframe(
            stab_df.style.background_gradient(subset=["Frecuencia (%)"], cmap="YlGn").format(
                {"Frecuencia (%)": "{:.1f}"}
            ),
            use_container_width=True,
            height=min(500, len(metabolites) * 38),
        )

    # Gráfico
    with col_chart:
        fig_stab = go.Figure(
            go.Bar(
                x=stab_df["Frecuencia (%)"],
                y=stab_df["Variable"],
                orientation="h",
                marker={
                    "color": ["#27ae60" if f >= stability_threshold else "#bdc3c7" for f in stab_df["Frecuencia (%)"]]
                },
            )
        )
        fig_stab.add_vline(
            x=stability_threshold, line_dash="dash", line_color="red", annotation_text=f"Umbral {stability_threshold}%"
        )
        fig_stab.update_layout(
            title="Frecuencia de Selección L1",
            xaxis_title="Frecuencia (%)",
            yaxis_title="",
            height=max(400, len(metabolites) * 25),
            yaxis={"categoryorder": "total ascending"},
        )
        st.plotly_chart(fig_stab, use_container_width=True)

    # Variables estables
    stable_features = [metabolites[i] for i in range(len(metabolites)) if selection_freq[i] >= stability_threshold]

    # Evaluar núcleo estable
    X_stable = get_feature_matrix(stable_features, df, {})
    ba_stable, std_stable, ba_tr_stable = evaluate_feature_set(X_stable, y, C_eval, n_splits)

    st.success(
        f"**Núcleo estable: {len(stable_features)} variables** — "
        f"BA = {ba_stable:.3f} ± {std_stable:.3f}, Gap = {ba_tr_stable - ba_stable:.3f}"
    )
    st.markdown(f"Variables: `{'`, `'.join(stable_features)}`")

    # ==================================================================
    # PASO 2: FORWARD SELECTION
    # ==================================================================
    st.divider()
    st.header("2️⃣ Forward Selection")
    st.caption(f"Desde el núcleo estable, añadir variables que mejoren BA > {delta_forward:.3f}.")

    progress.progress(30, text="Forward Selection...")

    selected = stable_features.copy()
    remaining = [m for m in metabolites if m not in selected]
    ba_current = ba_stable

    forward_history = []
    fwd_step = 1
    while remaining:
        progress.progress(
            min(30 + int(20 * fwd_step / max(len(remaining), 1)), 50),
            text=f"Forward step {fwd_step}: evaluando {len(remaining)} candidatas...",
        )

        best_feat = None
        best_ba = ba_current
        best_std = 0
        best_ba_tr = 0

        for feat in remaining:
            test_set = [*selected, feat]
            X_test_arr = get_feature_matrix(test_set, df, {})
            ba_f, std_f, ba_tr_f = evaluate_feature_set(X_test_arr, y, C_eval, n_splits)
            if ba_f > best_ba + delta_forward:
                best_ba = ba_f
                best_std = std_f
                best_ba_tr = ba_tr_f
                best_feat = feat

        if best_feat is None:
            break

        selected.append(best_feat)
        remaining.remove(best_feat)
        ba_current = best_ba

        forward_history.append(
            {
                "Step": fwd_step,
                "Variable añadida": best_feat,
                "N features": len(selected),
                "BA test": best_ba,
                "±std": best_std,
                "Gap": best_ba_tr - best_ba,
            }
        )
        fwd_step += 1

    # Mostrar historial forward
    if forward_history:
        fwd_df = pd.DataFrame(forward_history)
        st.dataframe(
            fwd_df.style.background_gradient(subset=["BA test"], cmap="Greens").format(
                {"BA test": "{:.3f}", "±std": "{:.3f}", "Gap": "{:.3f}"}
            ),
            use_container_width=True,
        )
    else:
        st.info("No se añadió ninguna variable.")

    st.success(f"**Después de Forward: {len(selected)} metabolitos** — BA = {ba_current:.3f}")
    st.markdown(f"Variables: `{'`, `'.join(selected)}`")

    # ==================================================================
    # PASO 3: FEATURE ENGINEERING
    # ==================================================================
    st.divider()
    st.header("3️⃣ Feature Engineering")
    st.caption("Crear ratios y productos con criterio dual (estadístico + biológico).")

    progress.progress(55, text="Feature Engineering...")

    # Definir las 4 features ingenieriles
    eng_definitions = {
        "DHEAS/lysoPC": (
            "DHEAS",
            "lysoPC.a.C18:2",
            "ratio",
            "↓/↑ → ratio amplifica | Neuroprotección/neuroinflamación",
        ),
        "DOPA*DHEAS": ("DOPA", "DHEAS", "product", "↓×↓ → producto doble déficit | Dopamina-esteroide"),
        "PC*DOPA": ("PC.aa.C40:4", "DOPA", "product", "↑×↓ → desacoplamiento | Membrana-neurotransmisión"),
        "DHEAS/Cer20": (
            "DHEAS",
            "Cer(d18:1/20:0)",
            "ratio",
            "↓/↑ → ratio amplifica | Supervivencia/apoptosis neuronal",
        ),
    }

    # Generar
    eng_data = {}
    eng_table = []
    for eng_name, (var_a, var_b, eng_type, justification) in eng_definitions.items():
        if var_a in df.columns and var_b in df.columns:
            if eng_type == "ratio":
                eng_data[eng_name] = df[var_a].values / (df[var_b].values + 1e-10)
            else:
                eng_data[eng_name] = df[var_a].values * df[var_b].values
            eng_table.append(
                {
                    "Feature": eng_name,
                    "Tipo": eng_type,
                    "Variables": f"{var_a}, {var_b}",
                    "Justificación": justification,
                }
            )

    st.dataframe(pd.DataFrame(eng_table), use_container_width=True, hide_index=True)

    # Evaluar incrementalmente
    st.subheader("Evaluación incremental")
    eng_names = list(eng_data.keys())

    base_met = selected.copy()
    X_base = get_feature_matrix(base_met, df, eng_data)
    ba_base, std_base, ba_tr_base = evaluate_feature_set(X_base, y, C_eval, n_splits)

    eng_eval_rows = [
        {
            "Config": f"{len(base_met)} met (base)",
            "N features": len(base_met),
            "BA test": ba_base,
            "Gap": ba_tr_base - ba_base,
        }
    ]

    current_eng = []
    for eng_name in eng_names:
        current_eng.append(eng_name)
        test_feats = base_met + current_eng
        X_test_arr = get_feature_matrix(test_feats, df, eng_data)
        ba_e, std_e, ba_tr_e = evaluate_feature_set(X_test_arr, y, C_eval, n_splits)
        eng_eval_rows.append(
            {
                "Config": f"+ {eng_name}",
                "N features": len(test_feats),
                "BA test": ba_e,
                "Gap": ba_tr_e - ba_e,
            }
        )

    eng_eval_df = pd.DataFrame(eng_eval_rows)
    st.dataframe(
        eng_eval_df.style.background_gradient(subset=["BA test"], cmap="Greens").format(
            {"BA test": "{:.3f}", "Gap": "{:.3f}"}
        ),
        use_container_width=True,
    )

    # Full set with all engineering
    selected_with_eng = base_met + eng_names
    X_full_eng = get_feature_matrix(selected_with_eng, df, eng_data)
    ba_full, std_full, ba_tr_full = evaluate_feature_set(X_full_eng, y, C_eval, n_splits)

    st.success(
        f"**Con 4 ingenieriles: {len(selected_with_eng)}f** — "
        f"BA = {ba_full:.3f} ± {std_full:.3f}, Gap = {ba_tr_full - ba_full:.3f}"
    )

    # ==================================================================
    # PASO 4: CLEANUP POST-ENGINEERING
    # ==================================================================
    st.divider()
    st.header("4️⃣ Cleanup Post-Engineering")
    st.caption(
        "Eliminar metabolitos base que se vuelven redundantes tras añadir las ingenieriles. "
        "Se elimina si BA mejora o se mantiene sin él."
    )

    progress.progress(75, text="Cleanup post-engineering...")

    # Intentar eliminar cada metabolito base
    ba_current_cleanup = ba_full
    final_features = selected_with_eng.copy()

    cleanup_history = []
    improved = True
    while improved:
        improved = False
        worst_feat = None
        best_ba_after = 0
        best_std_after = 0
        best_ba_tr_after = 0

        # Solo probar eliminar metabolitos base, no ingenieriles
        base_in_final = [f for f in final_features if f in metabolites]

        for feat in base_in_final:
            test_set = [f for f in final_features if f != feat]
            X_test_arr = get_feature_matrix(test_set, df, eng_data)
            ba_r, std_r, ba_tr_r = evaluate_feature_set(X_test_arr, y, C_eval, n_splits)

            if ba_r >= ba_current_cleanup and ba_r > best_ba_after:
                best_ba_after = ba_r
                best_std_after = std_r
                best_ba_tr_after = ba_tr_r
                worst_feat = feat

        if worst_feat and best_ba_after >= ba_current_cleanup:
            final_features.remove(worst_feat)
            ba_current_cleanup = best_ba_after
            cleanup_history.append(
                {
                    "Eliminada": worst_feat,
                    "N features": len(final_features),
                    "BA test": best_ba_after,
                    "±std": best_std_after,
                    "Gap": best_ba_tr_after - best_ba_after,
                    "Motivo": "Redundante con ingenieriles (BA mejora o se mantiene)",
                }
            )
            improved = True

    if cleanup_history:
        clean_df = pd.DataFrame(cleanup_history)
        st.dataframe(
            clean_df.style.format({"BA test": "{:.3f}", "±std": "{:.3f}", "Gap": "{:.3f}"}),
            use_container_width=True,
        )
    else:
        st.info("No se eliminó ningún metabolito (todos son necesarios).")

    # ==================================================================
    # RESULTADO FINAL
    # ==================================================================
    st.divider()
    st.header("🏆 Modelo Final")

    progress.progress(90, text="Evaluación final...")

    # Evaluación final con AUC
    X_final = get_feature_matrix(final_features, df, eng_data)
    ba_final, std_final, ba_tr_final, auc_final = evaluate_feature_set(X_final, y, C_eval, n_splits, return_auc=True)

    progress.progress(100, text="✅ Pipeline completado")

    # Separar base e ingenieriles
    final_base = [f for f in final_features if f in metabolites]
    final_eng = [f for f in final_features if f in eng_data]

    st.success(
        f"**{len(final_features)} features** ({len(final_base)} metabolitos + {len(final_eng)} ingenieriles) — "
        f"BA = {ba_final:.3f} ± {std_final:.3f}, Gap = {ba_tr_final - ba_final:.3f}, AUC = {auc_final:.3f}"
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Metabolitos base:**")
        for i, feat in enumerate(final_base, 1):
            st.markdown(f"{i}. `{feat}`")

    with col2:
        st.markdown("**Features ingenieriles:**")
        for i, feat in enumerate(final_eng, 1):
            info = eng_definitions.get(feat, ("", "", "", ""))
            st.markdown(f"{i}. `{feat}` — {info[3].split('|')[1].strip() if '|' in info[3] else ''}")

    # Métricas
    st.subheader("📊 Métricas")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("BA test", f"{ba_final:.3f} ± {std_final:.3f}")
    m2.metric("BA train", f"{ba_tr_final:.3f}")
    m3.metric("Gap", f"{ba_tr_final - ba_final:.3f}")
    m4.metric("AUC", f"{auc_final:.3f}")

    # Gráfico de importancia: evaluar eliminando cada feature
    st.subheader("📈 Importancia por eliminación")
    importance_rows = []
    for feat in final_features:
        test_set = [f for f in final_features if f != feat]
        X_test_arr = get_feature_matrix(test_set, df, eng_data)
        ba_without, _, _ = evaluate_feature_set(X_test_arr, y, C_eval, n_splits)
        importance_rows.append(
            {
                "Feature": feat,
                "BA sin ella": ba_without,
                "ΔBA": ba_final - ba_without,
                "Tipo": "Ingenieril" if feat in eng_data else "Metabolito",
            }
        )

    imp_df = pd.DataFrame(importance_rows).sort_values("ΔBA", ascending=False)

    fig_imp = go.Figure(
        go.Bar(
            x=imp_df["ΔBA"],
            y=imp_df["Feature"],
            orientation="h",
            marker={"color": ["#e74c3c" if t == "Metabolito" else "#3498db" for t in imp_df["Tipo"]]},
        )
    )
    fig_imp.update_layout(
        title="Importancia por eliminación (ΔBA al quitar cada feature)",
        xaxis_title="ΔBA (caída al eliminar)",
        height=max(350, len(final_features) * 35),
        yaxis={"categoryorder": "total ascending"},
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    st.dataframe(
        imp_df.style.background_gradient(subset=["ΔBA"], cmap="Reds").format(
            {"BA sin ella": "{:.3f}", "ΔBA": "{:.3f}"}
        ),
        use_container_width=True,
    )

    # Guardar en session state
    st.session_state["selected_features"] = final_features
    st.session_state["selected_base_features"] = final_base
    st.session_state["selected_eng_features"] = final_eng
    st.session_state["eng_data"] = eng_data
    st.session_state["custom_features"] = final_features
    # Actualizar el widget del selector manual para que refleje las features encontradas
    st.session_state["manual_var_selector"] = final_features

# ============================================================================
# Sección: Selector Manual de Variables + Métricas
# ============================================================================

st.divider()
st.header("📋 Selector Manual de Variables")

st.markdown("""
Selecciona manualmente las variables con las que deseas trabajar.
Tras ejecutar la búsqueda automática, el selector se actualiza con las features encontradas.
Se muestra **AUC univariante**, **correlación con la variable objetivo** y **VIF**.
""")

# Todas las variables disponibles (originales + engineered)
_ALL_SELECTABLE = metabolites + [
    c
    for c in df.columns
    if c not in [*metabolites, "ID", "Group", "Sample", "Age [y]", "Sex", "Fasting time [h]"]
    and pd.api.types.is_numeric_dtype(df[c])
]

# Inicializar el widget key si no existe
if "manual_var_selector" not in st.session_state:
    # Default: las 20 metabolites originales
    st.session_state["manual_var_selector"] = metabolites

selected_vars = st.multiselect(
    "Variables seleccionadas",
    options=_ALL_SELECTABLE,
    key="manual_var_selector",
    help="Añade o elimina variables. Tras la búsqueda automática se actualizan a las encontradas.",
)

if selected_vars:
    # Guardar selección manual en session state (para downstream pages)
    st.session_state["selected_features"] = selected_vars

    # Preparar target binario
    y_bin = (df["Group"] == "AD").astype(int)

    # --- Calcular métricas ---
    results = []
    for feat in selected_vars:
        vals = df[feat].values

        # AUC univariante
        try:
            auc = roc_auc_score(y_bin, vals)
            auc = max(auc, 1 - auc)
        except Exception:
            auc = np.nan

        # Correlación punto-biserial con target
        try:
            r_pb, p_pb = stats.pointbiserialr(y_bin, vals)
        except Exception:
            r_pb, p_pb = np.nan, np.nan

        results.append(
            {
                "Variable": feat,
                "AUC univariante": auc,
                "r (point-biserial)": r_pb,
                "p-valor": p_pb,
            }
        )

    metrics_df = pd.DataFrame(results).set_index("Variable")

    # --- VIF ---
    X_sel = df[selected_vars].dropna()
    if len(selected_vars) >= 2:
        X_sel_arr = X_sel.values
        vif_values = []
        for i in range(X_sel_arr.shape[1]):
            try:
                vif_values.append(variance_inflation_factor(X_sel_arr, i))
            except Exception:
                vif_values.append(np.nan)
        metrics_df["VIF"] = vif_values
    else:
        metrics_df["VIF"] = np.nan

    # --- Mostrar tabla ---
    st.subheader("📊 Métricas de las variables seleccionadas")

    styled = metrics_df.style.format(
        {
            "AUC univariante": "{:.3f}",
            "r (point-biserial)": "{:.3f}",
            "p-valor": "{:.2e}",
            "VIF": "{:.2f}",
        }
    ).background_gradient(subset=["AUC univariante"], cmap="Greens", vmin=0.5, vmax=1.0)

    if "VIF" in metrics_df.columns:
        styled = styled.background_gradient(subset=["VIF"], cmap="Reds", vmin=1, vmax=10)

    st.dataframe(styled, use_container_width=True)

    # --- Gráficos ---
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("🎯 AUC Univariante")
        auc_sorted = metrics_df["AUC univariante"].sort_values(ascending=True)
        fig = go.Figure(
            go.Bar(
                x=auc_sorted.values,
                y=auc_sorted.index,
                orientation="h",
                marker_color=[
                    "#2ecc71" if v >= 0.7 else "#f39c12" if v >= 0.6 else "#e74c3c" for v in auc_sorted.values
                ],
            )
        )
        fig.add_vline(x=0.5, line_dash="dash", line_color="gray")
        fig.update_layout(
            xaxis_title="AUC",
            yaxis_title="",
            xaxis_range=[0.4, 1.0],
            height=max(300, len(selected_vars) * 28),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("⚠️ VIF (Multicolinealidad)")
        if "VIF" in metrics_df.columns and len(selected_vars) >= 2:
            vif_sorted = metrics_df["VIF"].sort_values(ascending=True)
            fig = go.Figure(
                go.Bar(
                    x=vif_sorted.values,
                    y=vif_sorted.index,
                    orientation="h",
                    marker_color=[
                        "#e74c3c" if v > 5 else "#f39c12" if v > 2.5 else "#2ecc71" for v in vif_sorted.values
                    ],
                )
            )
            fig.add_vline(x=5, line_dash="dash", line_color="red", annotation_text="Umbral=5")
            fig.update_layout(
                xaxis_title="VIF",
                yaxis_title="",
                height=max(300, len(selected_vars) * 28),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Se necesitan ≥ 2 variables para calcular VIF")
else:
    st.warning("Selecciona al menos una variable para ver las métricas.")

# ============================================================================
# Sección informativa (siempre visible)
# ============================================================================

st.divider()

with st.expander("ℹ️ Descripción del pipeline"):
    st.markdown("""
    ### Metodología

    **PASO 1 — Stability Selection (L1/Lasso)**

    Se entrena LogReg con penalización L1 en cada split SSS. Para cada variable se cuenta
    la frecuencia con que tiene coeficiente ≠ 0. Las que superan el umbral (por defecto 50%)
    forman el **núcleo estable**. Esto garantiza que el punto de partida sean variables
    robustas, no dependientes de un subconjunto particular del dataset.

    **PASO 2 — Forward Selection**

    Desde el núcleo estable, se evalúa añadir cada variable del pool restante. Se selecciona
    la que más mejora el BA, siempre que supere ΔBA > umbral. Se repite hasta que ninguna
    candidata mejore.

    **PASO 3 — Feature Engineering**

    Se crean 4 features basadas en la convergencia de dos criterios:
    - **Criterio estadístico**: dirección del efecto (Cohen's d) determina si ratio o producto amplifica la señal
    - **Criterio biológico**: cada combinación tiene interpretación fisiopatológica

    | Feature | Fórmula | Justificación |
    |---|---|---|
    | DHEAS/lysoPC | DHEAS ÷ lysoPC.a.C18:2 | Neuroprotección vs neuroinflamación |
    | DOPA*DHEAS | DOPA × DHEAS | Doble déficit dopaminérgico-esteroideo |
    | PC*DOPA | PC.aa.C40:4 × DOPA | Desacoplamiento membrana-neurotransmisión |
    | DHEAS/Cer20 | DHEAS ÷ Cer(d18:1/20:0) | Supervivencia vs apoptosis neuronal |

    **PASO 4 — Cleanup post-engineering**

    Las ingenieriles pueden hacer redundantes algunos metabolitos base (la ratio captura
    la misma información). Se elimina iterativamente cualquier metabolito cuya eliminación
    no baje el BA. Típicamente TG(18:2_38:5) y CE(22:5) se eliminan porque DHEAS/lysoPC
    y el bloque de ingenieriles capturan su señal.

    ---

    **Resultado esperado** (con parámetros por defecto):
    - 7 metabolitos: DOPA, Cer(d18:1/20:0), lysoPC.a.C18:2, PC.aa.C40:4, DHEAS, Arg, HexCer(d18:1/26:1)
    - 4 ingenieriles: DHEAS/lysoPC, DOPA*DHEAS, PC*DOPA, DHEAS/Cer20
    - BA ≈ 0.863, Gap ≈ 0.015, AUC ≈ 0.911
    """)

# ============================================================================
# Footer
# ============================================================================

st.markdown("---")
st.caption("Módulo 2: Selección de Variables | TFM Alzheimer Classification")
