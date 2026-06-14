"""Módulo 5: Análisis de Errores.

Análisis detallado de falsos positivos y falsos negativos con alertas metabólicas.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Añadir path al módulo principal
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app_utils import (
    calculate_effect_size,
    check_atypical_markers,
    get_metabolite_columns,
    initialize_session_state,
    load_data,
)

# ============================================================================
# Configuración de la página
# ============================================================================

st.set_page_config(page_title="Análisis de Errores", page_icon="⚠️", layout="wide")

initialize_session_state()

# ============================================================================
# Header
# ============================================================================

st.title("⚠️ Análisis de Errores")
st.markdown("""
Identificación y análisis detallado de errores de clasificación (FP y FN).
Sistema de alertas metabólicas para detectar marcadores que contradicen la predicción.
""")

st.info("📊 **Análisis basado en predicciones de Cross-Validation** (sin data leakage)")

# ============================================================================
# Verificar que hay modelo entrenado
# ============================================================================

if "model_trained" not in st.session_state or not st.session_state.get("model_trained", False):
    st.error("❌ No hay modelo entrenado")
    st.info("👉 Ve al módulo **📈 Evaluación y Resultados** para entrenar el modelo")
    st.stop()

# ============================================================================
# Cargar datos y resultados
# ============================================================================

df = load_data()
metabolites = get_metabolite_columns(df)

# Obtener predicciones y etiquetas verdaderas
# IMPORTANTE: usar y_true guardado en session_state que corresponde al mismo subconjunto que las predicciones
y_pred = st.session_state["predictions"]
y_proba = st.session_state["probabilities"]
y_true = st.session_state.get("y_true", (df["Group"] == "AD").astype(int).values)

# Verificar que las dimensiones coinciden
if len(y_true) != len(y_pred):
    st.error(f"❌ Error: Dimensiones no coinciden. y_true={len(y_true)}, y_pred={len(y_pred)}")
    st.info("Esto puede ocurrir si se usó train/test split. Asegúrate de entrenar el modelo primero.")
    st.stop()

sample_ids = df["ID"].values if "ID" in df.columns else np.arange(len(df))

# Ajustar sample_ids y df si las predicciones son de un subconjunto
if len(y_pred) < len(df):
    # Las predicciones son de un subconjunto (probablemente train set)
    # Usar los índices guardados para alinear correctamente
    st.warning(f"⚠️ Análisis basado en {len(y_pred)} muestras (subconjunto de entrenamiento)")

    train_indices = st.session_state.get("train_indices", None)
    if train_indices is not None:
        # Usar los índices exactos del train set
        df = df.iloc[train_indices]
        sample_ids = sample_ids[train_indices]
    else:
        # Fallback: usar primeros índices (no recomendado pero mantiene compatibilidad)
        st.warning("⚠️ No se encontraron índices de entrenamiento. Usando primeras muestras.")
        df = df.iloc[: len(y_pred)]
        sample_ids = sample_ids[: len(y_pred)]

    # CRÍTICO: Resetear índices para que coincidan con posiciones en arrays (0, 1, 2, ...)
    df = df.reset_index(drop=True)
    sample_ids = sample_ids if isinstance(sample_ids, np.ndarray) else np.array(sample_ids)

# Identificar errores
tn_mask = (y_true == 0) & (y_pred == 0)
tp_mask = (y_true == 1) & (y_pred == 1)
fn_mask = (y_true == 1) & (y_pred == 0)  # Real AD, predicho NC
fp_mask = (y_true == 0) & (y_pred == 1)  # Real NC, predicho AD

# Contar
n_tn = tn_mask.sum()
n_tp = tp_mask.sum()
n_fn = fn_mask.sum()
n_fp = fp_mask.sum()

# ============================================================================
# Sidebar: Configuración
# ============================================================================

with st.sidebar:
    st.header("⚙️ Configuración")

    st.metric("Total Muestras", len(df))
    st.metric("True Positives (TP)", n_tp)
    st.metric("True Negatives (TN)", n_tn)
    st.metric("False Negatives (FN)", n_fn)
    st.metric("False Positives (FP)", n_fp)

    st.markdown("---")

    effect_size_threshold = st.slider(
        "Umbral de Cohen's d para alertas",
        min_value=0.3,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Solo se muestran alertas para marcadores con |d| mayor a este umbral",
    )

# ============================================================================
# Sección 1: Resumen de Errores
# ============================================================================

st.header("1️⃣ Resumen de Errores")

col1, col2 = st.columns(2)

with col1:
    # Falsos Negativos
    st.subheader(f"🚨 Falsos Negativos (FN = {n_fn})")
    st.markdown("""
    **Real**: AD | **Predicción**: NC
    Casos de Alzheimer que el modelo no detectó.
    """)

    if n_fn > 0:
        fn_rate = n_fn / (n_fn + n_tp)
        st.error(f"Tasa FN: {fn_rate:.1%}")
    else:
        st.success("✓ No se detectaron falsos negativos")

with col2:
    # Falsos Positivos
    st.subheader(f"⚠️ Falsos Positivos (FP = {n_fp})")
    st.markdown("""
    **Real**: NC | **Predicción**: AD
    Controles sanos clasificados erróneamente como AD.
    """)

    if n_fp > 0:
        fp_rate = n_fp / (n_fp + n_tn)
        st.warning(f"Tasa FP: {fp_rate:.1%}")
    else:
        st.success("✓ No se detectaron falsos positivos")

# Gráfico de distribución de probabilidades por categoría
st.subheader("📊 Distribución de Probabilidades por Categoría")

prob_df = pd.DataFrame(
    {
        "ID": sample_ids,
        "Grupo Real": ["AD" if y == 1 else "NC" for y in y_true],
        "Predicción": ["AD" if y == 1 else "NC" for y in y_pred],
        "P(AD)": y_proba,
        "Categoría": [
            "True Positive" if tp else "True Negative" if tn else "False Negative" if fn else "False Positive"
            for tp, tn, fn, fp in zip(tp_mask, tn_mask, fn_mask, fp_mask, strict=False)
        ],
    }
)

fig = px.box(
    prob_df,
    x="Categoría",
    y="P(AD)",
    color="Categoría",
    points="all",
    hover_data=["ID"],
    color_discrete_map={
        "True Positive": "green",
        "True Negative": "lightblue",
        "False Negative": "red",
        "False Positive": "orange",
    },
)

fig.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="Umbral=0.5")

fig.update_layout(
    title="Distribución de Probabilidades por Categoría de Predicción", yaxis_title="Probabilidad P(AD)", height=500
)

st.plotly_chart(fig, width="stretch")

# ============================================================================
# Calcular estadísticas de referencia (TP vs TN)
# ============================================================================


@st.cache_data
def compute_reference_stats(_df, _metabolites):
    """Calcula estadísticas de referencia para AD y NC usando TP y TN."""
    tp_mask = (_df["Group"] == "AD") & (st.session_state["predictions"] == 1)
    tn_mask = (_df["Group"] == "NC") & (st.session_state["predictions"] == 0)

    stats = {}

    for metabolite in _metabolites:
        ad_values = _df.loc[tp_mask, metabolite].values
        nc_values = _df.loc[tn_mask, metabolite].values

        stats[metabolite] = {
            "ad_mean": ad_values.mean(),
            "ad_std": ad_values.std(),
            "nc_mean": nc_values.mean(),
            "nc_std": nc_values.std(),
            "effect_size": calculate_effect_size(ad_values, nc_values),
        }

    return pd.DataFrame(stats).T


reference_stats = compute_reference_stats(df, metabolites)

# ============================================================================
# Sección 2: Análisis de Falsos Negativos
# ============================================================================

st.header("2️⃣ Análisis Detallado de Falsos Negativos")

if n_fn == 0:
    st.success("✓ No hay falsos negativos en este modelo")
else:
    st.markdown(f"""
    Análisis de {n_fn} muestras clasificadas como **NC** pero que son realmente **AD**.
    Se muestran alertas para marcadores metabólicos que van en contra de la predicción.
    """)

    # Tabla de FN
    fn_df = df[fn_mask].copy()
    fn_df["P(AD)"] = y_proba[fn_mask]
    fn_df["Predicción"] = "NC"
    fn_df["Real"] = "AD"

    # Calcular número de alertas para cada muestra
    fn_alerts_count = []

    for idx in fn_df.index:
        sample_values = df.loc[idx, metabolites].to_dict()
        alerts = check_atypical_markers(
            sample_values,
            predicted_class="NC",  # Predicción errónea
            reference_stats=reference_stats,
            threshold=effect_size_threshold,
        )
        fn_alerts_count.append(len(alerts))

    fn_df["N_Alertas"] = fn_alerts_count

    # Ordenar por número de alertas (descendente)
    fn_df_display = fn_df[["ID", "Real", "Predicción", "P(AD)", "N_Alertas"]].sort_values("N_Alertas", ascending=False)

    st.dataframe(fn_df_display.style.background_gradient(subset=["N_Alertas"], cmap="Reds"), width="stretch")

    st.info(f"💡 **Promedio de alertas por FN**: {np.mean(fn_alerts_count):.1f}")

    # Selector de muestra FN para análisis detallado
    st.subheader("🔍 Análisis Individual de Muestra FN")

    selected_fn_id = st.selectbox(
        "Selecciona una muestra FN para análisis detallado", fn_df["ID"].values, key="fn_selector"
    )

    # Obtener índice de la muestra (usar iloc para obtener posición, no índice del DF)
    selected_fn_pos = fn_df[fn_df["ID"] == selected_fn_id].index[0]
    selected_fn_proba = y_proba[selected_fn_pos]

    # Información general
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("ID", selected_fn_id)
    with col2:
        st.metric("P(AD)", f"{selected_fn_proba:.3f}")
    with col3:
        n_alerts_sample = fn_df[fn_df["ID"] == selected_fn_id]["N_Alertas"].values[0]
        st.metric("Alertas", n_alerts_sample)

    # Generar alertas
    sample_values = df.loc[selected_fn_pos, metabolites].to_dict()
    alerts = check_atypical_markers(
        sample_values, predicted_class="NC", reference_stats=reference_stats, threshold=effect_size_threshold
    )

    # Ordenar alertas por severidad (effect size descendente)
    alerts = sorted(alerts, key=lambda x: abs(x["effect_size"]), reverse=True)

    # Mostrar alertas (solo las más relevantes)
    if alerts:
        n_alerts_to_show = min(len(alerts), 5)  # Mostrar máximo 5 alertas más relevantes
        st.warning(f"⚠️ {len(alerts)} marcador(es) atípico(s) detectado(s) - Mostrando top {n_alerts_to_show}")

        for alert in alerts[:n_alerts_to_show]:
            severity_icon = "🔴" if alert["severity"] == "HIGH" else "🟡" if alert["severity"] == "MEDIUM" else "🔵"
            severity_color = (
                "red" if alert["severity"] == "HIGH" else "orange" if alert["severity"] == "MEDIUM" else "lightblue"
            )

            with st.expander(f"{severity_icon} {alert['severity']} - {alert['metabolito']}", expanded=False):
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown(f"**Valor de la muestra**: `{alert['valor']:.3f}`")
                    st.markdown(f"**Media en AD (TP)**: `{alert['ad_mean']:.3f}`")
                    st.markdown(f"**Media en NC (TN)**: `{alert['nc_mean']:.3f}`")
                    st.markdown(f"**Cohen's d**: `{alert['effect_size']:.3f}`")
                    st.markdown("---")
                    st.info(alert["mensaje"])

                with col2:
                    # Mini gráfico comparativo
                    fig_mini = go.Figure()

                    fig_mini.add_trace(
                        go.Bar(
                            x=["AD (TP)", "NC (TN)", "Muestra"],
                            y=[alert["ad_mean"], alert["nc_mean"], alert["valor"]],
                            marker_color=["green", "blue", "red"],
                        )
                    )

                    fig_mini.update_layout(
                        title=f"{alert['metabolito']}", yaxis_title="Valor", height=250, showlegend=False
                    )

                    st.plotly_chart(fig_mini, width="stretch")
    else:
        st.success("✓ No se detectaron marcadores atípicos (con el umbral actual)")

    # Comparación con TP
    st.subheader("📊 Comparación FN vs TP")

    st.markdown("""
    Comparación de valores metabólicos entre **Falsos Negativos** y **True Positives** (AD correctamente clasificados).
    """)

    # Calcular medias usando boolean indexing (más robusto que .loc)
    fn_means = df[fn_mask][metabolites].mean()
    tp_means = df[tp_mask][metabolites].mean()

    comparison_df = pd.DataFrame(
        {
            "Metabolito": metabolites,
            "Media FN": fn_means.values,
            "Media TP": tp_means.values,
            "Diferencia": fn_means.values - tp_means.values,
        }
    )

    # Calcular effect sizes
    effect_sizes = []
    for metabolite in metabolites:
        fn_values = df[fn_mask][metabolite].values
        tp_values = df[tp_mask][metabolite].values
        d = calculate_effect_size(fn_values, tp_values)
        effect_sizes.append(d)

    comparison_df["Cohen's d"] = effect_sizes
    comparison_df = comparison_df.sort_values("Cohen's d", key=abs, ascending=False)

    st.dataframe(
        comparison_df.style.format(
            {"Media FN": "{:.3f}", "Media TP": "{:.3f}", "Diferencia": "{:.3f}", "Cohen's d": "{:.3f}"}
        ).background_gradient(subset=["Cohen's d"], cmap="RdBu_r", vmin=-1, vmax=1),
        width="stretch",
    )

# ============================================================================
# Sección 3: Análisis de Falsos Positivos
# ============================================================================

st.header("3️⃣ Análisis Detallado de Falsos Positivos")

if n_fp == 0:
    st.success("✓ No hay falsos positivos en este modelo")
else:
    st.markdown(f"""
    Análisis de {n_fp} muestras clasificadas como **AD** pero que son realmente **NC**.
    Se muestran alertas para marcadores metabólicos que van en contra de la predicción.
    """)

    # Tabla de FP
    fp_df = df[fp_mask].copy()
    fp_df["P(AD)"] = y_proba[fp_mask]
    fp_df["Predicción"] = "AD"
    fp_df["Real"] = "NC"

    # Calcular número de alertas
    fp_alerts_count = []

    for idx in fp_df.index:
        sample_values = df.loc[idx, metabolites].to_dict()
        alerts = check_atypical_markers(
            sample_values,
            predicted_class="AD",  # Predicción errónea
            reference_stats=reference_stats,
            threshold=effect_size_threshold,
        )
        fp_alerts_count.append(len(alerts))

    fp_df["N_Alertas"] = fp_alerts_count

    # Ordenar por número de alertas
    fp_df_display = fp_df[["ID", "Real", "Predicción", "P(AD)", "N_Alertas"]].sort_values("N_Alertas", ascending=False)

    st.dataframe(fp_df_display.style.background_gradient(subset=["N_Alertas"], cmap="Oranges"), width="stretch")

    st.info(f"💡 **Promedio de alertas por FP**: {np.mean(fp_alerts_count):.1f}")

    # Selector de muestra FP
    st.subheader("🔍 Análisis Individual de Muestra FP")

    selected_fp_id = st.selectbox(
        "Selecciona una muestra FP para análisis detallado", fp_df["ID"].values, key="fp_selector"
    )

    # Obtener índice de la muestra (usar posición en el array, no índice del DF)
    selected_fp_pos = fp_df[fp_df["ID"] == selected_fp_id].index[0]
    selected_fp_proba = y_proba[selected_fp_pos]

    # Información general
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("ID", selected_fp_id)
    with col2:
        st.metric("P(AD)", f"{selected_fp_proba:.3f}")
    with col3:
        n_alerts_sample = fp_df[fp_df["ID"] == selected_fp_id]["N_Alertas"].values[0]
        st.metric("Alertas", n_alerts_sample)

    # Generar alertas
    sample_values = df.loc[selected_fp_pos, metabolites].to_dict()
    alerts = check_atypical_markers(
        sample_values, predicted_class="AD", reference_stats=reference_stats, threshold=effect_size_threshold
    )

    # Ordenar alertas por severidad (effect size descendente)
    alerts = sorted(alerts, key=lambda x: abs(x["effect_size"]), reverse=True)

    # Mostrar alertas (solo las más relevantes)
    if alerts:
        n_alerts_to_show = min(len(alerts), 5)  # Mostrar máximo 5 alertas más relevantes
        st.warning(f"⚠️ {len(alerts)} marcador(es) atípico(s) detectado(s) - Mostrando top {n_alerts_to_show}")

        for alert in alerts[:n_alerts_to_show]:
            severity_icon = "🔴" if alert["severity"] == "HIGH" else "🟡" if alert["severity"] == "MEDIUM" else "🔵"

            with st.expander(f"{severity_icon} {alert['severity']} - {alert['metabolito']}", expanded=False):
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown(f"**Valor de la muestra**: `{alert['valor']:.3f}`")
                    st.markdown(f"**Media en AD (TP)**: `{alert['ad_mean']:.3f}`")
                    st.markdown(f"**Media en NC (TN)**: `{alert['nc_mean']:.3f}`")
                    st.markdown(f"**Cohen's d**: `{alert['effect_size']:.3f}`")
                    st.markdown("---")
                    st.info(alert["mensaje"])

                with col2:
                    # Mini gráfico
                    fig_mini = go.Figure()

                    fig_mini.add_trace(
                        go.Bar(
                            x=["AD (TP)", "NC (TN)", "Muestra"],
                            y=[alert["ad_mean"], alert["nc_mean"], alert["valor"]],
                            marker_color=["green", "blue", "orange"],
                        )
                    )

                    fig_mini.update_layout(
                        title=f"{alert['metabolito']}", yaxis_title="Valor", height=250, showlegend=False
                    )

                    st.plotly_chart(fig_mini, width="stretch")
    else:
        st.success("✓ No se detectaron marcadores atípicos (con el umbral actual)")

    # Comparación con TN
    st.subheader("📊 Comparación FP vs TN")

    st.markdown("""
    Comparación de valores metabólicos entre **Falsos Positivos** y **True Negatives** (NC correctamente clasificados).
    """)

    # Calcular medias usando boolean indexing (más robusto que .loc)
    fp_means = df[fp_mask][metabolites].mean()
    tn_means = df[tn_mask][metabolites].mean()

    comparison_df = pd.DataFrame(
        {
            "Metabolito": metabolites,
            "Media FP": fp_means.values,
            "Media TN": tn_means.values,
            "Diferencia": fp_means.values - tn_means.values,
        }
    )

    # Effect sizes
    effect_sizes = []
    for metabolite in metabolites:
        fp_values = df[fp_mask][metabolite].values
        tn_values = df[tn_mask][metabolite].values
        d = calculate_effect_size(fp_values, tn_values)
        effect_sizes.append(d)

    comparison_df["Cohen's d"] = effect_sizes
    comparison_df = comparison_df.sort_values("Cohen's d", key=abs, ascending=False)

    st.dataframe(
        comparison_df.style.format(
            {"Media FP": "{:.3f}", "Media TN": "{:.3f}", "Diferencia": "{:.3f}", "Cohen's d": "{:.3f}"}
        ).background_gradient(subset=["Cohen's d"], cmap="RdBu_r", vmin=-1, vmax=1),
        width="stretch",
    )

# ============================================================================
# Sección 4: Resumen y Exportación
# ============================================================================

st.header("4️⃣ Resumen y Exportación")

# Generar reporte completo
if st.button("📄 Generar Reporte Completo de Errores"):
    with st.spinner("Generando reporte..."):
        # FN report
        fn_report = []
        for idx in df[fn_mask].index:
            sample_values = df.loc[idx, metabolites].to_dict()
            alerts = check_atypical_markers(
                sample_values, predicted_class="NC", reference_stats=reference_stats, threshold=effect_size_threshold
            )

            fn_report.append(
                {
                    "ID": df.loc[idx, "ID"] if "ID" in df.columns else idx,
                    "Tipo": "Falso Negativo",
                    "P(AD)": y_proba[idx],
                    "N_Alertas": len(alerts),
                    "Alertas": "; ".join([f"{a['metabolito']} ({a['severity']})" for a in alerts]),
                }
            )

        # FP report
        fp_report = []
        for idx in df[fp_mask].index:
            sample_values = df.loc[idx, metabolites].to_dict()
            alerts = check_atypical_markers(
                sample_values, predicted_class="AD", reference_stats=reference_stats, threshold=effect_size_threshold
            )

            fp_report.append(
                {
                    "ID": df.loc[idx, "ID"] if "ID" in df.columns else idx,
                    "Tipo": "Falso Positivo",
                    "P(AD)": y_proba[idx],
                    "N_Alertas": len(alerts),
                    "Alertas": "; ".join([f"{a['metabolito']} ({a['severity']})" for a in alerts]),
                }
            )

        # Combinar
        report_df = pd.DataFrame(fn_report + fp_report)

        st.dataframe(report_df, width="stretch")

        # Descarga
        csv = report_df.to_csv(index=False)
        st.download_button(
            label="📥 Descargar Reporte de Errores (CSV)", data=csv, file_name="reporte_errores.csv", mime="text/csv"
        )

        st.success("✓ Reporte generado correctamente")

# ============================================================================
# Sección 5: Interpretación y Conclusiones
# ============================================================================

st.header("5️⃣ Interpretación y Conclusiones")

st.markdown("""
### 📊 Análisis de Patrones de Error

Esta sección proporciona interpretaciones automáticas sobre los errores del modelo.
""")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🚨 Falsos Negativos (FN)")

    if n_fn > 0:
        fn_avg_proba = y_proba[fn_mask].mean()
        st.metric("Probabilidad media P(AD)", f"{fn_avg_proba:.3f}")

        if fn_avg_proba < 0.4:
            st.warning("""
            **⚠️ Baja confianza en errores FN**
            - El modelo tiene baja confianza (P(AD) < 0.4) en estos casos
            - Sugiere que estos pacientes AD tienen perfil metabólico más similar a NC
            - Posibles casos de AD temprano o con presentación atípica
            """)
        elif fn_avg_proba >= 0.4 and fn_avg_proba < 0.5:
            st.info("""
            **ℹ️ Casos cercanos al umbral**
            - El modelo está cerca de clasificarlos correctamente
            - Pequeños ajustes en el umbral podrían reducir estos errores
            - Considerar ajustar threshold o usar características adicionales
            """)

        # Análisis de alertas
        fn_alerts = []
        for idx in df[fn_mask].index:
            sample_values = df.loc[idx, metabolites].to_dict()
            alerts = check_atypical_markers(
                sample_values, predicted_class="NC", reference_stats=reference_stats, threshold=0.5
            )
            fn_alerts.extend([a["metabolito"] for a in alerts])

        if fn_alerts:
            from collections import Counter

            fn_alert_counts = Counter(fn_alerts)
            top_fn_markers = fn_alert_counts.most_common(3)

            st.markdown("**Marcadores más frecuentes en FN:**")
            for marker, count in top_fn_markers:
                st.markdown(f"- `{marker}`: {count} casos ({count / n_fn * 100:.0f}%)")

            st.info(f"""
            💡 **Interpretación**: Los marcadores {", ".join([m[0] for m in top_fn_markers[:2]])}
            son los que más frecuentemente contradicen la predicción en casos FN. Estos pacientes
            AD tienen perfiles metabólicos mixtos que confunden al modelo.
            """)
    else:
        st.success("✓ No hay falsos negativos. El modelo detecta correctamente todos los casos AD.")

with col2:
    st.subheader("⚠️ Falsos Positivos (FP)")

    if n_fp > 0:
        fp_avg_proba = y_proba[fp_mask].mean()
        st.metric("Probabilidad media P(AD)", f"{fp_avg_proba:.3f}")

        if fp_avg_proba > 0.7:
            st.error("""
            **🔴 Alta confianza en errores FP**
            - El modelo está muy confiado (P(AD) > 0.7) pero se equivoca
            - Estos controles NC tienen perfil metabólico similar a AD
            - Posibles casos de MCI (deterioro cognitivo leve) no diagnosticado
            - Considerar seguimiento clínico de estos pacientes
            """)
        elif fp_avg_proba >= 0.5 and fp_avg_proba <= 0.7:
            st.warning("""
            **⚠️ Confianza moderada en errores FP**
            - El modelo tiene dudas razonables en estos casos
            - Perfil metabólico limítrofe entre NC y AD
            - Pueden ser casos de NC con alteraciones metabólicas subclínicas
            """)

        # Análisis de alertas
        fp_alerts = []
        for idx in df[fp_mask].index:
            sample_values = df.loc[idx, metabolites].to_dict()
            alerts = check_atypical_markers(
                sample_values, predicted_class="AD", reference_stats=reference_stats, threshold=0.5
            )
            fp_alerts.extend([a["metabolito"] for a in alerts])

        if fp_alerts:
            from collections import Counter

            fp_alert_counts = Counter(fp_alerts)
            top_fp_markers = fp_alert_counts.most_common(3)

            st.markdown("**Marcadores más frecuentes en FP:**")
            for marker, count in top_fp_markers:
                st.markdown(f"- `{marker}`: {count} casos ({count / n_fp * 100:.0f}%)")

            st.info(f"""
            💡 **Interpretación**: Los marcadores {", ".join([m[0] for m in top_fp_markers[:2]])}
            muestran valores similares a AD a pesar de ser controles NC. Podrían indicar
            vulnerabilidad metabólica o factores de riesgo preclínicos.
            """)
    else:
        st.success("✓ No hay falsos positivos. El modelo no confunde controles NC con AD.")

# Recomendaciones generales
st.markdown("---")
st.subheader("💡 Recomendaciones")

if n_fn > 0 or n_fp > 0:
    st.markdown("""
    ### Estrategias para mejorar el modelo:

    1. **Ajuste de umbral**: Considerar threshold adaptativo según el contexto clínico
       - Screening: Umbral más bajo (0.4) para maximizar sensibilidad (reducir FN)
       - Diagnóstico: Umbral estándar (0.5) para balance
       - Confirmatorio: Umbral más alto (0.6) para maximizar especificidad (reducir FP)

    2. **Features adicionales**: Los metabolitos identificados en alertas podrían incluirse para mejorar discriminación

    3. **Análisis de subgrupos**: Pacientes con errores consistentes podrían formar subgrupos clínicos distintos

    4. **Validación clínica**: Casos con alta confianza pero erróneos merecen revisión clínica detallada
    """)
else:
    st.success("""
    ✅ **Clasificación perfecta**: El modelo no presenta errores en este conjunto de datos.
    Esto es excelente pero revisa si el dataset es suficientemente desafiante o si hay overfitting.
    """)

# ============================================================================
# Footer
# ============================================================================

st.markdown("---")
st.caption("Módulo 5: Análisis de Errores | TFM Alzheimer Classification")
