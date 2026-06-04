"""
Módulo 6: Predicción Individual
================================
Predicción para nuevos pacientes con sistema de alertas metabólicas.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Añadir path al módulo principal
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app_utils import (
    load_data,
    get_metabolite_columns,
    get_feature_stats,
    check_atypical_markers,
    calculate_effect_size,
    get_bio_e_features,
    get_minimal7_features,
    get_engineered_features,
    get_base_features_needed,
    compute_engineered_feature,
    compute_confidence_assessment,
    initialize_session_state,
)

# ============================================================================
# Configuración de la página
# ============================================================================

st.set_page_config(page_title="Predicción Individual", page_icon="🔮", layout="wide")

initialize_session_state()

# ============================================================================
# Header
# ============================================================================

st.title("🔮 Predicción Individual")
st.markdown("""
Realiza predicciones para nuevos pacientes ingresando sus valores metabolómicos.
El sistema generará una predicción y alertas metabólicas si hay marcadores atípicos.
""")

# ============================================================================
# Verificar modelo entrenado
# ============================================================================

if "model_trained" not in st.session_state or not st.session_state.get("model_trained", False):
    st.error("❌ No hay modelo entrenado")
    st.info("👉 Ve al módulo **📈 Evaluación y Resultados** para entrenar el modelo")
    st.stop()

# ============================================================================
# Cargar datos y modelo
# ============================================================================

df = load_data()
metabolites = get_metabolite_columns(df)
selected_features = st.session_state["selected_features"]
trained_pipeline = st.session_state["trained_pipeline"]
model_config = st.session_state["model_config"]


# Calcular estadísticas de referencia (usando TP y TN)
@st.cache_data
def get_reference_statistics(_df, _metabolites):
    """Calcula estadísticas de referencia para AD y NC."""
    y_pred = st.session_state["predictions"]
    y_true = st.session_state.get("y_true", (_df["Group"] == "AD").astype(int).values)

    # Verificar dimensiones y ajustar si es necesario
    if len(y_pred) != len(_df):
        # Las predicciones son de un subconjunto, usar los índices correctos
        train_indices = st.session_state.get("train_indices", None)
        if train_indices is not None:
            # y_true ya corresponde al subconjunto (almacenado como y_train_full)
            # Solo necesitamos filtrar el DataFrame
            _df = _df.iloc[train_indices].copy()
        else:
            # Fallback: usar primeros índices
            _df = _df.iloc[: len(y_pred)].copy()
            y_true = y_true[: len(y_pred)]

    tp_mask = (y_true == 1) & (y_pred == 1)
    tn_mask = (y_true == 0) & (y_pred == 0)

    stats = {}

    for metabolite in _metabolites:
        ad_values = _df.loc[_df.index[tp_mask], metabolite].values
        nc_values = _df.loc[_df.index[tn_mask], metabolite].values

        stats[metabolite] = {
            "ad_mean": ad_values.mean() if len(ad_values) > 0 else 0,
            "ad_std": ad_values.std() if len(ad_values) > 0 else 0,
            "nc_mean": nc_values.mean() if len(nc_values) > 0 else 0,
            "nc_std": nc_values.std() if len(nc_values) > 0 else 0,
            "effect_size": calculate_effect_size(ad_values, nc_values)
            if len(ad_values) > 0 and len(nc_values) > 0
            else 0,
        }

    return pd.DataFrame(stats).T


reference_stats = get_reference_statistics(df, metabolites)

# ============================================================================
# Sidebar: Información del Modelo
# ============================================================================

with st.sidebar:
    st.header("ℹ️ Información del Modelo")

    st.metric("Algoritmo", model_config["model_type"])
    st.metric("Features", len(selected_features))

    if "cv_results" in st.session_state:
        bal_acc = st.session_state["cv_results"]["test_balanced_accuracy"].mean()
        st.metric("Balanced Accuracy", f"{bal_acc:.3f}")

    with st.expander("Ver features del modelo"):
        for feat in selected_features:
            st.caption(f"• {feat}")

    st.markdown("---")
    st.header("⚙️ Umbral de Confianza")
    confidence_margin = st.slider(
        "Margen de incertidumbre",
        min_value=0.05,
        max_value=0.30,
        value=0.15,
        step=0.05,
        help=(
            "Muestras con P(AD) entre 0.5±margen se consideran **inciertas** "
            "y se recomiendan para análisis adicional. "
            "Un margen de 0.15 significa que P(AD) entre 0.35-0.65 es zona incierta."
        ),
    )
    st.caption(f"Zona incierta: P(AD) ∈ [{0.5 - confidence_margin:.2f}, {0.5 + confidence_margin:.2f}]")

# ============================================================================
# Sección 1: Métodos de Entrada
# ============================================================================

st.header("1️⃣ Entrada de Datos del Paciente")

input_method = st.radio(
    "Método de entrada",
    ["Manual (formulario)", "Cargar desde archivo CSV", "Usar muestra del dataset"],
    horizontal=True,
)

new_sample_values = {}

# ====================
# Método 1: Manual
# ====================
if input_method == "Manual (formulario)":
    st.markdown("### Ingresa los valores para cada metabolito")

    st.info("""
    💡 **Tip**: Los valores deben estar en la misma escala que los datos de entrenamiento (residuales).
    Puedes ver el rango de valores en las estadísticas descriptivas.
    """)

    # Identificar features base y engineered
    base_features_needed = get_base_features_needed(selected_features)
    engineered_features = [f for f in selected_features if f in get_engineered_features()]

    if engineered_features:
        st.info(f"""
        ℹ️ **Features calculadas automáticamente**: {", ".join(engineered_features)}

        Solo necesitas ingresar las features base. Las features derivadas se calcularán automáticamente.
        """)

    # Botón para cargar valores de ejemplo
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Cargar ejemplo: Perfil AD típico"):
            for feature in base_features_needed:
                if feature in reference_stats.index:
                    ad_mean = reference_stats.loc[feature, "ad_mean"]
                    st.session_state[f"input_{feature}"] = ad_mean
            st.rerun()

    with col2:
        if st.button("Cargar ejemplo: Perfil NC típico"):
            for feature in base_features_needed:
                if feature in reference_stats.index:
                    nc_mean = reference_stats.loc[feature, "nc_mean"]
                    st.session_state[f"input_{feature}"] = nc_mean
            st.rerun()

    with col3:
        if st.button("Resetear formulario"):
            for feature in base_features_needed:
                if f"input_{feature}" in st.session_state:
                    del st.session_state[f"input_{feature}"]
            st.rerun()

    # Formulario
    st.markdown("---")

    # Dividir en columnas
    n_cols = 3
    cols = st.columns(n_cols)

    # Diccionario temporal para features base
    base_values = {}

    for i, feature in enumerate(base_features_needed):
        col_idx = i % n_cols

        with cols[col_idx]:
            # Obtener rango de referencia
            if feature in reference_stats.index:
                ad_mean = reference_stats.loc[feature, "ad_mean"]
                nc_mean = reference_stats.loc[feature, "nc_mean"]
            else:
                ad_mean = df[feature].mean()
                nc_mean = df[feature].mean()

            feature_min = df[feature].min()
            feature_max = df[feature].max()

            # Valor por defecto (media global)
            default_value = df[feature].mean()

            # Input
            value = st.number_input(
                f"{feature}",
                value=st.session_state.get(f"input_{feature}", float(default_value)),
                format="%.4f",
                help=f"AD: {ad_mean:.3f} | NC: {nc_mean:.3f} | Rango: [{feature_min:.3f}, {feature_max:.3f}]",
                key=f"input_{feature}",
            )

            base_values[feature] = value

    # Construir diccionario completo con features base y engineered
    new_sample_values = base_values.copy()

    # Calcular features engineered
    for eng_feat in engineered_features:
        new_sample_values[eng_feat] = compute_engineered_feature(eng_feat, base_values)

# ====================
# Método 2: CSV
# ====================
elif input_method == "Cargar desde archivo CSV":
    st.markdown("### Carga un archivo CSV con los valores")

    base_features_needed = get_base_features_needed(selected_features)
    engineered_in_selection = [f for f in selected_features if f in get_engineered_features()]

    st.info(f"""
    El archivo CSV debe contener las **features base** necesarias: {", ".join(base_features_needed)}

    {"Las features engineered (" + ", ".join(engineered_in_selection) + ") se calcularán automáticamente." if engineered_in_selection else ""}
    """)

    # Ejemplo de formato (solo features base)
    with st.expander("Ver formato de ejemplo"):
        example_df = pd.DataFrame([{feat: 0.0 for feat in base_features_needed}])
        st.dataframe(example_df)

        csv_example = example_df.to_csv(index=False)
        st.download_button(
            label="📥 Descargar plantilla CSV", data=csv_example, file_name="plantilla_prediccion.csv", mime="text/csv"
        )

    # Upload
    uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)

            # Verificar que tenga todas las features BASE necesarias
            missing_features = set(base_features_needed) - set(uploaded_df.columns)

            if missing_features:
                st.error(f"❌ Faltan las siguientes features base: {missing_features}")
            else:
                st.success("✓ Archivo cargado correctamente")

                # Tomar primera fila - solo features base
                base_values = {}
                for feature in base_features_needed:
                    base_values[feature] = uploaded_df[feature].iloc[0]

                # Construir diccionario completo
                new_sample_values = base_values.copy()

                # Calcular features engineered
                for eng_feat in engineered_in_selection:
                    new_sample_values[eng_feat] = compute_engineered_feature(eng_feat, base_values)

                # Mostrar valores (todas las features)
                display_df = pd.DataFrame([new_sample_values])
                st.dataframe(display_df[selected_features])

        except Exception as e:
            st.error(f"❌ Error al cargar archivo: {e}")

# ====================
# Método 3: Muestra del dataset
# ====================
else:  # Usar muestra del dataset
    st.markdown("### Selecciona una muestra existente del dataset")

    sample_ids = df["ID"].values if "ID" in df.columns else df.index.values

    selected_sample_id = st.selectbox("ID de la muestra", sample_ids)

    selected_sample_idx = df[df["ID"] == selected_sample_id].index[0] if "ID" in df.columns else selected_sample_id

    # Mostrar info de la muestra
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Grupo Real", df.loc[selected_sample_idx, "Group"])

    with col2:
        if "predictions" in st.session_state:
            pred = "AD" if st.session_state["predictions"][selected_sample_idx] == 1 else "NC"
            st.metric("Predicción Actual", pred)

    with col3:
        if "probabilities" in st.session_state:
            proba = st.session_state["probabilities"][selected_sample_idx]
            st.metric("P(AD)", f"{proba:.3f}")

    # Cargar valores
    for feature in selected_features:
        new_sample_values[feature] = df.loc[selected_sample_idx, feature]

    # Mostrar tabla
    st.markdown("**Valores de la muestra:**")
    values_df = pd.DataFrame([new_sample_values])
    st.dataframe(values_df.T.rename(columns={0: "Valor"}), width="stretch")

# ============================================================================
# Sección 2: Realizar Predicción
# ============================================================================

st.header("2️⃣ Realizar Predicción")

if len(new_sample_values) == len(selected_features):
    col1, col2 = st.columns([3, 1])

    with col1:
        predict_button = st.button("🔮 Predecir", type="primary")

    if predict_button or "last_prediction" in st.session_state:
        # Preparar datos
        X_new = np.array([[new_sample_values[feat] for feat in selected_features]])

        # Predecir
        prediction = trained_pipeline.predict(X_new)[0]

        # Probabilidad
        if hasattr(trained_pipeline.named_steps["classifier"], "predict_proba"):
            proba = trained_pipeline.named_steps["classifier"].predict_proba(
                trained_pipeline.named_steps["scaler"].transform(X_new)
            )[0, 1]
        else:
            proba = trained_pipeline.decision_function(X_new)[0]

        # Guardar en session state
        st.session_state["last_prediction"] = {
            "prediction": prediction,
            "probability": proba,
            "values": new_sample_values.copy(),
        }

        # ============================================================================
        # Sección 3: Resultados de la Predicción con Evaluación de Confianza
        # ============================================================================

        st.markdown("---")
        st.header("3️⃣ Resultados de la Predicción")

        # Evaluar confianza
        confidence = compute_confidence_assessment(
            proba,
            new_sample_values,
            reference_stats,
            threshold_margin=confidence_margin,
        )

        # --- Resultado principal con semáforo ---
        col1, col2, col3, col4 = st.columns([1.2, 1, 1, 1])

        with col1:
            if confidence["needs_review"]:
                st.warning("🟡 **INCIERTA — Requiere revisión**")
            elif prediction == 1:
                st.error("🔴 **ALZHEIMER (AD)**")
            else:
                st.success("🟢 **NORMAL (NC)**")

        with col2:
            st.metric("P(AD)", f"{proba:.3f}")

        with col3:
            margin_pct = confidence["margin"] * 200  # 0-100%
            st.metric("Margen", f"{margin_pct:.1f}%")

        with col4:
            level_icons = {"alta": "🟢", "moderada": "🟡", "baja": "🔴"}
            icon = level_icons.get(confidence["confidence_level"], "⚪")
            st.metric("Confianza", f"{icon} {confidence['confidence_level'].capitalize()}")

        # --- Gráfico de probabilidad con zonas de confianza ---
        st.subheader("📊 Probabilidad y Zona de Confianza")

        fig = go.Figure()

        # Zona NC (confiable)
        fig.add_vrect(
            x0=0,
            x1=0.5 - confidence_margin,
            fillcolor="rgba(0, 180, 0, 0.1)",
            line_width=0,
            annotation_text="NC confiable",
            annotation_position="top left",
        )

        # Zona incierta
        fig.add_vrect(
            x0=0.5 - confidence_margin,
            x1=0.5 + confidence_margin,
            fillcolor="rgba(255, 165, 0, 0.15)",
            line_width=0,
            annotation_text="⚠️ Zona incierta",
            annotation_position="top left",
        )

        # Zona AD (confiable)
        fig.add_vrect(
            x0=0.5 + confidence_margin,
            x1=1,
            fillcolor="rgba(255, 0, 0, 0.1)",
            line_width=0,
            annotation_text="AD confiable",
            annotation_position="top right",
        )

        # Línea de umbral
        fig.add_vline(x=0.5, line_dash="dash", line_color="gray", line_width=2)

        # Punto de la predicción
        marker_color = "orange" if confidence["needs_review"] else ("red" if prediction == 1 else "green")
        fig.add_trace(
            go.Scatter(
                x=[proba],
                y=["Predicción"],
                mode="markers+text",
                marker=dict(size=20, color=marker_color, line=dict(color="black", width=2)),
                text=f"P(AD)={proba:.3f}",
                textposition="top center",
                showlegend=False,
            )
        )

        fig.update_layout(
            title="Posición de la muestra en el espacio de decisión",
            xaxis=dict(title="P(AD)", range=[0, 1], dtick=0.1),
            yaxis=dict(showticklabels=False),
            height=220,
            margin=dict(t=60, b=40),
        )

        st.plotly_chart(fig, use_container_width=True)

        # --- Bloque de confianza detallado ---
        if confidence["needs_review"]:
            st.error(confidence["recommendation"])

            if confidence["factors"]:
                st.markdown("**Factores de incertidumbre detectados:**")
                for factor in confidence["factors"]:
                    st.markdown(f"- {factor}")

            st.info(
                "💡 **¿Qué significa?** Esta muestra tiene un perfil metabólico ambiguo: "
                "los biomarcadores no apuntan claramente hacia AD ni NC. "
                "En la validación cruzada, muestras con este nivel de incertidumbre "
                "tienen una tasa de error ~40-50%. Clasificar únicamente las muestras "
                "con confianza alta/moderada eleva la precisión del modelo a ~87-90%."
            )
        elif confidence["confidence_level"] == "moderada":
            st.warning(confidence["recommendation"])
            if confidence["factors"]:
                for factor in confidence["factors"]:
                    st.markdown(f"- {factor}")
        else:
            st.success(confidence["recommendation"])

        # ============================================================================
        # Sección 4: Alertas Metabólicas
        # ============================================================================

        st.header("4️⃣ Alertas Metabólicas")

        st.markdown("""
        Análisis de marcadores que contradicen la predicción.
        Se muestran solo metabolitos atípicos respecto al grupo predicho.
        """)

        # Determinar clase predicha
        predicted_class = "AD" if prediction == 1 else "NC"

        # Generar alertas
        threshold = 0.5
        alerts = check_atypical_markers(new_sample_values, predicted_class, reference_stats, threshold=threshold)

        # Mostrar alertas
        if alerts:
            st.warning(f"⚠️ {len(alerts)} marcador(es) atípico(s) detectado(s)")

            st.markdown("""
            Los siguientes marcadores tienen valores que van en contra de la distribución
            típica del grupo predicho:
            """)

            for alert in alerts:
                severity_icon = "🟡" if alert["severity"] == "MEDIUM" else "🔵"
                severity_label = alert["severity"]

                with st.expander(f"{severity_icon} {severity_label} - {alert['metabolito']}", expanded=True):
                    col1, col2 = st.columns([3, 2])

                    with col1:
                        st.markdown(f"**Valor del paciente**: `{alert['valor']:.3f}`")
                        st.markdown(
                            f"**Media en AD (típico)**: `{alert['ad_mean']:.3f} ± {reference_stats.loc[alert['metabolito'], 'ad_std']:.3f}`"
                        )
                        st.markdown(
                            f"**Media en NC (típico)**: `{alert['nc_mean']:.3f} ± {reference_stats.loc[alert['metabolito'], 'nc_std']:.3f}`"
                        )
                        st.markdown(f"**Effect Size (Cohen's d)**: `{alert['effect_size']:.3f}`")
                        st.markdown("---")
                        st.info(f"💡 {alert['mensaje']}")

                    with col2:
                        # Gráfico comparativo
                        fig_alert = go.Figure()

                        # Barras
                        fig_alert.add_trace(
                            go.Bar(
                                x=["AD (típico)", "NC (típico)", "Paciente"],
                                y=[alert["ad_mean"], alert["nc_mean"], alert["valor"]],
                                marker=dict(
                                    color=["lightgreen", "lightblue", "red"], line=dict(color="black", width=1)
                                ),
                                showlegend=False,
                            )
                        )

                        # Error bars
                        fig_alert.add_trace(
                            go.Scatter(
                                x=["AD (típico)", "NC (típico)"],
                                y=[alert["ad_mean"], alert["nc_mean"]],
                                error_y=dict(
                                    type="data",
                                    array=[
                                        reference_stats.loc[alert["metabolito"], "ad_std"],
                                        reference_stats.loc[alert["metabolito"], "nc_std"],
                                    ],
                                    visible=True,
                                ),
                                mode="markers",
                                marker=dict(size=0.1, color="black"),
                                showlegend=False,
                            )
                        )

                        fig_alert.update_layout(title=f"{alert['metabolito']}", yaxis_title="Valor", height=300)

                        st.plotly_chart(fig_alert, width="stretch")

            # Resumen de severidad
            st.markdown("---")
            st.subheader("📋 Resumen de Alertas")

            severity_counts = {}
            for alert in alerts:
                sev = alert["severity"]
                severity_counts[sev] = severity_counts.get(sev, 0) + 1

            col1, col2 = st.columns(2)

            with col1:
                if "MEDIUM" in severity_counts:
                    st.metric("🟡 Severidad MEDIA", severity_counts["MEDIUM"])

            with col2:
                if "LOW" in severity_counts:
                    st.metric("🔵 Severidad BAJA", severity_counts["LOW"])

        else:
            st.success("✓ No se detectaron marcadores atípicos")
            st.markdown("""
            Todos los marcadores son consistentes con el perfil metabólico típico
            del grupo predicho.
            """)

        # ============================================================================
        # Sección 5: Interpretación Clínica
        # ============================================================================

        st.header("5️⃣ Interpretación Clínica")

        st.markdown(f"""
        ### Resumen

        | | |
        |---|---|
        | **Predicción** | {"Alzheimer (AD)" if prediction == 1 else "Normal (NC)"} |
        | **Probabilidad P(AD)** | {proba:.3f} |
        | **Confianza** | {confidence["confidence_level"].capitalize()} ({confidence["confidence_color"]}) |
        | **Margen al umbral** | {confidence["margin"]:.3f} |
        | **Biomarcadores contradictorios** | {confidence["n_contradictions"]}/6 |
        | **Alertas metabólicas** | {len(alerts)} |
        | **Requiere revisión** | {"✅ Sí" if confidence["needs_review"] else "❌ No"} |
        """)

        st.markdown("### Recomendaciones")

        if confidence["needs_review"]:
            st.error("""
            🔴 **CLASIFICACIÓN INCIERTA — NECESITA MÁS ANÁLISIS**

            Esta muestra se encuentra en la **zona de incertidumbre** del modelo.
            No se puede emitir un diagnóstico fiable basado únicamente en el perfil metabolómico.

            **Acciones recomendadas:**
            - Evaluación cognitiva detallada (MMSE, MoCA)
            - Neuroimagen (PET amiloide, RMN)
            - Biomarcadores en LCR (Aβ42, tau, p-tau)
            - Seguimiento longitudinal (repetir análisis en 6-12 meses)
            """)
        elif prediction == 1:  # AD
            if proba > 0.8:
                st.error("""
                🔴 **RIESGO ALTO**
                - Probabilidad muy alta de Alzheimer con alta confianza
                - Se recomienda evaluación clínica completa
                - Considerar biomarcadores adicionales (neuroimagen, LCR)
                """)
            elif proba > 0.65:
                st.warning("""
                🟡 **RIESGO MODERADO**
                - Probabilidad moderada de Alzheimer
                - Realizar seguimiento periódico
                - Evaluación cognitiva detallada recomendada
                """)
        else:  # NC
            if proba < 0.2:
                st.success("""
                🟢 **RIESGO BAJO**
                - Probabilidad muy baja de Alzheimer con alta confianza
                - Perfil metabólico normal
                - Seguimiento rutinario
                """)
            elif proba < 0.35:
                st.info("""
                🟡 **RIESGO LEVE**
                - Probabilidad baja pero con margen moderado
                - Considerar seguimiento preventivo
                """)

        if alerts and not confidence["needs_review"]:
            st.warning(f"""
            ⚠️ **{len(alerts)} marcador(es) atípico(s)**: Algunos metabolitos muestran valores
            que se desvían del perfil típico del grupo {"AD" if prediction == 1 else "NC"}.
            Esto podría indicar variabilidad biológica o un estado de transición.
            """)

        # ============================================================================
        # Sección 6: Exportar Resultados
        # ============================================================================

        st.header("6️⃣ Exportar Resultados")

        # Preparar reporte
        report_data = {
            "Predicción": ["AD" if prediction == 1 else "NC"],
            "Probabilidad P(AD)": [proba],
            "Confianza": [confidence["confidence_level"]],
            "Requiere_Revisión": [confidence["needs_review"]],
            "Margen": [confidence["margin"]],
            "Contradicciones": [confidence["n_contradictions"]],
            "N_Alertas": [len(alerts)],
        }

        for feature in selected_features:
            report_data[feature] = [new_sample_values[feature]]

        if alerts:
            report_data["Alertas_Detalle"] = ["; ".join([f"{a['metabolito']} ({a['severity']})" for a in alerts])]

        report_df = pd.DataFrame(report_data)

        # Descarga
        col1, col2 = st.columns(2)

        with col1:
            csv_report = report_df.to_csv(index=False)
            st.download_button(
                label="📥 Descargar Reporte (CSV)", data=csv_report, file_name="reporte_prediccion.csv", mime="text/csv"
            )

        with col2:
            json_report = report_df.to_json(orient="records", indent=2)
            st.download_button(
                label="📥 Descargar Reporte (JSON)",
                data=json_report,
                file_name="reporte_prediccion.json",
                mime="application/json",
            )

else:
    st.warning(f"⚠️ Faltan valores. Completa todos los campos ({len(new_sample_values)}/{len(selected_features)})")

# ============================================================================
# Footer
# ============================================================================

st.markdown("---")
st.caption("Módulo 6: Predicción Individual | TFM Alzheimer Classification")
