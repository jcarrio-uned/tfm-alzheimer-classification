"""Módulo 1: Exploración de Datos.

Visualización y análisis exploratorio del dataset de metabolómica.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Añadir path al módulo principal
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app_utils import (
    ORIGINAL_METABS,
    calculate_effect_size,
    initialize_session_state,
    load_data,
)

# ============================================================================
# Configuración de la página
# ============================================================================

st.set_page_config(page_title="Exploración de Datos", page_icon="📊", layout="wide")

initialize_session_state()

# ============================================================================
# Header
# ============================================================================

st.title("📊 Exploración de Datos")
st.markdown("""
Análisis exploratorio del dataset de metabolómica para clasificación AD/NC.
Este módulo permite entender las distribuciones, relaciones y estructura de los datos.
""")

# ============================================================================
# Cargar datos
# ============================================================================

with st.spinner("Cargando datos..."):
    df = load_data()
    # Solo metabolitos originales (sin engineered features)
    metabolites = [m for m in ORIGINAL_METABS if m in df.columns]

# Almacenar en session state si no existe
if "data" not in st.session_state:
    st.session_state["data"] = df

# ============================================================================
# Sidebar: Configuración
# ============================================================================

st.sidebar.header("⚙️ Configuración")

# Filtros
st.sidebar.subheader("Filtros")
grupos_disponibles = df["Group"].unique().tolist()
grupos_seleccionados = st.sidebar.multiselect("Grupos a visualizar", grupos_disponibles, default=grupos_disponibles)

# Opciones de visualización
st.sidebar.subheader("Opciones")
show_outliers = st.sidebar.checkbox("Marcar outliers", value=True)
alpha_level = st.sidebar.slider("Nivel alpha para outliers", 0.01, 0.1, 0.05, 0.01)

# Aplicar filtros
df_filtered = df[df["Group"].isin(grupos_seleccionados)].copy()

# ============================================================================
# Sección 1: Resumen del Dataset
# ============================================================================

st.header("1️⃣ Resumen del Dataset")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Muestras", len(df))

with col2:
    st.metric("Variables Metabolómicas", len(metabolites))

with col3:
    if "AD" in df["Group"].values:
        st.metric("Muestras AD", len(df[df["Group"] == "AD"]))

with col4:
    if "NC" in df["Group"].values:
        st.metric("Muestras NC", len(df[df["Group"] == "NC"]))

# Mostrar primeras filas
with st.expander("👁️ Ver datos crudos"):
    st.dataframe(df_filtered, height=300)

# Estadísticas descriptivas
with st.expander("📈 Estadísticas descriptivas"):
    stats_df = df_filtered[metabolites].describe()
    st.dataframe(stats_df)

    # Opción de descarga
    csv = stats_df.to_csv()
    st.download_button(
        label="📥 Descargar estadísticas (CSV)", data=csv, file_name="estadisticas_descriptivas.csv", mime="text/csv"
    )

# ============================================================================
# Sección 2: Distribuciones por Grupo
# ============================================================================

st.header("2️⃣ Distribuciones por Grupo")

# Selector de metabolito
metabolite_selected = st.selectbox("Selecciona un metabolito para visualizar", metabolites, index=0)

# Tabs para diferentes tipos de visualización
tab1, tab2, tab3 = st.tabs(["📊 Violin Plot", "📦 Box Plot", "📈 Histograma"])

with tab1:
    # Violin plot con Plotly
    fig = go.Figure()

    for group in grupos_seleccionados:
        data_group = df_filtered[df_filtered["Group"] == group][metabolite_selected]

        fig.add_trace(
            go.Violin(
                y=data_group,
                name=group,
                box_visible=True,
                meanline_visible=True,
                fillcolor="lightseagreen" if group == "AD" else "lightcoral",
                opacity=0.6,
                x0=group,
            )
        )

    fig.update_layout(
        title=f"Distribución de {metabolite_selected}",
        yaxis_title="Valor (residual)",
        xaxis_title="Grupo",
        showlegend=True,
        height=500,
    )

    st.plotly_chart(fig, width="stretch")

with tab2:
    # Box plot con Plotly
    fig = px.box(
        df_filtered,
        x="Group",
        y=metabolite_selected,
        color="Group",
        points="all" if show_outliers else False,
        hover_data=["ID"] if "ID" in df_filtered.columns else None,
    )

    fig.update_layout(title=f"Box Plot de {metabolite_selected}", yaxis_title="Valor (residual)", height=500)

    st.plotly_chart(fig, width="stretch")

with tab3:
    # Histogramas superpuestos
    fig = go.Figure()

    for group in grupos_seleccionados:
        data_group = df_filtered[df_filtered["Group"] == group][metabolite_selected]

        fig.add_trace(go.Histogram(x=data_group, name=group, opacity=0.6, nbinsx=20))

    fig.update_layout(
        title=f"Histograma de {metabolite_selected}",
        xaxis_title="Valor (residual)",
        yaxis_title="Frecuencia",
        barmode="overlay",
        height=500,
    )

    st.plotly_chart(fig, width="stretch")

# Test estadístico entre grupos
if len(grupos_seleccionados) == 2:
    st.markdown("---")
    st.subheader("🔬 Test Estadístico")

    group1, group2 = grupos_seleccionados
    data1 = df_filtered[df_filtered["Group"] == group1][metabolite_selected]
    data2 = df_filtered[df_filtered["Group"] == group2][metabolite_selected]

    # t-test
    t_stat, p_value = stats.ttest_ind(data1, data2)

    # Effect size (Cohen's d)
    cohen_d = calculate_effect_size(data1, data2)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("t-statistic", f"{t_stat:.4f}")

    with col2:
        st.metric("p-value", f"{p_value:.4f}")
        if p_value < 0.001:
            st.success("***")
        elif p_value < 0.01:
            st.success("**")
        elif p_value < 0.05:
            st.success("*")
        else:
            st.info("ns")

    with col3:
        st.metric("Cohen's d", f"{cohen_d:.4f}")
        if abs(cohen_d) > 0.8:
            st.success("Grande")
        elif abs(cohen_d) > 0.5:
            st.info("Medio")
        elif abs(cohen_d) > 0.2:
            st.warning("Pequeño")
        else:
            st.error("Muy pequeño")

# ============================================================================
# Sección 3: Matriz de Correlación
# ============================================================================

st.header("3️⃣ Matriz de Correlación")

# Selector de features para correlación
features_for_corr = st.multiselect(
    "Selecciona variables para matriz de correlación (vacío = todas)", metabolites, default=[]
)

if not features_for_corr:
    features_for_corr = metabolites

# Calcular correlación
corr_matrix = df_filtered[features_for_corr].corr()

# Plotly heatmap
fig = go.Figure(
    data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale="RdBu_r",
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size": 8},
        colorbar={"title": "Correlación"},
    )
)

fig.update_layout(title="Matriz de Correlación", width=800, height=800)

st.plotly_chart(fig, width="stretch")

# Pares altamente correlacionados
st.subheader("🔗 Correlaciones Altas")

threshold = st.slider("Umbral de correlación", 0.5, 0.95, 0.7, 0.05)

# Extraer pares con correlación alta
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        corr_val = corr_matrix.iloc[i, j]
        if abs(corr_val) >= threshold:
            high_corr_pairs.append(
                {"Variable 1": corr_matrix.columns[i], "Variable 2": corr_matrix.columns[j], "Correlación": corr_val}
            )

if high_corr_pairs:
    high_corr_df = pd.DataFrame(high_corr_pairs)
    high_corr_df = high_corr_df.sort_values("Correlación", key=abs, ascending=False)
    st.dataframe(high_corr_df, width="stretch")
else:
    st.info(f"No se encontraron correlaciones ≥ {threshold}")

# ============================================================================
# Sección 4: PCA (Análisis de Componentes Principales)
# ============================================================================

st.header("4️⃣ PCA - Análisis de Componentes Principales")

# Preparar datos para PCA
X = df_filtered[metabolites].values
y = df_filtered["Group"].values
sample_ids = df_filtered["ID"].values if "ID" in df_filtered.columns else np.arange(len(df_filtered))

# Estandarizar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
n_components = st.slider("Número de componentes", 2, min(len(metabolites), 10), 3)
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

# Varianza explicada
st.subheader("📊 Varianza Explicada")

fig = go.Figure()

fig.add_trace(
    go.Bar(
        x=[f"PC{i + 1}" for i in range(n_components)],
        y=pca.explained_variance_ratio_ * 100,
        name="Individual",
        marker_color="lightblue",
    )
)

fig.add_trace(
    go.Scatter(
        x=[f"PC{i + 1}" for i in range(n_components)],
        y=np.cumsum(pca.explained_variance_ratio_) * 100,
        name="Acumulada",
        mode="lines+markers",
        marker={"size": 10, "color": "red"},
        yaxis="y2",
    )
)

fig.update_layout(
    title="Varianza Explicada por Componente",
    xaxis_title="Componente",
    yaxis_title="Varianza Explicada (%)",
    yaxis2={"title": "Varianza Acumulada (%)", "overlaying": "y", "side": "right"},
    height=400,
)

st.plotly_chart(fig, width="stretch")

# Scatter plot de primeras 2 componentes
st.subheader("🔵 Proyección PCA")

col1, col2 = st.columns(2)

with col1:
    pc_x = st.selectbox("Componente X", [f"PC{i + 1}" for i in range(n_components)], index=0)
with col2:
    pc_y = st.selectbox("Componente Y", [f"PC{i + 1}" for i in range(n_components)], index=1)

pc_x_idx = int(pc_x.replace("PC", "")) - 1
pc_y_idx = int(pc_y.replace("PC", "")) - 1

# Crear scatter plot
pca_df = pd.DataFrame({pc_x: X_pca[:, pc_x_idx], pc_y: X_pca[:, pc_y_idx], "Group": y, "ID": sample_ids})

fig = px.scatter(pca_df, x=pc_x, y=pc_y, color="Group", hover_data=["ID"], title=f"PCA: {pc_x} vs {pc_y}")

fig.update_layout(height=600)

st.plotly_chart(fig, width="stretch")

# Loadings (contribución de variables)
with st.expander("🔍 Ver Loadings (Contribución de Variables)"):
    st.markdown("""
    Los **loadings** muestran qué variables contribuyen más a cada componente principal.
    """)

    loadings = pd.DataFrame(pca.components_.T, columns=[f"PC{i + 1}" for i in range(n_components)], index=metabolites)

    st.dataframe(loadings.style.background_gradient(cmap="RdBu_r", axis=None))

    # Top contribuidores al PC seleccionado
    st.markdown(f"**Top 10 contribuidores a {pc_x}:**")
    top_loadings = loadings[pc_x].abs().sort_values(ascending=False).head(10)

    fig = go.Figure(go.Bar(x=top_loadings.values, y=top_loadings.index, orientation="h", marker_color="steelblue"))

    fig.update_layout(
        title=f"Top 10 Variables que contribuyen a {pc_x}",
        xaxis_title="Loading Absoluto",
        yaxis_title="Variable",
        height=400,
    )

    st.plotly_chart(fig, width="stretch")

# ============================================================================
# Sección 5: Detección de Outliers
# ============================================================================

st.header("5️⃣ Detección de Outliers")

st.markdown("""
Identificación de muestras con valores extremos usando el método de **Z-score**.
Un valor se considera outlier si |Z| > umbral.
""")

# Umbral para Z-score
z_threshold = st.slider("Umbral Z-score", 2.0, 4.0, 3.0, 0.5)

# Calcular Z-scores
z_scores = np.abs(stats.zscore(df_filtered[metabolites]))

# Identificar outliers
outlier_mask = (z_scores > z_threshold).any(axis=1)
outliers_df = df_filtered[outlier_mask].copy()

st.metric("Muestras con Outliers", f"{len(outliers_df)} / {len(df_filtered)}")

if len(outliers_df) > 0:
    # Mostrar tabla de outliers
    with st.expander("👁️ Ver muestras outlier"):
        # Contar outliers por muestra
        outlier_counts = (z_scores[outlier_mask] > z_threshold).sum(axis=1)
        outliers_df["N_Outliers"] = outlier_counts

        display_cols = ["ID", "Group", "N_Outliers"] if "ID" in outliers_df.columns else ["Group", "N_Outliers"]
        st.dataframe(outliers_df[display_cols], width="stretch")

    # Heatmap de Z-scores para outliers
    st.subheader("🔥 Heatmap de Z-scores (solo outliers)")

    z_scores_outliers = z_scores[outlier_mask]

    fig = go.Figure(
        data=go.Heatmap(
            z=z_scores_outliers, x=metabolites, y=outliers_df.index, colorscale="Reds", colorbar={"title": "Z-score"}
        )
    )

    fig.update_layout(
        title="Z-scores de Muestras Outlier",
        xaxis_title="Variable",
        yaxis_title="Muestra",
        height=max(400, len(outliers_df) * 20),
    )

    st.plotly_chart(fig, width="stretch")
else:
    st.success("✓ No se detectaron outliers con el umbral seleccionado")

# ============================================================================
# Sección 6: Detección de Outliers — 3×IQR (Multivariante)
# ============================================================================

st.header("6️⃣ Detección de Outliers — Método IQR")

st.markdown("""
Método robusto basado en el **rango intercuartílico (IQR)**.
Un valor es outlier si cae fuera del intervalo $[Q_1 - k \\times IQR,\\ Q_3 + k \\times IQR]$.

Este método es **no paramétrico** (no asume normalidad) y resistente a los propios outliers.
""")

col_iqr1, col_iqr2 = st.columns(2)
with col_iqr1:
    iqr_k = st.slider(
        "Factor k (multiplicador IQR)", 1.5, 4.0, 3.0, 0.5, help="1.5 = estándar Tukey, 3.0 = conservador"
    )
with col_iqr2:
    min_metab_outlier = st.number_input(
        "Mín. metabolitos fuera de rango para marcar muestra",
        min_value=1,
        max_value=5,
        value=1,
        help="Cuántos metabolitos deben ser extremos para considerar la muestra como outlier.",
    )

# Calcular IQR por metabolito
X_metab = df_filtered[metabolites].values
Q1 = np.percentile(X_metab, 25, axis=0)
Q3 = np.percentile(X_metab, 75, axis=0)
IQR = Q3 - Q1
lower_bound = Q1 - iqr_k * IQR
upper_bound = Q3 + iqr_k * IQR

# Matriz booleana: True si fuera de rango
outside_iqr = (X_metab < lower_bound) | (X_metab > upper_bound)
n_metab_outlier_per_sample = outside_iqr.sum(axis=1)
iqr_outlier_mask = n_metab_outlier_per_sample >= min_metab_outlier

# Métricas resumen
col_m1, col_m2, col_m3 = st.columns(3)
col_m1.metric("Muestras marcadas", f"{iqr_outlier_mask.sum()} / {len(df_filtered)}")
col_m2.metric("% outliers", f"{iqr_outlier_mask.sum() / len(df_filtered) * 100:.1f}%")
col_m3.metric("Metabolitos con algún outlier", f"{(outside_iqr.sum(axis=0) > 0).sum()} / {len(metabolites)}")

# Tabla de outliers por metabolito
with st.expander("📊 Outliers por metabolito"):
    outlier_per_metab = outside_iqr.sum(axis=0)
    metab_iqr_df = pd.DataFrame(
        {
            "Metabolito": metabolites,
            "N outliers": outlier_per_metab,
            "% del total": [f"{v / len(df_filtered) * 100:.1f}%" for v in outlier_per_metab],
            "Q1": [f"{v:.3f}" for v in Q1],
            "Q3": [f"{v:.3f}" for v in Q3],
            "IQR": [f"{v:.3f}" for v in IQR],
            f"Límite inf ({iqr_k}×IQR)": [f"{v:.3f}" for v in lower_bound],
            f"Límite sup ({iqr_k}×IQR)": [f"{v:.3f}" for v in upper_bound],
        }
    ).sort_values("N outliers", ascending=False)
    st.dataframe(metab_iqr_df, hide_index=True, use_container_width=True)

# Detalle de muestras outlier
if iqr_outlier_mask.sum() > 0:
    st.subheader("🔍 Muestras identificadas como outlier")

    outlier_detail = []
    for i in np.where(iqr_outlier_mask)[0]:
        row = df_filtered.iloc[i]
        metabs_fuera = [metabolites[j] for j in range(len(metabolites)) if outside_iqr[i, j]]
        outlier_detail.append(
            {
                "ID": row.get("ID", f"#{i}"),
                "Grupo": row.get("Group", "—"),
                "N metab fuera": n_metab_outlier_per_sample[i],
                "Metabolitos": ", ".join(metabs_fuera),
            }
        )

    df_outlier_detail = pd.DataFrame(outlier_detail)
    st.dataframe(df_outlier_detail, hide_index=True, use_container_width=True)

    # Visualización: boxplots con outliers marcados
    st.subheader("📦 Boxplots con outliers")
    # Solo metabolitos que tienen outliers
    metabs_with_outliers = [metabolites[j] for j in range(len(metabolites)) if outside_iqr[:, j].sum() > 0]

    if metabs_with_outliers:
        selected_metab_bp = st.selectbox("Metabolito a visualizar", metabs_with_outliers)
        j_sel = metabolites.index(selected_metab_bp)

        fig_bp = go.Figure()
        # Boxplot del metabolito
        fig_bp.add_trace(
            go.Box(
                y=X_metab[:, j_sel],
                name=selected_metab_bp,
                boxpoints="suspectedoutliers",
                marker_color="steelblue",
            )
        )
        # Marcar límites IQR
        fig_bp.add_hline(
            y=upper_bound[j_sel],
            line_dash="dash",
            line_color="red",
            annotation_text=f"Q3 + {iqr_k}×IQR = {upper_bound[j_sel]:.3f}",
        )
        fig_bp.add_hline(
            y=lower_bound[j_sel],
            line_dash="dash",
            line_color="red",
            annotation_text=f"Q1 − {iqr_k}×IQR = {lower_bound[j_sel]:.3f}",
        )
        # Marcar outliers como puntos rojos
        outlier_vals = X_metab[outside_iqr[:, j_sel], j_sel]
        outlier_ids = (
            df_filtered.iloc[np.where(outside_iqr[:, j_sel])[0]]["ID"].values if "ID" in df_filtered.columns else None
        )
        fig_bp.add_trace(
            go.Scatter(
                y=outlier_vals,
                x=[selected_metab_bp] * len(outlier_vals),
                mode="markers+text",
                marker={"color": "red", "size": 10, "symbol": "x"},
                text=outlier_ids if outlier_ids is not None else None,
                textposition="top center",
                name="Outliers IQR",
            )
        )
        fig_bp.update_layout(height=400, title=f"Distribución de {selected_metab_bp} con límites {iqr_k}×IQR")
        st.plotly_chart(fig_bp, use_container_width=True)

    # Conclusión
    st.markdown(f"""
    **Resumen**: Con k={iqr_k} y criterio de ≥{min_metab_outlier} metabolito(s) fuera de rango,
    se identifican **{iqr_outlier_mask.sum()} muestras** como potenciales outliers.

    > 💡 **Tratamiento aplicado**: No se eliminan observaciones. Se emplea **winsorización al percentil 5-95**
    > durante el entrenamiento como protección robusta, preservando la información de todas las muestras
    > dado el reducido tamaño muestral (N={len(df_filtered)}).
    """)
else:
    st.success(f"✓ No se detectaron outliers con k={iqr_k} y ≥{min_metab_outlier} metabolito(s) fuera de rango.")

# ============================================================================
# Footer
# ============================================================================

st.markdown("---")
st.caption("Módulo 1: Exploración de Datos | TFM Alzheimer Classification")
