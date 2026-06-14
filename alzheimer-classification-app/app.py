"""Herramienta Web Interactiva para Análisis de Clasificación AD/NC.

=================================================================

Aplicación Streamlit para análisis completo del pipeline de clasificación
de Alzheimer, desde selección de variables hasta evaluación de modelos.

Autor: TFM Alzheimer Classification
Fecha: 2026
"""

import streamlit as st
from app_utils import ORIGINAL_METABS, initialize_session_state, load_data

# Configuración de página
st.set_page_config(page_title="AD Classification Tool", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

# Inicializar session state
initialize_session_state()

# Título principal
st.title("🧠 Herramienta de Análisis: Clasificación AD/NC")
st.markdown("---")

# ============================================================================
# PÁGINA DE INICIO
# ============================================================================

st.header("Bienvenido a la Herramienta de Análisis AD/NC")

st.markdown("""
Esta herramienta web interactiva permite realizar análisis completos del proceso de
clasificación de Alzheimer (AD) vs Normal (NC) usando datos metabolómicos.
""")

# Cargar datos para mostrar estadísticas
df = load_data()
n_metab = sum(1 for m in ORIGINAL_METABS if m in df.columns)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Dataset", f"{len(df)} muestras", f"{len(df[df['Group'] == 'AD'])} AD / {len(df[df['Group'] == 'NC'])} NC"
    )

with col2:
    st.metric("Metabolitos", f"{n_metab} variables", "datos residuales")

with col3:
    # Mostrar BA del modelo entrenado si existe, si no mostrar placeholder
    if "cv_results" in st.session_state:
        ba_val = st.session_state["cv_results"].get("test_balanced_accuracy", None)
        if ba_val is not None:
            import numpy as np

            ba_mean = np.mean(ba_val)
            st.metric("Performance", f"{ba_mean * 100:.1f}%", "Balanced Accuracy (CV)")
        else:
            st.metric("Performance", "—", "Entrenar modelo en Evaluación")
    else:
        st.metric("Performance", "—", "Entrenar modelo en Evaluación")

st.markdown("---")

st.subheader("📋 Módulos Disponibles")

st.markdown("""
### 📊 **Exploración de Datos**
Visualiza las distribuciones de metabolitos, correlaciones, y separación entre grupos AD/NC.
- Distribuciones por grupo (violin, box, histogramas)
- Matriz de correlaciones
- PCA para visualización de estructura
- Detección de outliers

### 🔍 **Selección de Variables**
Analiza qué variables son más discriminantes y explora diferentes subsets **(Bio-E, Minimal-7)**.
- Comparación de subsets (Bio-E vs Minimal-7)
- Forward/Backward selection interactiva

### 🤖 **Configuración de Modelos**
Configura y entrena modelos de clasificación con hiperparámetros personalizados mediante **sliders interactivos**.
- Logistic Regression (C, penalty, class_weight)
- SVM (C, kernel, gamma)
- Random Forest (n_estimators, max_death, etc.)
- Cross-validation personalizada (K-folds, repeticiones)

### 📈 **Evaluación y Resultados**
Visualiza métricas de performance, curvas ROC/PR, y análisis detallado de resultados.
- Métricas agregadas con intervalos de confianza (95% CI)
- Curvas ROC y Precision-Recall interactivas
- Matriz de confusión
- Distribución de probabilidades predichas
- Feature importance

### ⚠️ **Análisis de Errores**
Identifica falsos negativos y falsos positivos, con **alertas de marcadores metabólicos atípicos** (sin scoring de confianza).
- Lista de casos mal clasificados (FN y FP)
- Análisis detallado muestra por muestra
- Alertas cuando metabolitos van contra distribución del grupo predicho
- Comparación FN vs TP y FP vs TN

### 🔮 **Predicción Individual**
Predice la clase de un nuevo paciente y detecta marcadores atípicos.
- Entrada manual o carga desde CSV
- Ver predicción y probabilidad P(AD)
- Alertas de marcadores que difieren del patrón esperado
- Interpretación clínica y recomendaciones

### 📚 **Metodología TFM** *(nuevo)*
Proceso completo y documentado de selección de modelo y variables del TFM.
- Fase 1: Comparativa de 15 clasificadores → LogReg seleccionado
- Fase 2: Stability Selection → Forward Selection → Feature Engineering → 11 features
- Fase 3: Enfoque de triaje (P<0.20 / P>0.70)
- Reproducción interactiva de resultados

### 🏥 **Triaje Clínico** *(nuevo)*
Sistema de triaje en cascada: metabolómica + reglas clínicas.
- Optimización del hiperparámetro C
- Sistema de tres zonas (verde/amarilla/roja) con umbrales optimizados
- Cascada con reglas clínicas (APOE, depresión, etc.)
- Métricas globales de clasificación
- Simulador interactivo de triaje
""")

st.markdown("---")

# Instrucciones de uso
st.subheader("🚀 Cómo Usar Esta Herramienta")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    #### Flujo de Trabajo Recomendado:

    1. **Exploración** → Familiarízate con los datos
    2. **Selección** → Elige o crea tu conjunto de features
    3. **Configuración** → Ajusta hiperparámetros del modelo
    4. **Evaluación** → Entrena y visualiza resultados
    5. **Análisis** → Revisa errores y alertas
    6. **Predicción** → Usa el modelo con nuevos pacientes
    """)

with col2:
    st.markdown("""
    #### Características Clave:

    - ✅ **Sin Confidence Scoring**: Solo alertas metabólicas
    - ✅ **Sliders interactivos**: Configuración intuitiva
    - ✅ **Visualizaciones dinámicas**: Plotly interactivo
    - ✅ **Exportación**: Descarga resultados y reportes
    - ✅ **Session State**: Mantiene configuración entre módulos
    """)

st.info("👈 **Selecciona un módulo en el menú lateral para comenzar**")

# ============================================================================
# SIDEBAR: Información del Sistema
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ Información del Sistema")

st.sidebar.markdown(f"""
**Dataset**: residuales_grupos3.xlsx
- {len(df)} muestras totales
- {len(df[df["Group"] == "AD"])} AD ({len(df[df["Group"] == "AD"]) / len(df) * 100:.1f}%)
- {len(df[df["Group"] == "NC"])} NC ({len(df[df["Group"] == "NC"]) / len(df) * 100:.1f}%)
- 20 metabolitos (valores residuales)

**Modelos Disponibles**:
- Logistic Regression ✅
- SVM (Linear/RBF/Poly) ✅
- Random Forest ✅

**Validación**:
- Repeated Stratified K-Fold
- Configurable (default: 5×10)
""")

# ============================================================================
# Footer
# ============================================================================

st.markdown("---")
st.caption("TFM Alzheimer Classification | Herramienta Web Interactiva v1.0")
