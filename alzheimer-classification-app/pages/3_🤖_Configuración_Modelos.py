"""
Módulo 3: Configuración de Modelos
===================================
Selección de algoritmo y configuración de hiperparámetros.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
import sys
import os

# Añadir path al módulo principal
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app_utils import (
    load_data,
    get_metabolite_columns,
    get_bio_e_features,
    get_minimal7_features,
    get_stable_features,
    initialize_session_state,
)

# ============================================================================
# Configuración de la página
# ============================================================================

st.set_page_config(page_title="Configuración de Modelos", page_icon="🤖", layout="wide")

initialize_session_state()

# ============================================================================
# Header
# ============================================================================

st.title("🤖 Configuración de Modelos")
st.markdown("""
Selecciona el algoritmo de clasificación y configura sus hiperparámetros.
Los modelos se entrenarán con los parámetros elegidos en el módulo de evaluación.
""")

# ============================================================================
# Cargar datos y features
# ============================================================================

df = load_data()
metabolites = get_metabolite_columns(df)

# Verificar si hay features seleccionadas
if "selected_features" not in st.session_state or not st.session_state["selected_features"]:
    st.warning("⚠️ No has seleccionado variables aún")
    st.info("👉 Ve al módulo **🔍 Selección de Variables** para seleccionar features")

    # Ofrecer presets como fallback
    st.markdown("### Presets Rápidos")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Usar Bio-E (9 variables)"):
            st.session_state["selected_features"] = get_bio_e_features()
            st.rerun()

    with col2:
        if st.button("Usar Minimal-7 (7 variables)"):
            st.session_state["selected_features"] = get_minimal7_features()
            st.rerun()

    with col3:
        if st.button("Usar Stable (9 variables)"):
            st.session_state["selected_features"] = get_stable_features()
            st.rerun()

    st.stop()

# ============================================================================
# Mostrar features seleccionadas
# ============================================================================

selected_features = st.session_state["selected_features"]

with st.expander("✓ Variables Seleccionadas", expanded=False):
    st.markdown(f"**{len(selected_features)} variables:**")
    st.markdown("• " + "  \n• ".join(selected_features))

# ============================================================================
# Sección 1: Selección de Algoritmo
# ============================================================================

st.header("1️⃣ Selección de Algoritmo")

model_type = st.selectbox(
    "Tipo de Modelo",
    ["Logistic Regression", "Support Vector Machine (SVM)", "Random Forest"],
    index=0,
    help="Selecciona el algoritmo de clasificación",
)

# Descripciones
if model_type == "Logistic Regression":
    st.info("""
    **Regresión Logística**: Modelo lineal que predice probabilidades mediante función sigmoide.
    - ✅ Rápido y eficiente
    - ✅ Resultados interpretables
    - ✅ Bien calibrado
    - ⚠️ Asume separabilidad lineal
    """)
elif model_type == "Support Vector Machine (SVM)":
    st.info("""
    **Support Vector Machine**: Encuentra hiperplano óptimo que separa clases con máximo margen.
    - ✅ Efectivo en espacios de alta dimensión
    - ✅ Robusto a outliers (con C bajo)
    - ✅ Versátil con diferentes kernels
    - ⚠️ Más lento con datasets grandes
    """)
else:
    st.info("""
    **Random Forest**: Ensemble de árboles de decisión con bagging.
    - ✅ No asume linealidad
    - ✅ Captura interacciones no lineales
    - ✅ Menos propenso a overfitting
    - ⚠️ Menos interpretable (caja negra)
    """)

# ============================================================================
# Sección 2: Configuración de Hiperparámetros
# ============================================================================

st.header("2️⃣ Hiperparámetros del Modelo")

model_config = {"model_type": model_type}

# ====================
# Logistic Regression
# ====================
if model_type == "Logistic Regression":
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Regularización")

        # Grid Search toggle
        use_grid_search = st.checkbox(
            "🔍 Usar Grid Search para C",
            value=False,
            help="Busca automáticamente el mejor valor de C mediante validación cruzada",
        )
        model_config["use_grid_search"] = use_grid_search

        if use_grid_search:
            st.info("""
            **Grid Search**: Probará múltiples valores de C y seleccionará el mejor mediante CV.
            El mejor modelo se usará para predicciones en el conjunto de test.
            """)

            # Definir rango de búsqueda
            st.markdown("**Rango de búsqueda para C:**")
            grid_c_values = st.multiselect(
                "Valores de C a probar",
                options=[0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 1000.0],
                default=[0.01, 0.1, 1.0, 10.0, 100.0],
                help="Selecciona los valores de C que se probarán en el grid search",
            )

            if not grid_c_values:
                st.warning("⚠️ Selecciona al menos un valor de C")
                grid_c_values = [1.0]

            model_config["grid_c_values"] = sorted(grid_c_values)
            st.caption(f"Se probarán {len(grid_c_values)} valores: {sorted(grid_c_values)}")
        else:
            # C manual (inverse of regularization strength)
            c_value = st.number_input(
                "C (Parámetro de Regularización)",
                min_value=0.0001,
                max_value=1000.0,
                value=1.0,
                step=0.001,
                format="%.3f",
                help="Trade-off entre margen y error de clasificación. C alto = menos regularización, margen más pequeño.",
            )
            model_config["C"] = c_value

        # Penalty
        penalty = st.selectbox(
            "Tipo de Penalización",
            ["l2", "l1", "none"],
            index=0,
            help="l2: Ridge (preferido con muchas features). l1: Lasso (selección automática de features).",
        )
        model_config["penalty"] = penalty

        if penalty == "l1":
            st.warning("⚠️ L1 requiere solver='liblinear' o 'saga'")
            solver = st.selectbox("Solver", ["liblinear", "saga"], index=0)
            model_config["solver"] = solver
        else:
            model_config["solver"] = "lbfgs"

    with col2:
        st.subheader("Opciones Adicionales")

        # Class weight
        class_weight = st.radio(
            "Peso de Clases",
            ["balanced", "none"],
            index=0,
            help="'balanced': Ajusta pesos inversamente proporcionales a frecuencias. Recomendado para datos desbalanceados.",
        )
        model_config["class_weight"] = class_weight if class_weight != "none" else None

        # Max iterations
        max_iter = st.number_input(
            "Máximo de Iteraciones",
            min_value=100,
            max_value=10000,
            value=2000,
            step=100,
            help="Número máximo de iteraciones para convergencia",
        )
        model_config["max_iter"] = max_iter

        # Tolerance
        tol = st.select_slider(
            "Tolerancia para Convergencia",
            options=[1e-5, 1e-4, 1e-3, 1e-2],
            value=1e-4,
            format_func=lambda x: f"{x:.0e}",
        )
        model_config["tol"] = tol

# ====================
# SVM
# ====================
elif model_type == "Support Vector Machine (SVM)":
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Kernel y Regularización")

        # Kernel
        kernel = st.selectbox(
            "Kernel",
            ["rbf", "linear", "poly", "sigmoid"],
            index=0,
            help="rbf: Radial Basis Function (más común). linear: Útil con muchas features. poly: Polinomial.",
        )
        model_config["kernel"] = kernel

        # C
        c_value = st.slider(
            "C (Regularization Parameter)",
            min_value=0.0001,
            max_value=1000.0,
            value=1.0,
            step=0.001,
            format="%.3f",
            help="Trade-off entre margen y error de clasificación. C alto = menos errores, margen más pequeño.",
        )
        model_config["C"] = c_value

        # Gamma (solo para rbf, poly, sigmoid)
        if kernel in ["rbf", "poly", "sigmoid"]:
            gamma_option = st.radio(
                "Gamma",
                ["scale", "auto", "custom"],
                index=0,
                help="Influencia de un solo ejemplo de entrenamiento. 'scale': 1/(n_features*X.var())",
            )

            if gamma_option == "custom":
                gamma_value = st.slider(
                    "Valor de Gamma", min_value=0.001, max_value=10.0, value=0.1, step=0.001, format="%.3f"
                )
                model_config["gamma"] = gamma_value
            else:
                model_config["gamma"] = gamma_option

        # Degree (solo para poly)
        if kernel == "poly":
            degree = st.slider("Grado del Polinomio", min_value=2, max_value=5, value=3, step=1)
            model_config["degree"] = degree

    with col2:
        st.subheader("Opciones Adicionales")

        # Class weight
        class_weight = st.radio("Peso de Clases", ["balanced", "none"], index=0)
        model_config["class_weight"] = class_weight if class_weight != "none" else None

        # Probability
        probability = st.checkbox(
            "Habilitar Estimación de Probabilidades",
            value=True,
            help="Necesario para ROC curves. Añade tiempo de entrenamiento.",
        )
        model_config["probability"] = probability

        # Max iterations
        max_iter = st.number_input("Máximo de Iteraciones", min_value=100, max_value=10000, value=1000, step=100)
        model_config["max_iter"] = max_iter

# ====================
# Random Forest
# ====================
else:  # Random Forest
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Configuración del Bosque")

        # Number of trees
        n_estimators = st.slider(
            "Número de Árboles",
            min_value=10,
            max_value=500,
            value=100,
            step=10,
            help="Más árboles = más estable pero más lento",
        )
        model_config["n_estimators"] = n_estimators

        # Max depth
        max_depth_option = st.radio("Profundidad Máxima", ["None (sin límite)", "Custom"], index=0)

        if max_depth_option == "Custom":
            max_depth = st.slider("Profundidad", min_value=2, max_value=20, value=10, step=1)
            model_config["max_depth"] = max_depth
        else:
            model_config["max_depth"] = None

        # Min samples split
        min_samples_split = st.slider(
            "Min Samples Split",
            min_value=2,
            max_value=20,
            value=2,
            step=1,
            help="Mínimo de muestras para dividir un nodo. Valores altos previenen overfitting.",
        )
        model_config["min_samples_split"] = min_samples_split

    with col2:
        st.subheader("Opciones Adicionales")

        # Min samples leaf
        min_samples_leaf = st.slider(
            "Min Samples Leaf",
            min_value=1,
            max_value=20,
            value=1,
            step=1,
            help="Mínimo de muestras en nodo hoja. Valores altos previenen overfitting.",
        )
        model_config["min_samples_leaf"] = min_samples_leaf

        # Max features
        max_features_option = st.selectbox(
            "Max Features",
            ["sqrt", "log2", "None (todas)"],
            index=0,
            help="Número de features a considerar para el mejor split",
        )

        if max_features_option == "sqrt":
            model_config["max_features"] = "sqrt"
        elif max_features_option == "log2":
            model_config["max_features"] = "log2"
        else:
            model_config["max_features"] = None

        # Class weight
        class_weight = st.radio("Peso de Clases", ["balanced", "balanced_subsample", "none"], index=0)

        if class_weight == "none":
            model_config["class_weight"] = None
        else:
            model_config["class_weight"] = class_weight

# ============================================================================
# Sección 3: Parámetros Generales
# ============================================================================

st.header("3️⃣ Parámetros Generales")

st.markdown("""
Configuración de parámetros generales del experimento.

ℹ️ **Nota**: La configuración de validación cruzada (CV) ahora se realiza en el módulo de **Evaluación**,
donde puedes elegir entre diferentes métodos: RepeatedStratifiedKFold, StratifiedKFold, o StratifiedShuffleSplit (Holdout).
""")

# Random state
random_state = st.number_input(
    "Random State (seed para reproducibilidad)",
    min_value=0,
    max_value=9999,
    value=42,
    step=1,
    help="Semilla aleatoria para garantizar reproducibilidad de los resultados",
)
model_config["random_state"] = random_state

# ============================================================================
# Sección 4: Resumen y Guardar Configuración
# ============================================================================

st.header("4️⃣ Resumen de Configuración")

# Crear tabla resumen
summary_data = {"Parámetro": [], "Valor": []}

summary_data["Parámetro"].append("**Modelo**")
summary_data["Valor"].append(model_type)

summary_data["Parámetro"].append("**N Features**")
summary_data["Valor"].append(len(selected_features))

summary_data["Parámetro"].append("**Random State**")
summary_data["Valor"].append(random_state)

for key, value in model_config.items():
    if key not in ["model_type", "random_state"]:
        summary_data["Parámetro"].append(key)
        summary_data["Valor"].append(str(value))

summary_df = pd.DataFrame(summary_data)
st.dataframe(summary_df, width="stretch", hide_index=True)

# Botón para guardar configuración
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("")  # Spacer

with col2:
    if st.button("💾 Guardar Configuración", type="primary"):
        st.session_state["model_config"] = model_config
        st.success("✓ Configuración guardada!")
        st.info("👉 Ve al módulo **📈 Evaluación y Resultados** para entrenar y evaluar")

# Mostrar configuración guardada si existe
if "model_config" in st.session_state:
    st.success("✓ Hay una configuración guardada en memoria")

    with st.expander("Ver configuración guardada"):
        st.json(st.session_state["model_config"])

# ============================================================================
# Sección 5: Comparador de Configuraciones (Opcional)
# ============================================================================

with st.expander("🔬 Comparador de Configuraciones (Avanzado)"):
    st.markdown("""
    ### Comparación Rápida
    Puedes entrenar múltiples configuraciones y comparar resultados.
    Útil para encontrar la mejor configuración de hiperparámetros.
    """)

    st.info("🚧 Funcionalidad en desarrollo. Por ahora, guarda tu configuración y ve al módulo de evaluación.")

# ============================================================================
# Footer con Tips
# ============================================================================

st.markdown("---")

with st.expander("💡 Tips para Configuración de Hiperparámetros"):
    st.markdown("""
    ### Logistic Regression
    - **C pequeño (0.001-0.1)**: Más regularización, modelo más simple
    - **C grande (10-100)**: Menos regularización, puede sobreajustar
    - **L1 penalty**: Hace selección automática de features (algunos coeficientes → 0)
    - **L2 penalty**: Preferred, todos los features contribuyen

    ### SVM
    - **Kernel RBF**: Default, bueno para relaciones no lineales
    - **Kernel Linear**: Más rápido, bueno con muchas features
    - **C bajo**: Margen más amplio, más generalizaciónC alto**: Clasifica mejor training data, puede sobreajustar
    - **Gamma bajo**: Influencia amplia de cada punto
    - **Gamma alto**: Influencia solo puntos cercanos

    ### Random Forest
    - **Más árboles**: Más estable, pero más lento
    - **Max depth pequeña**: Previene overfitting
    - **Min samples split/leaf alto**: Árboles más conservadores
    - **Max features='sqrt'**: Reduce correlación entre árboles
    """)

st.caption("Módulo 3: Configuración de Modelos | TFM Alzheimer Classification")
