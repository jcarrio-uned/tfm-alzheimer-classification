"""Módulo 4: Evaluación y Resultados.

Entrenamiento, evaluación y visualización de resultados del modelo.
"""

import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Añadir path al módulo principal
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app_utils import (
    WinsorizerTransformer,
    get_metabolite_columns,
    initialize_session_state,
    load_data,
)

# ============================================================================
# Configuración de la página
# ============================================================================

st.set_page_config(page_title="Evaluación y Resultados", page_icon="📈", layout="wide")

initialize_session_state()

# ============================================================================
# Header
# ============================================================================

st.title("📈 Evaluación y Resultados")
st.markdown("""
Entrena el modelo con la configuración seleccionada y visualiza resultados completos.
""")

# ============================================================================
# Verificar configuración previa
# ============================================================================

df = load_data()

# Verificar features
if "selected_features" not in st.session_state or not st.session_state["selected_features"]:
    st.error("❌ No has seleccionado variables")
    st.info("👉 Ve al módulo **🔍 Selección de Variables**")
    st.stop()

# Verificar configuración de modelo
if "model_config" not in st.session_state:
    st.error("❌ No has configurado el modelo")
    st.info("👉 Ve al módulo **🤖 Configuración de Modelos**")
    st.stop()

# ============================================================================
# Cargar configuración
# ============================================================================

selected_features = st.session_state["selected_features"]
model_config = st.session_state["model_config"]

# Generar features ingenieriles si están en session_state pero no en df
eng_data = st.session_state.get("eng_data", {})
for feat_name, feat_values in eng_data.items():
    if feat_name not in df.columns:
        df[feat_name] = feat_values

# Validar que las features existen en el DataFrame
available_features = get_metabolite_columns(df)
missing_features = [f for f in selected_features if f not in df.columns]

if missing_features:
    st.error(f"❌ Las siguientes features no están en el dataset: {missing_features}")
    st.info("💡 Las features disponibles son las que aparecen en el dataset actual. Vuelve a Selección de Variables.")
    with st.expander("Ver features disponibles"):
        st.write(available_features)
    st.stop()

# Preparar datos
X = df[selected_features].values
y = (df["Group"] == "AD").astype(int).values
sample_ids = df["ID"].values if "ID" in df.columns else np.arange(len(df))

# NOTA: La winsorización ahora se aplica DENTRO del pipeline (WinsorizerTransformer)
# para evitar data leakage. Los bounds se aprenden solo de los datos de train.

# ============================================================================
# Sidebar: Resumen de configuración
# ============================================================================

with st.sidebar:
    st.header("⚙️ Configuración Actual")

    st.metric("Modelo", model_config["model_type"])
    st.metric("Features", len(selected_features))
    st.metric("Random State", model_config["random_state"])

    with st.expander("Ver features"):
        for feat in selected_features:
            st.caption(f"• {feat}")

# ============================================================================
# Función para crear modelo
# ============================================================================


def create_model(config, use_grid_search=False):
    """Crea modelo según configuración."""
    model_type = config["model_type"]

    if model_type == "Logistic Regression":
        # Parámetros base (sin C si es grid search)
        if use_grid_search:
            params = {
                "penalty": config["penalty"],
                "solver": config.get("solver", "lbfgs"),
                "class_weight": config["class_weight"],
                "max_iter": config["max_iter"],
                "tol": config["tol"],
                "random_state": config["random_state"],
            }
        else:
            params = {
                "C": config["C"],
                "penalty": config["penalty"],
                "solver": config.get("solver", "lbfgs"),
                "class_weight": config["class_weight"],
                "max_iter": config["max_iter"],
                "tol": config["tol"],
                "random_state": config["random_state"],
            }
        model = LogisticRegression(**params)

    elif model_type == "Support Vector Machine (SVM)":
        if use_grid_search:
            params = {
                "kernel": config["kernel"],
                "class_weight": config["class_weight"],
                "probability": config.get("probability", True),
                "max_iter": config["max_iter"],
                "random_state": config["random_state"],
            }
        else:
            params = {
                "C": config["C"],
                "kernel": config["kernel"],
                "class_weight": config["class_weight"],
                "probability": config.get("probability", True),
                "max_iter": config["max_iter"],
                "random_state": config["random_state"],
            }

        if config["kernel"] in ["rbf", "poly", "sigmoid"]:
            params["gamma"] = config["gamma"]

        if config["kernel"] == "poly":
            params["degree"] = config["degree"]

        model = SVC(**params)

    else:  # Random Forest
        params = {
            "n_estimators": config["n_estimators"],
            "max_depth": config["max_depth"],
            "min_samples_split": config["min_samples_split"],
            "min_samples_leaf": config["min_samples_leaf"],
            "max_features": config["max_features"],
            "class_weight": config["class_weight"],
            "random_state": config["random_state"],
        }
        model = RandomForestClassifier(**params)

    # Crear pipeline: Winsorización (train-only bounds) → Scaler → Clasificador
    return Pipeline(
        [
            ("winsorizer", WinsorizerTransformer(percentile=0.05)),
            ("scaler", StandardScaler()),
            ("classifier", model),
        ]
    )


# ============================================================================
# Sección 1: Entrenar y Evaluar
# ============================================================================

st.header("1️⃣ Entrenamiento y Evaluación")

# Opción de método de evaluación
st.subheader("⚙️ Método de Evaluación")

evaluation_method = st.selectbox(
    "Selecciona método de validación:",
    [
        "RepeatedStratifiedKFold",
        "StratifiedKFold",
        "StratifiedShuffleSplit (Holdout)",
    ],
    index=0,
    help="Elige el método de validación cruzada o holdout",
)

# Configuración específica según el método
st.markdown("##### Configuración del método:")
col_config1, col_config2, col_config3 = st.columns(3)

if evaluation_method == "RepeatedStratifiedKFold":
    with col_config1:
        cv_n_splits = st.number_input("N° folds (n_splits)", 2, 10, 5, step=1)
    with col_config2:
        cv_n_repeats = st.number_input("N° repeticiones (n_repeats)", 1, 20, 10, step=1)
    with col_config3:
        st.metric("Total evaluaciones", cv_n_splits * cv_n_repeats)

    st.info(
        f"""
        **RepeatedStratifiedKFold** - {cv_n_splits} folds × {cv_n_repeats} repeticiones
        - Cada muestra se predice {cv_n_repeats} veces (1 vez por repetición)
        - Los folds se entrelazan: usa el 100% de los datos
        - ⚡ Rápido y eficiente
        - ⚠️ Puede ser ~1-3 puntos optimista vs holdout puro
        """
    )

elif evaluation_method == "StratifiedKFold":
    with col_config1:
        cv_n_splits = st.number_input("N° folds (n_splits)", 2, 10, 5, step=1)
    with col_config2:
        cv_n_repeats = 1  # Sin repeticiones
        st.metric("N° repeticiones", 1)
    with col_config3:
        st.metric("Total evaluaciones", cv_n_splits)

    st.info(
        f"""
        **StratifiedKFold** - {cv_n_splits} folds (sin repeticiones)
        - Cada muestra se predice exactamente 1 vez
        - Usa el 100% de los datos
        - ⚡ Más rápido que Repeated
        - Útil para datasets pequeños
        """
    )

else:  # StratifiedShuffleSplit
    with col_config1:
        holdout_n_splits = st.number_input(
            "N° repeticiones", 10, 100, 50, step=10, help="Número de splits independientes (recomendado: 50)"
        )
    with col_config2:
        holdout_test_size = st.slider("Test size (%)", 10, 40, 20, step=5)
    with col_config3:
        st.metric("Splits independientes", holdout_n_splits)

    st.info(
        f"""
        **StratifiedShuffleSplit** - {holdout_n_splits} splits independientes ({100 - holdout_test_size}/{holdout_test_size})
        - Método usado en el pipeline original
        - Cada split es completamente independiente
        - Calcula Train BA y Test BA → Generalization Gap
        - 🎯 Más conservador y realista
        - 📊 BA esperado: Bio-E 0.857±0.064, Minimal-7 0.860±0.058
        """
    )

st.markdown("---")

# Opción de train/test split adicional (solo para métodos CV, no para StratifiedShuffleSplit)
use_train_test_split = False
if evaluation_method != "StratifiedShuffleSplit (Holdout)":
    use_train_test_split = st.checkbox(
        "🎯 Usar Train/Test Split adicional",
        value=False,
        help="Divide los datos en entrenamiento y prueba antes de CV. Permite evaluar la generalización del modelo en datos no vistos.",
    )

    if use_train_test_split:
        col1, col2 = st.columns(2)
        with col1:
            test_size_pct = st.slider("Test size (%)", 10, 40, 20, step=5)
        with col2:
            split_seed = st.number_input("Semilla para split", 0, 999, model_config["random_state"], step=1)

        st.info(
            f"📊 Se usará {100 - test_size_pct}% para entrenamiento (con CV) y {test_size_pct}% para evaluación final (holdout)"
        )

st.markdown("---")

# Botón de entrenamiento
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    train_button = st.button("🚀 Entrenar y Evaluar Modelo", type="primary")

with col2:
    if "cv_results" in st.session_state:
        st.success("✓ Modelo entrenado")

with col3:
    if "cv_results" in st.session_state and st.button("🗑️ Limpiar resultados"):
        for key in ["cv_results", "trained_pipeline", "model_trained", "predictions", "probabilities"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# Ejecutar entrenamiento
if train_button:
    # Limpiar estado anterior para evitar mezclar resultados de diferentes métodos
    for key in [
        "cv_results",
        "holdout_results",
        "test_metrics",
        "used_train_test_split",
        "fold_roc_curves",
        "fold_pr_curves",
        "predictions",
        "probabilities",
        "test_predictions",
        "test_probabilities",
        "test_y_true",
    ]:
        if key in st.session_state:
            del st.session_state[key]

    # =========================================================================
    # MÉTODO 1: STRATIFIED SHUFFLE SPLIT (Holdout independiente)
    # =========================================================================
    if evaluation_method == "StratifiedShuffleSplit (Holdout)":
        with st.spinner(
            f"Evaluando con {holdout_n_splits} holdout splits ({100 - holdout_test_size}/{holdout_test_size})..."
        ):
            n_repeats = holdout_n_splits
            test_size = holdout_test_size / 100.0

            # Verificar si se usa grid search
            use_grid_search = model_config.get("use_grid_search", False)
            best_c_per_split = []  # Para almacenar el mejor C de cada split (si hay grid search)

            # Arrays para almacenar resultados de cada split
            test_scores = []
            train_scores = []
            sens_scores = []
            spec_scores = []

            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Crear objeto SSS con todos los splits (reproducible con page 2)
            sss = StratifiedShuffleSplit(
                n_splits=n_repeats, test_size=test_size, random_state=model_config["random_state"]
            )

            for i, (train_idx, test_idx) in enumerate(sss.split(X, y)):
                status_text.text(f"Evaluando split {i + 1}/{n_repeats}...")

                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # Entrenar modelo (con o sin grid search)
                if use_grid_search and model_config["model_type"] in [
                    "Logistic Regression",
                    "Support Vector Machine (SVM)",
                ]:
                    # Grid search para encontrar mejor C
                    pipeline_base = create_model(model_config, use_grid_search=True)
                    param_grid = {"classifier__C": model_config["grid_c_values"]}

                    # CV interno para grid search (5-fold)
                    cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=model_config["random_state"])

                    grid_search = GridSearchCV(
                        pipeline_base,
                        param_grid,
                        cv=cv_inner,
                        scoring="balanced_accuracy",
                        n_jobs=1,  # Para no saturar en el loop
                    )
                    grid_search.fit(X_train, y_train)

                    pipeline = grid_search.best_estimator_
                    best_c_per_split.append(grid_search.best_params_["classifier__C"])
                else:
                    pipeline = create_model(model_config)
                    pipeline.fit(X_train, y_train)

                # Evaluar en train
                y_pred_train = pipeline.predict(X_train)
                ba_train = balanced_accuracy_score(y_train, y_pred_train)
                train_scores.append(ba_train)

                # Evaluar en test
                y_pred_test = pipeline.predict(X_test)
                ba_test = balanced_accuracy_score(y_test, y_pred_test)
                test_scores.append(ba_test)

                # Calcular sensibilidad y especificidad
                cm = confusion_matrix(y_test, y_pred_test)
                tn, fp, fn, tp = cm.ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                sens_scores.append(sensitivity)
                spec_scores.append(specificity)

                progress_bar.progress((i + 1) / n_repeats)

            progress_bar.progress(1.0)
            status_text.text("✓ Evaluación completada!")

            # Convertir a arrays
            test_scores = np.array(test_scores)
            train_scores = np.array(train_scores)
            sens_scores = np.array(sens_scores)
            spec_scores = np.array(spec_scores)

            # Si hubo grid search, mostrar resultados
            if use_grid_search and best_c_per_split:
                best_c_per_split = np.array(best_c_per_split)
                most_common_c = np.median(best_c_per_split)  # Usar mediana como representativo
                st.session_state["grid_search_results"] = {
                    "best_C": most_common_c,
                    "best_C_per_split": best_c_per_split,
                    "best_score": test_scores.mean(),
                }
                st.success(f"✓ **C más frecuente**: {most_common_c:.3f} (mediana de {n_repeats} splits)")

            # Almacenar resultados en formato compatible con CV
            st.session_state["holdout_results"] = {
                "test_scores": test_scores,
                "train_scores": train_scores,
                "sens_scores": sens_scores,
                "spec_scores": spec_scores,
                "n_repeats": n_repeats,
            }

            # Crear cv_results simulado para compatibilidad con código existente
            st.session_state["cv_results"] = {
                "test_balanced_accuracy": test_scores,
                "test_recall": sens_scores,
                "test_precision": np.zeros(n_repeats),  # No calculado en holdout simple
                "test_f1": np.zeros(n_repeats),
                "test_roc_auc": np.zeros(n_repeats),
                "test_average_precision": np.zeros(n_repeats),
            }

            # Entrenar modelo final en TODOS los datos (usando mejor C si hubo grid search)
            status_text.text("Entrenando modelo final en todos los datos...")
            if use_grid_search and best_c_per_split is not None and len(best_c_per_split) > 0:
                model_config_final = model_config.copy()
                model_config_final["C"] = most_common_c
                model_config_final["use_grid_search"] = False
                pipeline_final = create_model(model_config_final)
            else:
                pipeline_final = create_model(model_config)

            pipeline_final.fit(X, y)
            st.session_state["trained_pipeline"] = pipeline_final
            st.session_state["model_trained"] = True
            st.session_state["evaluation_method"] = "holdout"

            # Predicciones en todo el dataset (para análisis)
            y_pred_all = pipeline_final.predict(X)
            if hasattr(pipeline_final.named_steps["classifier"], "predict_proba"):
                y_proba_all = pipeline_final.predict_proba(X)[:, 1]
            else:
                y_proba_all = pipeline_final.decision_function(X)

            st.session_state["predictions"] = y_pred_all
            st.session_state["probabilities"] = y_proba_all
            st.session_state["y_true"] = y  # Etiquetas verdaderas de todo el dataset
            st.session_state["train_indices"] = None  # Holdout usa todo el dataset
            st.session_state["cv_config"] = {
                "method": "StratifiedShuffleSplit",
                "n_splits": n_repeats,
                "test_size": int(test_size * 100),
            }

    # =========================================================================
    # MÉTODO 2: CROSS-VALIDATION (StratifiedKFold o RepeatedStratifiedKFold)
    # =========================================================================
    else:
        # Determinar mensaje según el método
        if evaluation_method == "RepeatedStratifiedKFold":
            cv_msg = f"Cross-validation con {cv_n_splits} folds × {cv_n_repeats} repeticiones"
        else:  # StratifiedKFold
            cv_msg = f"Cross-validation con {cv_n_splits} folds"

        with st.spinner(cv_msg + "..."):
            # Split train/test si está activado
            if use_train_test_split:
                test_size_frac = test_size_pct / 100.0
                indices = np.arange(len(y))
                train_idx, test_idx = train_test_split(
                    indices, test_size=test_size_frac, stratify=y, random_state=split_seed
                )
                X_train_full, X_test_full = X[train_idx], X[test_idx]
                y_train_full, y_test_full = y[train_idx], y[test_idx]
                sample_ids_train, sample_ids_test = sample_ids[train_idx], sample_ids[test_idx]

                st.info(f"📊 Train: {len(y_train_full)} muestras | Test: {len(y_test_full)} muestras")

                # Guardar índices de entrenamiento para alinear predicciones con datos reales
                train_indices_to_save = train_idx
            else:
                # Sin split: usa todos los datos para CV
                X_train_full, y_train_full = X, y
                X_test_full, y_test_full = None, None
                sample_ids_train = sample_ids
                train_indices_to_save = None

            # Configurar CV en el conjunto de entrenamiento
            if evaluation_method == "RepeatedStratifiedKFold":
                cv = RepeatedStratifiedKFold(
                    n_splits=cv_n_splits,
                    n_repeats=cv_n_repeats,
                    random_state=model_config["random_state"],
                )
                total_folds = cv_n_splits * cv_n_repeats
            else:  # StratifiedKFold
                cv = StratifiedKFold(
                    n_splits=cv_n_splits,
                    shuffle=True,
                    random_state=model_config["random_state"],
                )
                total_folds = cv_n_splits

            # Progress bar y status text
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Crear modelo base
            use_grid_search = model_config.get("use_grid_search", False)

            if use_grid_search and model_config["model_type"] in [
                "Logistic Regression",
                "Support Vector Machine (SVM)",
            ]:
                st.info("🔍 **Grid Search activado**: Buscando el mejor valor de C mediante CV...")

                # Crear pipeline base sin C especificado
                pipeline_base = create_model(model_config, use_grid_search=True)

                # Definir grid de parámetros
                param_grid = {"classifier__C": model_config["grid_c_values"]}

                # Crear GridSearchCV
                grid_search = GridSearchCV(
                    pipeline_base,
                    param_grid,
                    cv=cv,
                    scoring="balanced_accuracy",
                    n_jobs=1,
                    return_train_score=False,
                    verbose=0,
                )

                # Entrenar con grid search
                status_text.text(f"Probando {len(model_config['grid_c_values'])} valores de C...")
                grid_search.fit(X_train_full, y_train_full)
                progress_bar.progress(0.5)

                # Obtener mejor modelo
                best_pipeline = grid_search.best_estimator_
                best_c = grid_search.best_params_["classifier__C"]
                best_score = grid_search.best_score_

                # Guardar información del grid search
                st.session_state["grid_search_results"] = {
                    "best_C": best_c,
                    "best_score": best_score,
                    "best_params": grid_search.best_params_,
                    "cv_results": grid_search.cv_results_,
                }

                st.success(f"✓ **Mejor C encontrado**: {best_c} (BA = {best_score:.3f})")

                # Ahora usar el mejor modelo para obtener predicciones de CV
                # y calcular métricas completas
                status_text.text("Calculando métricas con mejor modelo...")

                # Usar el mejor C para crear resultados detallados
                model_config_best = model_config.copy()
                model_config_best["C"] = best_c
                model_config_best["use_grid_search"] = False

                # Actualizar model_config original para que tenga el mejor C
                # (necesario para crear modelos en el fold loop posterior)
                model_config["C"] = best_c
                model_config["use_grid_search"] = False

                pipeline = create_model(model_config_best)

            else:
                # Modelo normal sin grid search
                pipeline = create_model(model_config)
                status_text.text("Iniciando validación cruzada...")

            # Scoring metrics
            scoring = {
                "balanced_accuracy": "balanced_accuracy",
                "roc_auc": "roc_auc",
                "average_precision": "average_precision",
                "recall": "recall",  # Sensitivity
                "precision": "precision",
                "f1": "f1",
            }

            # Cross-validation en entrenamiento (con mejor modelo si hubo grid search)
            cv_results = cross_validate(
                pipeline, X_train_full, y_train_full, cv=cv, scoring=scoring, return_train_score=False, n_jobs=1
            )

            progress_bar.progress(0.7)
            status_text.text("Calculando curvas ROC y PR...")

            # Obtener predicciones y probabilidades para cada fold
            # Inicializar arrays para reconstruir predicciones completas (sin data leakage)
            cv_predictions = np.zeros(len(y_train_full))  # Predicciones de CV
            cv_probabilities = np.zeros(len(y_train_full))  # Probabilidades de CV
            cv_prediction_counts = np.zeros(len(y_train_full))  # Contador para promediar repeticiones

            fold_roc_curves = []
            fold_pr_curves = []

            fold_counter = 0

            for fold_counter, (train_idx, test_idx) in enumerate(cv.split(X_train_full, y_train_full)):
                X_train, X_test = X_train_full[train_idx], X_train_full[test_idx]
                y_train, y_test = y_train_full[train_idx], y_train_full[test_idx]

                # Entrenar
                pipeline_fold = create_model(model_config)
                pipeline_fold.fit(X_train, y_train)

                # Predecir
                y_pred = pipeline_fold.predict(X_test)

                # Probabilidades
                if hasattr(pipeline_fold.named_steps["classifier"], "predict_proba"):
                    y_proba = pipeline_fold.predict_proba(X_test)[:, 1]
                else:
                    y_proba = pipeline_fold.decision_function(X_test)

                # Acumular predicciones y probabilidades (para promediar repeticiones)
                cv_predictions[test_idx] += y_pred
                cv_probabilities[test_idx] += y_proba
                cv_prediction_counts[test_idx] += 1

                # Curvas ROC y PR
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                precision, recall, _ = precision_recall_curve(y_test, y_proba)

                fold_roc_curves.append((fpr, tpr))
                fold_pr_curves.append((precision, recall))

                progress_bar.progress(0.7 + 0.3 * ((fold_counter + 1) / total_folds))

            # Promediar predicciones de repeticiones de CV
            cv_predictions = cv_predictions / cv_prediction_counts
            cv_probabilities = cv_probabilities / cv_prediction_counts

            # Convertir probabilidades promediadas a predicciones binarias (threshold 0.5)
            cv_predictions_binary = (cv_predictions > 0.5).astype(int)

            progress_bar.progress(1.0)
            status_text.text("✓ Evaluación completada!")

            # Almacenar resultados de CV
            st.session_state["cv_results"] = cv_results
            st.session_state["fold_roc_curves"] = fold_roc_curves
            st.session_state["fold_pr_curves"] = fold_pr_curves

            # Predicciones válidas de CV (SIN data leakage) - solo en conjunto de entrenamiento
            st.session_state["predictions"] = cv_predictions_binary
            st.session_state["probabilities"] = cv_probabilities
            st.session_state["y_true"] = y_train_full  # Etiquetas verdaderas correspondientes
            st.session_state["train_indices"] = train_indices_to_save  # Índices para alinear con DataFrame original

            # Si hay test set, entrenar modelo final y evaluar en test
            if use_train_test_split and X_test_full is not None:
                status_text.text("Evaluando en Test Set...")

                # Entrenar en TODOS los datos de entrenamiento
                pipeline_final = create_model(model_config)
                pipeline_final.fit(X_train_full, y_train_full)

                # Predecir en test
                y_pred_test = pipeline_final.predict(X_test_full)
                if hasattr(pipeline_final.named_steps["classifier"], "predict_proba"):
                    y_proba_test = pipeline_final.predict_proba(X_test_full)[:, 1]
                else:
                    y_proba_test = pipeline_final.decision_function(X_test_full)

                # Calcular métricas en test
                test_metrics = {
                    "balanced_accuracy": balanced_accuracy_score(y_test_full, y_pred_test),
                    "precision": precision_score(y_test_full, y_pred_test, zero_division=0),
                    "recall": recall_score(y_test_full, y_pred_test),
                    "f1": f1_score(y_test_full, y_pred_test),
                    "roc_auc": roc_auc_score(y_test_full, y_proba_test),
                    "average_precision": average_precision_score(y_test_full, y_proba_test),
                }

                # Guardar métricas y predicciones de test
                st.session_state["test_metrics"] = test_metrics
                st.session_state["test_predictions"] = y_pred_test
                st.session_state["test_probabilities"] = y_proba_test
                st.session_state["test_y_true"] = y_test_full
                st.session_state["used_train_test_split"] = True

                # Entrenar modelo final con TODOS los datos (train + test) para predicciones futuras
                pipeline_all = create_model(model_config)
                pipeline_all.fit(X, y)
                st.session_state["trained_pipeline"] = pipeline_all
            else:
                # Sin split: entrenar modelo final con TODOS los datos
                pipeline_final = create_model(model_config)
                pipeline_final.fit(X, y)
                st.session_state["trained_pipeline"] = pipeline_final
                st.session_state["used_train_test_split"] = False

            st.session_state["model_trained"] = True
            st.session_state["evaluation_method"] = "cv"
            st.session_state["cv_config"] = {
                "method": evaluation_method,
                "n_splits": cv_n_splits,
                "n_repeats": cv_n_repeats if evaluation_method == "RepeatedStratifiedKFold" else 1,
            }

            # Timestamp
            st.session_state["training_timestamp"] = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M:%S")

            st.rerun()

# ============================================================================
# Mostrar resultados si existen
# ============================================================================

if "cv_results" not in st.session_state:
    st.info("👆 Haz clic en 'Entrenar y Evaluar Modelo' para comenzar")
    st.stop()

# Cargar resultados
cv_results = st.session_state["cv_results"]
evaluation_method = st.session_state.get("evaluation_method", "cv")
cv_config = st.session_state.get("cv_config", {})

# Cargar curvas si existen (solo en CV)
fold_roc_curves = st.session_state.get("fold_roc_curves", [])
fold_pr_curves = st.session_state.get("fold_pr_curves", [])

# ============================================================================
# Sección 2: Métricas Principales
# ============================================================================

st.header("2️⃣ Métricas de Performance")

# Mostrar método de evaluación usado
if evaluation_method == "holdout":
    n_splits = cv_config.get("n_splits", 50)
    test_size = cv_config.get("test_size", 20)
    st.info(
        f"📊 **Método:** StratifiedShuffleSplit - {n_splits} splits ({100 - test_size}/{test_size}) - Compatible con pipeline original"
    )
    holdout_results = st.session_state.get("holdout_results", {})
else:
    method_name = cv_config.get("method", "Cross-Validation")
    n_splits = cv_config.get("n_splits", 5)
    n_repeats = cv_config.get("n_repeats", 10)
    if n_repeats > 1:
        st.info(f"📊 **Método:** {method_name} - {n_splits} folds × {n_repeats} repeticiones")
    else:
        st.info(f"📊 **Método:** {method_name} - {n_splits} folds")

st.caption(f"Entrenado: {st.session_state.get('training_timestamp', 'N/A')}")

# KPIs principales
col1, col2, col3, col4 = st.columns(4)

if evaluation_method == "holdout":
    # Mostrar métricas de holdout
    holdout_results = st.session_state.get("holdout_results", {})

    with col1:
        ba_mean = holdout_results["test_scores"].mean()
        ba_std = holdout_results["test_scores"].std()
        st.metric("🎯 Test BA", f"{ba_mean:.3f}", delta=f"±{ba_std:.3f}")

    with col2:
        train_mean = holdout_results["train_scores"].mean()
        train_std = holdout_results["train_scores"].std()
        st.metric("📚 Train BA", f"{train_mean:.3f}", delta=f"±{train_std:.3f}")

    with col3:
        sens_mean = holdout_results["sens_scores"].mean()
        sens_std = holdout_results["sens_scores"].std()
        st.metric("🔍 Sensitivity", f"{sens_mean:.3f}", delta=f"±{sens_std:.3f}")

    with col4:
        spec_mean = holdout_results["spec_scores"].mean()
        spec_std = holdout_results["spec_scores"].std()
        st.metric("🎯 Specificity", f"{spec_mean:.3f}", delta=f"±{spec_std:.3f}")

    # Gap de generalización
    gap_mean = (holdout_results["train_scores"] - holdout_results["test_scores"]).mean()
    gap_std = (holdout_results["train_scores"] - holdout_results["test_scores"]).std()

    st.markdown("---")
    col_gap1, col_gap2, col_gap3 = st.columns(3)
    with col_gap1:
        st.metric("📉 Generalization Gap (Train - Test)", f"{gap_mean:.3f}", delta=f"±{gap_std:.3f}")
    with col_gap2:
        if abs(gap_mean) < 0.03:
            st.success("✅ Excelente generalización (gap < 0.03)")
        elif abs(gap_mean) < 0.05:
            st.info("✓ Buena generalización (gap < 0.05)")
        else:
            st.warning("⚠️ Posible overfitting (gap > 0.05)")
    with col_gap3:
        st.metric("N Repeticiones", holdout_results["n_repeats"])

else:
    # Mostrar métricas de CV (código original)
    metrics_data = {
        "balanced_accuracy": ("Balanced Accuracy", "🎯"),
        "roc_auc": ("ROC AUC", "📊"),
        "average_precision": ("Avg Precision", "📏"),
        "recall": ("Sensitivity", "🔍"),
    }

    for _i, (col, (metric_key, (label, icon))) in enumerate(
        zip([col1, col2, col3, col4], metrics_data.items(), strict=False)
    ):
        with col:
            scores = cv_results[f"test_{metric_key}"]
            mean_score = scores.mean()
            std_score = scores.std()

            st.metric(f"{icon} {label}", f"{mean_score:.3f}", delta=f"±{std_score:.3f}")

# Tabla detallada de métricas
st.subheader("📋 Tabla Completa de Métricas")

if evaluation_method == "holdout":
    # Tabla para holdout
    st.info(
        "📊 **Métricas de 50 Holdout Splits**: Promedio de 50 repeticiones de train/test (80/20) con diferentes semillas"
    )
    holdout_results = st.session_state.get("holdout_results", {})
    metrics_df = pd.DataFrame(
        {
            "Métrica": [
                "Test Balanced Accuracy",
                "Train Balanced Accuracy",
                "Generalization Gap",
                "Sensitivity (AD Detection)",
                "Specificity (NC Detection)",
            ],
            "Media": [
                holdout_results["test_scores"].mean(),
                holdout_results["train_scores"].mean(),
                (holdout_results["train_scores"] - holdout_results["test_scores"]).mean(),
                holdout_results["sens_scores"].mean(),
                holdout_results["spec_scores"].mean(),
            ],
            "Std": [
                holdout_results["test_scores"].std(),
                holdout_results["train_scores"].std(),
                (holdout_results["train_scores"] - holdout_results["test_scores"]).std(),
                holdout_results["sens_scores"].std(),
                holdout_results["spec_scores"].std(),
            ],
            "Min": [
                holdout_results["test_scores"].min(),
                holdout_results["train_scores"].min(),
                (holdout_results["train_scores"] - holdout_results["test_scores"]).min(),
                holdout_results["sens_scores"].min(),
                holdout_results["spec_scores"].min(),
            ],
            "Max": [
                holdout_results["test_scores"].max(),
                holdout_results["train_scores"].max(),
                (holdout_results["train_scores"] - holdout_results["test_scores"]).max(),
                holdout_results["sens_scores"].max(),
                holdout_results["spec_scores"].max(),
            ],
        }
    )
else:
    # Tabla para CV (código original)
    st.info("📊 **Métricas de Cross-Validation**: Promedio de folds de CV (sin data leakage)")
    metrics_df = pd.DataFrame(
        {
            "Métrica": [
                "Balanced Accuracy",
                "ROC AUC",
                "Average Precision",
                "Sensitivity (Recall)",
                "Precision",
                "F1-Score",
            ],
            "Media": [
                cv_results["test_balanced_accuracy"].mean(),
                cv_results["test_roc_auc"].mean(),
                cv_results["test_average_precision"].mean(),
                cv_results["test_recall"].mean(),
                cv_results["test_precision"].mean(),
                cv_results["test_f1"].mean(),
            ],
            "Std": [
                cv_results["test_balanced_accuracy"].std(),
                cv_results["test_roc_auc"].std(),
                cv_results["test_average_precision"].std(),
                cv_results["test_recall"].std(),
                cv_results["test_precision"].std(),
                cv_results["test_f1"].std(),
            ],
            "Min": [
                cv_results["test_balanced_accuracy"].min(),
                cv_results["test_roc_auc"].min(),
                cv_results["test_average_precision"].min(),
                cv_results["test_recall"].min(),
                cv_results["test_precision"].min(),
                cv_results["test_f1"].min(),
            ],
            "Max": [
                cv_results["test_balanced_accuracy"].max(),
                cv_results["test_roc_auc"].max(),
                cv_results["test_average_precision"].max(),
                cv_results["test_recall"].max(),
                cv_results["test_precision"].max(),
                cv_results["test_f1"].max(),
            ],
        }
    )

metrics_df["95% CI"] = metrics_df.apply(
    lambda row: f"[{row['Media'] - 1.96 * row['Std']:.3f}, {row['Media'] + 1.96 * row['Std']:.3f}]", axis=1
)

st.dataframe(
    metrics_df.style.format({"Media": "{:.3f}", "Std": "{:.3f}", "Min": "{:.3f}", "Max": "{:.3f}"}).background_gradient(
        subset=["Media"], cmap="YlGn"
    ),
    width="stretch",
)

# ============================================================================
# Grid Search Results y Model Coefficients
# ============================================================================

if "grid_search_results" in st.session_state:
    st.markdown("---")
    st.subheader("🔍 Resultados de Grid Search")

    grid_results = st.session_state["grid_search_results"]
    best_c = grid_results["best_C"]
    best_score = grid_results["best_score"]

    col1, col2 = st.columns(2)

    with col1:
        st.metric("✨ Mejor C encontrado", f"{best_c:.4f}")
        st.caption("Valor de C que maximiza el Balanced Accuracy en CV")

    with col2:
        st.metric("📊 BA con mejor C", f"{best_score:.3f}")
        st.caption("Balanced Accuracy obtenido con el mejor C")

    # Mostrar distribución de C si es holdout con múltiples splits
    if "best_C_per_split" in grid_results:
        best_c_per_split = grid_results["best_C_per_split"]
        with st.expander(f"📈 Distribución de mejor C en {len(best_c_per_split)} splits"):
            # Crear histograma
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=best_c_per_split, nbinsx=20, name="Frecuencia", marker_color="steelblue"))
            fig.update_layout(
                title=f"Distribución de mejor C en {len(best_c_per_split)} splits",
                xaxis_title="Valor de C",
                yaxis_title="Frecuencia",
                height=300,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Estadísticas
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                st.metric("Min", f"{best_c_per_split.min():.4f}")
            with col_b:
                st.metric("Mediana", f"{np.median(best_c_per_split):.4f}")
            with col_c:
                st.metric("Media", f"{best_c_per_split.mean():.4f}")
            with col_d:
                st.metric("Max", f"{best_c_per_split.max():.4f}")

    # Mostrar grid completo si está disponible
    if "cv_results" in grid_results and grid_results["cv_results"] is not None:
        with st.expander("📊 Ver resultados completos del Grid Search"):
            grid_cv_results = grid_results["cv_results"]

            # Crear dataframe con resultados
            grid_df = pd.DataFrame(
                {
                    "C": grid_cv_results["param_classifier__C"],
                    "Mean BA": grid_cv_results["mean_test_score"],
                    "Std BA": grid_cv_results["std_test_score"],
                }
            )
            grid_df = grid_df.sort_values("Mean BA", ascending=False)

            st.dataframe(
                grid_df.style.format({"C": "{:.4f}", "Mean BA": "{:.3f}", "Std BA": "{:.3f}"}).background_gradient(
                    subset=["Mean BA"], cmap="YlGn"
                ),
                width="stretch",
            )

            # Gráfico de C vs BA
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=grid_df["C"],
                    y=grid_df["Mean BA"],
                    mode="lines+markers",
                    name="BA",
                    error_y={"type": "data", "array": grid_df["Std BA"], "visible": True},
                    marker={"size": 10, "color": "steelblue"},
                    line={"color": "steelblue", "width": 2},
                )
            )

            # Marcar el mejor C
            best_idx = grid_df["C"] == best_c
            if best_idx.any():
                fig.add_trace(
                    go.Scatter(
                        x=[best_c],
                        y=[grid_df[best_idx]["Mean BA"].values[0]],
                        mode="markers",
                        name="Mejor C",
                        marker={"size": 15, "color": "red", "symbol": "star"},
                    )
                )

            fig.update_layout(
                title="Balanced Accuracy vs C",
                xaxis_title="C (regularización)",
                yaxis_title="Balanced Accuracy",
                xaxis_type="log",  # Escala logarítmica para C
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

# Mostrar coeficientes del modelo (solo para Logistic Regression)
if "trained_pipeline" in st.session_state:
    trained_pipeline = st.session_state["trained_pipeline"]

    # Verificar si es Logistic Regression
    if hasattr(trained_pipeline.named_steps["classifier"], "coef_"):
        st.markdown("---")
        st.subheader("📊 Coeficientes del Modelo")

        coef = trained_pipeline.named_steps["classifier"].coef_[0]
        intercept = trained_pipeline.named_steps["classifier"].intercept_[0]

        # Crear dataframe con coeficientes
        coef_df = pd.DataFrame({"Variable": selected_features, "Coeficiente": coef, "Abs(Coef)": np.abs(coef)})
        coef_df = coef_df.sort_values("Abs(Coef)", ascending=False)

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("##### Coeficientes por Variable")
            st.dataframe(
                coef_df[["Variable", "Coeficiente"]]
                .style.format({"Coeficiente": "{:.4f}"})
                .background_gradient(subset=["Coeficiente"], cmap="RdYlGn"),
                width="stretch",
            )

        with col2:
            st.markdown("##### Información del Modelo")
            st.metric("Intercept", f"{intercept:.4f}")
            st.metric("N° Features", len(selected_features))
            st.metric("Features con coef > 0", (coef > 0).sum())
            st.metric("Features con coef < 0", (coef < 0).sum())

        # Gráfico de importancia de coeficientes
        fig = go.Figure()

        colors = ["green" if c > 0 else "red" for c in coef_df["Coeficiente"]]

        fig.add_trace(
            go.Bar(
                x=coef_df["Coeficiente"],
                y=coef_df["Variable"],
                orientation="h",
                marker={"color": colors},
                text=coef_df["Coeficiente"].apply(lambda x: f"{x:.3f}"),
                textposition="outside",
            )
        )

        fig.update_layout(
            title="Importancia de Variables (Coeficientes estandarizados)",
            xaxis_title="Coeficiente (después de StandardScaler)",
            yaxis_title="Variable",
            height=max(400, len(selected_features) * 30),
            showlegend=False,
        )

        st.plotly_chart(fig, use_container_width=True)

        st.caption("""
        💡 **Interpretación**:
        - Coeficientes positivos (verde): Aumentan la probabilidad de AD
        - Coeficientes negativos (rojo): Disminuyen la probabilidad de AD
        - Magnitud absoluta indica importancia relativa de cada variable
        - Coeficientes están estandarizados (después de StandardScaler), por lo que son comparables
        """)

# Nota explicativa según el método
if evaluation_method == "holdout":
    st.info("""
📊 **Metodología de 50 Holdout Splits**
- Se realizan 50 repeticiones con diferentes splits train/test (80/20)
- **Test BA**: Promedio del Balanced Accuracy en los 50 test sets
- **Train BA**: Promedio del Balanced Accuracy en los 50 train sets
- **Generalization Gap**: Diferencia Train - Test (idealmente < 0.05)
- **Sin data leakage**: Cada split evalúa en datos NO usados para entrenar
""")
else:
    st.info("""
📊 **Métricas de Validación Cruzada (Train Set)**
- Estas métricas provienen de **Repeated Stratified K-Fold CV** en el conjunto de entrenamiento
- Cada muestra se predijo cuando **NO** estaba en el conjunto de entrenamiento
- **No hay data leakage**: predicciones válidas y generalizables
""")

# Mostrar métricas de Test Set si están disponibles (SOLO para CV + train/test split)
if (
    evaluation_method == "cv"
    and st.session_state.get("used_train_test_split", False)
    and "test_metrics" in st.session_state
):
    st.markdown("---")
    st.subheader("🎯 Evaluación en Test Set (Holdout Final)")
    st.caption("Evaluación en conjunto de test reservado después de hacer CV en el conjunto de train")

    test_metrics = st.session_state["test_metrics"]

    # Comparación Train (CV) vs Test
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔄 Train (CV)")
        st.metric(
            "Balanced Accuracy",
            f"{cv_results['test_balanced_accuracy'].mean():.3f} ± {cv_results['test_balanced_accuracy'].std():.3f}",
        )
        st.metric("Precision", f"{cv_results['test_precision'].mean():.3f} ± {cv_results['test_precision'].std():.3f}")
        st.metric("Recall", f"{cv_results['test_recall'].mean():.3f} ± {cv_results['test_recall'].std():.3f}")
        st.metric("F1-Score", f"{cv_results['test_f1'].mean():.3f} ± {cv_results['test_f1'].std():.3f}")

    with col2:
        st.markdown("### 🎯 Test (Holdout)")
        delta_ba = test_metrics["balanced_accuracy"] - cv_results["test_balanced_accuracy"].mean()
        delta_precision = test_metrics["precision"] - cv_results["test_precision"].mean()
        delta_recall = test_metrics["recall"] - cv_results["test_recall"].mean()
        delta_f1 = test_metrics["f1"] - cv_results["test_f1"].mean()

        st.metric("Balanced Accuracy", f"{test_metrics['balanced_accuracy']:.3f}", delta=f"{delta_ba:+.3f}")
        st.metric("Precision", f"{test_metrics['precision']:.3f}", delta=f"{delta_precision:+.3f}")
        st.metric("Recall", f"{test_metrics['recall']:.3f}", delta=f"{delta_recall:+.3f}")
        st.metric("F1-Score", f"{test_metrics['f1']:.3f}", delta=f"{delta_f1:+.3f}")

    # Indicador de generalización
    if abs(delta_ba) > 0.10:
        if delta_ba < -0.10:
            st.warning(f"⚠️ Posible overfitting: Test BA es {abs(delta_ba):.3f} menor que Train CV")
        else:
            st.info(f"✓ Test BA es {delta_ba:+.3f} mayor que Train CV (excelente generalización)")
    else:
        st.success(f"✅ Buena generalización: |Δ BA| = {abs(delta_ba):.3f} < 0.10")

# Message sobre modelo final
st.info("""
🔮 **Modelo Final**
- El modelo guardado para predicciones futuras está entrenado con **todos los datos disponibles**
- Se usa **solo** para predecir nuevos pacientes (módulo de Predicción)
""")

# Descarga
csv = metrics_df.to_csv(index=False)
st.download_button(
    label="📥 Descargar métricas (CSV)",
    data=csv,
    file_name=f"metricas_{model_config['model_type'].replace(' ', '_')}.csv",
    mime="text/csv",
)

# ============================================================================
# Sección 3: Curvas ROC y Precision-Recall (SOLO para CV)
# ============================================================================

# Solo mostrar curvas si usamos CV (tienen fold_roc_curves disponibles)
if evaluation_method == "cv" and len(fold_roc_curves) > 0:
    st.header("3️⃣ Curvas de Evaluación")

    tab1, tab2 = st.tabs(["📊 ROC Curve", "📏 Precision-Recall Curve"])

    with tab1:
        st.markdown("### ROC Curve (Receiver Operating Characteristic)")

        # Crear figura
        fig = go.Figure()

        # Curvas individuales (transparentes)
        for fpr, tpr in fold_roc_curves:
            fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    line={"color": "lightblue", "width": 1},
                    opacity=0.2,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        # Curva promedio
        mean_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.mean([np.interp(mean_fpr, fpr, tpr) for fpr, tpr in fold_roc_curves], axis=0)
        mean_auc = cv_results["test_roc_auc"].mean()
        std_auc = cv_results["test_roc_auc"].std()

        fig.add_trace(
            go.Scatter(
                x=mean_fpr,
                y=mean_tpr,
                mode="lines",
                line={"color": "darkblue", "width": 3},
                name=f"Mean ROC (AUC={mean_auc:.3f}±{std_auc:.3f})",
            )
        )

        # Línea diagonal (random)
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line={"color": "gray", "width": 2, "dash": "dash"},
                name="Random (AUC=0.500)",
            )
        )

        fig.update_layout(
            title="ROC Curves - Repeated Cross-Validation",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate (Sensitivity)",
            width=700,
            height=600,
            legend={"x": 0.6, "y": 0.1},
        )

        st.plotly_chart(fig, width="stretch")

        st.info("""
        **Interpretación**:
        - AUC = 1.0: Clasificador perfecto
        - AUC = 0.5: Random (línea diagonal)
        - Mejor cuanto más cerca del ángulo superior izquierdo
        """)

    with tab2:
        st.markdown("### Precision-Recall Curve")

        # Crear figura
        fig = go.Figure()

        # Curvas individuales
        for precision, recall in fold_pr_curves:
            fig.add_trace(
                go.Scatter(
                    x=recall,
                    y=precision,
                    mode="lines",
                    line={"color": "lightcoral", "width": 1},
                    opacity=0.2,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        # Curva promedio
        mean_recall = np.linspace(0, 1, 100)
        mean_precision = np.mean(
            [np.interp(mean_recall[::-1], recall[::-1], precision[::-1]) for precision, recall in fold_pr_curves],
            axis=0,
        )
        mean_ap = cv_results["test_average_precision"].mean()
        std_ap = cv_results["test_average_precision"].std()

        fig.add_trace(
            go.Scatter(
                x=mean_recall,
                y=mean_precision,
                mode="lines",
                line={"color": "darkred", "width": 3},
                name=f"Mean PR (AP={mean_ap:.3f}±{std_ap:.3f})",
            )
        )

        # Baseline (prevalencia)
        prevalence = y.mean()
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[prevalence, prevalence],
                mode="lines",
                line={"color": "gray", "width": 2, "dash": "dash"},
                name=f"Baseline (prevalencia={prevalence:.3f})",
            )
        )

        fig.update_layout(
            title="Precision-Recall Curves - Repeated Cross-Validation",
            xaxis_title="Recall (Sensitivity)",
            yaxis_title="Precision",
            width=700,
            height=600,
            legend={"x": 0.6, "y": 0.9},
        )

        st.plotly_chart(fig, width="stretch")

        st.info("""
        **Interpretación**:
        - Útil para datasets desbalanceados
        - Average Precision resume el área bajo la curva
        - Mejor cuanto más alta y a la derecha esté la curva
        """)

else:
    # Cuando se usa holdout, no generamos curvas ROC/PR
    st.header("3️⃣ Curvas de Evaluación")
    st.info("""
    📊 **Curvas ROC/PR no disponibles con método de 50 Holdout Splits**

    Las curvas ROC y Precision-Recall se generan solo cuando usas Cross-Validation.

    Con el método de 50 Holdout Splits, obtienes métricas más robustas (Test BA, Train BA,
    Sensitivity, Specificity) promediadas sobre 50 repeticiones independientes, lo cual es
    más comparable con el pipeline original.
    """)

# ============================================================================
# Sección 4: Matriz de Confusión
# ============================================================================

st.header("4️⃣ Matriz de Confusión")

st.caption("📊 Basada en predicciones del modelo (sin data leakage)")

# Construir matriz de confusión agregada
y_pred_full = st.session_state["predictions"]
y_true_full = st.session_state.get("y_true", y)  # Usar y_true guardado o todo el dataset
cm = confusion_matrix(y_true_full, y_pred_full)

# Calcular porcentajes
cm_pct = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

# Plotly heatmap
fig = go.Figure(
    data=go.Heatmap(
        z=cm,
        x=["Predicted NC", "Predicted AD"],
        y=["Actual NC", "Actual AD"],
        text=[[f"{cm[i, j]}<br>({cm_pct[i, j]:.1f}%)" for j in range(2)] for i in range(2)],
        texttemplate="%{text}",
        textfont={"size": 16},
        colorscale="Blues",
        showscale=True,
        colorbar={"title": "Count"},
    )
)

fig.update_layout(
    title="Confusion Matrix (Full Dataset Prediction)",
    xaxis_title="Predicted Label",
    yaxis_title="True Label",
    width=600,
    height=500,
)

st.plotly_chart(fig, width="stretch")

# Métricas derivadas
col1, col2, col3, col4 = st.columns(4)

tn, fp, fn, tp = cm.ravel()

with col1:
    sensitivity = tp / (tp + fn)
    st.metric("Sensitivity", f"{sensitivity:.3f}")
    st.caption(f"TP={tp}, FN={fn}")

with col2:
    specificity = tn / (tn + fp)
    st.metric("Specificity", f"{specificity:.3f}")
    st.caption(f"TN={tn}, FP={fp}")

with col3:
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    st.metric("PPV (Precision)", f"{ppv:.3f}")

with col4:
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    st.metric("NPV", f"{npv:.3f}")

# ============================================================================
# Sección 5: Distribución de Probabilidades
# ============================================================================

st.header("5️⃣ Distribución de Probabilidades Predichas")

y_proba_full = st.session_state["probabilities"]
y_true_full = st.session_state.get("y_true", y)  # Usar y_true guardado
y_pred_full = st.session_state["predictions"]

# Crear DataFrame para plotting
prob_df = pd.DataFrame(
    {
        "Probabilidad": y_proba_full,
        "Clase Real": ["NC" if label == 0 else "AD" for label in y_true_full],
        "Predicción": ["NC" if pred == 0 else "AD" for pred in y_pred_full],
    }
)

# Histograma superpuesto
fig = go.Figure()

for class_label in ["NC", "AD"]:
    data_class = prob_df[prob_df["Clase Real"] == class_label]["Probabilidad"]

    fig.add_trace(go.Histogram(x=data_class, name=f"Real: {class_label}", opacity=0.6, nbinsx=30))

fig.add_vline(x=0.5, line_dash="dash", line_color="red", annotation_text="Umbral=0.5")

fig.update_layout(
    title="Distribución de Probabilidades P(AD)",
    xaxis_title="Probabilidad de AD",
    yaxis_title="Frecuencia",
    barmode="overlay",
    height=400,
)

st.plotly_chart(fig, width="stretch")

st.info("""
**Interpretación**:
- **Buena calibración**: Dos distribuciones bien separadas
- **Overlap**: Región de incertidumbre donde el modelo duda
- Muestras cerca de 0.5 son las más difíciles de clasificar
""")

# ============================================================================
# Sección 6: Feature Importances (si disponible)
# ============================================================================

if model_config["model_type"] in ["Logistic Regression", "Random Forest"]:
    st.header("6️⃣ Importancia de Variables")

    pipeline_final = st.session_state["trained_pipeline"]
    classifier = pipeline_final.named_steps["classifier"]

    if hasattr(classifier, "coef_"):  # Logistic Regression
        importances = np.abs(classifier.coef_[0])
        importance_label = "Coeficiente (absoluto)"
    elif hasattr(classifier, "feature_importances_"):  # Random Forest
        importances = classifier.feature_importances_
        importance_label = "Feature Importance"

    # Crear DataFrame
    importance_df = pd.DataFrame({"Variable": selected_features, importance_label: importances}).sort_values(
        importance_label, ascending=False
    )

    # Gráfico de barras
    fig = go.Figure(
        go.Bar(
            x=importance_df[importance_label],
            y=importance_df["Variable"],
            orientation="h",
            marker={"color": importance_df[importance_label], "colorscale": "Viridis", "showscale": True},
        )
    )

    fig.update_layout(
        title=f"Importancia de Variables ({model_config['model_type']})",
        xaxis_title=importance_label,
        yaxis_title="Variable",
        height=max(400, len(selected_features) * 30),
        yaxis={"categoryorder": "total ascending"},
    )

    st.plotly_chart(fig, width="stretch")

    # Tabla
    st.dataframe(importance_df, width="stretch")

# ============================================================================
# Footer
# ============================================================================

st.markdown("---")
st.caption("Módulo 4: Evaluación y Resultados | TFM Alzheimer Classification")
