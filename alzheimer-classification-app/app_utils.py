"""
Utilidades para la aplicación Streamlit
========================================

Funciona compartidas entre módulos y gestión de estado de sesión.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.stats.mstats import winsorize
from sklearn.base import BaseEstimator, TransformerMixin


def load_data():
    """Carga el dataset principal."""
    if "data" not in st.session_state:
        data_path = Path(__file__).parent / "data" / "residuales_grupos3.xlsx"
        df = pd.read_excel(data_path)
        # Crear features engineered automáticamente
        df = create_engineered_features(df)
        st.session_state.data = df
    return st.session_state.data


def create_engineered_features(df):
    """
    Crea features engineered (ratios, sumas, interacciones).
    Incluye las mejores features encontradas en experimentos de feature engineering.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame original

    Returns
    -------
    pd.DataFrame
        DataFrame con features engineered añadidas
    """
    df = df.copy()

    # === ORIGINAL BIO-E FEATURES (V2) ===

    # CerSum: Suma de ceramidas (neuroinflamación)
    # CORRECTO: Cer(d18:1/20:0) + Cer(d18:0/24:1) según pipeline original
    if "Cer(d18:1/20:0)" in df.columns and "Cer(d18:0/24:1)" in df.columns:
        df["CerSum"] = df["Cer(d18:1/20:0)"] + df["Cer(d18:0/24:1)"]

    # DHEAS/lysoPC: Ratio esteroide/fosfolípido
    if "DHEAS" in df.columns and "lysoPC.a.C18:2" in df.columns:
        df["DHEAS/lysoPC"] = df["DHEAS"] / (df["lysoPC.a.C18:2"] + 1e-10)

    # DOPA*lysoPC: Interacción catecolamina×lípido
    if "DOPA" in df.columns and "lysoPC.a.C18:2" in df.columns:
        df["DOPA*lysoPC"] = df["DOPA"] * df["lysoPC.a.C18:2"]

    # === TOP INTERACCIONES (Forward Selection - BA +2.8%) ===

    # lysoPC*DOPA: Mejor interacción encontrada (BA 0.862, +2.0%)
    if "lysoPC.a.C18:2" in df.columns and "DOPA" in df.columns:
        df["lysoPC*DOPA"] = df["lysoPC.a.C18:2"] * df["DOPA"]

    # DOPA*DHEAS: Segunda mejor interacción (BA 0.855, +1.4%)
    if "DOPA" in df.columns and "DHEAS" in df.columns:
        df["DOPA*DHEAS"] = df["DOPA"] * df["DHEAS"]

    # PC*DOPA: Tercera mejor interacción (BA 0.849, +0.8%)
    if "PC.aa.C40:4" in df.columns and "DOPA" in df.columns:
        df["PC*DOPA"] = df["PC.aa.C40:4"] * df["DOPA"]

    # lysoPC*PC: Cuarta mejor (BA 0.846, +0.5%)
    if "lysoPC.a.C18:2" in df.columns and "PC.aa.C40:4" in df.columns:
        df["lysoPC*PC"] = df["lysoPC.a.C18:2"] * df["PC.aa.C40:4"]

    # DHEAS/Cer20: Ratio supervivencia/apoptosis neuronal
    if "DHEAS" in df.columns and "Cer(d18:1/20:0)" in df.columns:
        df["DHEAS/Cer20"] = df["DHEAS"] / (df["Cer(d18:1/20:0)"] + 1e-10)

    return df


def apply_winsorization(X, percentile=0.05):
    """
    Aplica winsorización columna por columna al percentil especificado.

    La winsorización trunca valores extremos empujándolos hacia los percentiles
    especificados. Reduce el impacto de outliers sin eliminar muestras.

    Parameters
    ----------
    X : np.ndarray
        Matriz de features (n_samples × n_features)
    percentile : float, default=0.05
        Percentil de truncamiento en cada extremo (0.05 = truncar al 5%)

    Returns
    -------
    X_winsor : np.ndarray
        Matriz winsorizada con la misma shape

    Notes
    -----
    Ejemplo con percentile=0.05:
    - Valores < percentil 5 → reemplazados por valor del percentil 5
    - Valores > percentil 95 → reemplazados por valor del percentil 95
    - Valores intermedios → sin cambios

    Orden crítico de transformaciones (como pipeline original):
    1. WINSORIZATION (5%) ← aquí
    2. STANDARDSCALER ← en pipeline
    3. LOGISTICREGRESSION ← en pipeline
    """
    X_winsor = np.empty_like(X)
    n_modified_total = 0

    for j in range(X.shape[1]):
        X_winsor[:, j] = winsorize(X[:, j], limits=[percentile, percentile])
        n_modified = np.sum(X[:, j] != X_winsor[:, j])
        n_modified_total += n_modified

    return X_winsor


class WinsorizerTransformer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible winsorizer that learns percentile bounds from training data.

    Parameters
    ----------
    percentile : float, default=0.05
        Percentile to clip at each tail (0.05 = 5th and 95th percentiles).
    clip_test : bool, default=False
        If True, clips test data using train-derived bounds.
        If False, transform() is a pass-through (only fit_transform clips).
        Default=False matches the TFM pipeline methodology where only
        training data is winsorized and test data stays raw.
    """

    def __init__(self, percentile=0.05, clip_test=False):
        self.percentile = percentile
        self.clip_test = clip_test

    def fit(self, X, y=None):
        X = np.asarray(X)
        self.lower_ = np.percentile(X, self.percentile * 100, axis=0)
        self.upper_ = np.percentile(X, (1 - self.percentile) * 100, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        if self.clip_test:
            for j in range(X.shape[1]):
                X[:, j] = np.clip(X[:, j], self.lower_[j], self.upper_[j])
        # If clip_test=False, return data unchanged (test stays raw)
        return X

    def fit_transform(self, X, y=None):
        """Fit and transform — always clips (this is training data)."""
        self.fit(X, y)
        X = np.asarray(X, dtype=float).copy()
        for j in range(X.shape[1]):
            X[:, j] = np.clip(X[:, j], self.lower_[j], self.upper_[j])
        return X


def get_metabolite_columns(df):
    """Obtiene las columnas de metabolitos (excluye demográficas, incluye engineered)."""
    exclude = ["ID", "Group", "Age [y]", "Sex", "Fasting time [h]"]
    return [col for col in df.columns if col not in exclude and pd.api.types.is_numeric_dtype(df[col])]


def calculate_effect_size(group1, group2):
    """
    Calcula Cohen's d (effect size) entre dos grupos.

    Parameters
    ----------
    group1, group2 : array-like
        Valores de los dos grupos a comparar

    Returns
    -------
    float
        Cohen's d
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0

    return (np.mean(group1) - np.mean(group2)) / pooled_std


def get_feature_stats(df, metabolites):
    """
    Calcula estadísticas de cada feature por grupo.

    Returns
    -------
    pd.DataFrame
        DataFrame con estadísticas por feature
    """
    ad_data = df[df["Group"] == "AD"]
    nc_data = df[df["Group"] == "NC"]

    stats_list = []

    for met in metabolites:
        ad_vals = ad_data[met].dropna()
        nc_vals = nc_data[met].dropna()

        if len(ad_vals) > 0 and len(nc_vals) > 0:
            # Test t
            t_stat, p_val = stats.ttest_ind(ad_vals, nc_vals)

            # Effect size
            cohens_d = calculate_effect_size(ad_vals, nc_vals)

            stats_list.append(
                {
                    "Metabolito": met,
                    "AD_mean": ad_vals.mean(),
                    "AD_std": ad_vals.std(),
                    "NC_mean": nc_vals.mean(),
                    "NC_std": nc_vals.std(),
                    "Cohen_d": cohens_d,
                    "p_value": p_val,
                    "significativo": "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "",
                }
            )

    return pd.DataFrame(stats_list).sort_values("Cohen_d", key=abs, ascending=False)


def get_bio_e_features():
    """Retorna las 9 features de Bio-E (6 core + 3 engineered).

    BA reportado en experimentos: 0.857 ± 0.064 (LogReg)
    """
    return [
        # 6 core metabolites
        "lysoPC.a.C18:2",
        "Cer(d18:1/20:0)",
        "DHEAS",
        "DOPA",
        "PC.aa.C40:4",
        "GCDCA",
        # 3 engineered features
        "DHEAS/lysoPC",  # Ratio esteroide/fosfolípido
        "DOPA*lysoPC",  # Interacción catecolamina×lípido
        "CerSum",  # Suma de ceramidas
    ]


def get_minimal7_features():
    """Retorna las 7 features de Minimal-7.

    BA reportado en experimentos: 0.860 ± 0.058 (LogReg)
    """
    return [
        "DOPA",
        "CerSum",
        "PC.aa.C40:4",
        "DHEAS/lysoPC",
        "lysoPC.a.C18:2",
        "DHEAS",
        "lysoPC*DOPA",  # Mejor interacción (era DOPA*lysoPC)
    ]


def get_best_forward_features():
    """Retorna el conjunto óptimo de Forward Selection (9 features).

    BA reportado en experimentos: 0.870 ± 0.061 (LogReg), 0.861 ± 0.058 (SVM)
    Mejor configuración encontrada en experimentos.
    """
    return [
        # 6 core
        "lysoPC.a.C18:2",
        "PC.aa.C40:4",
        "DHEAS",
        "DOPA",
        "Cer(d18:1/20:0)",
        "GCDCA",
        # 3 mejores interacciones (Forward Selection)
        "lysoPC*DOPA",  # BA +2.0%
        "DOPA*DHEAS",  # BA +1.4%
        "lysoPC*PC",  # BA +0.5%
    ]


def get_stable_features():
    """Retorna las 9 features del conjunto Stable.

    Conjunto de marcadores metabólicos estables seleccionados.
    """
    return [
        "lysoPC.a.C18:2",
        "DOPA",
        "Cer(d18:1/20:0)",
        "DHEAS/lysoPC",  # Engineered feature
        "DHEAS",
        "PC.aa.C40:4",
        "Arg",
        "HexCer(d18:1/26:1)",
        "Ind-SO4",
    ]


def get_golden_features():
    """Retorna las 10 features del conjunto Golden (Stable-Opt).

    Optimización del conjunto Stable: se elimina Ind-SO4 y se añaden
    dos interacciones DOPA-centradas. Validado con paired t-test
    (p=0.019 vs Bio-E, p=0.002 vs Stable).

    BA reportado: 0.863 ± 0.056 (LogReg C=0.18, balanced)
    Sens=85.4%, Spec=87.2%
    Configuración recomendada: LogReg C=0.18, class_weight='balanced'
    """
    return [
        # 8 core metabolites
        "lysoPC.a.C18:2",  # Fosfolípido de membrana
        "DOPA",  # Catecolamina (dopamina)
        "Cer(d18:1/20:0)",  # Ceramida (neuroinflamación)
        "DHEAS/lysoPC",  # Ratio esteroide/fosfolípido (engineered)
        "DHEAS",  # Esteroide neuroactivo
        "PC.aa.C40:4",  # Fosfatidilcolina
        "Arg",  # Aminoácido (vía NO)
        "HexCer(d18:1/26:1)",  # Hexosilceramida
        # 2 interacciones DOPA-centradas
        "DOPA*DHEAS",  # Interacción catecolamina×esteroide
        "PC*DOPA",  # Interacción fosfolípido×catecolamina
    ]


def get_engineered_features():
    """Retorna lista de TODAS las features engineered disponibles."""
    return [
        # Bio-E originales
        "CerSum",
        "DHEAS/lysoPC",
        "DOPA*lysoPC",
        # Top interacciones (experimentos previos)
        "lysoPC*DOPA",
        "DOPA*DHEAS",
        "PC*DOPA",
        "lysoPC*PC",
        # Ratios
        "DHEAS/Cer20",
        # Top interacciones (búsqueda exhaustiva 2026)
        "DOPA*PC.aa.C40:4",
        "PC.aa.C40:4/DHEAS",
        "lysoPC.a.C18:2*DOPA",
    ]


def get_base_features_for_engineered(engineered_feature):
    """
    Retorna las features base necesarias para calcular una feature engineered.

    Parameters
    ----------
    engineered_feature : str
        Nombre de la feature engineered

    Returns
    -------
    list
        Lista de features base necesarias
    """
    mapping = {
        "CerSum": ["Cer(d18:1/20:0)", "Cer(d18:0/24:1)"],
        "DHEAS/lysoPC": ["DHEAS", "lysoPC.a.C18:2"],
        "DOPA*lysoPC": ["DOPA", "lysoPC.a.C18:2"],
        "lysoPC*DOPA": ["lysoPC.a.C18:2", "DOPA"],
        "DOPA*DHEAS": ["DOPA", "DHEAS"],
        "PC*DOPA": ["PC.aa.C40:4", "DOPA"],
        "lysoPC*PC": ["lysoPC.a.C18:2", "PC.aa.C40:4"],
        "DHEAS/Cer20": ["DHEAS", "Cer(d18:1/20:0)"],
        "DOPA*PC.aa.C40:4": ["DOPA", "PC.aa.C40:4"],
        "PC.aa.C40:4/DHEAS": ["PC.aa.C40:4", "DHEAS"],
        "lysoPC.a.C18:2*DOPA": ["lysoPC.a.C18:2", "DOPA"],
    }
    return mapping.get(engineered_feature, [])


def compute_engineered_feature(feature_name, values_dict):
    """
    Calcula el valor de una feature engineered a partir de features base.

    Parameters
    ----------
    feature_name : str
        Nombre de la feature engineered
    values_dict : dict
        Diccionario con valores de features base

    Returns
    -------
    float
        Valor calculado de la feature engineered
    """
    if feature_name == "CerSum":
        return values_dict.get("Cer(d18:1/20:0)", 0) + values_dict.get("Cer(d18:0/24:1)", 0)

    elif feature_name == "DHEAS/Cer20":
        cer20 = values_dict.get("Cer(d18:1/20:0)", 0)
        return values_dict.get("DHEAS", 0) / (cer20 + 1e-10)

    elif feature_name == "DHEAS/lysoPC":
        lysopc = values_dict.get("lysoPC.a.C18:2", 0)
        return values_dict.get("DHEAS", 0) / (lysopc + 1e-10)  # Evitar división por cero

    elif feature_name == "DOPA*lysoPC":
        return values_dict.get("DOPA", 0) * values_dict.get("lysoPC.a.C18:2", 0)

    elif feature_name == "lysoPC*DOPA":
        return values_dict.get("lysoPC.a.C18:2", 0) * values_dict.get("DOPA", 0)

    elif feature_name == "DOPA*DHEAS":
        return values_dict.get("DOPA", 0) * values_dict.get("DHEAS", 0)

    elif feature_name == "PC*DOPA":
        return values_dict.get("PC.aa.C40:4", 0) * values_dict.get("DOPA", 0)

    elif feature_name == "lysoPC*PC":
        return values_dict.get("lysoPC.a.C18:2", 0) * values_dict.get("PC.aa.C40:4", 0)

    elif feature_name == "DOPA*PC.aa.C40:4":
        return values_dict.get("DOPA", 0) * values_dict.get("PC.aa.C40:4", 0)

    elif feature_name == "PC.aa.C40:4/DHEAS":
        dheas = values_dict.get("DHEAS", 0)
        return values_dict.get("PC.aa.C40:4", 0) / (dheas + 1e-10)

    elif feature_name == "lysoPC.a.C18:2*DOPA":
        return values_dict.get("lysoPC.a.C18:2", 0) * values_dict.get("DOPA", 0)

    else:
        return 0.0


def get_base_features_needed(selected_features):
    """
    Obtiene todas las features base necesarias (incluyendo las necesarias para engineered).

    Parameters
    ----------
    selected_features : list
        Lista de features seleccionadas (puede incluir engineered)

    Returns
    -------
    list
        Lista de features base que el usuario debe proporcionar
    """
    engineered = get_engineered_features()
    base_features = []

    for feat in selected_features:
        if feat in engineered:
            # Añadir las features base necesarias para calcular esta
            base_features.extend(get_base_features_for_engineered(feat))
        else:
            # Es una feature base, añadirla directamente
            base_features.append(feat)

    # Eliminar duplicados manteniendo orden
    return list(dict.fromkeys(base_features))


def check_atypical_markers(sample_values, predicted_class, reference_stats, threshold=0.5):
    """
    Identifica marcadores que van en contra de la distribución del grupo predicho.

    Parameters
    ----------
    sample_values : dict
        Valores de metabolitos de la muestra
    predicted_class : str
        Clase predicha ("AD" o "NC")
    reference_stats : pd.DataFrame
        Estadísticas de referencia por grupo
    threshold : float, optional
        Umbral mínimo de Cohen's d para considerar un metabolito (default: 0.5)

    Returns
    -------
    list
        Lista de alertas con marcadores atípicos
    """
    alerts = []

    for metabolite, value in sample_values.items():
        # Metabolites are in the index, not in a column
        if metabolite not in reference_stats.index:
            continue

        row = reference_stats.loc[metabolite]

        ad_mean = row["ad_mean"]
        nc_mean = row["nc_mean"]
        effect_size = row["effect_size"]

        # Solo considerar metabolitos con effect size significativo
        if abs(effect_size) < threshold:
            continue

        # Determinar qué lado es "AD" y qué lado es "NC"
        decision_boundary = (ad_mean + nc_mean) / 2

        if effect_size > 0:  # AD tiene valores mayores
            is_on_ad_side = value > decision_boundary
            expected_for_ad = "mayor"
            expected_for_nc = "menor"
        else:  # AD tiene valores menores
            is_on_ad_side = value < decision_boundary
            expected_for_ad = "menor"
            expected_for_nc = "mayor"

        # Calcular distancia a la media esperada (qué tan atípico es el valor)
        if predicted_class == "AD":
            distance_from_expected = abs(value - ad_mean) / (row.get("ad_std", 1.0) + 1e-10)
        else:
            distance_from_expected = abs(value - nc_mean) / (row.get("nc_std", 1.0) + 1e-10)

        # Verificar si hay discrepancia
        if predicted_class == "AD" and not is_on_ad_side:
            # Determinar severidad basada en effect size y distancia
            if abs(effect_size) > 1.2:
                severity = "HIGH"
            elif abs(effect_size) > 0.8:
                severity = "MEDIUM"
            else:
                severity = "LOW"

            alerts.append(
                {
                    "metabolito": metabolite,
                    "valor": value,
                    "ad_mean": ad_mean,
                    "nc_mean": nc_mean,
                    "effect_size": effect_size,
                    "distance": distance_from_expected,
                    "severity": severity,
                    "mensaje": f"🔍 Valor contradice predicción AD. Está en lado NC ({value:.3f} vs AD típico: {ad_mean:.3f}). "
                    f"Effect size: {abs(effect_size):.2f}",
                }
            )
        elif predicted_class == "NC" and is_on_ad_side:
            # Determinar severidad basada en effect size y distancia
            if abs(effect_size) > 1.2:
                severity = "HIGH"
            elif abs(effect_size) > 0.8:
                severity = "MEDIUM"
            else:
                severity = "LOW"

            alerts.append(
                {
                    "metabolito": metabolite,
                    "valor": value,
                    "ad_mean": ad_mean,
                    "nc_mean": nc_mean,
                    "effect_size": effect_size,
                    "distance": distance_from_expected,
                    "severity": severity,
                    "mensaje": f"🔍 Valor contradice predicción NC. Está en lado AD ({value:.3f} vs NC típico: {nc_mean:.3f}). "
                    f"Effect size: {abs(effect_size):.2f}",
                }
            )

    # Ordenar alertas por severidad (effect size descendente) antes de retornar
    alerts = sorted(alerts, key=lambda x: abs(x["effect_size"]), reverse=True)

    return alerts


def compute_confidence_assessment(proba, sample_values, reference_stats, threshold_margin=0.15):
    """
    Evalúa el nivel de confianza de una predicción combinando múltiples indicadores.

    Factores de incertidumbre analizados:
    1. Margen de probabilidad: distancia de P(AD) al umbral 0.5
    2. Contradicciones entre biomarcadores: nº de metabolitos clave en el lado opuesto
    3. Alertas metabólicas: marcadores atípicos respecto al grupo predicho

    Parameters
    ----------
    proba : float
        Probabilidad P(AD) del modelo
    sample_values : dict
        Valores de metabolitos de la muestra
    reference_stats : pd.DataFrame
        Estadísticas de referencia (ad_mean, nc_mean, effect_size por metabolito)
    threshold_margin : float
        Margen de probabilidad para zona de incertidumbre (default: 0.15)

    Returns
    -------
    dict con claves:
        - confidence_level: "alta", "moderada", "baja"
        - confidence_color: "green", "orange", "red"
        - margin: distancia al umbral 0.5
        - n_contradictions: nº de biomarcadores contradictorios
        - contradiction_details: lista de metabolitos contradictorios
        - recommendation: texto de recomendación clínica
        - needs_review: bool, si la muestra debería ser revisada
        - factors: lista de factores que contribuyen a la incertidumbre
    """
    margin = abs(proba - 0.5)
    predicted_class = "AD" if proba >= 0.5 else "NC"

    # --- Factor 1: Margen de probabilidad ---
    if margin >= 0.30:
        margin_level = "alta"
    elif margin >= 0.15:
        margin_level = "moderada"
    else:
        margin_level = "baja"

    # --- Factor 2: Contradicciones entre biomarcadores clave ---
    # Biomarcadores clave y su dirección esperada en AD
    key_markers = {
        "DOPA": "low_in_AD",
        "lysoPC.a.C18:2": "high_in_AD",
        "PC.aa.C40:4": "high_in_AD",
        "Cer(d18:1/20:0)": "high_in_AD",
        "DHEAS": "low_in_AD",
        "GCDCA": "low_in_AD",
    }

    contradictions = []
    for met, direction in key_markers.items():
        if met not in reference_stats.index or met not in sample_values:
            continue

        ad_mean = reference_stats.loc[met, "ad_mean"]
        nc_mean = reference_stats.loc[met, "nc_mean"]
        val = sample_values[met]
        midpoint = (ad_mean + nc_mean) / 2

        # Comprobar si el valor está en el lado incorrecto
        if predicted_class == "AD":
            if direction == "low_in_AD" and val > nc_mean:
                contradictions.append(met)
            elif direction == "high_in_AD" and val < nc_mean:
                contradictions.append(met)
        else:  # NC
            if direction == "low_in_AD" and val < ad_mean:
                contradictions.append(met)
            elif direction == "high_in_AD" and val > ad_mean:
                contradictions.append(met)

    n_contradictions = len(contradictions)

    # --- Determinar nivel global de confianza ---
    factors = []

    if margin < threshold_margin:
        factors.append(f"Probabilidad cercana al umbral (P(AD)={proba:.3f}, margen={margin:.3f})")

    if n_contradictions >= 3:
        factors.append(
            f"{n_contradictions} biomarcadores clave contradicen la predicción: " + ", ".join(contradictions)
        )
    elif n_contradictions >= 2:
        factors.append(f"{n_contradictions} biomarcadores contradictorios: " + ", ".join(contradictions))

    # Clasificación final
    needs_review = False
    if margin < threshold_margin or n_contradictions >= 4:
        confidence_level = "baja"
        confidence_color = "red"
        needs_review = True
    elif margin < 0.25 or n_contradictions >= 3:
        confidence_level = "moderada"
        confidence_color = "orange"
    else:
        confidence_level = "alta"
        confidence_color = "green"

    # Recomendación
    if needs_review:
        recommendation = (
            "⚠️ **Clasificación incierta**. La probabilidad está demasiado cerca del umbral de decisión"
            + (f" y {n_contradictions} biomarcadores contradicen el diagnóstico" if n_contradictions >= 2 else "")
            + ". Se recomienda **análisis complementario** (neuroimagen, LCR, evaluación cognitiva detallada) "
            "antes de emitir un diagnóstico."
        )
    elif confidence_level == "moderada":
        recommendation = (
            "🟡 **Confianza moderada**. La predicción es consistente pero con margen limitado. "
            "Se recomienda validar con pruebas complementarias."
        )
    else:
        recommendation = (
            "🟢 **Alta confianza**. Los biomarcadores son consistentes con la predicción "
            "y la probabilidad está alejada del umbral."
        )

    return {
        "confidence_level": confidence_level,
        "confidence_color": confidence_color,
        "margin": margin,
        "margin_level": margin_level,
        "n_contradictions": n_contradictions,
        "contradiction_details": contradictions,
        "recommendation": recommendation,
        "needs_review": needs_review,
        "factors": factors,
        "predicted_class": predicted_class,
    }


def format_metric_with_ci(mean, std, n_folds=50):
    """Formatea una métrica con intervalo de confianza."""
    ci = 1.96 * std / np.sqrt(n_folds)
    return f"{mean:.3f} ± {std:.3f} (95% CI: [{mean - ci:.3f}, {mean + ci:.3f}])"


def initialize_session_state():
    """Inicializa variables de estado de la sesión."""
    if "model_trained" not in st.session_state:
        st.session_state.model_trained = False

    if "current_results" not in st.session_state:
        st.session_state.current_results = None

    if "selected_features" not in st.session_state:
        st.session_state.selected_features = get_minimal7_features()

    if "model_config" not in st.session_state:
        st.session_state.model_config = {
            "model_type": "LogisticRegression",
            "C": 1.0,
            "penalty": "l2",
            "class_weight": "balanced",
            "cv_folds": 5,
            "cv_repeats": 10,
        }
