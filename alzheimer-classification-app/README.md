# 🧠 Alzheimer Classification - Web Application

Aplicación web interactiva basada en Streamlit para clasificación de Alzheimer mediante biomarcadores metabolómicos.

## 📋 Características

### 6 Módulos Interactivos

1. **📊 Exploración de Datos**
   - Análisis exploratorio del dataset
   - Visualización de distribuciones por grupo (AD vs NC)
   - Estadísticas descriptivas y correlaciones

2. **🔍 Selección de Variables**
   - Conjuntos predefinidos: Bio-E, Minimal-7, Best Forward, Stable
   - Ranking por importancia (Random Forest)
   - Evaluación de performance por conjunto
   - Selección personalizada de features

3. **🤖 Configuración de Modelos**
   - Regresión Logística
   - Support Vector Machine (SVM)
   - Configuración de hiperparámetros
   - Grid Search para optimización automática

4. **📈 Evaluación y Resultados**
   - Cross-Validation (RepeatedStratifiedKFold, StratifiedKFold)
   - Holdout (StratifiedShuffleSplit)
   - Train/Test Split opcional
   - Métricas: Balanced Accuracy, Sensibilidad, Especificidad, F1-Score, ROC-AUC
   - Curvas ROC y Precision-Recall
   - Matriz de confusión
   - Visualización de Grid Search
   - Coeficientes del modelo

5. **⚠️ Análisis de Errores**
   - Detección de Falsos Negativos (FN) y Falsos Positivos (FP)
   - Sistema de alertas metabólicas (HIGH/MEDIUM/LOW)
   - Comparación FN vs TP y FP vs TN
   - Interpretación automática de patrones
   - Recomendaciones clínicas
   - Exportación de reportes

6. **🔮 Predicción Individual**
   - Entrada manual de valores
   - Carga desde CSV
   - Uso de muestras del dataset
   - Predicción con alertas metabólicas
   - Visualización de perfiles

## 🚀 Instalación

### Requisitos Previos
- Python 3.9 o superior
- pip (gestor de paquetes de Python)

### Opción 1: Instalación Rápida (Recomendada)

```bash
# Clonar el repositorio
git clone <URL_DEL_REPO>
cd alzheimer-classification-app

# Ejecutar script de instalación y arranque
bash run_app.sh
```

El script `run_app.sh` detecta automáticamente Python, instala dependencias si es necesario, y lanza la aplicación.

### Opción 2: Instalación Manual

```bash
# Crear entorno virtual (recomendado)
python3 -m venv venv
source venv/bin/activate  # En Linux/Mac
# venv\Scripts\activate    # En Windows

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar aplicación
streamlit run app.py
```

## 📁 Estructura del Proyecto

```
alzheimer-classification-app/
├── app.py                        # Aplicación principal
├── app_utils.py                  # Utilidades y funciones compartidas
├── requirements-app.txt          # Dependencias de Python
├── run_app.sh                    # Script de instalación y arranque
├── README.md                     # Este archivo
├── data/                         # Datasets
│   ├── residuales_grupos3.xlsx   # Dataset principal
│   └── variables.csv             # Lista de variables
└── pages/                        # Módulos de la aplicación
    ├── 1_📊_Exploración.py
    ├── 2_🔍_Selección_Variables.py
    ├── 3_🤖_Configuración_Modelos.py
    ├── 4_📈_Evaluación.py
    ├── 5_⚠️_Análisis_Errores.py
    └── 6_🔮_Predicción.py
```

## 🎯 Uso

### Flujo de Trabajo Recomendado

1. **Exploración** → Familiarízate con los datos
2. **Selección de Variables** → Elige o evalúa conjuntos de features
3. **Configuración** → Selecciona algoritmo y parámetros
4. **Evaluación** → Entrena y evalúa el modelo
5. **Análisis de Errores** → Revisa FP/FN y alertas metabólicas
6. **Predicción** → Usa el modelo para nuevos pacientes

### Conjuntos Predefinidos de Features

- **Bio-E (9 features)**: 6 metabolitos core + 3 engineered
  - BA: 0.857 ± 0.064

- **Minimal-7 (7 features)**: Conjunto optimizado
  - BA: 0.860 ± 0.058

- **Best Forward (9 features)**: Forward Selection óptimo
  - BA: 0.870 ± 0.061

- **Stable (9 features)**: Marcadores metabólicos estables
  - Incluye: lysoPC.a.C18:2, DOPA, Cer(d18:1/20:0), DHEAS/lysoPC, DHEAS, PC.aa.C40:4, Arg, HexCer(d18:1/26:1), Ind-SO4

## 🔬 Metodología

### Datos
- **148 muestras**: 74 AD (Alzheimer) + 74 NC (Controles Normales)
- **Residuales metabólicos**: Valores ajustados por edad, sexo, IMC, enfermedades
- **Features engineered**: Ratios, sumas e interacciones metabólicas

### Modelos Disponibles
1. **Logistic Regression**
   - Regularización L1/L2
   - Optimización: liblinear, lbfgs, saga
   - Grid Search para C

2. **Support Vector Machine (SVM)**
   - Kernel: linear, rbf, poly
   - Grid Search para C y gamma

### Evaluación
- **RepeatedStratifiedKFold**: 5 folds × 10 repeticiones
- **StratifiedKFold**: 5-10 folds
- **StratifiedShuffleSplit**: 50 splits (80/20)
- **Métricas balanceadas**: Especial énfasis en Balanced Accuracy
- **Sin data leakage**: Predicciones solo en conjunto de validación

### Sistema de Alertas Metabólicas
- **HIGH (🔴)**: |Cohen's d| > 1.2 - Contradicción muy fuerte
- **MEDIUM (🟡)**: |Cohen's d| > 0.8 - Contradicción moderada
- **LOW (🔵)**: |Cohen's d| > 0.5 - Contradicción leve

Identifica metabolitos que contradicen la predicción del modelo, útil para interpretación clínica.

## 📊 Resultados Esperados

Con configuración óptima (Best Forward + GridSearch):
- **Balanced Accuracy**: ~0.87
- **Sensibilidad**: ~0.85-0.90 (detección de AD)
- **Especificidad**: ~0.85-0.90 (identificación de NC)
- **AUC-ROC**: ~0.92

## 🛠️ Tecnologías

- **Frontend**: Streamlit 1.30+
- **Visualización**: Plotly, Matplotlib, Seaborn
- **Machine Learning**: scikit-learn 1.4+
- **Data**: Pandas, NumPy
- **Estadística**: SciPy

## 📝 Notas Importantes

### Interpretación de Resultados
- Los modelos están entrenados en datos residuales (ajustados por confounders)
- La app es educativa/investigación, NO para uso clínico sin validación
- Las alertas metabólicas son indicativas, no diagnósticas

### Limitaciones
- Dataset de 148 muestras (validación externa recomendada)
- Modelos lineales (pueden no capturar relaciones complejas)
- Features engineered requieren metabolitos base

### Privacidad
- Los datos están anonimizados (IDs: PX001, PX002, ...)
- No contiene información personal identificable

## 🐛 Troubleshooting

**Error: "ModuleNotFoundError"**
```bash
pip install -r requirements-app.txt
```

**Error: "No module named streamlit"**
```bash
pip install streamlit>=1.30.0
```

**Puerto ocupado**
```bash
streamlit run app.py --server.port 8502
```

**Datos no encontrados**
Verifica que la carpeta `data/` contenga `residuales_grupos3.xlsx`

## 📧 Contacto y Contribuciones

Para reportar bugs o sugerir mejoras, por favor abre un issue en el repositorio.

## 📄 Licencia

Este proyecto es parte de un Trabajo de Fin de Máster (TFM) en Bioinformática.

## 🙏 Agradecimientos

Desarrollado como parte del TFM sobre clasificación de Alzheimer mediante metabolómica.

---

**Version**: 1.0.0
**Última actualización**: Marzo 2026
