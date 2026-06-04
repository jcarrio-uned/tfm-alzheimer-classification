# 🚀 Guía de Inicio Rápido

## Instalación en 3 Pasos

### 1. Clonar el Repositorio
```bash
git clone <URL_DEL_REPO>
cd alzheimer-classification-app
```

### 2. Instalar Dependencias
```bash
pip install -r requirements.txt
```

### 3. Ejecutar la Aplicación
```bash
streamlit run app.py
```

O simplemente ejecuta el script automático:
```bash
bash run_app.sh
```

## 🎯 Flujo de Trabajo Rápido

1. **Módulo 2** - Selecciona "Minimal-7" como conjunto de features
2. **Módulo 3** - Mantén configuración por defecto (Logistic Regression)
3. **Módulo 4** - Presiona "🚀 Entrenar Modelo" con RepeatedStratifiedKFold
4. **Módulo 5** - Revisa errores y alertas metabólicas
5. **Módulo 6** - Prueba predicciones con datos del dataset

## ⚡ Configuración Recomendada

**Para mejores resultados**:
- Features: Best Forward (9 features)
- Modelo: Logistic Regression
- Regularización: L2
- CV: RepeatedStratifiedKFold (5 folds × 10 repeticiones)
- Grid Search: Activado con C = [0.01, 0.1, 1.0, 10.0, 100.0]

**Resultado esperado**: Balanced Accuracy ~0.87

## 🐛 Solución de Problemas Rápida

**ImportError**
```bash
pip install --upgrade -r requirements.txt
```

**Puerto ocupado**
```bash
streamlit run app.py --server.port 8502
```

**Error en datos**
Verifica que `data/residuales_grupos3.xlsx` existe

## 📖 Documentación Completa

Ver [README.md](README.md) para información detallada.
