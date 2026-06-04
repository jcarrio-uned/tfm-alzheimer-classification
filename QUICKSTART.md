# 🚀 Guía de Inicio Rápido - Aplicación Web

## ⚡ Inicio Rápido (3 pasos)

### Opción 1: Script Automático (Recomendado)

```bash
# 1. Ir al directorio del proyecto
cd /workspace/tfm-alzheimer-classification

# 2. Ejecutar script de inicio
bash run_app.sh
```

### Opción 2: Manual con Entorno Virtual

```bash
# 1. Crear entorno virtual (solo primera vez)
python3 -m venv venv

# 2. Activar entorno
source venv/bin/activate      # Linux/Mac
# O en Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements-app.txt

# 4. Ejecutar aplicación
streamlit run app.py
```

### Opción 3: Instalación Global (no recomendado)

```bash
# Instalar dependencias globalmente
pip install streamlit plotly scikit-learn pandas numpy seaborn matplotlib openpyxl scipy

# Ejecutar
streamlit run app.py
```

---

## 🌐 Acceso a la Aplicación

Una vez iniciada, la aplicación estará disponible en:

**http://localhost:8501**

El navegador debería abrirse automáticamente. Si no:
1. Abre tu navegador web
2. Navega a `http://localhost:8501`

---

## 📋 Requisitos del Sistema

- **Python**: 3.10, 3.11 o 3.12
- **RAM**: Mínimo 4 GB (recomendado 8 GB)
- **Espacio**: ~500 MB para dependencias

### Verificar Python

```bash
python3 --version  # Debe mostrar 3.10+
```

### Instalar Python (si no está instalado)

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

**macOS:**
```bash
brew install python@3.12
```

**Windows:**
Descargar desde [python.org](https://www.python.org/downloads/)

---

## 🐳 Opción Docker (Alternativa)

Si prefieres usar Docker:

### 1. Crear Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Copiar archivos
COPY requirements-app.txt .
COPY app.py .
COPY app_utils.py .
COPY pages/ pages/
COPY data/ data/
COPY tfm_alzheimer_classification/ tfm_alzheimer_classification/

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements-app.txt

# Exponer puerto
EXPOSE 8501

# Comando de inicio
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### 2. Construir imagen

```bash
docker build -t ad-classifier-app .
```

### 3. Ejecutar contenedor

```bash
docker run -p 8501:8501 ad-classifier-app
```

Luego abrir: **http://localhost:8501**

---

## 🔧 Solución de Problemas

### Error: "streamlit: command not found"

**Causa**: Streamlit no está instalado o no está en PATH

**Solución**:
```bash
pip install streamlit
# O usar: python3 -m streamlit run app.py
```

### Error: "No module named 'plotly'" (u otro módulo)

**Causa**: Dependencias no instaladas

**Solución**:
```bash
pip install -r requirements-app.txt
```

### Error: "FileNotFoundError: residuales_grupos3.xlsx"

**Causa**: Dataset no encontrado

**Solución**: Asegúrate de que el archivo esté en `data/residuales_grupos3.xlsx`

### Error: "Address already in use"

**Causa**: Puerto 8501 ocupado

**Solución**:
```bash
# Usar puerto diferente
streamlit run app.py --server.port 8502
```

### La aplicación se ve mal / sin estilos

**Causa**: Navegador antiguo o cache

**Solución**:
1. Refrescar con Ctrl+Shift+R (fuerza recarga)
2. Limpiar cache del navegador
3. Probar en navegador diferente (Chrome/Firefox)

### Aplicación muy lenta

**Causas y soluciones**:
- **Muchos datos**: Normal en primera carga (cache ayuda después)
- **Poca RAM**: Cerrar otros programas
- **Muchos repeticiones CV**: Reducir en Configuración de Modelos

---

## 📁 Estructura de Archivos Necesarios

```
tfm-alzheimer-classification/
├── app.py                    # ← Archivo principal
├── app_utils.py              # ← Utilidades
├── requirements-app.txt      # ← Dependencias
├── run_app.sh                # ← Script de inicio
├── data/
│   └── residuales_grupos3.xlsx  # ← DATASET REQUERIDO
├── pages/                    # ← Módulos Streamlit
│   ├── 1_📊_Exploración.py
│   ├── 2_🔍_Selección_Variables.py
│   ├── 3_🤖_Configuración_Modelos.py
│   ├── 4_📈_Evaluación.py
│   ├── 5_⚠️_Análisis_Errores.py
│   └── 6_🔮_Predicción.py
└── tfm_alzheimer_classification/
    └── final_pipeline/       # ← Código auxiliar
```

---

## 🎯 Primeros Pasos en la Aplicación

Una vez que la aplicación esté corriendo:

### 1. **Página de Inicio** 🏠
- Revisa el resumen del dataset
- Familiarízate con los módulos disponibles

### 2. **Exploración de Datos** 📊
- Visualiza distribuciones de metabolitos
- Explora correlaciones
- Ejecuta PCA

### 3. **Selección de Variables** 🔍
- Carga **Bio-E** o **Minimal-7**
- O crea tu propio conjunto custom
- Evalúa performance con CV rápido

### 4. **Configuración de Modelos** 🤖
- Elige algoritmo (LogReg, SVM, RF)
- Ajusta hiperparámetros con sliders
- Configura cross-validation
- **Guardar configuración** (botón importante)

### 5. **Evaluación y Resultados** 📈
- Botón **"Entrenar y Evaluar"**
- Espera 10-60s (según modelo y CV)
- Visualiza métricas, ROC curves, confusion matrix

### 6. **Análisis de Errores** ⚠️
- Revisa FN y FP
- Analiza alertas metabólicas
- Identifica patrones

### 7. **Predicción Individual** 🔮
- Ingresa valores de nuevo paciente
- Obtén predicción + alertas
- Interpreta resultados clínicos

---

## 💡 Consejos de Uso

### Performance
- Primera carga: ~5-10s (carga dataset y cache)
- Siguiente navegación: <1s (gracias al cache)
- Entrenar modelo LogReg: ~10-20s
- Entrenar SVM RBF: ~30-60s

### Best Practices
1. **Siempre guardar configuración** antes de entrenar
2. **Comenzar con CV pequeño** (3 folds, 3 repeats) para tests rápidos
3. **Usar LogReg primero** (más rápido), luego probar SVM/RF
4. **Revisar alertas** con umbral 0.5 primero, ajustar si necesario

### Shortcuts
- **Ctrl+C** en terminal: Detener aplicación
- **R** en navegador: Rerun app (refresca todo)
- **C** en navegador: Clear cache

---

## 📚 Documentación Adicional

- **APP_README.md** - Documentación completa de módulos (600+ líneas)
- **ARCHITECTURE.md** - Arquitectura técnica y componentes
- **DEPLOYMENT_STATUS.md** - Estado de implementación

---

## 🆘 Soporte

### Errores Comunes

| Error | Solución |
|-------|----------|
| `ModuleNotFoundError` | `pip install -r requirements-app.txt` |
| `Port already in use` | Cambiar puerto: `--server.port 8502` |
| `Dataset not found` | Verificar `data/residuales_grupos3.xlsx` |
| `pip: command not found` | Instalar pip: `apt install python3-pip` |
| `Streamlit won't start` | Probar: `python3 -m streamlit run app.py` |

### Logs y Debug

Para ver logs detallados:
```bash
streamlit run app.py --logger.level=debug
```

---

## ✅ Checklist Pre-Ejecución

- [ ] Python 3.10+ instalado (`python3 --version`)
- [ ] pip disponible (`pip --version`)
- [ ] Dataset en `data/residuales_grupos3.xlsx`
- [ ] Dependencias instaladas (`pip install -r requirements-app.txt`)
- [ ] Puerto 8501 libre (`lsof -i :8501`)
- [ ] Navegador web disponible

---

## 🎉 ¡Listo para Usar!

```bash
# Todo listo? Ejecuta:
bash run_app.sh

# O directamente:
streamlit run app.py
```

**La aplicación está completamente funcional y lista para producción** ✨

---

*Última actualización: 2026-02-23*
