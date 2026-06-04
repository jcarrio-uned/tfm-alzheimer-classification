#!/bin/bash
# Script de inicio rápido para la aplicación web

echo "🧠 Alzheimer Classification - Web Application Setup"
echo "=================================================="
echo ""

# Verificar que estamos en el directorio correcto
if [ ! -f "app.py" ]; then
    echo "❌ Error: app.py no encontrado"
    echo "   Por favor ejecuta este script desde el directorio raíz del proyecto"
    exit 1
fi

# Detectar Python
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo "❌ Error: Python no encontrado"
    exit 1
fi

echo "🐍 Usando: $($PYTHON --version)"
echo ""

# Verificar pip
if ! $PYTHON -m pip --version &> /dev/null; then
    echo "❌ Error: pip no está disponible"
    echo ""
    echo "Instala pip o crea un entorno virtual:"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate  # Linux/Mac"
    echo "  venv\\Scripts\\activate     # Windows"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Verificar Streamlit
if ! $PYTHON -m streamlit version &> /dev/null; then
    echo "📦 Instalando dependencias..."
    $PYTHON -m pip install -q -r requirements.txt

    if [ $? -ne 0 ]; then
        echo "❌ Error instalando dependencias"
        exit 1
    fi
    echo "✅ Dependencias instaladas"
else
    echo "✅ Streamlit ya instalado"
fi

echo ""
echo "🚀 Iniciando aplicación Streamlit..."
echo ""
echo "La aplicación se abrirá en tu navegador en http://localhost:8501"
echo ""
echo "Para detener la aplicación, presiona Ctrl+C"
echo ""
echo "=================================================="
echo ""

# Ejecutar Streamlit
$PYTHON -m streamlit run app.py
