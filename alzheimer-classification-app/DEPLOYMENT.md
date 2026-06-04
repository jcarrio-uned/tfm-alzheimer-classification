# 🚀 Opciones de Despliegue

## Despliegues Rápidos (Gratis)

### 1. Streamlit Community Cloud (Recomendado) ⭐

**Ventajas**: Gratuito, fácil, integrado con GitHub
**Limitaciones**: 1GB de recursos

**Pasos**:
1. Sube tu código a GitHub (ver GIT_INSTRUCTIONS.md)
2. Ve a https://share.streamlit.io
3. Conecta tu cuenta de GitHub
4. Selecciona tu repositorio
5. Branch: `main`
6. Main file: `app.py`
7. Click en "Deploy"

**Configuración adicional** (si es necesario):
Crea `.streamlit/config.toml`:
```toml
[server]
maxUploadSize = 200

[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

**URL resultante**: `https://[app-name]-[random].streamlit.app`

---

### 2. Hugging Face Spaces 🤗

**Ventajas**: Gratuito, soporte para múltiples frameworks
**Limitaciones**: 16GB de almacenamiento

**Pasos**:
1. Crea cuenta en https://huggingface.co
2. Ve a Spaces: https://huggingface.co/spaces
3. Click en "Create new Space"
4. Nombre: `alzheimer-classification`
5. SDK: Streamlit
6. Visibilidad: Public/Private
7. Sube tu código o conecta con GitHub

Crea `app.py` en la raíz (ya lo tienes) y `requirements.txt` (ya lo tienes).

**URL resultante**: `https://huggingface.co/spaces/[username]/alzheimer-classification`

---

### 3. Render.com

**Ventajas**: Gratuito, buen soporte, fácil configuración
**Limitaciones**: Instancia se duerme después de 15 min de inactividad

**Pasos**:
1. Sube código a GitHub
2. Ve a https://render.com
3. "New" → "Web Service"
4. Conecta GitHub y selecciona tu repo
5. Configuración:
   - Name: `alzheimer-classification`
   - Environment: `Python 3`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
6. Plan: Free
7. Click en "Create Web Service"

**URL resultante**: `https://alzheimer-classification.onrender.com`

---

### 4. Railway.app 🚂

**Ventajas**: $5 gratis al mes, despliegue automático desde Git
**Limitaciones**: Después del crédito gratis es de pago

**Pasos**:
1. Ve a https://railway.app
2. Conecta GitHub
3. "New Project" → "Deploy from GitHub repo"
4. Selecciona tu repositorio
5. Railway detecta automáticamente Streamlit
6. Configuración automática

**URL resultante**: `https://[project-name].up.railway.app`

---

## Despliegues Profesionales (Pago)

### 5. AWS EC2 + Nginx

**Costo**: ~$5-10/mes (t2.micro)

**Ventajas**: Control completo, escalable
**Requisitos**: Conocimientos de Linux/DevOps

**Pasos básicos**:
```bash
# En EC2 instance
sudo apt update
sudo apt install python3-pip nginx
git clone <tu-repo>
cd alzheimer-classification-app
pip3 install -r requirements.txt

# Crear servicio systemd
sudo nano /etc/systemd/system/streamlit.service
```

```ini
[Unit]
Description=Streamlit Alzheimer App
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/alzheimer-classification-app
ExecStart=/usr/local/bin/streamlit run app.py --server.port 8501
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable streamlit
sudo systemctl start streamlit

# Configurar Nginx como reverse proxy
sudo nano /etc/nginx/sites-available/streamlit
```

---

### 6. Docker Container 🐳

**Ventajas**: Portabilidad, fácil de escalar

Crea `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Crea `.dockerignore`:
```
__pycache__
*.pyc
.git
.gitignore
venv
.venv
*.md
!README.md
```

**Build y Run**:
```bash
docker build -t alzheimer-app .
docker run -p 8501:8501 alzheimer-app
```

**Deploy en Docker Hub**:
```bash
docker tag alzheimer-app username/alzheimer-app:v1.0.0
docker push username/alzheimer-app:v1.0.0
```

---

### 7. Google Cloud Run

**Costo**: Pay-as-you-go (gratis hasta 2M requests/mes)

```bash
# Instalar gcloud CLI
gcloud auth login
gcloud config set project [PROJECT-ID]

# Deploy desde Dockerfile
gcloud run deploy alzheimer-app \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

---

### 8. Azure Web Apps

**Costo**: ~$13/mes (Basic tier)

```bash
# Instalar Azure CLI
az login
az webapp up --name alzheimer-classification --runtime PYTHON:3.11
```

---

## Comparación Rápida

| Plataforma | Costo | Dificultad | Uptime | Recursos |
|------------|-------|------------|--------|----------|
| Streamlit Cloud | Gratis | ⭐ Fácil | 99% | 1GB RAM |
| Hugging Face | Gratis | ⭐ Fácil | 99% | 16GB storage |
| Render | Gratis* | ⭐⭐ Media | 95%** | 512MB RAM |
| Railway | $5/mes | ⭐⭐ Media | 99% | 512MB RAM |
| AWS EC2 | $5-10/mes | ⭐⭐⭐ Difícil | 99.9% | Variable |
| Docker | Variable | ⭐⭐⭐ Difícil | Variable | Variable |
| GCP Cloud Run | Pay-per-use | ⭐⭐ Media | 99.9% | 2GB RAM |

*Se duerme después de inactividad
**Con cold starts

---

## Recomendación por Caso de Uso

### 🎓 Proyecto Académico / TFM
→ **Streamlit Community Cloud** o **Hugging Face**
Razón: Gratis, fácil, suficiente para demos y presentaciones

### 👨‍💻 Proyecto Personal
→ **Render** o **Railway**
Razón: Balance entre facilidad y control

### 🏢 Producción / Uso Clínico
→ **AWS/Azure/GCP** con Docker
Razón: Escalabilidad, seguridad, cumplimiento normativo

### 🔬 Investigación Colaborativa
→ **Hugging Face Spaces**
Razón: Comunidad científica, fácil compartir

---

## Configuración de Dominio Personalizado

Si quieres usar tu propio dominio (ej: `alzheimer-app.tudominio.com`):

1. **Compra un dominio** (Namecheap, GoDaddy, ~$10/año)
2. **Configura DNS**:
   - En tu proveedor de dominio
   - Añade registro CNAME apuntando a tu app
3. **SSL/HTTPS**:
   - La mayoría de plataformas lo incluyen gratis (Let's Encrypt)

---

## Monitoreo y Logs

### Streamlit Cloud
- Panel de control en share.streamlit.io
- Logs en tiempo real

### Render/Railway
- Dashboard con métricas
- Logs integrados

### AWS/GCP/Azure
- CloudWatch, Stackdriver, Application Insights
- Configuración personalizada

---

## Backup y Actualizaciones

### Actualizar la app:
```bash
# Hacer cambios en local
git add -A
git commit -m "Update: descripción"
git push

# En la mayoría de plataformas, se despliega automáticamente
```

### Rollback:
```bash
git revert HEAD
git push
```

---

## Seguridad

### Variables de Entorno
Para información sensible, usa variables de entorno:

```python
import os
API_KEY = os.getenv('API_KEY')
```

En Streamlit Cloud:
1. Settings → Secrets
2. Añade en formato TOML:
```toml
API_KEY = "tu-clave-secreta"
```

### Autenticación
Para añadir login, considera:
- `streamlit-authenticator`
- OAuth con Google/GitHub
- Basic Auth con Nginx

---

## 📊 Analítica

Para trackear uso:
- Google Analytics
- Plausible (privacy-friendly)
- Streamlit Cloud Analytics (built-in)

---

## ✅ Checklist Pre-Despliegue

- [ ] Código probado localmente
- [ ] requirements.txt actualizado
- [ ] README con instrucciones claras
- [ ] .gitignore configurado
- [ ] Sin credenciales en el código
- [ ] Datos anonimizados
- [ ] Licencia incluida
- [ ] URLs actualizadas en documentación
- [ ] Tests básicos pasados
- [ ] Performance optimizada

---

**¿Dudas?** Consulta la documentación específica de cada plataforma.
