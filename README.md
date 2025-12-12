# MHealth HAR — Proyecto Final

Guía rápida para preparar y ejecutar el backend (FastAPI), el frontend (Vite + React + Tailwind), y el notebook de entrenamiento de modelos.

---

## Requisitos
- Windows 10/11
- Python 3.10+ (recomendado 3.10/3.11)
- Node.js 18+ y npm
- Git (opcional)

Estructura relevante:
- `backend/` — API FastAPI
- `my-frontend/` — Vite + React frontend
- `Producto3_MHealth_Modelos_Jorge_Migueles.ipynb` — Notebook que genera los artefactos del modelo
- `models/` — Carpeta donde se guardan los `.pkl` del modelo y el `scaler`

---

## 1) Entrenar y exportar modelos (prerrequisito del backend)
El backend requiere los artefactos del modelo en `models/`:
- `models/random_forest_rf.pkl`
- `models/logistic_regression_lr.pkl` (opcional para comparar)
- `models/scaler.pkl`

Pasos en el notebook:
1. Abra el notebook: `Producto3_MHealth_Modelos_Jorge_Migueles.ipynb`.
2. Ejecute todas las celdas (Run All).
3. Verifique el mensaje final: "Modelos y scaler guardados" y/o "Modelos exportados a ../models/".
   - Si corre el notebook desde la raíz del workspace, los archivos quedarán en `models/`.
   - Si los artefactos no aparecen, busque el mensaje y ajuste la ruta o copie los `.pkl` a `models/`.

> Nota: El notebook descarga el dataset MHEALTH automáticamente si no existe.

---

## 2) Backend (FastAPI)
Ubicación: `backend/`

### Configurar entorno Python
Se recomienda usar un entorno virtual.

```powershell
# Desde la carpeta backend
cd backend

# Crear y activar venv
python -m venv .venv
.\.venv\Scripts\activate

# Instalar dependencias
# Importante: el archivo en el repo se llama "requierements.txt"
pip install -r requierements.txt
```

> Atención: El script `run_backend.bat` intenta instalar desde `requirements.txt` (sin la "e") y puede fallar. Use el comando anterior o edite el BAT para que apunte a `requierements.txt`.

### Ejecutar el servidor
```powershell
# Con el venv activo, dentro de backend/
uvicorn main:app --reload --port 8000
```

- API viva en: http://localhost:8000
- Swagger UI: http://localhost:8000/docs
- Salud: http://localhost:8000/health

Si prefieres el BAT:
```powershell
# (Opcional) Ejecutar el script de ayuda
# Edita run_backend.bat si quieres corregir el nombre del requirements
.\nrun_backend.bat
```

### Problemas comunes
- "Faltan modelos en carpeta models/": ejecuta el notebook y asegúrate de que `models/random_forest_rf.pkl` y `models/scaler.pkl` existan.
- "Feature mismatch" en `/predict/json`: verifica que el orden y cantidad de columnas coincidan con el entrenamiento (usa `feature_names` si envías JSON).

---

## 3) Frontend (Vite + React + Tailwind)
Ubicación: `my-frontend/`

### Instalar y arrancar
```powershell
cd my-frontend
npm install
npm run dev
```

- Abrirá el servidor de desarrollo (por defecto en http://localhost:5173).
- El frontend llama al backend en `http://localhost:8000`. Asegúrate de tener el backend corriendo.

### Uso
- Pestaña "Carga de datos":
  - Modo `Archivo .log`: selecciona un `.log` del dataset MHEALTH. El backend procesa y devuelve la predicción agregada.
  - Modo `JSON`: pega un arreglo de `rows` (`[[...],[...], ...]`). Opcionalmente especifica `feature_names` separadas por coma para alinear columnas.
- Pestaña "Resultados": visualiza la distribución y métricas devueltas por la API.

### Ajustes opcionales
- Si cambias el puerto del backend, actualiza las URLs de `fetch` en `my-frontend/src/App.tsx` (`http://localhost:8000`).
- Tailwind está habilitado vía plugin Vite. Si observas estilos faltantes, reinstala dependencias y reinicia `npm run dev`.

---

## 4) Pruebas rápidas de API
Con backend corriendo:

```powershell
# Predicción desde JSON minimal (PowerShell)
$payload = @{ rows = @(@(0,0,0), @(1,1,1)) } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/predict/json -Method Post -ContentType 'application/json' -Body $payload
```

> Para archivos `.log`, usa el frontend o herramientas como Postman.

---

## 5) Solución de problemas
- CORS: ya está permitido en backend (`allow_origins=["*"]`). Si despliegas, restringe orígenes.
- Versiones de Python: si hay errores binarios, prueba con Python 3.10/3.11.
- Dependencias: si `pip install -r requierements.txt` falla, borra `.venv` y reinstala.

---

## 6) Despliegue (opcional)
- Backend: cualquier servidor ASGI (Uvicorn/Gunicorn) detrás de un proxy (NGINX). Ajusta CORS.
- Frontend: `npm run build` y sirve `dist/` (Vercel, Netlify o NGINX).

```powershell
cd my-frontend
npm run build
```

---

## Créditos
- Dataset: UCI MHEALTH
- Autor notebook: Jorge Migueles
- Tech: FastAPI, scikit-learn, Vite, React, Tailwind
