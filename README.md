# Proyecto Integración de Datos — MHealth HAR

Backend FastAPI + Notebook para entrenar, exportar y servir modelos de reconocimiento de actividad humana (HAR) sobre el dataset MHEALTH.

## Resumen
- Entrenas en el notebook y exportas artefactos directamente a `backend/models/`.
- El backend carga el modelo activo, alinea columnas con `features_{name}.json` y expone endpoints para predicción.
- Los archivos de modelo pesados (`.pkl/.joblib`) están ignorados en Git para evitar problemas al subirlos.

## Requisitos
- Python 3.10+
- Windows PowerShell o CMD

## Backend — Inicio rápido (Windows)
```powershell
cd .\backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
# Swagger UI: http://localhost:8000/docs
# Salud:      http://localhost:8000/health
```

Opcional: fijar el modelo activo por variable de entorno
```powershell
$env:BACKEND_MODEL_NAME = "HistGradientBoosting"  # o el nombre que exportaste
uvicorn main:app --reload --port 8000
```

## Exportar modelo desde el Notebook
1. Ejecuta el entrenamiento y la selección de features.
2. Ejecuta la celda “Exportar artefactos al backend/models” al final del notebook.
   - Guarda:
     - `backend/models/{name}.pkl`
     - `backend/models/scaler_{name}.pkl`
     - `backend/models/features_{name}.json`
   - El scaler se alinea automáticamente a las mismas columnas exportadas (mismas dimensiones).

## Endpoints principales (FastAPI)
- `GET /health`: estado del servicio
- `GET /schema`: descripción breve
- `GET /models/list`: lista de modelos en `backend/models/` y el activo
- `POST /models/load` `{ "name": "HistGradientBoosting" }`: cambiar modelo activo sin reiniciar
- `POST /predict/json`: predicción desde filas numéricas
- `POST /predict/file`: predicción desde archivo `.log` del dataset

Ejemplos rápidos con PowerShell (Invoke-RestMethod):
```powershell
# Listar modelos
Invoke-RestMethod -Method GET http://localhost:8000/models/list | Format-List

# Seleccionar modelo
Invoke-RestMethod -Method POST http://localhost:8000/models/load -Body '{"name":"HistGradientBoosting"}' -ContentType 'application/json'

# Predicción JSON (ejemplo)
$body = @{ rows = @(@(0,1,2), @(3,4,5)); feature_names = @("f1","f2","f3") } | ConvertTo-Json
Invoke-RestMethod -Method POST http://localhost:8000/predict/json -Body $body -ContentType 'application/json'

# Predicción desde archivo .log
Invoke-RestMethod -Method POST http://localhost:8000/predict/file -InFile "MHEALTHDATASET\mHealth_subject1.log" -ContentType 'multipart/form-data'
```

## Pureza vs. Confianza (resumen)
- **Pureza**: fracción de la clase mayoritaria dentro de cada ventana, calculada sobre el tamaño total de ventana (las filas sin etiqueta cuentan como “no coinciden”). Umbral configurable por env: `MHEALTH_MIN_LABEL_PURITY` (default 0.6).
- **Confianza**: probabilidad máxima que el modelo asigna a su predicción por ventana. Se reporta por ventana, y también agregada (promedio) por archivo/actividad.

## Modelos pesados y Git
- Por defecto se ignoran `backend/models/*.pkl` y `*.joblib` mediante `.gitignore` para evitar errores de tamaño en GitHub.
- Si necesitas versionar modelos grandes, usa Git LFS (opcional): https://git-lfs.github.com/

## Problemas comunes
- `run_backend.bat` falla: instala dependencias manualmente y arranca con `uvicorn` (ver “Inicio rápido”).
- Error por desajuste de columnas: asegúrate de exportar con la celda del notebook; el backend usará `features_{name}.json` para alinear columnas.

## Estructura relevante
```
backend/
  main.py                # API FastAPI
  mhealth_pipeline.py    # Extracción de ventanas y features + predicción
  model_registry.py      # Carga/guardado de artefactos (modelo/scaler/features)
  models/                # Artefactos exportados desde el notebook
Producto_MHealth_Modelos_Jorge_Migueles.ipynb  # Notebook de entrenamiento/exportación
```
