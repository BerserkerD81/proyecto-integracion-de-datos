# MHEALTH HAR — Backend + Notebook

Servir y exportar modelos de reconocimiento de actividad humana (HAR) para MHEALTH.

## Inicio Rápido
- Requisitos: Python 3.10+ (Windows / PowerShell o CMD)
- Backend:
  ```powershell
  cd .\backend
  pip install -r requirements.txt
  uvicorn main:app --reload --port 8000
  ```
  - Swagger: http://localhost:8000/docs
  - Salud:   http://localhost:8000/health

## Exportar desde el Notebook
1) Entrena el modelo y selecciona features.
2) Ejecuta la celda “Exportar artefactos al backend/models”. Genera:
   - `backend/models/{name}.pkl`
   - `backend/models/scaler_{name}.pkl`
   - `backend/models/features_{name}.json`

## Uso del Backend
- Modelos: `GET /models/list`, cambiar activo: `POST /models/load`.
- Predicción: `POST /predict/json` o `POST /predict/file` (ver Swagger para esquemas y ejemplos).
- Opcional: `BACKEND_MODEL_NAME` para fijar el modelo al iniciar.

## Notas
- Los artefactos de modelos (`.pkl/.joblib`) se manejan con Git LFS y/o `.gitignore` para evitar límites de tamaño.
- Las columnas se alinean con `features_{name}.json` automáticamente.

## Estructura
```
backend/
  main.py
  mhealth_pipeline.py
  model_registry.py
  models/
Producto_MHealth_Modelos_Jorge_Migueles.ipynb
```
