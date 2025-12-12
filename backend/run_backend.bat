@echo off
REM Script para ejecutar el backend localmente
echo.
echo ========================================
echo  MHealth HAR - Backend (FastAPI)
echo ========================================
echo.
echo Instalando dependencias...
pip install -r requirements.txt

echo.
echo Iniciando servidor uvicorn...
echo Accede a: http://localhost:8000/docs para Swagger UI
echo Endpoint de salud: http://localhost:8000/health
echo.

uvicorn main:app --reload --port 8000
