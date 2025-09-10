# Requisitos
Python 3.10+

PostgreSQL (local o en la nube)

(Opcional) DBeaver/pgAdmin para verificar tablas

# Instalación
## 1) Crear y activar venv (Windows)
python -m venv venv
venv\Scripts\activate

##  En macOS/Linux:
- python3 -m venv venv
- source venv/bin/activate

## 2) Instalar dependencias
pip install -r requirements.txt

# Configuración

Crea un archivo .env en la raíz:

## 3) PostgreSQL local (ejemplo)
PG_URI=postgresql+psycopg2://usuario:password@localhost:5432/tu_db

## 4) Crea un usuario en Kaggle 

KAGGLE_USERNAME=tu_usuario
KAGGLE_KEY=tu_contraseña


### Importante: asegúrate de que PG_URI apunta a la misma base que abres en DBeaver/pgAdmin.

# 5) Ejecutar el codigo 

venv\Scripts\activate


python -m etl.load


