# etl/extract.py
import os
import glob
import subprocess
import shutil
import pandas as pd
from dotenv import load_dotenv

# Cargar .env
load_dotenv()

def _ensure_kaggle_env():
    user = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")
    if not user or not key:
        raise EnvironmentError("Faltan KAGGLE_USERNAME/KAGGLE_KEY en .env")
    return user, key

DATASET = "yashdevladdha/uber-ride-analytics-dashboard"

def _ensure_kaggle_env():
    # Debes tener KAGGLE_USERNAME y KAGGLE_KEY en .env
    if not os.getenv("KAGGLE_USERNAME") or not os.getenv("KAGGLE_KEY"):
        raise EnvironmentError("Faltan KAGGLE_USERNAME/KAGGLE_KEY en .env")

def _kaggle_cli_exists():
    return shutil.which("kaggle") is not None

def download_from_kaggle(data_dir: str = "data") -> None:
    load_dotenv()
    _ensure_kaggle_env()

    os.makedirs(data_dir, exist_ok=True)
    if not _kaggle_cli_exists():
        raise RuntimeError(
           
        )
    # Descarga y descomprime
    cmd = ["kaggle", "datasets", "download", "-d", DATASET, "-p", data_dir, "--unzip"]
    print("⬇ Descargando dataset Kaggle…")
    subprocess.run(cmd, check=True)
    print("Descarga Kaggle completa.")

def read_csvs_from_data(data_dir: str = "data") -> pd.DataFrame:
    # Leer todos los CSV del directorio (después de la descarga)
    csvs = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No hay CSV en {data_dir}. Verifica la descarga.")
    dfs = []
    for f in csvs:
        try:
            df = pd.read_csv(f)
        except UnicodeDecodeError:
            df = pd.read_csv(f, encoding="latin-1")
        dfs.append(df)
        print(f"✓ {os.path.basename(f)} ({len(df)} filas)")
    return pd.concat(dfs, ignore_index=True)

def extract() -> pd.DataFrame:
    # Descarga (idempotente: si ya descargaste, simplemente leerá)
    download_from_kaggle("data")
    raw = read_csvs_from_data("data")
    print(f" Extraído total: {raw.shape[0]} filas, {raw.shape[1]} columnas")
    return raw
