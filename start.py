#!/usr/bin/env python3
"""
Script de inicio mejorado para la aplicación de clasificación de movimientos.
Verifica dependencias, modelo exportado y ejecuta la aplicación completa.
"""

import subprocess
import sys
import os
import time
import threading
from pathlib import Path


def print_header():
    """Mostrar header de la aplicación"""
    print("=" * 70)
    print("🤖 CLASIFICADOR DE MOVIMIENTOS CORPORALES")
    print("   Detección y clasificación en tiempo real con IA")
    print("=" * 70)


def install_requirements():
    """Instalar dependencias de requirements.txt"""
    print("\n📦 INSTALACIÓN DE DEPENDENCIAS")
    print("-" * 40)

    if not Path("requirements.txt").exists():
        print("❌ Error: No se encontró requirements.txt")
        return False

    try:
        print("Instalando dependencias...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode == 0:
            print("✅ Dependencias instaladas correctamente")
            return True
        else:
            print("❌ Error instalando dependencias:")
            print(result.stderr)
            return False

    except subprocess.TimeoutExpired:
        print("❌ Timeout instalando dependencias")
        return False
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando dependencias: {e}")
        return False


def check_model_files():
    """Verificar que existen los archivos del modelo exportado"""
    print("\n🧠 VERIFICACIÓN DEL MODELO")
    print("-" * 35)

    models_dir = Path("notebooks/models")
    if not models_dir.exists():
        print(f"❌ Directorio {models_dir} no encontrado")
        print("   📋 Instrucciones:")
        print("   1. Ejecuta el notebook 'model_training.ipynb' completo")
        print("   2. Asegúrate de que se exportaron todos los archivos del modelo")
        return False

    # Buscar archivos del modelo
    model_files = list(models_dir.glob("motion_classifier_*.joblib"))
    scaler_files = list(models_dir.glob("scaler_*.joblib"))
    encoder_files = list(models_dir.glob("label_encoder_*.joblib"))
    metadata_files = list(models_dir.glob("metadata_*.json"))

    if not model_files:
        print("❌ No se encontraron archivos de modelo (.joblib)")
        print("   📋 Ejecuta el notebook model_training.ipynb para entrenar el modelo")
        return False

    if not scaler_files:
        print("❌ No se encontró el scaler del modelo")
        return False

    if not encoder_files:
        print("❌ No se encontró el label encoder")
        return False

    if not metadata_files:
        print("❌ No se encontraron metadatos del modelo")
        return False

    # Mostrar información del modelo encontrado
    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
    model_size = latest_model.stat().st_size / 1024**2

    print(f"✅ Modelo encontrado: {latest_model.name}")
    print(f"   📊 Tamaño: {model_size:.2f} MB")
    print(f"   📅 Última modificación: {time.ctime(latest_model.stat().st_mtime)}")
    print(
        f"   📁 Archivos del modelo: {len(model_files + scaler_files + encoder_files + metadata_files)}"
    )

    return True


def check_required_files():
    """Verificar que existen todos los archivos necesarios"""
    print("\n📁 VERIFICACIÓN DE ARCHIVOS")
    print("-" * 35)

    required_files = {
        "server.py": "Servidor web principal",
        "model_api.py": "API del modelo de clasificación",
        "index.html": "Interfaz web principal",
        "script.js": "Lógica del frontend",
        "styles.css": "Estilos de la interfaz",
        "requirements.txt": "Lista de dependencias",
    }

    missing_files = []
    for file_path, description in required_files.items():
        if Path(file_path).exists():
            print(f"   ✅ {file_path} - {description}")
        else:
            print(f"   ❌ {file_path} - {description} (FALTANTE)")
            missing_files.append(file_path)

    if missing_files:
        print(f"\n❌ Faltan archivos necesarios: {', '.join(missing_files)}")
        return False

    print("✅ Todos los archivos necesarios están presentes")
    return True


def test_model_api():
    """Probar que la API del modelo funciona"""
    print("\n🔧 PRUEBA DE LA API DEL MODELO")
    print("-" * 40)

    try:
        # Importar Flask para verificar que está disponible
        import flask
        import joblib
        import pandas as pd
        import numpy as np

        print("✅ Librerías de ML disponibles")

        # Intentar cargar el modelo rápidamente
        models_dir = Path("notebooks/models")
        model_files = list(models_dir.glob("motion_classifier_*.joblib"))

        if model_files:
            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
            model = joblib.load(latest_model)
            print("✅ Modelo se puede cargar correctamente")
            print(f"   📊 Tipo de modelo: {type(model).__name__}")

            # Verificar que tiene los métodos necesarios
            if hasattr(model, "predict"):
                print("✅ Modelo tiene método de predicción")
            else:
                print("❌ Modelo no tiene método de predicción")
                return False

        return True

    except ImportError as e:
        print(f"❌ Error importando librerías: {e}")
        return False
    except Exception as e:
        print(f"❌ Error probando el modelo: {e}")
        return False


def start_model_api():
    """Iniciar la API del modelo en hilo separado"""
    try:
        print("🤖 Iniciando API del modelo...")
        subprocess.run([sys.executable, "model_api.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error en API del modelo: {e}")
    except KeyboardInterrupt:
        print("🛑 API del modelo detenida")


def start_web_server():
    """Iniciar el servidor web"""
    try:
        print("🌐 Iniciando servidor web...")
        subprocess.run([sys.executable, "server.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error en servidor web: {e}")
    except KeyboardInterrupt:
        print("🛑 Servidor web detenido")


def main():
    """Función principal"""
    print_header()

    # Verificar que estamos en el directorio correcto
    if not Path("server.py").exists():
        print("❌ Error: Ejecuta este script desde el directorio raíz del proyecto")
        print("   El directorio debe contener server.py, model_api.py, etc.")
        sys.exit(1)

    # Verificaciones previas
    checks = [
        ("Archivos requeridos", check_required_files),
        ("Instalación de dependencias", install_requirements),
        ("Verificación de archivos del modelo", check_model_files),
        ("Prueba de API del modelo", test_model_api),
    ]

    for name, check_func in checks:
        print(f"\n🔍 {name}")
        if not check_func():
            print("❌ Verificación fallida")
            sys.exit(1)

    # Todas las verificaciones pasaron
    print("\n✅ TODAS LAS VERIFICACIONES COMPLETADAS")
    print("=" * 50)

    # Iniciar los servidores
    print("\n🚀 INICIANDO APLICACIÓN")
    print("-" * 30)

    try:
        # Crear hilos para ejecutar ambos servidores
        print("Iniciando API del modelo en puerto 5001...")
        api_thread = threading.Thread(target=start_model_api, daemon=True)
        api_thread.start()

        # Esperar un momento para que la API se inicie
        time.sleep(3)

        print("Iniciando servidor web en puerto 5000...")
        print("\n🌐 APLICACIÓN LISTA")
        print("=" * 40)
        print("📱 Abre tu navegador en: http://localhost:5000")
        print("🤖 API del modelo en: http://localhost:5001")
        print("⌨️  Presiona Ctrl+C para detener")
        print("=" * 40)

        # Iniciar servidor web (bloquea hasta Ctrl+C)
        start_web_server()

    except KeyboardInterrupt:
        print("\n\n🛑 DETENIENDO APLICACIÓN")
        print("   Cerrando servidores...")
        print("   ¡Hasta luego! 👋")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
