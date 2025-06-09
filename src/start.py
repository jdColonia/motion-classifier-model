#!/usr/bin/env python3
"""
Script de inicio para la aplicación de clasificación de movimientos.
Verifica dependencias y archivos necesarios, luego inicia la aplicación.
"""

import subprocess
import sys
from pathlib import Path


def print_header():
    """Mostrar header de la aplicación"""
    print("=" * 70)
    print("🤖 CLASIFICADOR DE MOVIMIENTOS CORPORALES")
    print("   Detección y clasificación en tiempo real con IA")
    print("=" * 70)


def check_required_files():
    """Verificar que existen todos los archivos necesarios"""
    print("\n📁 VERIFICACIÓN DE ARCHIVOS")
    print("-" * 35)

    required_files = {
        "src/web/server.py": "Servidor web principal",
        "src/api/model_api.py": "API del modelo de clasificación",
        "templates/index.html": "Interfaz web principal",
        "static/js/script.js": "Lógica del frontend",
        "static/css/styles.css": "Estilos de la interfaz",
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


def check_model_files():
    """Verificar que existen los archivos del modelo"""
    print("\n🧠 VERIFICACIÓN DEL MODELO")
    print("-" * 35)

    models_dir = Path("notebooks/models")
    if not models_dir.exists():
        print(f"❌ Directorio {models_dir} no encontrado")
        print(
            "   📋 Ejecuta el notebook 'model_training.ipynb' para entrenar el modelo"
        )
        return False

    # Buscar archivos del modelo
    model_files = list(models_dir.glob("motion_classifier_*.joblib"))
    scaler_files = list(models_dir.glob("scaler_*.joblib"))
    encoder_files = list(models_dir.glob("label_encoder_*.joblib"))
    metadata_files = list(models_dir.glob("metadata_*.json"))

    missing_components = []
    if not model_files:
        missing_components.append("modelo principal")
    if not scaler_files:
        missing_components.append("scaler")
    if not encoder_files:
        missing_components.append("label encoder")
    if not metadata_files:
        missing_components.append("metadatos")

    if missing_components:
        print(f"❌ Faltan componentes del modelo: {', '.join(missing_components)}")
        print(
            "   📋 Ejecuta el notebook model_training.ipynb para generar todos los componentes"
        )
        return False

    # Mostrar información del modelo
    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
    model_size = latest_model.stat().st_size / 1024**2
    total_files = len(model_files + scaler_files + encoder_files + metadata_files)

    print(f"✅ Modelo encontrado: {latest_model.name}")
    print(f"   📊 Tamaño: {model_size:.2f} MB")
    print(f"   📁 Archivos del modelo: {total_files}")
    return True


def install_requirements():
    """Instalar dependencias si es necesario"""
    print("\n📦 VERIFICACIÓN DE DEPENDENCIAS")
    print("-" * 40)

    if not Path("requirements.txt").exists():
        print("❌ Error: No se encontró requirements.txt")
        return False

    try:
        # Verificar si las dependencias ya están instaladas
        import flask, joblib, pandas, numpy, sklearn

        print("✅ Dependencias principales ya están instaladas")
        return True
    except ImportError:
        print("📦 Instalando dependencias faltantes...")
        try:
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
        except Exception as e:
            print(f"❌ Error instalando dependencias: {e}")
            return False


def main():
    """Función principal"""
    print_header()

    # Verificaciones previas
    checks = [
        ("Archivos requeridos", check_required_files),
        ("Dependencias", install_requirements),
        ("Modelo de clasificación", check_model_files),
    ]

    print("\n🔍 Realizando verificaciones...")
    for name, check_func in checks:
        if not check_func():
            print(f"\n❌ Verificación '{name}' fallida")
            sys.exit(1)

    # Todas las verificaciones pasaron
    print("\n✅ TODAS LAS VERIFICACIONES COMPLETADAS")
    print("🚀 Iniciando aplicación...")
    print("=" * 50)

    # Delegar al servidor web para iniciar todo
    try:
        subprocess.run([sys.executable, "src/web/server.py"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Aplicación detenida por el usuario")
        print("¡Hasta luego! 👋")
    except Exception as e:
        print(f"\n❌ Error iniciando la aplicación: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
