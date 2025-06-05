#!/usr/bin/env python3
"""
Script de inicio para la aplicación de clasificación de movimientos.
Instala dependencias y ejecuta tanto el servidor web como la API del modelo.
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Instalar dependencias de requirements.txt"""
    print("📦 Instalando dependencias...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencias instaladas correctamente")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando dependencias: {e}")
        return False

def check_model_files():
    """Verificar que existen los archivos del modelo"""
    models_dir = Path("notebooks/models")
    if not models_dir.exists():
        print(f"⚠️  Directorio {models_dir} no encontrado")
        print("   Ejecuta primero el notebook model_training.ipynb para entrenar el modelo")
        return False
    
    # Buscar archivos del modelo
    model_files = list(models_dir.glob("motion_classifier_*.joblib"))
    if not model_files:
        print("⚠️  No se encontraron archivos de modelo entrenado")
        print("   Ejecuta primero el notebook model_training.ipynb para entrenar el modelo")
        return False
    
    print(f"✅ Modelo encontrado: {model_files[0].name}")
    return True

def main():
    """Función principal"""
    print("🚀 Iniciando aplicación de clasificación de movimientos...")
    print("=" * 60)
    
    # Verificar que estamos en el directorio correcto
    if not Path("server.py").exists():
        print("❌ Error: Ejecuta este script desde el directorio raíz del proyecto")
        sys.exit(1)
    
    # Instalar dependencias
    if not install_requirements():
        print("❌ No se pudieron instalar las dependencias")
        sys.exit(1)
    
    # Verificar archivos del modelo
    model_available = check_model_files()
    if not model_available:
        print("\n⚠️  Continuando sin modelo de clasificación...")
        print("   Solo estará disponible la detección de poses")
    
    print("\n🎯 Iniciando aplicación...")
    print("   • Servidor web: http://localhost:8000")
    print("   • API del modelo: http://localhost:5000")
    print("\n🛑 Presiona Ctrl+C para detener\n")
    
    # Ejecutar servidor principal
    try:
        subprocess.run([sys.executable, "server.py"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Aplicación detenida por el usuario")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error ejecutando la aplicación: {e}")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")

if __name__ == "__main__":
    main() 