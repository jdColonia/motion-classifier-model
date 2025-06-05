#!/usr/bin/env python3
"""
Servidor HTTP simple para servir la aplicación de detección de poses.
Ejecutar con: python server.py
"""

import http.server
import socketserver
import webbrowser
import os
import sys
import subprocess
import threading
import time
from pathlib import Path

# Configuración del servidor
PORT = 8000
HOST = 'localhost'

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Handler personalizado para servir archivos con las cabeceras correctas."""
    
    def end_headers(self):
        # Agregar cabeceras CORS para permitir acceso a MediaPipe
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        super().end_headers()
    
    def guess_type(self, path):
        """Determinar el tipo MIME del archivo."""
        # Asegurar tipos MIME correctos primero
        if path.endswith('.js'):
            return 'application/javascript'
        elif path.endswith('.css'):
            return 'text/css'
        elif path.endswith('.html'):
            return 'text/html'
        
        # Usar el método padre como fallback
        result = super().guess_type(path)
        if isinstance(result, tuple) and len(result) >= 1:
            return result[0]
        return result

def start_model_api():
    """Iniciar la API del modelo en un proceso separado."""
    try:
        print("🤖 Iniciando API del modelo de clasificación...")
        # Ejecutar la API del modelo
        subprocess.run([sys.executable, "model_api.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error iniciando API del modelo: {e}")
    except FileNotFoundError:
        print("❌ Error: No se encontró model_api.py")
    except Exception as e:
        print(f"❌ Error inesperado en API del modelo: {e}")

def check_model_api():
    """Verificar si la API del modelo está disponible."""
    try:
        import requests
        response = requests.get("http://localhost:5000/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def start_server():
    """Iniciar el servidor HTTP y la API del modelo."""
    try:
        # Cambiar al directorio donde están los archivos web
        web_dir = Path(__file__).parent
        os.chdir(web_dir)
        
        # Verificar que existen los archivos necesarios
        required_files = ['index.html', 'styles.css', 'script.js', 'model_api.py']
        missing_files = [f for f in required_files if not Path(f).exists()]
        
        if missing_files:
            print(f"❌ Error: Faltan archivos necesarios: {', '.join(missing_files)}")
            sys.exit(1)
        
        # Verificar si existe el directorio de modelos
        models_dir = Path("notebooks/models")
        if not models_dir.exists():
            print(f"⚠️  Advertencia: No se encontró el directorio {models_dir}")
            print("   La clasificación de movimientos no estará disponible")
        
        # Iniciar la API del modelo en un hilo separado
        model_thread = threading.Thread(target=start_model_api, daemon=True)
        model_thread.start()
        
        # Esperar un poco para que la API se inicie
        print("⏳ Esperando que la API del modelo se inicie...")
        time.sleep(3)
        
        # Verificar si la API está funcionando
        if check_model_api():
            print("✅ API del modelo iniciada correctamente")
        else:
            print("⚠️  API del modelo no disponible - solo detección de poses")
        
        # Crear y configurar el servidor web
        with socketserver.TCPServer((HOST, PORT), CustomHTTPRequestHandler) as httpd:
            server_url = f"http://{HOST}:{PORT}"
            
            print("\n🚀 Iniciando servidor de detección de poses...")
            print(f"📍 Servidor web: {server_url}")
            print(f"🤖 API del modelo: http://localhost:5000")
            print(f"📁 Directorio servido: {web_dir.absolute()}")
            print("\n📋 Instrucciones:")
            print("   1. Abre tu navegador y ve a la URL mostrada arriba")
            print("   2. Permite el acceso a la cámara cuando se solicite")
            print("   3. Haz clic en 'Iniciar Cámara' para comenzar")
            print("   4. Los movimientos se clasificarán automáticamente")
            print("\n⚠️  Nota: Usa HTTPS en producción para acceso a cámara")
            print("🛑 Presiona Ctrl+C para detener ambos servidores\n")
            
            # Abrir automáticamente en el navegador
            try:
                webbrowser.open(server_url)
            except Exception as e:
                print(f"No se pudo abrir el navegador automáticamente: {e}")
            
            # Iniciar el servidor
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\n🛑 Servidores detenidos por el usuario")
    except OSError as e:
        if e.errno == 48:  # Puerto en uso
            print(f"❌ Error: El puerto {PORT} ya está en uso")
            print("   Intenta con un puerto diferente o cierra la aplicación que usa este puerto")
        else:
            print(f"❌ Error del sistema: {e}")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")

if __name__ == "__main__":
    start_server() 