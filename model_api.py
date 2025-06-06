#!/usr/bin/env python3
"""
API Flask para el clasificador de movimientos corporales.
Carga el modelo entrenado y proporciona endpoints para clasificación en tiempo real.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import json
from pathlib import Path
import logging

# Definir la clase FeatureEngineer para que joblib pueda deserializarla
class FeatureEngineer:
    """Clase para crear características avanzadas a partir de landmarks de pose"""
    
    def __init__(self):
        # Mapeo de landmarks según MediaPipe
        self.landmark_names = {
            0: "nose", 11: "left_shoulder", 12: "right_shoulder",
            23: "left_hip", 24: "right_hip", 25: "left_knee", 
            26: "right_knee", 27: "left_ankle", 28: "right_ankle",
            31: "left_foot", 32: "right_foot"
        }
        
        # Histórico para calcular velocidades y secuencias (para tiempo real)
        self.previous_frames = {}  # Por landmark_index
        self.frame_count = 0
        self.sequence_count = {}  # Por landmark_index para sequence_position
    
    def calculate_angle(self, p1, p2, p3):
        """Calcula el ángulo entre tres puntos"""
        # Vectores
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        
        # Ángulo en radianes
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1, 1)  # Evitar errores numéricos
        angle = np.arccos(cos_angle)
        
        return np.degrees(angle)  # Convertir a grados
    
    def calculate_distance(self, p1, p2):
        """Calcula distancia euclidiana entre dos puntos"""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)
    
    def pivot_landmarks(self, df):
        """Convierte formato largo a ancho para facilitar cálculos"""
        # Crear tabla pivotada con coordenadas por landmark
        pivot_data = []
        
        for frame in df['frame'].unique():
            frame_data = df[df['frame'] == frame]
            row_data = {'frame': frame}
            
            # Agregar coordenadas para cada landmark
            for _, landmark_row in frame_data.iterrows():
                landmark_idx = landmark_row['landmark_index']
                
                row_data[f'x_{landmark_idx}'] = landmark_row['x']
                row_data[f'y_{landmark_idx}'] = landmark_row['y'] 
                row_data[f'z_{landmark_idx}'] = landmark_row['z']
                row_data[f'vis_{landmark_idx}'] = landmark_row['visibility']
            
            # Agregar movimiento si existe
            if 'movement' in frame_data.columns:
                row_data['movement'] = frame_data['movement'].iloc[0]
            
            pivot_data.append(row_data)
        
        return pd.DataFrame(pivot_data)
    
    def create_enhanced_features(self, df):
        """
        Crear todas las características mejoradas siguiendo exactamente la lógica del entrenamiento
        """
        df_enhanced = df.copy()
        self.frame_count += 1
        
        # PASO 1: Normalizar coordenadas por frame (centrar respecto al centroide)
        logger.info("Normalizando coordenadas por frame...")
        
        for frame in df_enhanced['frame'].unique():
            frame_mask = df_enhanced['frame'] == frame
            frame_data = df_enhanced[frame_mask]
            
            if len(frame_data) > 0:
                # Centrar coordenadas respecto al centroide del frame
                x_center = frame_data['x'].mean()
                y_center = frame_data['y'].mean()
                z_center = frame_data['z'].mean()
                
                df_enhanced.loc[frame_mask, 'x_normalized'] = df_enhanced.loc[frame_mask, 'x'] - x_center
                df_enhanced.loc[frame_mask, 'y_normalized'] = df_enhanced.loc[frame_mask, 'y'] - y_center
                df_enhanced.loc[frame_mask, 'z_normalized'] = df_enhanced.loc[frame_mask, 'z'] - z_center
        
        # PASO 2: Crear features de velocidad (diferencias temporales)
        logger.info("Creando features de velocidad...")
        
        # Ordenar por frame y landmark para calcular velocidades
        df_enhanced = df_enhanced.sort_values(['landmark_index', 'frame'])
        
        # Inicializar columnas de velocidad
        df_enhanced['velocity_x'] = 0.0
        df_enhanced['velocity_y'] = 0.0
        df_enhanced['velocity_z'] = 0.0
        df_enhanced['velocity_magnitude'] = 0.0
        
        # Calcular velocidades por landmark usando el histórico
        for idx, row in df_enhanced.iterrows():
            landmark_idx = row['landmark_index']
            
            if landmark_idx in self.previous_frames:
                # Calcular diferencias temporales (velocidad)
                prev_row = self.previous_frames[landmark_idx]
                
                velocity_x = row['x'] - prev_row['x']
                velocity_y = row['y'] - prev_row['y']
                velocity_z = row['z'] - prev_row['z']
                velocity_magnitude = np.sqrt(velocity_x**2 + velocity_y**2 + velocity_z**2)
                
                df_enhanced.loc[idx, 'velocity_x'] = velocity_x
                df_enhanced.loc[idx, 'velocity_y'] = velocity_y
                df_enhanced.loc[idx, 'velocity_z'] = velocity_z
                df_enhanced.loc[idx, 'velocity_magnitude'] = velocity_magnitude
            
            # Actualizar histórico
            self.previous_frames[landmark_idx] = {
                'x': row['x'],
                'y': row['y'],
                'z': row['z']
            }
        
        # PASO 3: Crear features específicas por landmark
        logger.info("Creando features específicas por landmark...")
        
        # Crear features one-hot para landmarks importantes (según la imagen)
        important_landmarks = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
        for landmark_idx in important_landmarks:
            df_enhanced[f'is_landmark_{landmark_idx}'] = (df_enhanced['landmark_index'] == landmark_idx).astype(int)
        
        # PASO 4: Crear features de contexto temporal
        logger.info("Creando features de contexto temporal...")
        
        # Frame normalizado (para tiempo real, normalizar por ventana deslizante)
        # Como es tiempo real, usar una normalización basada en el contador de frames
        df_enhanced['frame_normalized'] = self.frame_count / 100.0  # Escala arbitraria
        
        # Sequence position (posición en la secuencia por landmark)
        for idx, row in df_enhanced.iterrows():
            landmark_idx = row['landmark_index']
            
            if landmark_idx not in self.sequence_count:
                self.sequence_count[landmark_idx] = 0
            else:
                self.sequence_count[landmark_idx] += 1
            
            df_enhanced.loc[idx, 'sequence_position'] = self.sequence_count[landmark_idx]
        
        # Sequence position normalizado
        max_sequence = max(self.sequence_count.values()) if self.sequence_count else 1
        df_enhanced['sequence_position_normalized'] = df_enhanced['sequence_position'] / (max_sequence + 1e-8)
        
        # PASO 5: Crear características de visibility específicas
        logger.info("Creando características de visibility...")
        
        # y_visibility y z_visibility como promedios de visibility
        df_enhanced['y_visibility'] = df_enhanced['visibility']
        df_enhanced['z_visibility'] = df_enhanced['visibility']
        
        return df_enhanced

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Permitir CORS para requests desde el frontend

class MovementPredictor:
    def __init__(self, models_dir="notebooks/models"):
        """Inicializar el predictor cargando todos los componentes del modelo"""
        self.models_dir = Path(models_dir)
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_engineer = None
        self.metadata = None
        self.selected_features = None
        
        # Cargar el modelo más reciente
        self.load_latest_model()
    
    def load_latest_model(self):
        """Cargar el modelo más reciente basado en timestamp"""
        try:
            # Buscar archivos del modelo
            model_files = list(self.models_dir.glob("motion_classifier_*.joblib"))
            if not model_files:
                raise FileNotFoundError("No se encontraron archivos de modelo")
            
            # Obtener el más reciente
            latest_model_file = max(model_files, key=lambda x: x.stat().st_mtime)
            
            # Extraer la versión completa del nombre del archivo
            # Formato: motion_classifier_random_forest_v1.0_20250605_180729.joblib
            model_name = latest_model_file.stem
            # Buscar el patrón v1.0_YYYYMMDD_HHMMSS
            import re
            version_match = re.search(r'(v\d+\.\d+_\d{8}_\d{6})', model_name)
            if version_match:
                version = version_match.group(1)
            else:
                # Fallback: usar las últimas partes del nombre
                parts = model_name.split('_')
                version = '_'.join(parts[-3:])  # v1.0_20250605_180729
            
            logger.info(f"Cargando modelo versión: {version}")
            
            # Cargar componentes con el formato correcto
            self.model = joblib.load(latest_model_file)
            self.scaler = joblib.load(self.models_dir / f"scaler_{version}.joblib")
            self.label_encoder = joblib.load(self.models_dir / f"label_encoder_{version}.joblib")
            
            # Crear una nueva instancia del FeatureEngineer en lugar de cargarlo desde joblib
            # porque puede haber problemas de deserialización
            self.feature_engineer = FeatureEngineer()
            
            # Para predicciones en tiempo real, necesitamos una instancia por predicción
            # para manejar correctamente el estado de velocidades
            self.feature_engineer_template = FeatureEngineer()
            
            # Cargar metadatos
            metadata_file = self.models_dir / f"model_metadata_{version}.json"
            with open(metadata_file, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            
            self.selected_features = self.metadata['feature_info']['selected_features']
            
            logger.info(f"✅ Modelo cargado exitosamente")
            logger.info(f"   • Algoritmo: {self.metadata['model_info']['name']}")
            logger.info(f"   • Accuracy: {self.metadata['model_info']['accuracy']:.4f}")
            logger.info(f"   • Features: {len(self.selected_features)}")
            
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            raise
    
    def landmarks_to_dataframe(self, landmarks_data):
        """Convertir landmarks de JavaScript a DataFrame de pandas"""
        try:
            logger.info(f"Convirtiendo {len(landmarks_data)} landmarks a DataFrame")
            
            # Crear DataFrame desde los landmarks recibidos
            data = []
            frame = 0  # Frame único para predicción en tiempo real
            
            for landmark in landmarks_data:
                if not isinstance(landmark, dict):
                    logger.error(f"Landmark no es un diccionario: {type(landmark)}")
                    continue
                
                required_keys = ['landmark_index', 'x', 'y', 'z', 'visibility']
                if not all(key in landmark for key in required_keys):
                    logger.error(f"Landmark falta claves requeridas. Claves encontradas: {landmark.keys()}")
                    continue
                
                landmark_idx = landmark['landmark_index']
                
                # Verificar que es un landmark relevante
                # Incluir todos los landmarks que el modelo podría necesitar
                relevant_landmarks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
                if landmark_idx in relevant_landmarks:
                    data.append({
                        'frame': frame,
                        'landmark_index': landmark_idx,
                        'x': landmark['x'],
                        'y': landmark['y'],
                        'z': landmark['z'],
                        'visibility': landmark['visibility'],
                        'movement': 'unknown'  # Placeholder
                    })
                # Remover el logging excesivo para landmarks no relevantes
            
            logger.info(f"DataFrame creado con {len(data)} landmarks válidos")
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Error convirtiendo landmarks: {e}")
            raise
    
    def predict(self, landmarks_data):
        """
        Predecir movimiento a partir de landmarks
        
        Args:
            landmarks_data: Lista de landmarks con formato {x, y, z, visibility}
        
        Returns:
            dict: Predicción y probabilidades
        """
        try:
            # Convertir a DataFrame
            df = self.landmarks_to_dataframe(landmarks_data)
            
            if len(df) == 0:
                return {"error": "No hay landmarks válidos"}
            
            # Feature engineering completo usando el nuevo método
            logger.info("Iniciando feature engineering...")
            
            # Crear todas las características usando el método mejorado
            df_with_features = self.feature_engineer.create_enhanced_features(df)
            
            # Verificar qué features tenemos vs las que necesitamos
            available_features = set(df_with_features.columns)
            required_features = set(self.selected_features)
            missing_features = required_features - available_features
            
            if missing_features:
                logger.warning(f"Features faltantes: {missing_features}")
                # Agregar features faltantes con valor por defecto
                for feature in missing_features:
                    df_with_features[feature] = 0.0
            
            # Seleccionar solo las features que el modelo necesita
            X = df_with_features[self.selected_features]
            
            logger.info(f"Shape de X: {X.shape}")
            logger.info(f"Features utilizadas: {list(X.columns)[:10]}...")  # Solo mostrar las primeras 10
            
            # Manejar NaN e infinitos
            X = X.fillna(0)
            X = X.replace([np.inf, -np.inf], 0)
            
            # Verificar que no hay problemas con los datos
            if X.isnull().sum().sum() > 0:
                logger.warning("Se encontraron valores NaN después de limpiar")
                X = X.fillna(0)
            
            # Escalar
            X_scaled = self.scaler.transform(X)
            
            # Predecir
            prediction_encoded = self.model.predict(X_scaled)[0]
            prediction = self.label_encoder.inverse_transform([prediction_encoded])[0]
            
            # Obtener probabilidades si es posible
            probabilities = {}
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(X_scaled)[0]
                for i, class_name in enumerate(self.label_encoder.classes_):
                    probabilities[class_name] = float(proba[i])
            
            return {
                "prediction": prediction,
                "probabilities": probabilities,
                "confidence": max(probabilities.values()) if probabilities else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error en predicción: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {"error": str(e)}

# Inicializar predictor global
try:
    predictor = MovementPredictor()
    logger.info("🤖 Predictor inicializado correctamente")
except Exception as e:
    logger.error(f"❌ Error inicializando predictor: {e}")
    predictor = None

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint para verificar el estado del servidor"""
    if predictor and predictor.model:
        return jsonify({
            "status": "healthy",
            "model_loaded": True,
            "model_info": predictor.metadata['model_info'] if predictor.metadata else None
        })
    else:
        return jsonify({
            "status": "error",
            "model_loaded": False,
            "error": "Modelo no cargado"
        }), 500

@app.route('/predict', methods=['POST'])
def predict_movement():
    """Endpoint para predecir movimiento desde landmarks"""
    if not predictor:
        return jsonify({"error": "Modelo no disponible"}), 500
    
    try:
        data = request.get_json()
        
        if not data or 'landmarks' not in data:
            logger.error("Datos de landmarks faltantes en request")
            return jsonify({"error": "Datos de landmarks requeridos"}), 400
        
        landmarks = data['landmarks']
        
        if not isinstance(landmarks, list) or len(landmarks) == 0:
            logger.error(f"Landmarks inválidos: tipo={type(landmarks)}, longitud={len(landmarks) if isinstance(landmarks, list) else 'N/A'}")
            return jsonify({"error": "Landmarks inválidos"}), 400
        
        logger.info(f"Recibidos {len(landmarks)} landmarks para predicción")
        
        # Realizar predicción
        result = predictor.predict(landmarks)
        
        if "error" in result:
            logger.error(f"Error en predicción: {result['error']}")
            return jsonify(result), 400
        
        logger.info(f"Predicción exitosa: {result['prediction']} (confianza: {result['confidence']:.3f})")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error en endpoint predict: {e}")
        return jsonify({"error": "Error interno del servidor"}), 500

@app.route('/model-info', methods=['GET'])
def get_model_info():
    """Endpoint para obtener información del modelo"""
    if not predictor or not predictor.metadata:
        return jsonify({"error": "Modelo no disponible"}), 500
    
    return jsonify({
        "model_info": predictor.metadata['model_info'],
        "classes": predictor.metadata['training_info']['classes'],
        "features_count": len(predictor.selected_features),
        "accuracy": predictor.metadata['performance_metrics']['accuracy']
    })

if __name__ == '__main__':
    if predictor:
        print("🚀 Iniciando API del clasificador de movimientos...")
        print(f"📊 Modelo: {predictor.metadata['model_info']['name']}")
        print(f"🎯 Accuracy: {predictor.metadata['model_info']['accuracy']:.4f}")
        print(f"🔗 API disponible en: http://localhost:5000")
        print("\n📋 Endpoints disponibles:")
        print("   • GET  /health     - Estado del servidor")
        print("   • POST /predict    - Clasificar movimiento")
        print("   • GET  /model-info - Información del modelo")
        
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        print("❌ No se pudo inicializar el predictor") 