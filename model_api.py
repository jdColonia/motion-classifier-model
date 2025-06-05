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
                prefix = f"{landmark_row['x']}_{landmark_idx}"
                
                row_data[f'x_{landmark_idx}'] = landmark_row['x']
                row_data[f'y_{landmark_idx}'] = landmark_row['y'] 
                row_data[f'z_{landmark_idx}'] = landmark_row['z']
                row_data[f'vis_{landmark_idx}'] = landmark_row['visibility']
            
            # Agregar movimiento si existe
            if 'movement' in frame_data.columns:
                row_data['movement'] = frame_data['movement'].iloc[0]
            
            pivot_data.append(row_data)
        
        return pd.DataFrame(pivot_data)
    
    def create_angle_features(self, df_pivot):
        """Crea características basadas en ángulos entre articulaciones"""
        df_angles = df_pivot.copy()
        
        # Definir ángulos importantes
        angle_definitions = [
            # Ángulos del torso
            ('torso_left', [11, 23, 25]),    # hombro_izq -> cadera_izq -> rodilla_izq
            ('torso_right', [12, 24, 26]),   # hombro_der -> cadera_der -> rodilla_der
            
            # Ángulos de las piernas
            ('leg_left', [23, 25, 27]),      # cadera_izq -> rodilla_izq -> tobillo_izq
            ('leg_right', [24, 26, 28]),     # cadera_der -> rodilla_der -> tobillo_der
            
            # Ángulos de postura
            ('posture_left', [0, 11, 23]),   # nariz -> hombro_izq -> cadera_izq
            ('posture_right', [0, 12, 24]),  # nariz -> hombro_der -> cadera_der
        ]
        
        for angle_name, landmarks in angle_definitions:
            angles = []
            
            for _, row in df_pivot.iterrows():
                try:
                    # Extraer coordenadas de los landmarks
                    p1 = (row[f'x_{landmarks[0]}'], row[f'y_{landmarks[0]}'])
                    p2 = (row[f'x_{landmarks[1]}'], row[f'y_{landmarks[1]}'])
                    p3 = (row[f'x_{landmarks[2]}'], row[f'y_{landmarks[2]}'])
                    
                    # Verificar que los landmarks existen
                    if not (pd.isna(p1[0]) or pd.isna(p2[0]) or pd.isna(p3[0])):
                        angle = self.calculate_angle(p1, p2, p3)
                    else:
                        angle = 0  # Valor por defecto si faltan datos
                    
                    angles.append(angle)
                    
                except:
                    angles.append(0)
            
            df_angles[f'angle_{angle_name}'] = angles
        
        return df_angles
    
    def create_distance_features(self, df_pivot):
        """Crea características basadas en distancias entre landmarks"""
        df_distances = df_pivot.copy()
        
        # Distancias importantes
        distance_definitions = [
            # Anchura de hombros y caderas
            ('shoulder_width', [11, 12]),
            ('hip_width', [23, 24]),
            
            # Altura del centro de masa
            ('torso_height', [0, 23]),  # Nariz a cadera promedio
            
            # Longitud de extremidades
            ('left_leg', [25, 27]),     # rodilla_izq -> tobillo_izq
            ('right_leg', [26, 28]),    # rodilla_der -> tobillo_der
            ('feet_distance', [31, 32]), # pie_izq -> pie_der
        ]
        
        for dist_name, landmarks in distance_definitions:
            distances = []
            
            for _, row in df_pivot.iterrows():
                try:
                    # Extraer coordenadas 3D
                    p1 = (row[f'x_{landmarks[0]}'], row[f'y_{landmarks[0]}'], row[f'z_{landmarks[0]}'])
                    p2 = (row[f'x_{landmarks[1]}'], row[f'y_{landmarks[1]}'], row[f'z_{landmarks[1]}'])
                    
                    # Verificar que los landmarks existen
                    if not (pd.isna(p1[0]) or pd.isna(p2[0])):
                        distance = self.calculate_distance(p1, p2)
                    else:
                        distance = 0
                
                    distances.append(distance)
                    
                except:
                    distances.append(0)
            
            df_distances[f'dist_{dist_name}'] = distances
        
        return df_distances

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
            self.feature_engineer = joblib.load(self.models_dir / f"feature_engineer_{version}.joblib")
            
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
            # Crear DataFrame desde los landmarks recibidos
            data = []
            frame = 0  # Frame único para predicción en tiempo real
            
            for landmark_idx, landmark in enumerate(landmarks_data):
                if landmark_idx in [0, 11, 12, 23, 24, 25, 26, 27, 28, 31, 32]:  # Solo landmarks relevantes
                    data.append({
                        'frame': frame,
                        'landmark_index': landmark_idx,
                        'x': landmark['x'],
                        'y': landmark['y'],
                        'z': landmark['z'],
                        'visibility': landmark['visibility'],
                        'movement': 'unknown'  # Placeholder
                    })
            
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
            
            # Feature engineering
            df_pivot = self.feature_engineer.pivot_landmarks(df)
            df_with_angles = self.feature_engineer.create_angle_features(df_pivot)
            df_with_features = self.feature_engineer.create_distance_features(df_with_angles)
            
            # Verificar que tenemos las features necesarias
            missing_features = set(self.selected_features) - set(df_with_features.columns)
            if missing_features:
                # Agregar features faltantes con valor 0
                for feature in missing_features:
                    df_with_features[feature] = 0
            
            # Seleccionar features
            X = df_with_features[self.selected_features]
            
            # Manejar NaN e infinitos
            X = X.fillna(0)
            X = X.replace([np.inf, -np.inf], 0)
            
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
            return jsonify({"error": "Datos de landmarks requeridos"}), 400
        
        landmarks = data['landmarks']
        
        if not isinstance(landmarks, list) or len(landmarks) == 0:
            return jsonify({"error": "Landmarks inválidos"}), 400
        
        # Realizar predicción
        result = predictor.predict(landmarks)
        
        if "error" in result:
            return jsonify(result), 400
        
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