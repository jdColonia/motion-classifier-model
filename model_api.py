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
import re

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

            # Extraer la versión del nombre del archivo
            model_name = latest_model_file.stem
            version_match = re.search(r"(v\d+\.\d+_\d{8}_\d{6})", model_name)
            if version_match:
                version = version_match.group(1)
            else:
                # Fallback: usar las últimas partes del nombre
                parts = model_name.split("_")
                version = "_".join(parts[-3:])

            logger.info(f"Cargando modelo versión: {version}")

            # Cargar componentes
            self.model = joblib.load(latest_model_file)
            self.scaler = joblib.load(self.models_dir / f"scaler_{version}.joblib")
            self.label_encoder = joblib.load(
                self.models_dir / f"label_encoder_{version}.joblib"
            )

            # Cargar metadatos
            metadata_file = self.models_dir / f"metadata_{version}.json"
            with open(metadata_file, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

            # Usar las features del metadata (modelo más reciente)
            self.selected_features = self.metadata["feature_info"]["selected_features"]

            logger.info(f"✅ Modelo cargado exitosamente")
            logger.info(f"   • Algoritmo: {self.metadata['model_info']['name']}")
            logger.info(f"   • Accuracy: {self.metadata['model_info']['accuracy']:.4f}")
            logger.info(f"   • Features: {len(self.selected_features)}")

        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            raise

    def landmarks_to_wide_format(self, landmarks_data):
        """
        Convertir landmarks de JavaScript al formato wide que espera el modelo
        Esto simula el formato que sale del EDA procesado
        """
        try:
            logger.info(f"Convirtiendo {len(landmarks_data)} landmarks al formato wide")

            # Crear una fila en formato wide (una fila por frame)
            frame_data = {"frame": 0}  # Frame único para predicción en tiempo real

            # Mapeo de índices de MediaPipe a nombres de landmarks del dataset
            # Basado en el script generado automáticamente
            landmark_mapping = {
                0: "nose",
                11: "left_shoulder",
                12: "right_shoulder",
                13: "left_elbow",
                14: "right_elbow",
                15: "left_wrist",
                16: "right_wrist",
                23: "left_hip",
                24: "right_hip",
                25: "left_knee",
                26: "right_knee",
                27: "left_ankle",
                28: "right_ankle",
                29: "left_heel",
                30: "right_heel",  # Agregados según el script generado
                31: "left_foot_index",
                32: "right_foot_index",
            }

            # Convertir landmarks a formato wide
            for landmark in landmarks_data:
                if not isinstance(landmark, dict):
                    continue

                landmark_idx = landmark.get("landmark_index")
                if landmark_idx in landmark_mapping:
                    landmark_name = landmark_mapping[landmark_idx]

                    # Agregar coordenadas y visibilidad
                    frame_data[f"{landmark_name}_x"] = landmark.get("x", 0.0)
                    frame_data[f"{landmark_name}_y"] = landmark.get("y", 0.0)
                    frame_data[f"{landmark_name}_z"] = landmark.get("z", 0.0)
                    frame_data[f"{landmark_name}_v"] = landmark.get("visibility", 0.5)

            # Crear DataFrame con una sola fila
            df = pd.DataFrame([frame_data])

            logger.info(f"DataFrame wide creado con {len(df.columns)} columnas")
            return df

        except Exception as e:
            logger.error(f"Error convirtiendo landmarks: {e}")
            raise

    def create_biomechanical_features(self, df):
        """
        Crear TODAS las features biomecánicas que el modelo espera (104 features)
        """
        try:
            df_enhanced = df.copy()

            def safe_get_coords(row, landmark_name, coord):
                col_name = f"{landmark_name}_{coord}"
                if col_name in row.index and pd.notna(row[col_name]):
                    return row[col_name]
                return 0.0

            def calculate_angle_3_points(p1, p2, p3):
                v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
                v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
                cos_angle = np.dot(v1, v2) / (
                    np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8
                )
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)
                return np.degrees(angle)

            def calculate_distance(p1, p2):
                return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

            # Para cada fila del DataFrame
            for idx, row in df_enhanced.iterrows():

                # 1. ÁNGULOS BIOMECÁNICOS
                # Ángulo rodilla izquierda
                try:
                    hip_left = (
                        safe_get_coords(row, "left_hip", "x"),
                        safe_get_coords(row, "left_hip", "y"),
                    )
                    knee_left = (
                        safe_get_coords(row, "left_knee", "x"),
                        safe_get_coords(row, "left_knee", "y"),
                    )
                    ankle_left = (
                        safe_get_coords(row, "left_ankle", "x"),
                        safe_get_coords(row, "left_ankle", "y"),
                    )

                    if all(
                        coord != 0.0
                        for coord in [
                            hip_left[0],
                            hip_left[1],
                            knee_left[0],
                            knee_left[1],
                            ankle_left[0],
                            ankle_left[1],
                        ]
                    ):
                        angle = calculate_angle_3_points(
                            hip_left, knee_left, ankle_left
                        )
                        df_enhanced.loc[idx, "angle_left_knee"] = angle
                    else:
                        df_enhanced.loc[idx, "angle_left_knee"] = 90.0
                except:
                    df_enhanced.loc[idx, "angle_left_knee"] = 90.0

                # Ángulo rodilla derecha
                try:
                    hip_right = (
                        safe_get_coords(row, "right_hip", "x"),
                        safe_get_coords(row, "right_hip", "y"),
                    )
                    knee_right = (
                        safe_get_coords(row, "right_knee", "x"),
                        safe_get_coords(row, "right_knee", "y"),
                    )
                    ankle_right = (
                        safe_get_coords(row, "right_ankle", "x"),
                        safe_get_coords(row, "right_ankle", "y"),
                    )

                    if all(
                        coord != 0.0
                        for coord in [
                            hip_right[0],
                            hip_right[1],
                            knee_right[0],
                            knee_right[1],
                            ankle_right[0],
                            ankle_right[1],
                        ]
                    ):
                        angle = calculate_angle_3_points(
                            hip_right, knee_right, ankle_right
                        )
                        df_enhanced.loc[idx, "angle_right_knee"] = angle
                    else:
                        df_enhanced.loc[idx, "angle_right_knee"] = 90.0
                except:
                    df_enhanced.loc[idx, "angle_right_knee"] = 90.0

                # Inclinación del tronco
                try:
                    shoulder_left = (
                        safe_get_coords(row, "left_shoulder", "x"),
                        safe_get_coords(row, "left_shoulder", "y"),
                    )
                    shoulder_right = (
                        safe_get_coords(row, "right_shoulder", "x"),
                        safe_get_coords(row, "right_shoulder", "y"),
                    )
                    hip_left = (
                        safe_get_coords(row, "left_hip", "x"),
                        safe_get_coords(row, "left_hip", "y"),
                    )
                    hip_right = (
                        safe_get_coords(row, "right_hip", "x"),
                        safe_get_coords(row, "right_hip", "y"),
                    )

                    if all(
                        coord != 0.0
                        for coord in [
                            shoulder_left[0],
                            shoulder_right[0],
                            hip_left[0],
                            hip_right[0],
                        ]
                    ):
                        shoulder_center = (
                            (shoulder_left[0] + shoulder_right[0]) / 2,
                            (shoulder_left[1] + shoulder_right[1]) / 2,
                        )
                        hip_center = (
                            (hip_left[0] + hip_right[0]) / 2,
                            (hip_left[1] + hip_right[1]) / 2,
                        )

                        trunk_vector = [
                            shoulder_center[0] - hip_center[0],
                            shoulder_center[1] - hip_center[1],
                        ]
                        vertical_vector = [0, 1]

                        dot_product = (
                            trunk_vector[0] * vertical_vector[0]
                            + trunk_vector[1] * vertical_vector[1]
                        )
                        trunk_magnitude = (
                            np.sqrt(trunk_vector[0] ** 2 + trunk_vector[1] ** 2) + 1e-8
                        )

                        cos_angle = dot_product / trunk_magnitude
                        cos_angle = np.clip(cos_angle, -1, 1)
                        trunk_angle = np.degrees(np.arccos(cos_angle))

                        df_enhanced.loc[idx, "trunk_inclination"] = trunk_angle
                    else:
                        df_enhanced.loc[idx, "trunk_inclination"] = 0.0
                except:
                    df_enhanced.loc[idx, "trunk_inclination"] = 0.0

                # 2. DISTANCIAS CORPORALES
                # Ancho de hombros
                try:
                    shoulder_left = (
                        safe_get_coords(row, "left_shoulder", "x"),
                        safe_get_coords(row, "left_shoulder", "y"),
                    )
                    shoulder_right = (
                        safe_get_coords(row, "right_shoulder", "x"),
                        safe_get_coords(row, "right_shoulder", "y"),
                    )

                    if all(
                        coord != 0.0 for coord in [shoulder_left[0], shoulder_right[0]]
                    ):
                        width = calculate_distance(shoulder_left, shoulder_right)
                        df_enhanced.loc[idx, "shoulder_width"] = width
                    else:
                        df_enhanced.loc[idx, "shoulder_width"] = 0.0
                except:
                    df_enhanced.loc[idx, "shoulder_width"] = 0.0

                # Ancho de caderas
                try:
                    hip_left = (
                        safe_get_coords(row, "left_hip", "x"),
                        safe_get_coords(row, "left_hip", "y"),
                    )
                    hip_right = (
                        safe_get_coords(row, "right_hip", "x"),
                        safe_get_coords(row, "right_hip", "y"),
                    )

                    if all(coord != 0.0 for coord in [hip_left[0], hip_right[0]]):
                        width = calculate_distance(hip_left, hip_right)
                        df_enhanced.loc[idx, "hip_width"] = width
                    else:
                        df_enhanced.loc[idx, "hip_width"] = 0.0
                except:
                    df_enhanced.loc[idx, "hip_width"] = 0.0

                # Altura del torso
                try:
                    nose = (
                        safe_get_coords(row, "nose", "x"),
                        safe_get_coords(row, "nose", "y"),
                    )
                    hip_left = (
                        safe_get_coords(row, "left_hip", "x"),
                        safe_get_coords(row, "left_hip", "y"),
                    )
                    hip_right = (
                        safe_get_coords(row, "right_hip", "x"),
                        safe_get_coords(row, "right_hip", "y"),
                    )

                    if all(
                        coord != 0.0 for coord in [nose[0], hip_left[0], hip_right[0]]
                    ):
                        hip_center = (
                            (hip_left[0] + hip_right[0]) / 2,
                            (hip_left[1] + hip_right[1]) / 2,
                        )
                        height = calculate_distance(nose, hip_center)
                        df_enhanced.loc[idx, "torso_height"] = height
                    else:
                        df_enhanced.loc[idx, "torso_height"] = 0.0
                except:
                    df_enhanced.loc[idx, "torso_height"] = 0.0

                # 3. VELOCIDADES (para tiempo real, usar 0)
                joints = [
                    "nose",
                    "left_shoulder",
                    "right_shoulder",
                    "left_hip",
                    "right_hip",
                    "left_knee",
                    "right_knee",
                    "left_ankle",
                    "right_ankle",
                ]

                for joint in joints:
                    df_enhanced.loc[idx, f"{joint}_vel_x"] = 0.0
                    df_enhanced.loc[idx, f"{joint}_vel_y"] = 0.0
                    df_enhanced.loc[idx, f"{joint}_vel_magnitude"] = 0.0
                    df_enhanced.loc[idx, f"{joint}_acceleration"] = 0.0

                # 4. CARACTERÍSTICAS ESPECÍFICAS
                df_enhanced.loc[idx, "shoulder_velocity_diff"] = 0.0
                df_enhanced.loc[idx, "hip_velocity_diff"] = 0.0
                df_enhanced.loc[idx, "hip_center_y"] = (
                    safe_get_coords(row, "left_hip", "y")
                    + safe_get_coords(row, "right_hip", "y")
                ) / 2
                df_enhanced.loc[idx, "hip_height_change"] = 0.0
                df_enhanced.loc[idx, "ankle_velocity_ratio"] = 1.0
                df_enhanced.loc[idx, "leg_coordination"] = 0.0
                df_enhanced.loc[idx, "knee_angle_symmetry"] = abs(
                    df_enhanced.loc[idx, "angle_left_knee"]
                    - df_enhanced.loc[idx, "angle_right_knee"]
                )

                # 5. CARACTERÍSTICAS DE VISIBILIDAD
                try:
                    upper_body_vis = [
                        safe_get_coords(row, "nose", "v"),
                        safe_get_coords(row, "left_shoulder", "v"),
                        safe_get_coords(row, "right_shoulder", "v"),
                    ]
                    lower_body_vis = [
                        safe_get_coords(row, "left_hip", "v"),
                        safe_get_coords(row, "right_hip", "v"),
                        safe_get_coords(row, "left_knee", "v"),
                        safe_get_coords(row, "right_knee", "v"),
                        safe_get_coords(row, "left_ankle", "v"),
                        safe_get_coords(row, "right_ankle", "v"),
                    ]

                    df_enhanced.loc[idx, "upper_body_visibility"] = (
                        np.mean([v for v in upper_body_vis if v != 0.0])
                        if any(v != 0.0 for v in upper_body_vis)
                        else 0.5
                    )
                    df_enhanced.loc[idx, "lower_body_visibility"] = (
                        np.mean([v for v in lower_body_vis if v != 0.0])
                        if any(v != 0.0 for v in lower_body_vis)
                        else 0.5
                    )

                    all_vis = upper_body_vis + lower_body_vis
                    df_enhanced.loc[idx, "visibility_stability"] = (
                        np.std([v for v in all_vis if v != 0.0])
                        if len([v for v in all_vis if v != 0.0]) > 1
                        else 0.0
                    )
                except:
                    df_enhanced.loc[idx, "upper_body_visibility"] = 0.5
                    df_enhanced.loc[idx, "lower_body_visibility"] = 0.5
                    df_enhanced.loc[idx, "visibility_stability"] = 0.0

            logger.info(
                f"Features biomecánicas creadas. Shape final: {df_enhanced.shape}"
            )
            return df_enhanced

        except Exception as e:
            logger.error(f"Error creando features biomecánicas: {e}")
            raise

    def predict(self, landmarks_data):
        """
        Predecir movimiento a partir de landmarks
        """
        try:
            # 1. Convertir a formato wide
            df_wide = self.landmarks_to_wide_format(landmarks_data)

            if len(df_wide) == 0:
                return {"error": "No hay landmarks válidos"}

            # 2. Crear características biomecánicas
            logger.info("Creando características biomecánicas...")
            df_with_features = self.create_biomechanical_features(df_wide)

            # 3. Verificar features disponibles vs requeridas
            available_features = set(df_with_features.columns)
            required_features = set(self.selected_features)
            missing_features = required_features - available_features

            if missing_features:
                logger.warning(
                    f"Features faltantes ({len(missing_features)}): {list(missing_features)[:10]}..."
                )
                # Agregar features faltantes con valor por defecto
                for feature in missing_features:
                    df_with_features[feature] = 0.0

            # 4. Seleccionar solo las features que el modelo necesita
            X = df_with_features[self.selected_features]

            logger.info(f"Shape de X: {X.shape}")

            # 5. Limpiar datos
            X = X.fillna(0)
            X = X.replace([np.inf, -np.inf], 0)

            # 6. Escalar
            X_scaled = self.scaler.transform(X)

            # 7. Predecir
            prediction_encoded = self.model.predict(X_scaled)[0]
            prediction = self.label_encoder.inverse_transform([prediction_encoded])[0]

            # 8. Obtener probabilidades
            probabilities = {}
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(X_scaled)[0]
                for i, class_name in enumerate(self.label_encoder.classes_):
                    probabilities[class_name] = float(proba[i])

            confidence = max(probabilities.values()) if probabilities else 0.0

            logger.info(f"Predicción: {prediction} (confianza: {confidence:.3f})")

            return {
                "prediction": prediction,
                "probabilities": probabilities,
                "confidence": confidence,
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


@app.route("/health", methods=["GET"])
def health_check():
    """Endpoint para verificar el estado del servidor"""
    if predictor and predictor.model:
        return jsonify(
            {
                "status": "healthy",
                "model_loaded": True,
                "model_info": (
                    predictor.metadata["model_info"] if predictor.metadata else None
                ),
            }
        )
    else:
        return (
            jsonify(
                {"status": "error", "model_loaded": False, "error": "Modelo no cargado"}
            ),
            500,
        )


@app.route("/predict", methods=["POST"])
def predict_movement():
    """Endpoint para predecir movimiento desde landmarks"""
    if not predictor:
        return jsonify({"error": "Modelo no disponible"}), 500

    try:
        data = request.get_json()

        if not data or "landmarks" not in data:
            logger.error("Datos de landmarks faltantes en request")
            return jsonify({"error": "Datos de landmarks requeridos"}), 400

        landmarks = data["landmarks"]

        if not isinstance(landmarks, list) or len(landmarks) == 0:
            logger.error(f"Landmarks inválidos")
            return jsonify({"error": "Landmarks inválidos"}), 400

        logger.info(f"Recibidos {len(landmarks)} landmarks para predicción")

        # Realizar predicción
        result = predictor.predict(landmarks)

        if "error" in result:
            logger.error(f"Error en predicción: {result['error']}")
            return jsonify(result), 400

        logger.info(
            f"Predicción exitosa: {result['prediction']} (confianza: {result['confidence']:.3f})"
        )
        return jsonify(result)

    except Exception as e:
        logger.error(f"Error en endpoint predict: {e}")
        return jsonify({"error": "Error interno del servidor"}), 500


@app.route("/model-info", methods=["GET"])
def get_model_info():
    """Endpoint para obtener información del modelo"""
    if not predictor or not predictor.metadata:
        return jsonify({"error": "Modelo no disponible"}), 500

    return jsonify(
        {
            "model_info": predictor.metadata["model_info"],
            "classes": predictor.metadata["training_info"]["classes"],
            "features_count": len(predictor.selected_features),
            "accuracy": predictor.metadata["performance_metrics"]["accuracy"],
        }
    )


if __name__ == "__main__":
    if predictor:
        print("🚀 Iniciando API del clasificador de movimientos...")
        print(f"📊 Modelo: {predictor.metadata['model_info']['name']}")
        print(f"🎯 Accuracy: {predictor.metadata['model_info']['accuracy']:.4f}")
        print(f"🔗 API disponible en: http://localhost:5000")
        print("\n📋 Endpoints disponibles:")
        print("   • GET  /health     - Estado del servidor")
        print("   • POST /predict    - Clasificar movimiento")
        print("   • GET  /model-info - Información del modelo")

        app.run(host="0.0.0.0", port=5000, debug=False)
    else:
        print("❌ No se pudo inicializar el predictor")
