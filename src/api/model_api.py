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

# Mapeo de índices MediaPipe a nombres de landmarks
LANDMARK_MAPPING = {
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
    30: "right_heel",
    31: "left_foot_index",
    32: "right_foot_index",
}

# Articulaciones principales para cálculo de velocidades
MAIN_JOINTS = [
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


class BiomechanicalCalculator:
    """Clase auxiliar para cálculos biomecánicos"""

    @staticmethod
    def safe_get_coords(row, landmark_name, coord):
        """Obtener coordenadas de manera segura"""
        col_name = f"{landmark_name}_{coord}"
        if col_name in row.index and pd.notna(row[col_name]):
            return row[col_name]
        return 0.0

    @staticmethod
    def calculate_angle_3_points(p1, p2, p3):
        """Calcular ángulo entre 3 puntos"""
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1, 1)
        return np.degrees(np.arccos(cos_angle))

    @staticmethod
    def calculate_distance(p1, p2):
        """Calcular distancia euclidiana entre dos puntos"""
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    @classmethod
    def get_point_coords(cls, row, landmark_name):
        """Obtener coordenadas (x, y) de un landmark"""
        return (
            cls.safe_get_coords(row, landmark_name, "x"),
            cls.safe_get_coords(row, landmark_name, "y"),
        )


class MovementPredictor:
    def __init__(self, models_dir=None):
        """Inicializar el predictor cargando todos los componentes del modelo"""
        if models_dir is None:
            # Encontrar el directorio raíz del proyecto (tres niveles arriba desde src/api/)
            project_root = Path(__file__).parent.parent.parent
            models_dir = project_root / "notebooks" / "models"
        self.models_dir = Path(models_dir)
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.metadata = None
        self.selected_features = None
        self.calc = BiomechanicalCalculator()

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
            version = self._extract_version(model_name)

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

    def _extract_version(self, model_name):
        """Extraer versión del nombre del modelo"""
        version_match = re.search(r"(v\d+\.\d+)", model_name)
        if version_match:
            return version_match.group(1)

        parts = model_name.split("_")
        return "_".join(parts[-1:])

    def landmarks_to_wide_format(self, landmarks_data):
        """
        Convertir landmarks de JavaScript al formato wide que espera el modelo
        Esto simula el formato que sale del EDA procesado
        """
        try:
            logger.info(f"Convirtiendo {len(landmarks_data)} landmarks al formato wide")

            # Crear una fila en formato wide (una fila por frame)
            frame_data = {"frame": 0}  # Frame único para predicción en tiempo real

            # Convertir landmarks a formato wide
            for landmark in landmarks_data:
                if not isinstance(landmark, dict):
                    continue

                landmark_idx = landmark.get("landmark_index")
                if landmark_idx in LANDMARK_MAPPING:
                    landmark_name = LANDMARK_MAPPING[landmark_idx]

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

    def _calculate_joint_angles(self, df_enhanced, idx, row):
        """Calcular ángulos de las articulaciones"""
        # Ángulo rodilla izquierda
        try:
            hip_left = self.calc.get_point_coords(row, "left_hip")
            knee_left = self.calc.get_point_coords(row, "left_knee")
            ankle_left = self.calc.get_point_coords(row, "left_ankle")

            if all(
                coord != 0.0
                for point in [hip_left, knee_left, ankle_left]
                for coord in point
            ):
                angle = self.calc.calculate_angle_3_points(
                    hip_left, knee_left, ankle_left
                )
                df_enhanced.loc[idx, "angle_left_knee"] = angle
            else:
                df_enhanced.loc[idx, "angle_left_knee"] = 90.0
        except:
            df_enhanced.loc[idx, "angle_left_knee"] = 90.0

        # Ángulo rodilla derecha
        try:
            hip_right = self.calc.get_point_coords(row, "right_hip")
            knee_right = self.calc.get_point_coords(row, "right_knee")
            ankle_right = self.calc.get_point_coords(row, "right_ankle")

            if all(
                coord != 0.0
                for point in [hip_right, knee_right, ankle_right]
                for coord in point
            ):
                angle = self.calc.calculate_angle_3_points(
                    hip_right, knee_right, ankle_right
                )
                df_enhanced.loc[idx, "angle_right_knee"] = angle
            else:
                df_enhanced.loc[idx, "angle_right_knee"] = 90.0
        except:
            df_enhanced.loc[idx, "angle_right_knee"] = 90.0

    def _calculate_trunk_inclination(self, df_enhanced, idx, row):
        """Calcular inclinación del tronco"""
        try:
            shoulder_left = self.calc.get_point_coords(row, "left_shoulder")
            shoulder_right = self.calc.get_point_coords(row, "right_shoulder")
            hip_left = self.calc.get_point_coords(row, "left_hip")
            hip_right = self.calc.get_point_coords(row, "right_hip")

            if all(
                point[0] != 0.0
                for point in [shoulder_left, shoulder_right, hip_left, hip_right]
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

    def _calculate_body_distances(self, df_enhanced, idx, row):
        """Calcular distancias corporales"""
        # Ancho de hombros
        try:
            shoulder_left = self.calc.get_point_coords(row, "left_shoulder")
            shoulder_right = self.calc.get_point_coords(row, "right_shoulder")

            if shoulder_left[0] != 0.0 and shoulder_right[0] != 0.0:
                width = self.calc.calculate_distance(shoulder_left, shoulder_right)
                df_enhanced.loc[idx, "shoulder_width"] = width
            else:
                df_enhanced.loc[idx, "shoulder_width"] = 0.0
        except:
            df_enhanced.loc[idx, "shoulder_width"] = 0.0

        # Ancho de caderas
        try:
            hip_left = self.calc.get_point_coords(row, "left_hip")
            hip_right = self.calc.get_point_coords(row, "right_hip")

            if hip_left[0] != 0.0 and hip_right[0] != 0.0:
                width = self.calc.calculate_distance(hip_left, hip_right)
                df_enhanced.loc[idx, "hip_width"] = width
            else:
                df_enhanced.loc[idx, "hip_width"] = 0.0
        except:
            df_enhanced.loc[idx, "hip_width"] = 0.0

        # Altura del torso
        try:
            nose = self.calc.get_point_coords(row, "nose")
            hip_left = self.calc.get_point_coords(row, "left_hip")
            hip_right = self.calc.get_point_coords(row, "right_hip")

            if all(point[0] != 0.0 for point in [nose, hip_left, hip_right]):
                hip_center = (
                    (hip_left[0] + hip_right[0]) / 2,
                    (hip_left[1] + hip_right[1]) / 2,
                )
                height = self.calc.calculate_distance(nose, hip_center)
                df_enhanced.loc[idx, "torso_height"] = height
            else:
                df_enhanced.loc[idx, "torso_height"] = 0.0
        except:
            df_enhanced.loc[idx, "torso_height"] = 0.0

    def _add_velocity_features(self, df_enhanced, idx):
        """Agregar características de velocidad (cero para tiempo real)"""
        for joint in MAIN_JOINTS:
            df_enhanced.loc[idx, f"{joint}_vel_x"] = 0.0
            df_enhanced.loc[idx, f"{joint}_vel_y"] = 0.0
            df_enhanced.loc[idx, f"{joint}_vel_magnitude"] = 0.0
            df_enhanced.loc[idx, f"{joint}_acceleration"] = 0.0

    def _add_specific_features(self, df_enhanced, idx, row):
        """Agregar características específicas del modelo"""
        df_enhanced.loc[idx, "shoulder_velocity_diff"] = 0.0
        df_enhanced.loc[idx, "hip_velocity_diff"] = 0.0
        df_enhanced.loc[idx, "hip_center_y"] = (
            self.calc.safe_get_coords(row, "left_hip", "y")
            + self.calc.safe_get_coords(row, "right_hip", "y")
        ) / 2
        df_enhanced.loc[idx, "hip_height_change"] = 0.0
        df_enhanced.loc[idx, "ankle_velocity_ratio"] = 1.0
        df_enhanced.loc[idx, "leg_coordination"] = 0.0
        df_enhanced.loc[idx, "knee_angle_symmetry"] = abs(
            df_enhanced.loc[idx, "angle_left_knee"]
            - df_enhanced.loc[idx, "angle_right_knee"]
        )

    def _add_visibility_features(self, df_enhanced, idx, row):
        """Agregar características de visibilidad"""
        try:
            upper_body_vis = [
                self.calc.safe_get_coords(row, landmark, "v")
                for landmark in ["nose", "left_shoulder", "right_shoulder"]
            ]
            lower_body_vis = [
                self.calc.safe_get_coords(row, landmark, "v")
                for landmark in [
                    "left_hip",
                    "right_hip",
                    "left_knee",
                    "right_knee",
                    "left_ankle",
                    "right_ankle",
                ]
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
            valid_vis = [v for v in all_vis if v != 0.0]
            df_enhanced.loc[idx, "visibility_stability"] = (
                np.std(valid_vis) if len(valid_vis) > 1 else 0.0
            )
        except:
            df_enhanced.loc[idx, "upper_body_visibility"] = 0.5
            df_enhanced.loc[idx, "lower_body_visibility"] = 0.5
            df_enhanced.loc[idx, "visibility_stability"] = 0.0

    def create_biomechanical_features(self, df):
        """
        Crear TODAS las features biomecánicas que el modelo espera (104 features)
        """
        try:
            df_enhanced = df.copy()

            for idx, row in df_enhanced.iterrows():
                # Calcular todas las características
                self._calculate_joint_angles(df_enhanced, idx, row)
                self._calculate_trunk_inclination(df_enhanced, idx, row)
                self._calculate_body_distances(df_enhanced, idx, row)
                self._add_velocity_features(df_enhanced, idx)
                self._add_specific_features(df_enhanced, idx, row)
                self._add_visibility_features(df_enhanced, idx, row)

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
