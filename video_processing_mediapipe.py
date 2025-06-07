#!/usr/bin/env python3
"""
Procesamiento de Videos con MediaPipe

Este script procesa videos de movimientos corporales y extrae datos de seguimiento de poses usando MediaPipe.
Los datos se almacenan en formato CSV con UNA FILA POR FRAME y una columna para cada landmark.

Formato de salida:
frame, nose_x, nose_y, nose_z, nose_v, left_eye_inner_x, left_eye_inner_y, ..., right_foot_index_v

Estructura de datos:
- Entrada: data/raw/{nombre_accion}/ - Videos de diferentes acciones
- Salida: data/processed/{nombre_accion}/ - Archivos CSV con datos de poses
"""

import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm
import time


class PoseExtractor:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Inicializa el extractor de poses con MediaPipe.

        Args:
            min_detection_confidence: Confianza mínima para detectar poses
            min_tracking_confidence: Confianza mínima para rastrear poses
        """
        self.pose = mp.solutions.pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        # Nombres de los 33 landmarks de MediaPipe Pose
        self.landmark_names = [
            "nose",
            "left_eye_inner",
            "left_eye",
            "left_eye_outer",
            "right_eye_inner",
            "right_eye",
            "right_eye_outer",
            "left_ear",
            "right_ear",
            "mouth_left",
            "mouth_right",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_pinky",
            "right_pinky",
            "left_index",
            "right_index",
            "left_thumb",
            "right_thumb",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
            "left_heel",
            "right_heel",
            "left_foot_index",
            "right_foot_index",
        ]

        # Generar nombres de columnas para el CSV
        self.column_names = ["frame"]
        for landmark_name in self.landmark_names:
            self.column_names.extend(
                [
                    f"{landmark_name}_x",
                    f"{landmark_name}_y",
                    f"{landmark_name}_z",
                    f"{landmark_name}_v",  # visibility
                ]
            )

    def extract_pose_landmarks(self, image):
        """
        Extrae landmarks de pose de una imagen.

        Args:
            image: Imagen en formato BGR

        Returns:
            dict: Diccionario con coordenadas de landmarks o None si no se detecta pose
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks_data = {}

            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                landmark_name = self.landmark_names[idx]
                landmarks_data[f"{landmark_name}_x"] = landmark.x
                landmarks_data[f"{landmark_name}_y"] = landmark.y
                landmarks_data[f"{landmark_name}_z"] = landmark.z
                landmarks_data[f"{landmark_name}_v"] = landmark.visibility

            return landmarks_data
        else:
            return None

    def process_video(self, video_path, action_name):
        """
        Procesa un video completo y extrae poses de todos los frames.

        Args:
            video_path: Ruta al archivo de video
            action_name: Nombre de la acción para almacenar en los datos

        Returns:
            DataFrame: Datos con una fila por frame y columnas para cada landmark
        """
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print(f"Error: No se pudo abrir el video {video_path}")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Procesando: {video_path.name}")
        print(f"FPS: {fps}, Total frames: {total_frames}")

        # Lista para almacenar datos de cada frame
        frames_data = []
        frame_number = 0

        # Crear barra de progreso
        with tqdm(total=total_frames, desc="Procesando frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Extraer landmarks para este frame
                landmarks_data = self.extract_pose_landmarks(frame)

                # Crear fila de datos para este frame
                frame_data = {"frame": frame_number}

                if landmarks_data is not None:
                    # Si se detectaron poses, agregar todos los landmarks
                    frame_data.update(landmarks_data)
                else:
                    # Si no se detectaron poses, dejar todos los campos de landmarks vacíos
                    for landmark_name in self.landmark_names:
                        frame_data[f"{landmark_name}_x"] = ""
                        frame_data[f"{landmark_name}_y"] = ""
                        frame_data[f"{landmark_name}_z"] = ""
                        frame_data[f"{landmark_name}_v"] = ""

                frames_data.append(frame_data)
                frame_number += 1
                pbar.update(1)

        cap.release()

        # Crear DataFrame
        df = pd.DataFrame(frames_data)

        # Reordenar columnas para asegurar el orden correcto
        df = df[self.column_names]

        # Estadísticas de detección
        frames_with_pose = df[df[f"{self.landmark_names[0]}_x"] != ""].shape[0]
        detection_rate = (
            (frames_with_pose / total_frames) * 100 if total_frames > 0 else 0
        )

        print(f"✓ Procesado exitosamente")
        print(f"  Frames totales: {total_frames}")
        print(f"  Frames con pose detectada: {frames_with_pose}")
        print(f"  Tasa de detección: {detection_rate:.1f}%")

        return df

    def save_csv(self, df, output_dir, video_name):
        """
        Guarda los datos en formato CSV.

        Args:
            df: DataFrame con los datos
            output_dir: Directorio de salida
            video_name: Nombre base del video

        Returns:
            str: Ruta del archivo CSV guardado
        """
        os.makedirs(output_dir, exist_ok=True)

        # Guardar CSV
        csv_path = os.path.join(output_dir, f"{video_name}_poses.csv")
        df.to_csv(csv_path, index=False)

        # Mostrar información del archivo generado
        print(f"Datos guardados en: {csv_path}")
        print(f"  Dimensiones: {df.shape[0]} frames × {df.shape[1]} columnas")
        print(
            f"  Columnas: frame + {len(self.landmark_names)} landmarks × 4 coordenadas"
        )

        return csv_path


def process_all_videos(raw_data_path, processed_data_path):
    """
    Procesa todos los videos en las carpetas de acciones.

    Args:
        raw_data_path: Ruta a los datos sin procesar
        processed_data_path: Ruta donde guardar los datos procesados

    Returns:
        dict: Resumen del procesamiento
    """
    # Extensiones de video soportadas
    video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"]

    # Inicializar extractor de poses
    pose_extractor = PoseExtractor()

    # Obtener todas las carpetas de acciones
    action_folders = [
        d
        for d in os.listdir(raw_data_path)
        if os.path.isdir(os.path.join(raw_data_path, d))
    ]

    print(f"Acciones encontradas: {action_folders}")

    processing_summary = {}

    for action in action_folders:
        print(f"\n{'='*50}")
        print(f"Procesando acción: {action}")
        print(f"{'='*50}")

        action_raw_path = os.path.join(raw_data_path, action)
        action_processed_path = os.path.join(processed_data_path, action)

        # Crear directorio de salida para esta acción
        os.makedirs(action_processed_path, exist_ok=True)

        # Encontrar todos los videos en esta carpeta de acción
        video_files = []
        for file in os.listdir(action_raw_path):
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_files.append(file)

        print(f"Videos encontrados: {len(video_files)}")

        action_results = []

        for video_file in video_files:
            video_path = Path(os.path.join(action_raw_path, video_file))
            video_name = video_path.stem  # Nombre sin extensión

            print(f"\nProcesando: {video_file}")

            try:
                # Procesar video
                df = pose_extractor.process_video(video_path, action)

                if df is not None and not df.empty:
                    # Guardar datos
                    csv_path = pose_extractor.save_csv(
                        df, action_processed_path, video_name
                    )

                    # Estadísticas
                    total_frames = len(df)
                    frames_with_data = df[
                        df[f"{pose_extractor.landmark_names[0]}_x"] != ""
                    ].shape[0]

                    result = {
                        "video_file": video_file,
                        "total_frames": total_frames,
                        "frames_with_pose": frames_with_data,
                        "detection_rate": (
                            (frames_with_data / total_frames) * 100
                            if total_frames > 0
                            else 0
                        ),
                        "csv_path": csv_path,
                        "status": "success",
                    }

                else:
                    result = {
                        "video_file": video_file,
                        "status": "error",
                        "error": "No se pudo procesar el video o datos vacíos",
                    }
                    print(f"✗ Error al procesar el video")

            except Exception as e:
                result = {"video_file": video_file, "status": "error", "error": str(e)}
                print(f"✗ Error: {e}")

            action_results.append(result)

        processing_summary[action] = action_results

    return processing_summary


def main():
    """Función principal para ejecutar el procesamiento de videos."""

    # Definir rutas del proyecto
    PROJECT_PATH = os.getcwd()  # Usar directorio actual como base
    RAW_DATA_PATH = os.path.join(PROJECT_PATH, "data", "raw")
    PROCESSED_DATA_PATH = os.path.join(PROJECT_PATH, "data", "processed")

    # Crear directorio processed si no existe
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

    print(f"Iniciando procesamiento de videos...")
    print(f"Ruta del proyecto: {PROJECT_PATH}")
    print(f"Datos raw: {RAW_DATA_PATH}")
    print(f"Datos procesados: {PROCESSED_DATA_PATH}")

    # Verificar que existe el directorio de datos raw
    if not os.path.exists(RAW_DATA_PATH):
        print(f"Error: El directorio {RAW_DATA_PATH} no existe.")
        print("Por favor, asegúrate de que los videos estén organizados en:")
        print("  data/raw/{nombre_movimiento}/video1.mp4")
        print("  data/raw/{nombre_movimiento}/video2.mp4")
        print("  ...")
        return

    # Procesar todos los videos
    start_time = time.time()
    summary = process_all_videos(RAW_DATA_PATH, PROCESSED_DATA_PATH)
    end_time = time.time()

    # Mostrar resumen final
    print(f"\n{'='*60}")
    print("RESUMEN DEL PROCESAMIENTO")
    print(f"{'='*60}")
    print(f"Tiempo total: {end_time - start_time:.1f} segundos")

    total_videos = 0
    successful_videos = 0
    total_frames = 0
    total_frames_with_pose = 0

    for action, results in summary.items():
        action_success = sum(1 for r in results if r["status"] == "success")
        action_total = len(results)
        action_frames = sum(
            r.get("total_frames", 0) for r in results if r["status"] == "success"
        )
        action_frames_with_pose = sum(
            r.get("frames_with_pose", 0) for r in results if r["status"] == "success"
        )

        print(f"\n{action}:")
        print(f"  Videos procesados: {action_success}/{action_total}")
        print(f"  Frames totales: {action_frames}")
        print(f"  Frames con pose: {action_frames_with_pose}")
        if action_frames > 0:
            detection_rate = (action_frames_with_pose / action_frames) * 100
            print(f"  Tasa de detección: {detection_rate:.1f}%")

        total_videos += action_total
        successful_videos += action_success
        total_frames += action_frames
        total_frames_with_pose += action_frames_with_pose

    print(f"\nTotal general:")
    print(f"  Videos: {successful_videos}/{total_videos} procesados exitosamente")
    print(
        f"  Tasa de éxito: {(successful_videos/total_videos)*100:.1f}%"
        if total_videos > 0
        else "Tasa de éxito: 0%"
    )
    print(f"  Frames: {total_frames_with_pose}/{total_frames} con pose detectada")
    print(
        f"  Detección global: {(total_frames_with_pose/total_frames)*100:.1f}%"
        if total_frames > 0
        else "Detección: 0%"
    )


if __name__ == "__main__":
    main()
