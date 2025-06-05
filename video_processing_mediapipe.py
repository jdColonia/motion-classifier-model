#!/usr/bin/env python3
"""
Procesamiento de Videos con MediaPipe

Este script procesa videos de movimientos corporales y extrae datos de seguimiento de poses usando MediaPipe.
Los datos se almacenan en formato CSV con las columnas: frame, landmark_index, x, y, z, visibility.

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
            min_tracking_confidence=min_tracking_confidence
        )

    def extract_pose_landmarks(self, image):
        """
        Extrae landmarks de pose de una imagen.
        
        Args:
            image: Imagen en formato BGR
            
        Returns:
            tuple: (landmarks_list, results) donde landmarks_list contiene 
                   los landmarks en formato [x, y, z, visibility] × 33
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
            return landmarks, results
        else:
            # Si no se detectan poses, retornar landmarks vacíos
            return [0.0] * (33 * 4), results  # 33 landmarks × 4 valores (x,y,z,visibility)

    def process_video(self, video_path, action_name):
        """
        Procesa un video completo y extrae poses de todos los frames.
        
        Args:
            video_path: Ruta al archivo de video
            action_name: Nombre de la acción para almacenar en los datos
            
        Returns:
            DataFrame: Datos en formato frame, landmark_index, x, y, z, visibility
        """
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print(f"Error: No se pudo abrir el video {video_path}")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Procesando: {video_path.name}")
        print(f"FPS: {fps}, Total frames: {total_frames}")

        # Lista para almacenar todos los datos
        all_data = []
        frame_number = 0

        # Crear barra de progreso
        with tqdm(total=total_frames, desc="Procesando frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Extraer landmarks
                landmarks, results = self.extract_pose_landmarks(frame)

                # Procesar landmarks para el formato requerido
                # landmarks contiene [x0, y0, z0, v0, x1, y1, z1, v1, ...]
                for landmark_idx in range(33):  # 33 landmarks en MediaPipe Pose
                    base_idx = landmark_idx * 4
                    
                    # Extraer x, y, z, visibility para este landmark
                    x = landmarks[base_idx]
                    y = landmarks[base_idx + 1]
                    z = landmarks[base_idx + 2]
                    visibility = landmarks[base_idx + 3]
                    
                    # Agregar fila de datos
                    all_data.append({
                        'frame': frame_number,
                        'landmark_index': landmark_idx,
                        'x': x,
                        'y': y,
                        'z': z,
                        'visibility': visibility
                    })

                frame_number += 1
                pbar.update(1)

        cap.release()

        # Crear DataFrame
        df = pd.DataFrame(all_data)
        
        # Estadísticas de detección
        frames_with_pose = df[df['visibility'] > 0].groupby('frame').size().count()
        detection_rate = (frames_with_pose / total_frames) * 100 if total_frames > 0 else 0
        
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

        print(f"Datos guardados en: {csv_path}")
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
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']

    # Inicializar extractor de poses
    pose_extractor = PoseExtractor()

    # Obtener todas las carpetas de acciones
    action_folders = [d for d in os.listdir(raw_data_path)
                     if os.path.isdir(os.path.join(raw_data_path, d))]

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
                    csv_path = pose_extractor.save_csv(df, action_processed_path, video_name)

                    # Estadísticas
                    total_frames = df['frame'].nunique()
                    total_landmarks = len(df)

                    result = {
                        'video_file': video_file,
                        'total_frames': total_frames,
                        'total_landmarks': total_landmarks,
                        'csv_path': csv_path,
                        'status': 'success'
                    }

                else:
                    result = {
                        'video_file': video_file,
                        'status': 'error',
                        'error': 'No se pudo procesar el video o datos vacíos'
                    }
                    print(f"✗ Error al procesar el video")

            except Exception as e:
                result = {
                    'video_file': video_file,
                    'status': 'error',
                    'error': str(e)
                }
                print(f"✗ Error: {e}")

            action_results.append(result)

        processing_summary[action] = action_results

    return processing_summary


def main():
    """Función principal para ejecutar el procesamiento de videos."""
    
    # Definir rutas del proyecto
    PROJECT_PATH = os.getcwd()  # Usar directorio actual como base
    RAW_DATA_PATH = os.path.join(PROJECT_PATH, 'data', 'raw')
    PROCESSED_DATA_PATH = os.path.join(PROJECT_PATH, 'data', 'processed')

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
    
    for action, results in summary.items():
        action_success = sum(1 for r in results if r['status'] == 'success')
        action_total = len(results)
        
        print(f"\n{action}:")
        print(f"  Videos procesados: {action_success}/{action_total}")
        
        total_videos += action_total
        successful_videos += action_success
    
    print(f"\nTotal general: {successful_videos}/{total_videos} videos procesados exitosamente")
    print(f"Tasa de éxito: {(successful_videos/total_videos)*100:.1f}%" if total_videos > 0 else "Tasa de éxito: 0%")


if __name__ == "__main__":
    import time
    main() 