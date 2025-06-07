#!/usr/bin/env python3
"""
Combinador de Datos Procesados para Clasificador de Movimientos

Este script combina todos los archivos CSV procesados en un solo dataset
para entrenamiento del modelo de clasificación.

Estructura de entrada: data/processed/{nombre_accion}/{video}_poses.csv
Formato entrada: frame, nose_x, nose_y, nose_z, nose_v, left_eye_inner_x, ...

Estructura de salida: data/processed/combined_dataset.csv 
Formato salida: frame, video_id, movement, [landmarks_relevantes_x_y_z_v]
"""

import pandas as pd
import os
from pathlib import Path
import numpy as np


def get_relevant_landmark_columns():
    """
    Define las columnas de landmarks relevantes para clasificación de movimientos.

    Landmarks importantes para movimientos corporales:
    - nose (0) - orientación de la cabeza
    - shoulders (11-12) - orientación del torso
    - wrists (15-16) - movimiento de brazos
    - hips (23-24) - centro de masa y orientación
    - knees (25-26) - movimiento de piernas
    - ankles (27-28) - apoyo y equilibrio
    - heels (29-30) - apoyo y dirección
    """
    relevant_landmarks = [
        "nose",
        "left_shoulder",
        "right_shoulder",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "left_heel",
        "right_heel",
    ]

    # Generar nombres de columnas para cada landmark
    relevant_columns = []
    for landmark in relevant_landmarks:
        relevant_columns.extend(
            [f"{landmark}_x", f"{landmark}_y", f"{landmark}_z", f"{landmark}_v"]
        )

    return relevant_columns


def load_and_process_csv(file_path, movement_name):
    """
    Carga un archivo CSV con formato de una fila por frame y procesa los datos.

    Args:
        file_path: Ruta al archivo CSV
        movement_name: Nombre del movimiento (turning, sit_down, etc.)

    Returns:
        DataFrame procesado o None si hay error
    """
    try:
        df = pd.read_csv(file_path)

        # Verificar que tenga la columna frame
        if "frame" not in df.columns:
            print(f"Advertencia: {file_path} no tiene columna 'frame'")
            return None

        # Agregar información del movimiento
        df["movement"] = movement_name

        # Seleccionar solo las columnas relevantes
        relevant_columns = get_relevant_landmark_columns()
        base_columns = ["frame", "movement"]

        # Verificar qué columnas relevantes existen en el DataFrame
        available_relevant = [col for col in relevant_columns if col in df.columns]

        if len(available_relevant) == 0:
            print(f"Advertencia: No se encontraron landmarks relevantes en {file_path}")
            return None

        # Seleccionar columnas finales
        final_columns = base_columns + available_relevant
        df_final = df[final_columns].copy()

        return df_final

    except Exception as e:
        print(f"Error cargando {file_path}: {str(e)}")
        return None


def combine_processed_data(base_dir="data/processed"):
    """Combina todos los datos procesados en un solo DataFrame"""

    # Definir las carpetas de movimientos
    movement_folders = [
        "turning",
        "sit_down",
        "stand_up",
        "walk_forward",
        "walk_backward",
    ]

    combined_data = []
    processing_stats = {
        movement: {"files": 0, "frames": 0, "errors": 0}
        for movement in movement_folders
    }

    for movement in movement_folders:
        movement_path = Path(base_dir) / movement
        if not movement_path.exists():
            print(f"Advertencia: La carpeta {movement} no existe")
            continue

        print(f"\nProcesando movimiento: {movement}")

        # Procesar todos los archivos CSV en la carpeta
        csv_files = list(movement_path.glob("*_poses.csv"))

        for csv_file in csv_files:
            try:
                # Cargar y procesar datos
                df = load_and_process_csv(csv_file, movement)

                if df is not None and len(df) > 0:
                    combined_data.append(df)
                    processing_stats[movement]["files"] += 1
                    processing_stats[movement]["frames"] += len(df)
                    print(f"  ✓ {csv_file.name}: {len(df)} frames")
                else:
                    print(f"  ✗ {csv_file.name}: Sin datos válidos")
                    processing_stats[movement]["errors"] += 1

            except Exception as e:
                print(f"  ✗ Error procesando {csv_file.name}: {str(e)}")
                processing_stats[movement]["errors"] += 1

    # Combinar todos los DataFrames
    if combined_data:
        print(f"\n{'='*50}")
        print("COMBINANDO DATOS...")
        print(f"{'='*50}")

        final_df = pd.concat(combined_data, ignore_index=True)

        # Guardar el resultado
        output_path = Path(base_dir) / "combined_dataset.csv"
        final_df.to_csv(output_path, index=False)

        print(f"\n✓ Dataset combinado guardado en: {output_path}")
        print(f"✓ Dimensiones del dataset: {final_df.shape}")
        print(f"✓ Columnas: {list(final_df.columns)}")

        # Mostrar estadísticas detalladas
        print(f"\n{'='*50}")
        print("ESTADÍSTICAS DEL PROCESAMIENTO")
        print(f"{'='*50}")

        for movement in movement_folders:
            stats = processing_stats[movement]
            print(f"{movement}:")
            print(f"  Archivos procesados: {stats['files']}")
            print(f"  Frames totales: {stats['frames']}")
            print(f"  Errores: {stats['errors']}")

        print(f"\nDistribución de movimientos en el dataset final:")
        movement_counts = final_df["movement"].value_counts()
        for movement, count in movement_counts.items():
            percentage = (count / len(final_df)) * 100
            print(f"  {movement}: {count} frames ({percentage:.1f}%)")

        print(f"\nDistribución de videos por movimiento:")
        for movement in movement_folders:
            if movement in processing_stats:
                count = processing_stats[movement]["files"]
                print(f"  {movement}: {count} archivos procesados")

        print(f"\nFrames promedio por archivo:")
        for movement in movement_folders:
            if movement in processing_stats and processing_stats[movement]["files"] > 0:
                avg_frames = (
                    processing_stats[movement]["frames"]
                    / processing_stats[movement]["files"]
                )
                print(f"  {movement}: {avg_frames:.1f} frames promedio")

        # Verificar balance del dataset
        min_samples = movement_counts.min()
        max_samples = movement_counts.max()
        balance_ratio = min_samples / max_samples
        print(f"\nBalance del dataset: {balance_ratio:.2f}")
        if balance_ratio < 0.5:
            print("Advertencia: Dataset desbalanceado. Considera técnicas de balanceo.")
        else:
            print("Dataset relativamente balanceado.")

        # Mostrar información sobre calidad de datos
        print(f"\nCalidad de los datos:")

        # Verificar porcentaje de campos vacíos
        relevant_columns = get_relevant_landmark_columns()
        available_columns = [col for col in relevant_columns if col in final_df.columns]

        empty_count = 0
        total_count = 0

        for col in available_columns:
            if col.endswith("_v"):  # Solo contar visibility
                empty_in_col = (final_df[col] == "").sum()
                empty_count += empty_in_col
                total_count += len(final_df)

        if total_count > 0:
            completeness = ((total_count - empty_count) / total_count) * 100
            print(f"  Completeness de landmarks: {completeness:.1f}%")

        return final_df
    else:
        print("❌ No se encontraron datos para procesar")
        return None


def main():
    """Función principal"""
    print("Combinador de Datos para Formato Una Fila por Frame")
    print("=" * 60)

    # Verificar que existe el directorio base
    base_dir = "data/processed"
    if not os.path.exists(base_dir):
        print(f"Error: El directorio {base_dir} no existe.")
        print("Ejecuta primero el procesador de videos para generar los datos.")
        return

    print("Iniciando combinación de datos procesados...")

    # Combinar datos
    df = combine_processed_data(base_dir)

    if df is not None:
        print(f"\nProcesamiento completado exitosamente!")
        print(f"Dataset listo para entrenamiento: data/processed/combined_dataset.csv")
        print(f"Formato final: frame, movement, [landmarks_relevantes]")

        # Mostrar muestra de los datos
        print(f"\nMuestra de los datos combinados:")
        print(df.head(3).to_string())

    else:
        print(f"\nNo se pudo crear el dataset combinado.")


if __name__ == "__main__":
    main()
