#!/usr/bin/env python3
"""
Combinador de Datos Procesados para Clasificador de Movimientos

Este script combina todos los archivos CSV procesados por video_processing_mediapipe.py
en un solo dataset para entrenamiento del modelo de clasificación.

Estructura de entrada: data/processed/{nombre_accion}/{video}_poses.csv
Estructura de salida: data/processed/combined_dataset.csv con formato: frame, landmark_index, x, y, z, visibility, movement
"""

import pandas as pd
import os
from pathlib import Path
import numpy as np

def load_csv_file(file_path, movement_name):
    """
    Carga un archivo CSV con formato: frame, landmark_index, x, y, z, visibility
    y agrega la columna movement
    """
    try:
        df = pd.read_csv(file_path)
        
        # Verificar que tenga las columnas esperadas
        expected_columns = ['frame', 'landmark_index', 'x', 'y', 'z', 'visibility']
        if not all(col in df.columns for col in expected_columns):
            print(f"Advertencia: {file_path} no tiene las columnas esperadas")
            return None
        
        # Filtrar solo landmarks con alta visibilidad
        df_filtered = df[df['visibility'] > 0.5].copy()
        
        if len(df_filtered) == 0:
            print(f"Advertencia: No hay landmarks visibles en {file_path}")
            return None
        
        # Agregar columna de movimiento
        df_filtered['movement'] = movement_name
        
        return df_filtered
        
    except Exception as e:
        print(f"Error cargando {file_path}: {str(e)}")
        return None

def get_relevant_landmarks(df):
    """
    Filtra solo los landmarks relevantes para nuestro modelo
    Landmarks importantes para movimientos corporales:
    - 0 (nariz) - orientación de la cabeza
    - 11-12 (hombros) - orientación del torso
    - 23-24 (caderas) - centro de masa y orientación
    - 25-26 (rodillas) - movimiento de piernas
    - 27-28 (tobillos) - apoyo y equilibrio
    - 31-32 (pies) - dirección del movimiento
    """
    relevant_indices = [0, 11, 12, 23, 24, 25, 26, 27, 28, 31, 32]
    
    # Filtrar solo los landmarks relevantes
    df_relevant = df[df['landmark_index'].isin(relevant_indices)].copy()
    
    if len(df_relevant) == 0:
        print("Advertencia: No se encontraron landmarks relevantes")
        return None
    
    return df_relevant

def combine_processed_data(base_dir='data/processed'):
    """Combina todos los datos procesados en un solo DataFrame"""
    # Definir las carpetas de movimientos
    movement_folders = ['girar', 'sentar', 'parar', 'ir al frente', 'devolverse']
    
    combined_data = []
    processing_stats = {movement: {'files': 0, 'frames': 0, 'errors': 0} for movement in movement_folders}
    
    for movement in movement_folders:
        movement_path = Path(base_dir) / movement
        if not movement_path.exists():
            print(f"Advertencia: La carpeta {movement} no existe")
            continue
            
        print(f"\nProcesando movimiento: {movement}")
        
        # Procesar todos los archivos CSV en la carpeta
        csv_files = list(movement_path.glob('*_poses.csv'))
        
        for csv_file in csv_files:
            try:
                # Cargar datos
                df = load_csv_file(csv_file, movement)
                
                if df is not None and len(df) > 0:
                    combined_data.append(df)
                    processing_stats[movement]['files'] += 1
                    processing_stats[movement]['frames'] += len(df)
                    print(f"  ✓ {csv_file.name}: {len(df)} landmarks")
                    
                    # Seleccionar landmarks relevantes
                    # df_relevant = get_relevant_landmarks(df)
                    
                    # if df_relevant is not None:
                    #     combined_data.append(df_relevant)
                    #     processing_stats[movement]['files'] += 1
                    #     processing_stats[movement]['frames'] += len(df_relevant)
                    #     print(f"  ✓ {csv_file.name}: {len(df_relevant)} landmarks")
                    # else:
                    #     print(f"  ✗ {csv_file.name}: Sin landmarks relevantes")
                    #     processing_stats[movement]['errors'] += 1
                else:
                    print(f"  ✗ {csv_file.name}: Archivo vacío o sin poses válidas")
                    processing_stats[movement]['errors'] += 1
                
            except Exception as e:
                print(f"  ✗ Error procesando {csv_file.name}: {str(e)}")
                processing_stats[movement]['errors'] += 1
    
    # Combinar todos los DataFrames
    if combined_data:
        print(f"\n{'='*50}")
        print("COMBINANDO DATOS...")
        print(f"{'='*50}")
        
        final_df = pd.concat(combined_data, ignore_index=True)
        
        # Reordenar columnas para el formato final deseado
        column_order = ['frame', 'landmark_index', 'x', 'y', 'z', 'visibility', 'movement']
        final_df = final_df[column_order]
        
        # Guardar el resultado
        output_path = Path(base_dir) / 'combined_dataset.csv'
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
            print(f"  Landmarks totales: {stats['frames']}")
            print(f"  Errores: {stats['errors']}")
        
        print(f"\nDistribución de movimientos en el dataset final:")
        movement_counts = final_df['movement'].value_counts()
        for movement, count in movement_counts.items():
            percentage = (count / len(final_df)) * 100
            print(f"  {movement}: {count} landmarks ({percentage:.1f}%)")
        
        print(f"\nDistribución de landmarks por movimiento:")
        for movement in movement_folders:
            if movement in movement_counts.index:
                movement_data = final_df[final_df['movement'] == movement]
                unique_frames = movement_data['frame'].nunique()
                landmarks_per_frame = len(movement_data) / unique_frames if unique_frames > 0 else 0
                print(f"  {movement}: {unique_frames} frames únicos, {landmarks_per_frame:.1f} landmarks promedio por frame")
        
        # Verificar balance del dataset
        min_samples = movement_counts.min()
        max_samples = movement_counts.max()
        balance_ratio = min_samples / max_samples
        print(f"\nBalance del dataset: {balance_ratio:.2f}")
        if balance_ratio < 0.5:
            print("⚠️  Dataset desbalanceado. Considera técnicas de balanceo.")
        else:
            print("✓ Dataset relativamente balanceado.")
        
        # Mostrar información sobre landmarks
        landmark_counts = final_df['landmark_index'].value_counts().sort_index()
        print(f"\nLandmarks incluidos en el dataset:")
        landmark_names = {
            0: "nariz", 11: "hombro_izq", 12: "hombro_der",
            23: "cadera_izq", 24: "cadera_der", 25: "rodilla_izq",
            26: "rodilla_der", 27: "tobillo_izq", 28: "tobillo_der",
            31: "pie_izq", 32: "pie_der"
        }
        for landmark_idx, count in landmark_counts.items():
            name = landmark_names.get(landmark_idx, f"landmark_{landmark_idx}")
            print(f"  {landmark_idx} ({name}): {count} registros")
        
        return final_df
    else:
        print("❌ No se encontraron datos para procesar")
        return None

def main():
    """Función principal"""
    print("Iniciando combinación de datos procesados...")
    
    # Verificar que existe el directorio base
    base_dir = 'data/processed'
    if not os.path.exists(base_dir):
        print(f"Error: El directorio {base_dir} no existe.")
        print("Ejecuta primero video_processing_mediapipe.py para procesar los videos.")
        return
    
    # Combinar datos
    df = combine_processed_data(base_dir)
    
    if df is not None:
        print(f"\n✅ Procesamiento completado exitosamente!")
        print(f"Dataset listo para entrenamiento: data/processed/combined_dataset.csv")
        print(f"Formato final: frame, landmark_index, x, y, z, visibility, movement")
    else:
        print(f"\n❌ No se pudo crear el dataset combinado.")

if __name__ == "__main__":
    main() 