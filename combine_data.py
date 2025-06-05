import json
import pandas as pd
import os
from pathlib import Path
import numpy as np

def load_json_file(file_path):
    """Carga un archivo JSON y convierte los landmarks en un DataFrame"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Convertir los frames a un formato tabular
    frames_data = []
    for frame in data:
        if frame['pose_detected']:  # Solo incluir frames donde pose_detected es True
            landmarks = frame['landmarks']
            # Convertir la lista plana de landmarks a un diccionario con nombres significativos
            frame_dict = {}
            for i in range(0, len(landmarks), 4):
                landmark_idx = i // 4
                frame_dict[f'landmark_{landmark_idx}_x'] = landmarks[i]
                frame_dict[f'landmark_{landmark_idx}_y'] = landmarks[i + 1]
                frame_dict[f'landmark_{landmark_idx}_z'] = landmarks[i + 2]
                frame_dict[f'landmark_{landmark_idx}_visibility'] = landmarks[i + 3]
            
            frame_dict['frame'] = frame['frame']
            frame_dict['timestamp'] = frame['timestamp']
            frames_data.append(frame_dict)
    
    return pd.DataFrame(frames_data)

def get_relevant_landmarks(df):
    """
    Selecciona solo los landmarks relevantes para nuestro modelo
    Landmarks importantes: 
    - 0 (nariz)
    - 11-12 (hombros)
    - 23-24 (caderas)
    - 25-26 (rodillas)
    - 27-28 (tobillos)
    - 31-32 (pies)
    """
    relevant_indices = [0, 11, 12, 23, 24, 25, 26, 27, 28, 31, 32]
    relevant_columns = []
    
    for idx in relevant_indices:
        # Para cada landmark, tomamos x, y, z, visibility
        columns = [
            f"landmark_{idx}_x",
            f"landmark_{idx}_y",
            f"landmark_{idx}_z",
            f"landmark_{idx}_visibility"
        ]
        relevant_columns.extend(columns)
    
    # Agregar columnas de tiempo si existen
    if 'frame' in df.columns:
        relevant_columns.append('frame')
    if 'timestamp' in df.columns:
        relevant_columns.append('timestamp')
    
    return df[relevant_columns]

def combine_processed_data(base_dir='data/processed'):
    """Combina todos los datos procesados en un solo DataFrame"""
    # Definir las carpetas de movimientos y sus etiquetas
    movement_folders = {
        'voltear': 0,
        'sentar': 1,
        'parar': 2,
        'ir al frente': 3,
        'devolverse': 4
    }
    
    combined_data = []
    
    for movement, label in movement_folders.items():
        movement_path = Path(base_dir) / movement
        if not movement_path.exists():
            print(f"Advertencia: La carpeta {movement} no existe")
            continue
            
        # Procesar todos los archivos JSON en la carpeta
        for json_file in movement_path.glob('*.json'):
            try:
                # Cargar datos
                df = load_json_file(json_file)
                
                if len(df) > 0:  # Solo procesar si hay datos válidos
                    # Seleccionar landmarks relevantes
                    df_relevant = get_relevant_landmarks(df)
                    
                    # Agregar la etiqueta del movimiento
                    df_relevant['target'] = label
                    df_relevant['movement_name'] = movement
                    
                    combined_data.append(df_relevant)
                    print(f"Procesado: {json_file}")
                else:
                    print(f"Archivo vacío o sin poses válidas: {json_file}")
                
            except Exception as e:
                print(f"Error procesando {json_file}: {str(e)}")
    
    # Combinar todos los DataFrames
    if combined_data:
        final_df = pd.concat(combined_data, ignore_index=True)
        
        # Guardar el resultado
        output_path = Path(base_dir) / 'combined_dataset.csv'
        final_df.to_csv(output_path, index=False)
        print(f"\nDataset combinado guardado en: {output_path}")
        print(f"Dimensiones del dataset: {final_df.shape}")
        
        # Mostrar distribución de clases
        print("\nDistribución de movimientos:")
        print(final_df['movement_name'].value_counts())
        
        return final_df
    else:
        print("No se encontraron datos para procesar")
        return None

if __name__ == "__main__":
    df = combine_processed_data() 