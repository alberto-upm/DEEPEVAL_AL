#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convierte preguntas técnicas o especializadas en versiones más accesibles
utilizando un modelo de lenguaje a través de VLLM.

Este script:
1. Lee un archivo CSV que contiene preguntas técnicas
2. Detecta el campo/tema de cada pregunta
3. Reformula las preguntas para hacerlas más comprensibles
4. Guarda el resultado en un nuevo CSV con ambas versiones

Requisitos:
    pip install pandas openai tqdm

Antes de ejecutar, asegúrate de lanzar vLLM:
    vllm serve [modelo] --port 8000 --dtype float16

Autor: Alberto G. García  |  Fecha: 2025-04-29
"""

import os
import pandas as pd
import time
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
import argparse

# ---------------------------------------------------------------------------
# Configuración de la conexión a vLLM
# ---------------------------------------------------------------------------
VLLM_BASE_URL = "http://localhost:8000/v1/"  # Endpoint creado por vLLM
API_KEY = "not-needed"  # vLLM ignora este valor
VLLM_MODEL = "NousResearch/Meta-Llama-3-8B-Instruct"

# ---------------------------------------------------------------------------
# Funciones principales
# ---------------------------------------------------------------------------

def detectar_campo(cliente, pregunta):
    """
    Detecta el campo o tema al que pertenece una pregunta.
    
    Args:
        cliente: Cliente de OpenAI configurado para VLLM
        pregunta: Texto de la pregunta
        
    Returns:
        str: Campo o área temática detectada
    """
    prompt = f"""
    En Español. Analiza la siguiente pregunta y determina a qué campo o área temática pertenece.
    Responde ÚNICAMENTE con el nombre del campo (por ejemplo: "Medicina", "Derecho", "Tecnología", etc.). 
    No incluyas explicaciones adicionales.
    
    Pregunta: {pregunta} 
    
    Campo: 
    """
    
    
    try:
        respuesta = cliente.chat.completions.create(
            model=VLLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0.0
        )
        return respuesta.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error al detectar campo: {e}")
        return "General"

def reformular_pregunta(cliente, pregunta, campo):
    """
    Reformula una pregunta técnica en un lenguaje más accesible.
    
    Args:
        cliente: Cliente de OpenAI configurado para VLLM
        pregunta: Texto de la pregunta original
        campo: Campo o área temática de la pregunta
        
    Returns:
        str: Pregunta reformulada
    """
    prompt = f"""
    En Español. Eres un modelo que escribe y habla en español. 
    Necesito que reformules una pregunta técnica de {campo} para hacerla más comprensible 
    para una persona sin conocimientos especializados en ese campo.
    
    Cambia los términos técnicos por explicaciones simples o analogías donde sea necesario.
    Asegúrate de que la pregunta reformulada mantenga el mismo significado e intención 
    que la original, pero en un lenguaje más accesible.
    Ten encuenta que la persona no tiene conocimeintos sobre {campo}
    
    Pregunta original: {pregunta}
    
    Pregunta reformulada:

    Responde solo la pregunta reformulada!!
    """
    
    try:
        respuesta = cliente.chat.completions.create(
            model=VLLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0.7
        )
        return respuesta.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error al reformular pregunta: {e}")
        return pregunta

def procesar_csv(ruta_entrada, ruta_salida, batch_size=5):
    """
    Procesa un archivo CSV con preguntas y añade versiones reformuladas.
    
    Args:
        ruta_entrada: Ruta al archivo CSV de entrada
        ruta_salida: Ruta donde guardar el archivo CSV de salida
        batch_size: Número de preguntas a procesar en cada lote para mostrar progreso
    """
    # Configurar cliente para VLLM
    cliente = OpenAI(
        base_url=VLLM_BASE_URL,
        api_key=API_KEY
    )
    
    # Leer el CSV de entrada
    try:
        df = pd.read_csv(ruta_entrada)
        print(f"CSV cargado correctamente. Columnas: {df.columns.tolist()}")
    except Exception as e:
        print(f"Error al leer el archivo CSV: {e}")
        return
    
    # Verificar que existe la columna 'input'
    if 'input' not in df.columns:
        print("Error: El archivo CSV no contiene una columna 'input'")
        return
    
    # Crear columna para preguntas reformuladas si no existe
    if 'input_reformulado' not in df.columns:
        df['input_reformulado'] = ''
    
    # Detectar y reformular preguntas
    total_preguntas = len(df)
    print(f"Procesando {total_preguntas} preguntas...")
    
    # Procesar en lotes para mostrar progreso
    for i in tqdm(range(0, total_preguntas, batch_size), desc="Procesando lotes"):
        lote = df.iloc[i:min(i+batch_size, total_preguntas)]
        
        for idx, fila in lote.iterrows():
            pregunta = fila['input']
            
            # Skip if already processed
            if pd.notna(df.at[idx, 'input_reformulado']) and df.at[idx, 'input_reformulado'] != '':
                continue
                
            # Detectar campo de la pregunta
            campo = detectar_campo(cliente, pregunta)
            
            # Pequeña pausa para no sobrecargar la API
            time.sleep(0.5)
            
            # Reformular la pregunta
            reformulada = reformular_pregunta(cliente, pregunta, campo)
            
            # Guardar en el DataFrame
            df.at[idx, 'input_reformulado'] = reformulada
            
            # Guardar progreso incremental cada lote
            if (idx % batch_size == 0) or (idx == total_preguntas - 1):
                df.to_csv(ruta_salida, index=False)
                
            # Pequeña pausa para no sobrecargar la API
            time.sleep(0.5)
    
    # Guardar resultado final
    df.to_csv(ruta_salida, index=False)
    print(f"Proceso completado. Archivo guardado en: {ruta_salida}")

# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Convierte preguntas técnicas en versiones más comprensibles')
    parser.add_argument('--input', type=str, default="/home/jovyan/DEEPEVAL_AL/output/dataset_main_vllm.csv", help='Ruta al archivo CSV de entrada')
    parser.add_argument('--output', type=str, default="/home/jovyan/DEEPEVAL_AL/output/dataset_main_vllm_realistas.csv", help='Ruta donde guardar el archivo CSV de salida')
    parser.add_argument('--batch', type=int, default=20, help='Tamaño del lote para procesamiento')
    args = parser.parse_args()
    
    # Si no se especifica archivo de entrada, buscar en la ubicación por defecto
    if not args.input:
        # Buscar archivos CSV en la carpeta output
        output_dir = Path("/home/jovyan/DEEPEVAL_AL/output")
        csv_files = list(output_dir.glob("*.csv"))
        
        if not csv_files:
            print("Error: No se encontraron archivos CSV en la carpeta 'output'")
            return
        
        # Ordenar por fecha de modificación (más reciente primero)
        csv_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        archivo_entrada = csv_files[0]
        print(f"Usando el archivo CSV más reciente: {archivo_entrada}")
    else:
        archivo_entrada = Path(args.input)
        
    # Si no se especifica archivo de salida, crear uno basado en el de entrada
    if not args.output:
        nombre_base = archivo_entrada.stem
        archivo_salida = archivo_entrada.parent / f"{nombre_base}_reformulado.csv"
    else:
        archivo_salida = Path(args.output)
    
    # Procesar el CSV
    procesar_csv(archivo_entrada, archivo_salida, args.batch)

if __name__ == "__main__":
    main()