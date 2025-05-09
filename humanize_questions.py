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

    Esta es la pregunta original: {pregunta}

    Responde ÚNICAMENTE con Pregunta reformulada.
    Acorta la longitud de la pregunta guardando el mismo significado.
    Si es necesario parte la pregunta en dos preguntas.
    Se lo menos técnico posible.
    No incluyas explicaciones adicionales.
    
    Pregunta reformulada:

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

def reformular_pregunta_2(cliente, pregunta, campo):
    """
    Reformula una pregunta técnica en un lenguaje más accesible.
    
    Args:
        cliente: Cliente de OpenAI configurado para VLLM
        pregunta: Texto de la pregunta original
        campo: Campo o área temática de la pregunta
        
    Returns:
        str: Pregunta reformulada
    """
    '''
    prompt = f"""
    En Español. Eres un modelo que escribe y habla en español. 
    Necesito que reformules una pregunta técnica de {campo} para hacerla más comprensible 
    para una persona sin conocimientos especializados en ese campo.
    
    Cambia los términos técnicos por explicaciones simples o analogías donde sea necesario.
    Asegúrate de que la pregunta reformulada mantenga el mismo significado e intención 
    que la original, pero en un lenguaje más accesible.
    Ten encuenta que la persona no tiene conocimeintos sobre {campo}

    Esta es la pregunta original: {pregunta}

    Responde ÚNICAMENTE con Pregunta reformulada.
    Imagina que quien hace la pregunta es un adolestente de 16 años sin información del tema.
    Acorta la longitud de la pregunta guardando el mismo significado.
    Simplefica la pregunta al maximo exponente.
    Las pregnutas reformuladas tienen que ser cortas.
    No incluyas explicaciones adicionales.
    
    Pregunta reformulada:

    """
    '''

    prompt = f"""
    En Español. Eres un experto en simplificar y acortar preguntas.
    
    Tengo una pregunta ya reformulada sobre {campo}, pero necesito que sea aún más corta y simple.
    
    Reglas:
    1. Reduce la longitud a menos de la mitad sin perder el significado esencial
    2. Usa palabras más sencillas y frases más directas
    3. Elimina cualquier explicación o contexto innecesario
    4. Mantén la pregunta clara y comprensible
    5. Responde ÚNICAMENTE con la pregunta simplificada, sin añadir comentarios
    6. No incluyas explicaciones adicionales.
    
    Pregunta reformulada: {pregunta}
    
    Pregunta final (corta y simple):
    """
    
    try:
        respuesta = cliente.chat.completions.create(
            model=VLLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            #max_tokens=256,
            max_tokens=100,
            temperature=0.8
        )
        return respuesta.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error al reformular pregunta: {e}")
        return pregunta


def insertar_columna_despues_de(df, nueva_columna, columna_objetivo):
    """
    Mueve una columna para que aparezca justo después de otra columna específica.

    Args:
        df: DataFrame original.
        nueva_columna: Nombre de la columna a mover.
        columna_objetivo: Columna después de la cual insertar.

    Returns:
        Un nuevo DataFrame con la columna reordenada.
    """
    cols = df.columns.tolist()
    if nueva_columna in cols and columna_objetivo in cols:
        cols.remove(nueva_columna)
        idx = cols.index(columna_objetivo) + 1
        cols.insert(idx, nueva_columna)
    return df[cols]

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
    
    # Crear columnas para preguntas reformuladas si no existen
    if 'input_reformulado' not in df.columns:
        df['input_reformulado'] = ''
    if 'input_reformulado_2' not in df.columns:
        df['input_reformulado_2'] = ''
    
    # Detectar y reformular preguntas
    total_preguntas = len(df)
    print(f"Procesando {total_preguntas} preguntas (primera reformulación)...")
    
    # Procesar en lotes para mostrar progreso - PRIMERA REFORMULACIÓN
    for i in tqdm(range(0, total_preguntas, batch_size), desc="Procesando lotes (reformulación 1)"):
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
                # Reordenar columnas
                cols = df.columns.tolist()
                input_idx = cols.index('input')
                # Reordenar para poner input_reformulado después de input
                reordered_cols = cols[:input_idx+1] + ['input_reformulado'] + [col for col in cols if col not in ['input', 'input_reformulado']]
                df = df[reordered_cols]
                df.to_csv(ruta_salida, index=False)
                
            # Pequeña pausa para no sobrecargar la API
            time.sleep(0.5)
    
    # SEGUNDA REFORMULACIÓN
    print(f"Procesando {total_preguntas} preguntas (segunda reformulación)...")
    for i in tqdm(range(0, total_preguntas, batch_size), desc="Procesando lotes (reformulación 2)"):
        lote = df.iloc[i:min(i+batch_size, total_preguntas)]
        
        for idx, fila in lote.iterrows():
            # Solo procesar si ya existe una reformulación previa y falta la segunda
            if not pd.notna(fila['input_reformulado']) or fila['input_reformulado'] == '':
                continue
                
            # Skip if already processed the second step
            if pd.notna(df.at[idx, 'input_reformulado_2']) and df.at[idx, 'input_reformulado_2'] != '':
                continue
                
            pregunta_reformulada = fila['input_reformulado']
            
            # Detectar campo de la pregunta original para usarlo en la segunda reformulación
            campo = detectar_campo(cliente, fila['input'])
            
            # Pequeña pausa para no sobrecargar la API
            time.sleep(0.5)
            
            # Re-reformular la pregunta
            simplificada = reformular_pregunta_2(cliente, pregunta_reformulada, campo)
            
            # Guardar en el DataFrame
            df.at[idx, 'input_reformulado_2'] = simplificada
            
            # Guardar progreso incremental cada lote
            if (idx % batch_size == 0) or (idx == total_preguntas - 1):
                # Reordenar columnas
                cols = df.columns.tolist()
                input_idx = cols.index('input')
                reformulado_idx = cols.index('input_reformulado')
                # Reordenar para poner columnas en orden correcto
                reordered_cols = cols[:input_idx+1] + ['input_reformulado', 'input_reformulado_2'] + [col for col in cols if col not in ['input', 'input_reformulado', 'input_reformulado_2']]
                df = df[reordered_cols]
                df.to_csv(ruta_salida, index=False)
                
            # Pequeña pausa para no sobrecargar la API
            time.sleep(0.5)
    
    # Guardar resultado final con columnas en orden correcto
    cols = df.columns.tolist()
    input_idx = cols.index('input')
    # Reordenar para poner columnas en orden correcto
    reordered_cols = cols[:input_idx+1] + ['input_reformulado', 'input_reformulado_2'] + [col for col in cols if col not in ['input', 'input_reformulado', 'input_reformulado_2']]
    df = df[reordered_cols]
    df.to_csv(ruta_salida, index=False)
    print(f"Proceso completado. Archivo guardado en: {ruta_salida}")
'''
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
                df = insertar_columna_despues_de(df, 'input_reformulado', 'input')
                #df_reordenado = insertar_columna_despues_de(df, 'input_reformulado', 'input')
                df.to_csv(ruta_salida, index=False)
                
            # Pequeña pausa para no sobrecargar la API
            time.sleep(0.5)
            
    # Guardar resultado final
    df = insertar_columna_despues_de(df, 'input_reformulado', 'input')

    # Crear columna para preguntas reformuladas si no existe
    if 'input_reformulado_2' not in df.columns:
        df['input_reformulado_2'] = ''
            
     # Procesar en lotes para mostrar progreso
    for i in tqdm(range(0, total_preguntas, batch_size), desc="Procesando lotes"):
        lote = df.iloc[i:min(i+batch_size, total_preguntas)]
        
        for idx, fila in lote.iterrows():
            pregunta = fila['input_reformulado']
            
            # Skip if already processed
            if pd.notna(df.at[idx, 'input_reformulado_2']) and df.at[idx, 'input_reformulado_2'] != '':
                continue
                
            # Detectar campo de la pregunta
            campo = detectar_campo(cliente, pregunta)
            
            # Pequeña pausa para no sobrecargar la API
            time.sleep(0.5)
            
            # Reformular la pregunta
            reformulada = reformular_pregunta_2(cliente, pregunta, campo)
            
            # Guardar en el DataFrame
            df.at[idx, 'input_reformulado_2'] = reformulada
            
            # Guardar progreso incremental cada lote
            if (idx % batch_size == 0) or (idx == total_preguntas - 1):
                df = insertar_columna_despues_de(df, 'input_reformulado_2', 'input_reformulado')
                df.to_csv(ruta_salida, index=False)
                
            # Pequeña pausa para no sobrecargar la API
            time.sleep(0.5)
    
    # Guardar resultado final
    #df_reordenado = insertar_columna_despues_de(df, 'input_reformulado', 'input')
    df = insertar_columna_despues_de(df, 'input_reformulado_2', 'input_reformulado')
    df.to_csv(ruta_salida, index=False)
    print(f"Proceso completado. Archivo guardado en: {ruta_salida}")
'''

# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Convierte preguntas técnicas en versiones más comprensibles')
    parser.add_argument('--input', type=str, default="/home/jovyan/DEEPEVAL_AL/output/dataset_main_vllm.csv", help='Ruta al archivo CSV de entrada')
    parser.add_argument('--output', type=str, default="/home/jovyan/DEEPEVAL_AL/output/dataset_main_vllm_realistas7.csv", help='Ruta donde guardar el archivo CSV de salida')
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