#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convierte preguntas técnicas o especializadas en versiones más accesibles
utilizando un modelo de lenguaje a través de VLLM y evalúa la calidad mediante
sentence embeddings y métricas ROUGE.

Este script:
1. Lee un archivo CSV que contiene preguntas técnicas
2. Detecta el campo/tema de cada pregunta
3. Reformula las preguntas para hacerlas más comprensibles
4. Evalúa la calidad de la reformulación mediante sentence similarity y ROUGE
5. Regenera las reformulaciones que no cumplen los umbrales
6. Guarda el resultado en un nuevo CSV con ambas versiones

Requisitos:
    pip install pandas openai tqdm sentence-transformers rouge-score

Antes de ejecutar, asegúrate de lanzar vLLM:
    vllm serve [modelo] --port 8000 --dtype float16

Autor: Alberto G. García  |  Fecha: 2025-04-29
Modificado: 2025-05-15
"""

import os
import pandas as pd
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
import argparse
from sentence_transformers import SentenceTransformer, util
from rouge_score import rouge_scorer

# ---------------------------------------------------------------------------
# Configuración de la conexión a vLLM
# ---------------------------------------------------------------------------
VLLM_BASE_URL = "http://localhost:8000/v1/"  # Endpoint creado por vLLM
API_KEY = "not-needed"  # vLLM ignora este valor
VLLM_MODEL = "NousResearch/Meta-Llama-3-8B-Instruct"

# ---------------------------------------------------------------------------
# Configuración para evaluación semántica
# ---------------------------------------------------------------------------
# Umbrales de calidad para las reformulaciones
UMBRAL_SIMILARITY = 0.75  # Mínima similitud semántica requerida (0-1)
UMBRAL_ROUGE_MAX = 0.5    # Máxima similitud léxica permitida (0-1)

# Carga del modelo de sentence embeddings para español
def cargar_modelo_embeddings():
    """
    Carga el modelo de sentence embeddings para español.
    
    Returns:
        SentenceTransformer: Modelo cargado
    """
    try:
        # Intenta cargar primero un modelo multilingüe optimizado para español
        return SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    except Exception as e:
        print(f"Error al cargar el modelo principal, intentando alternativa: {e}")
        # Alternativa si el primer modelo falla
        return SentenceTransformer('distiluse-base-multilingual-cased-v1')

# Inicializar el modelo de embeddings y el evaluador ROUGE
modelo_embeddings = None  # Se inicializará bajo demanda
rouge_evaluador = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

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
    
    Reglas importantes:
    1. La reformulación debe mantener EXACTAMENTE el mismo significado e intención que la original (alta similitud semántica o "sentence similarity")
    2. Debes usar palabras diferentes y estructura de frase distinta (bajo solapamiento léxico o bajo valor de "ROUGE")
    3. Cambia los términos técnicos por explicaciones simples o analogías
    4. La persona no tiene conocimientos sobre {campo}
    5. Acorta la longitud de la pregunta guardando el mismo significado
    6. Si es necesario divide la pregunta en dos preguntas más simples
    7. Sé lo menos técnico posible

    Esta es la pregunta original: {pregunta}

    Responde ÚNICAMENTE con la pregunta reformulada.
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
    prompt = f"""
    En Español. Eres un experto en simplificar y acortar preguntas.
    
    Tengo una pregunta ya reformulada sobre {campo}, pero necesito que sea aún más corta y simple.
    
    Reglas importantes:
    1. La versión simplificada DEBE mantener el mismo significado que la pregunta original (alta similitud semántica o "sentence similarity")
    2. Usa vocabulario y estructura TOTALMENTE DIFERENTES (bajo solapamiento léxico o bajo valor de "ROUGE")
    3. Reduce la longitud a menos de la mitad sin perder el significado esencial
    4. Usa palabras más sencillas y frases más directas
    5. Elimina cualquier explicación o contexto innecesario
    6. Mantén la pregunta clara y comprensible
    
    Pregunta reformulada: {pregunta}
    
    Pregunta final (corta y simple):
    """
    
    try:
        respuesta = cliente.chat.completions.create(
            model=VLLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.8
        )
        return respuesta.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error al reformular pregunta: {e}")
        return pregunta

def calcular_similitud_semantica(texto1, texto2):
    """
    Calcula la similitud semántica entre dos textos utilizando sentence embeddings.
    
    Args:
        texto1: Primer texto
        texto2: Segundo texto
        
    Returns:
        float: Valor de similitud (0-1)
    """
    global modelo_embeddings
    
    # Cargar el modelo si aún no está inicializado
    if modelo_embeddings is None:
        modelo_embeddings = cargar_modelo_embeddings()
    
    try:
        # Codificar los textos
        embedding1 = modelo_embeddings.encode(texto1, convert_to_tensor=True)
        embedding2 = modelo_embeddings.encode(texto2, convert_to_tensor=True)
        
        # Calcular similitud de coseno
        similitud = util.pytorch_cos_sim(embedding1, embedding2).item()
        
        return similitud
    except Exception as e:
        print(f"Error al calcular similitud semántica: {e}")
        return 0.0

def calcular_rouge(texto1, texto2):
    """
    Calcula el valor ROUGE entre dos textos para medir la similitud léxica.
    
    Args:
        texto1: Texto original
        texto2: Texto reformulado
        
    Returns:
        float: Promedio de puntuaciones ROUGE (0-1)
    """
    try:
        scores = rouge_evaluador.score(texto1, texto2)
        
        # Calcular el promedio de varios tipos de ROUGE
        rouge_promedio = (
            scores['rouge1'].fmeasure + 
            scores['rouge2'].fmeasure + 
            scores['rougeL'].fmeasure
        ) / 3
        
        return rouge_promedio
    except Exception as e:
        print(f"Error al calcular ROUGE: {e}")
        return 1.0  # En caso de error, asumimos máxima similitud (peor caso)

def evaluar_calidad_reformulacion(pregunta_original, pregunta_reformulada):
    """
    Evalúa si la pregunta reformulada mantiene la semántica pero cambia el léxico.
    
    Args:
        pregunta_original: Texto de la pregunta original
        pregunta_reformulada: Texto de la pregunta reformulada
        
    Returns:
        bool: True si la reformulación es válida, False en caso contrario
        dict: Diccionario con métricas detalladas
    """
    # Calcular similitud semántica
    similitud = calcular_similitud_semantica(pregunta_original, pregunta_reformulada)
    
    # Calcular similitud léxica (ROUGE)
    rouge = calcular_rouge(pregunta_original, pregunta_reformulada)
    
    # Una buena reformulación tiene alta similitud semántica y baja similitud léxica
    es_valida = (similitud >= UMBRAL_SIMILARITY) and (rouge <= UMBRAL_ROUGE_MAX)
    
    metricas = {
        'similitud_semantica': similitud,
        'similitud_lexica': rouge,
        'es_valida': es_valida
    }
    
    return es_valida, metricas

def procesar_csv(ruta_entrada, ruta_salida, batch_size=5, max_intentos_reformulacion=3):
    """
    Procesa un archivo CSV con preguntas y añade versiones reformuladas.
    
    Args:
        ruta_entrada: Ruta al archivo CSV de entrada
        ruta_salida: Ruta donde guardar el archivo CSV de salida
        batch_size: Número de preguntas a procesar en cada lote para mostrar progreso
        max_intentos_reformulacion: Número máximo de intentos para reformular una pregunta
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
    
    # Crear columnas para preguntas reformuladas y métricas si no existen
    if 'campo_tematico' not in df.columns:
        df['campo_tematico'] = ''
    if 'input_reformulado' not in df.columns:
        df['input_reformulado'] = ''
    if 'input_reformulado_2' not in df.columns:
        df['input_reformulado_2'] = ''
    if 'similitud_semantica' not in df.columns:
        df['similitud_semantica'] = 0.0
    if 'similitud_lexica' not in df.columns:
        df['similitud_lexica'] = 0.0
    if 'intentos_reformulacion' not in df.columns:
        df['intentos_reformulacion'] = 0
    
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
            
            # Guardar el campo temático en el DataFrame
            df.at[idx, 'campo_tematico'] = campo
            
            # Pequeña pausa para no sobrecargar la API
            time.sleep(0.5)
            
            # Proceso de reformulación con validación de calidad
            reformulada = ""
            es_valida = False
            metricas = {}
            intentos = 0
            
            while not es_valida and intentos < max_intentos_reformulacion:
                intentos += 1
                
                # Reformular la pregunta
                reformulada = reformular_pregunta(cliente, pregunta, campo)
                
                # Evaluar calidad de la reformulación
                es_valida, metricas = evaluar_calidad_reformulacion(pregunta, reformulada)
                
                if es_valida:
                    print(f"Pregunta {idx} reformulada válidamente en intento {intentos}")
                    print(f"  - Similitud semántica: {metricas['similitud_semantica']:.2f}")
                    print(f"  - Similitud léxica: {metricas['similitud_lexica']:.2f}")
                    break
                else:
                    print(f"Intento {intentos}: Reformulación inválida, regenerando...")
                    print(f"  - Similitud semántica: {metricas['similitud_semantica']:.2f} (mínimo: {UMBRAL_SIMILARITY})")
                    print(f"  - Similitud léxica: {metricas['similitud_lexica']:.2f} (máximo: {UMBRAL_ROUGE_MAX})")
                
                # Pequeña pausa para no sobrecargar la API
                time.sleep(1.0)
            
            # Si después de todos los intentos no se consiguió una reformulación válida, usamos la última
            if not es_valida:
                print(f"ADVERTENCIA: No se logró una reformulación válida para la pregunta {idx} después de {max_intentos_reformulacion} intentos")
                df.at[idx, 'intentos_reformulacion'] = 0
            else:
                df.at[idx, 'intentos_reformulacion'] = intentos
            
            # Guardar en el DataFrame
            df.at[idx, 'input_reformulado'] = reformulada
            df.at[idx, 'similitud_semantica'] = metricas.get('similitud_semantica', 0.0)
            df.at[idx, 'similitud_lexica'] = metricas.get('similitud_lexica', 1.0)
            
            # Guardar progreso incremental cada lote
            if (idx % batch_size == 0) or (idx == total_preguntas - 1):
                # Reordenar columnas
                columnas_deseadas = [
                    'input', 'campo_tematico', 'input_reformulado', 
                    'similitud_semantica', 'similitud_lexica', 'intentos_reformulacion'
                ]
                # Añadir otras columnas que puedan existir
                otras_columnas = [col for col in df.columns if col not in columnas_deseadas]
                orden_final = columnas_deseadas + otras_columnas
                # Filtrar para incluir solo las que existen
                orden_final = [col for col in orden_final if col in df.columns]
                
                df = df[orden_final]
                df.to_csv(ruta_salida, index=False)
                
            # Pequeña pausa para no sobrecargar la API
            time.sleep(0.5)
    
    # SEGUNDA REFORMULACIÓN
    print(f"Procesando {total_preguntas} preguntas (segunda reformulación)...")
    
    # Añadir columnas para métricas de la segunda reformulación si no existen
    if 'similitud_semantica_2' not in df.columns:
        df['similitud_semantica_2'] = 0.0
    if 'similitud_lexica_2' not in df.columns:
        df['similitud_lexica_2'] = 0.0
    if 'intentos_reformulacion_2' not in df.columns:
        df['intentos_reformulacion_2'] = 0
    
    for i in tqdm(range(0, total_preguntas, batch_size), desc="Procesando lotes (reformulación 2)"):
        lote = df.iloc[i:min(i+batch_size, total_preguntas)]
        
        for idx, fila in lote.iterrows():
            # Solo procesar si ya existe una reformulación previa y falta la segunda
            if not pd.notna(fila['input_reformulado']) or fila['input_reformulado'] == '':
                continue
                
            # Skip if already processed the second step
            if pd.notna(df.at[idx, 'input_reformulado_2']) and df.at[idx, 'input_reformulado_2'] != '':
                continue
                
            pregunta_original = fila['input']
            pregunta_reformulada = fila['input_reformulado']
            
            # Detectar campo de la pregunta original para usarlo en la segunda reformulación
            campo = detectar_campo(cliente, pregunta_original)
            
            # Pequeña pausa para no sobrecargar la API
            time.sleep(0.5)
            
            # Proceso de reformulación con validación de calidad
            simplificada = ""
            es_valida = False
            metricas = {}
            intentos = 0
            
            while not es_valida and intentos < max_intentos_reformulacion:
                intentos += 1
                
                # Re-reformular la pregunta
                simplificada = reformular_pregunta_2(cliente, pregunta_reformulada, campo)
                
                # Evaluar calidad de la reformulación (comparando con la pregunta original)
                es_valida, metricas = evaluar_calidad_reformulacion(pregunta_original, simplificada)
                
                if es_valida:
                    print(f"Pregunta {idx} simplificada válidamente en intento {intentos}")
                    print(f"  - Similitud semántica: {metricas['similitud_semantica']:.2f}")
                    print(f"  - Similitud léxica: {metricas['similitud_lexica']:.2f}")
                    break
                else:
                    print(f"Intento {intentos}: Simplificación inválida, regenerando...")
                    print(f"  - Similitud semántica: {metricas['similitud_semantica']:.2f} (mínimo: {UMBRAL_SIMILARITY})")
                    print(f"  - Similitud léxica: {metricas['similitud_lexica']:.2f} (máximo: {UMBRAL_ROUGE_MAX})")
                
                # Pequeña pausa para no sobrecargar la API
                time.sleep(1.0)
            
            # Si después de todos los intentos no se consiguió una simplificación válida, usamos la última
            if not es_valida:
                print(f"ADVERTENCIA: No se logró una simplificación válida para la pregunta {idx} después de {max_intentos_reformulacion} intentos")
                df.at[idx, 'intentos_reformulacion_2'] = 0
            else:
                df.at[idx, 'intentos_reformulacion_2'] = intentos
            
            # Guardar en el DataFrame
            df.at[idx, 'input_reformulado_2'] = simplificada
            df.at[idx, 'similitud_semantica_2'] = metricas.get('similitud_semantica', 0.0)
            df.at[idx, 'similitud_lexica_2'] = metricas.get('similitud_lexica', 1.0)
            
            # Guardar progreso incremental cada lote
            if (idx % batch_size == 0) or (idx == total_preguntas - 1):
                # Guardar progreso incremental
                df.to_csv(ruta_salida, index=False)
                
            # Pequeña pausa para no sobrecargar la API
            time.sleep(0.5)
    
    # Reordenar columnas para que queden en un orden lógico
    columnas_deseadas = [
        'input', 'campo_tematico', 
        'input_reformulado', 'similitud_semantica', 'similitud_lexica', 'intentos_reformulacion', 
        'input_reformulado_2', 'similitud_semantica_2', 'similitud_lexica_2', 'intentos_reformulacion_2'
    ]
    
    # Añadir las columnas que no están en el orden deseado pero existen en el dataframe
    otras_columnas = [col for col in df.columns if col not in columnas_deseadas]
    orden_final = columnas_deseadas + otras_columnas
    
    # Filtrar para incluir solo las columnas que existen en el dataframe
    orden_final = [col for col in orden_final if col in df.columns]
    
    # Reordenar y guardar
    df = df[orden_final]
    df.to_csv(ruta_salida, index=False)
    print(f"Proceso completado. Archivo guardado en: {ruta_salida}")

# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Convierte preguntas técnicas en versiones más comprensibles')
    parser.add_argument('--input', type=str, default="/home/jovyan/DEEPEVAL_AL/output/dataset_main_vllm.csv", help='Ruta al archivo CSV de entrada')
    parser.add_argument('--output', type=str, default="/home/jovyan/DEEPEVAL_AL/output/dataset_main_vllm_evaluado_semantic.csv", help='Ruta donde guardar el archivo CSV de salida')
    parser.add_argument('--batch', type=int, default=20, help='Tamaño del lote para procesamiento')
    parser.add_argument('--max-intentos', type=int, default=20, help='Número máximo de intentos para reformular una pregunta')
    parser.add_argument('--sim-umbral', type=float, default=0.75, help='Umbral mínimo de similitud semántica (0-1)')
    parser.add_argument('--rouge-umbral', type=float, default=0.5, help='Umbral máximo de similitud léxica (0-1)')
    args = parser.parse_args()
    
    # Actualizar umbrales globales
    global UMBRAL_SIMILARITY, UMBRAL_ROUGE_MAX
    UMBRAL_SIMILARITY = args.sim_umbral
    UMBRAL_ROUGE_MAX = args.rouge_umbral
    
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
        archivo_salida = archivo_entrada.parent / f"{nombre_base}_evaluado_semantic.csv"
    else:
        archivo_salida = Path(args.output)
    
    # Procesar el CSV
    procesar_csv(archivo_entrada, archivo_salida, args.batch, args.max_intentos)

if __name__ == "__main__":
    main()
