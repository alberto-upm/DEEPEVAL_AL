#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genera un conjunto de datos sintético (pregunta-respuesta)
a partir de un directorio de documentos usando DeepEval
y un servidor vLLM local (OpenAI-compatible).

Requisitos:
    pip install deepeval vllm

Antes de ejecutar, asegúrate de lanzar vLLM:
    python -m vllm.entrypoints.openai.api_server \
           --model /ruta/al/modelo \
           --port 8000 --dtype float16

Autor: Alberto G. García  |  Fecha: 2025-04-24
"""

import os
import glob
from pathlib import Path

from deepeval.synthesizer import Synthesizer
from deepeval.models import LocalModel           # vLLM via OpenAI-compatible API
from deepeval.synthesizer.config import StylingConfig

os.environ["DEEPEVAL_PRESERVE_VECTOR_DB"] = "1"

# ---------------------------------------------------------------------------
# 1. Rutas de entrada / salida
# ---------------------------------------------------------------------------
DOCUMENTS_DIR = Path("//home/jovyan/Documentos/Docs_pdf")
OUTPUT_FILE   = Path("/home/jovyan/DEEPEVAL_AL/output/dataset_main_vllm.csv")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 2. Configurar la conexión a vLLM
#    Si ya ejecutaste `deepeval set-local-model ...` puedes omitir este bloque.
# ---------------------------------------------------------------------------
#VLLM_BASE_URL = "http://localhost:8000/v1/"      # Endpoint creado por vLLM
#os.environ.setdefault("OPENAI_API_KEY", "EMPTY") # vLLM ignora el valor

vllm_model = LocalModel(
    model="NousResearch/Meta-Llama-3-8B-Instruct", # alias que diste al arrancar vLLM
    #base_url=VLLM_BASE_URL,
    #openai_api_key=os.environ["OPENAI_API_KEY"],
)

print(f"🦙  Modelo vLLM configurado: {vllm_model.get_model_name()}")

# ---------------------------------------------------------------------------
# 3. Estilo de las preguntas y respuestas sintetizadas (🇪🇸 enunciados breves)
# ---------------------------------------------------------------------------
estilo_es = StylingConfig(
    input_format=(
        "Genera preguntas concisas EN ESPAÑOL que puedan responderse "
        "exclusivamente con la información del contexto proporcionado."
    ),
    expected_output_format="Respuesta correcta y breve en ESPAÑOL.",
    task="Responder consultas sobre los documentos, en ESPAÑOL.",
    scenario="Respondes a preguntas sobre documentos utilizando el ESPAÑOL.",
)

# ---------------------------------------------------------------------------
# 4. Crear el sintetizador
# ---------------------------------------------------------------------------
synthesizer = Synthesizer(
    model=vllm_model,
    async_mode=False, 
    max_concurrent=5,    # Ajusta según la VRAM / throughput de tu GPU
    cost_tracking=True,
    styling_config=estilo_es,
)

# ---------------------------------------------------------------------------
# 5. Localizar todos los documentos admitidos
# ---------------------------------------------------------------------------
document_paths = []
for ext in ("*.txt", "*.pdf", "*.docx"):
    document_paths += glob.glob(str(DOCUMENTS_DIR / ext))

print(f"📂  Documentos encontrados: {len(document_paths)}")

# ---------------------------------------------------------------------------
# 6. Generar preguntas + respuestas (goldens) y guardarlas en CSV
# ---------------------------------------------------------------------------
synthesizer.generate_goldens_from_docs(
    document_paths=document_paths,
    include_expected_output=True,
    # max_goldens_per_context=3,  # Descomenta para limitar a 3 por documento
)

print(f"✅  Goldens generados: {len(synthesizer.synthetic_goldens)}")

synthesizer.save_as(
    file_type="csv",
    directory=str(OUTPUT_FILE.parent),
    file_name=OUTPUT_FILE.stem,
)

print(f"💾  Dataset sintético guardado en: {OUTPUT_FILE}")

# ---------------------------------------------------------------------------
# 7. (Opcional) inspección rápida con pandas
# ---------------------------------------------------------------------------
df = synthesizer.to_pandas()
print(df.head())
