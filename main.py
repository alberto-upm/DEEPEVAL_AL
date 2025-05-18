"""
Genera dos conjuntos de datos sintéticos (pregunta-respuesta y metadatos) a partir de un directorio
# de documentos usando DeepEval y un servidor vLLM local (OpenAI-compatible).

Requisitos:
    pip install deepeval vllm pandas

Antes de ejecutar, asegúrate de lanzar vLLM:
    python -m vllm.entrypoints.openai.api_server \
        --model /ruta/al/modelo \
        --port 8000 \
        --dtype float16

Autor: Alberto G. García
Fecha: 2025-04-24
"""

import os
import glob
from pathlib import Path
import pandas as pd
from deepeval.synthesizer import Synthesizer
from deepeval.models import LocalModel
from deepeval.synthesizer.config import StylingConfig, FiltrationConfig, EvolutionConfig, Evolution, ContextConstructionConfig 

# Persistir base de vectores para acelerar embeddings
os.environ["DEEPEVAL_PRESERVE_VECTOR_DB"] = "1"

# ----------------------------------------------------------------------------
# 1. Rutas de entrada / salida
# ----------------------------------------------------------------------------
DOCUMENTS_DIR = Path("/home/jovyan/Documentos/Docs_pdf")
OUTPUT_DIR   = Path("/home/jovyan/DEEPEVAL_AL/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATASET_FILE  = OUTPUT_DIR / "2_dataset.csv"
METADATA_FILE = OUTPUT_DIR / "2_metadata_dataset.csv"

# ----------------------------------------------------------------------------
# 2. Conexión al modelo vLLM
# ----------------------------------------------------------------------------
os.environ.setdefault("OPENAI_API_KEY", "EMPTY")  # vLLM ignora la clave
vllm_model = LocalModel(
    model="NousResearch/Meta-Llama-3-8B-Instruct",
    #model="meta-llama/Llama-3.1-8B-Instruct",
    #base_url="http://localhost:8000/v1/",
    #openai_api_key=os.environ["OPENAI_API_KEY"]
)
print(f"🦙 Modelo vLLM configurado: {vllm_model.get_model_name()}")

# ----------------------------------------------------------------------------
# 3. Configuración de Filtrado
# ----------------------------------------------------------------------------
filtration_cfg = FiltrationConfig(
    synthetic_input_quality_threshold= 0.5,
    max_quality_retries= 3,
    #critic_model= vllm_model
)

# ----------------------------------------------------------------------------
# 4. Configuración de Evolución
# ----------------------------------------------------------------------------
evolution_cfg = EvolutionConfig(
    num_evolutions=1, #modificar hasta 3 para aumentar la complejidad
    evolutions={
        Evolution.REASONING:    1 / 7,  # Razonamiento lógico
        Evolution.MULTICONTEXT: 1 / 7,  # Multi-contexto
        Evolution.CONCRETIZING: 1 / 7,  # Concretizar detalle
        Evolution.CONSTRAINED:  1 / 7,  # Restringir límites
        Evolution.COMPARATIVE:  1 / 7,  # Comparativo (peso total)
        Evolution.HYPOTHETICAL: 1 / 7,  # Hipotético
        Evolution.IN_BREADTH:   1 / 7   # Cobertura amplia
    }
)

# ----------------------------------------------------------------------------
# 5. Configuración de Estilo
# ----------------------------------------------------------------------------
estilo_es = StylingConfig(
    input_format=(
        "Genera preguntas concisas EN ESPAÑOL que puedan responderse exclusivamente con la información del contexto proporcionado."
    ),
    expected_output_format="Respuesta corta y breve en ESPAÑOL. Responde UNICAMENTE con la respuesta, sin añadir comentarios.",
    task="Responder consultas sobre los documentos, en ESPAÑOL.",
    scenario="Evaluación de comprensión de documentos en ESPAÑOL."
)

# ----------------------------------------------------------------------------
# 6. Creación del Synthesizer
# ----------------------------------------------------------------------------
synthesizer = Synthesizer(
    model=vllm_model,
    async_mode=False,
    max_concurrent=5,
    filtration_config=filtration_cfg,
    evolution_config=evolution_cfg,
    styling_config=estilo_es,
    cost_tracking=True
)
# Ajuste de chunking
#synthesizer.context_generator.chunk_size = 1024
#synthesizer.context_generator.chunk_overlap = 0
#synthesizer.context_generator.max_contexts_per_document = 3

# ----------------------------------------------------------------------------
# 7. Carga de documentos
# ----------------------------------------------------------------------------
document_paths = []
for ext in ("*.txt", "*.pdf", "*.docx"):
    document_paths += glob.glob(str(DOCUMENTS_DIR / ext))
print(f"📄 Documentos encontrados: {len(document_paths)}")

# ----------------------------------------------------------------------------
# 8. Generación de Goldens
# ----------------------------------------------------------------------------
context_construction_cfg = ContextConstructionConfig(
    #embedder: Optional[Union[str, DeepEvalBaseEmbeddingModel]] = None,
    critic_model= vllm_model,
    #encoding: Optional[str] = None,
    max_contexts_per_document= 3,
    min_contexts_per_document= 1,
    max_context_length= 3,
    min_context_length= 1,
    chunk_size= 1024,
    chunk_overlap= 0,
    context_quality_threshold= 0.5,
    context_similarity_threshold= 0.0,
    max_retries= 3
)
    

synthesizer.generate_goldens_from_docs(
    document_paths=document_paths,
    include_expected_output=True,
    max_goldens_per_context=4,
    context_construction_config=context_construction_cfg
    
)
print(f"✅ Goldens generados: {len(synthesizer.synthetic_goldens)}")

# Guardar dataset principal
df = synthesizer.to_pandas()
df.to_csv(DATASET_FILE, index=False, encoding='utf-8')
print(f"💾 Dataset guardado en: {DATASET_FILE}")

# ----------------------------------------------------------------------------
# 9. Extracción y guardado de metadatos
# ----------------------------------------------------------------------------
df_meta = pd.DataFrame([
    {
        "input": g.input,
        "evolutions": g.additional_metadata.get("evolutions"),
        "synthetic_input_quality": g.additional_metadata.get("synthetic_input_quality"),
        "context_quality": g.additional_metadata.get("context_quality")
    }
    for g in synthesizer.synthetic_goldens
])
df_meta.to_csv(METADATA_FILE, index=False, encoding='utf-8')
print(f"💾 Metadatos guardados en: {METADATA_FILE}")
