from deepeval.synthesizer import Synthesizer
from deepeval.models import OllamaModel
from deepeval.models import LocalModel
from deepeval.synthesizer.config import StylingConfig
import os
import glob

# Define paths
documents_path = "/Users/albertog.garcia/Documents/UPM/TFG/Documentos"
output_path = '/Users/albertog.garcia/Documents/UPM/TFG/DEEPEVAL/output/dataset_main_espanol.csv'

# Ensure output directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)
'''
# Configure Ollama model
ollama_model = OllamaModel(
    #model="deepseek-r1:1.5b",
    model="llama3.1:8b",
    base_url="http://localhost:11434"
)
'''
lmstudio_model = LocalModel(
    model="lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    #base_url="http://localhost:1234/v1/",
    #openai_api_key="not-needed"
)

print(f"Model name: {lmstudio_model.get_model_name()}")

estilo_es = StylingConfig(
    input_format=(
        "Genera preguntas concisas EN ESPAÑOL que puedan responderse "
        "exclusivamente con la información del contexto proporcionado."
    ),
    expected_output_format="Respuesta correcta y breve en ESPAÑOL.",
    task="Responder consultas sobre los documentos en ESPAÑOL.",
    scenario="Sistema que responde a preguntas sobre documentos en ESPAÑOL.",
)

# Create synthesizer with Ollama model
synthesizer = Synthesizer(
    model=lmstudio_model,
    async_mode=True,
    max_concurrent=5,  # Lower value to avoid rate limiting with local models
    cost_tracking=True,
    styling_config=estilo_es,
)

# Get all document files from the directory
document_paths = []
for ext in ['*.txt', '*.pdf', '*.docx']:
    document_paths.extend(glob.glob(os.path.join(documents_path, ext)))

print(f"Found {len(document_paths)} documents")

'''
MAX_CTXS_PER_DOC   = 8    # hasta 8 contextos distintos por documento
MAX_GOLDENS_CTX    = 4    # hasta 4 preguntas por contexto
CHUNK_SIZE         = 512  # tokens por chunk → más chunks = más contextos
CHUNK_OVERLAP      = 50   # solape para no cortar info relevante

    context_construction_config = {
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "max_contexts_per_document": MAX_CTXS_PER_DOC,
    },
'''
# Generate synthetic goldens from documents
synthesizer.generate_goldens_from_docs(
    document_paths=document_paths,
    include_expected_output=True,
    max_goldens_per_context = 9,
)

# Print generated goldens
print(f"Generated {len(synthesizer.synthetic_goldens)} synthetic goldens")

# Save as CSV
synthesizer.save_as(
    file_type='csv',
    directory=os.path.dirname(output_path),
    file_name=os.path.basename(output_path).split('.')[0]
)

print(f"Saved synthetic dataset to {output_path}")

# You can also convert to pandas DataFrame for further processing
dataframe = synthesizer.to_pandas()
print(dataframe.head())
