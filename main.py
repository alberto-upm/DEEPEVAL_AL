from deepeval.synthesizer import Synthesizer
from deepeval.models import OllamaModel
import os
import glob

# Define paths
documents_path = "/Users/albertog.garcia/Documents/UPM/TFG/Documentos"
output_path = '/Users/albertog.garcia/Documents/UPM/TFG/DEEPEVAL/output/dataset_main.csv'

# Ensure output directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Configure Ollama model
ollama_model = OllamaModel(
    #model="deepseek-r1:1.5b",
    model="llama3.1:8b",
    base_url="http://localhost:11434"
)

# Create synthesizer with Ollama model
synthesizer = Synthesizer(
    model=ollama_model,
    async_mode=True,
    max_concurrent=5,  # Lower value to avoid rate limiting with local models
    cost_tracking=True
)

# Get all document files from the directory
document_paths = []
for ext in ['*.txt', '*.pdf', '*.docx']:
    document_paths.extend(glob.glob(os.path.join(documents_path, ext)))

print(f"Found {len(document_paths)} documents")

# Generate synthetic goldens from documents
synthesizer.generate_goldens_from_docs(
    document_paths=document_paths,
    include_expected_output=True,
    #max_goldens_per_context=3  # Generate 3 questions per document
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
