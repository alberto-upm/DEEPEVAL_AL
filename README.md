python3 -m venv venv
cd DEEPEVAL_AL/
source venv/bin/activate
pip install -U deepeval

deepeval login --confident-api-key 
pip install requirements.txt

# He modificado este fichero para que funcione en Jupyterlab:
deepeval/synthesizer/chunking/context_generator.py


Estoy utilizando LM studio para el LLM y Ollama para los embeddings 
en LM Studio tientes que lanzar el servidor
y en Ollama tiene que lanzar el servidor

## vllm
vllm serve NousResearch/Meta-Llama-3-8B-Instruct \
    --port 8000 --dtype float16 --max-model-len 4096

### para sabaner si está funcionando: 
curl http://localhost:8000/v1/models

deepeval set-local-model \
  --model-name="NousResearch/Meta-Llama-3-8B-Instruct" \
  --base-url="http://localhost:8000/v1/" \
  --api-key="not-needed"

## LM Studio
### para sabaner si está funcionando: 
curl http://localhost:1234/v1/models

deepeval set-local-model \
  --model-name="lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" \
  --base-url="http://localhost:1234/v1/" \
  --api-key="not-needed"


## Ollama (para el embeddign)
ollama run deepseek-r1:1.5b
deepeval set-ollama-embeddings deepseek-r1:1.5b \
  --base-url="http://localhost:11434"

# Verificación básica: uso de GPU con nvidia-smi
## Ejecuta el siguiente comando mientras haces una inferencia con Ollama:
watch -n 1 nvidia-smi



deepeval set-ollama-embeddings deepseek-r1:1.5b

deepeval unset-ollama
deepeval unset-ollama-embeddings


from deepeval.models import OllamaModel
from deepeval.metrics import AnswerRelevancyMetric

model = OllamaModel(
    model="deepseek-r1:1.5b",
    base_url="http://localhost:11434"
)

answer_relevancy = AnswerRelevancyMetric(model=model)

deepeval set-local-model --model-name=<model_name> \
    --base-url="http://localhost:8000/v1/" \
    --api-key=<api-key>


Using local LLM models
There are several local LLM providers that offer OpenAI API compatible endpoints, like vLLM or LM Studio. You can use them with deepeval by setting several parameters from the CLI. To configure any of those providers, you need to supply the base URL where the service is running. These are some of the most popular alternatives for base URLs:

LM Studio: http://localhost:1234/v1/
vLLM: http://localhost:8000/v1/
For example to use a local model from LM Studio, use the following command:

deepeval set-local-model --model-name=<model_name> \
    --base-url="http://localhost:1234/v1/" \
    --api-key=<api-key>

Then, run this to set the local Embeddings model:

deepeval set-local-embeddings --model-name=<embedding_model_name> \
    --base-url="http://localhost:1234/v1/" \
    --api-key=<api-key>

To revert back to the default OpenAI embeddings run:

deepeval unset-local-embeddings

For additional instructions about LLM model and embeddings model availability and base URLs, consult the provider's documentation.


# Para installar ollama se tienen que hacer estos pasos
sudo apt update 
sudo apt install curl -y

## para permitir la detección de hardware:
sudo apt install pciutils lshw -y 

## Instalar ollama
curl -fsSL https://ollama.com/install.sh | sh

## Ejecutar ollama
export CUDA_VISIBLE_DEVICES=0
export OLLAMA_NUM_GPU_LAYERS=999      # 999 = intenta todas
export OLLAMA_FLASH_ATTENTION=1
ollama serve

## en caso de que falle ollama serve para usar este comando para verificar qué está ocupando el puerto 11434:
por que me ha salido lo siguiente: (venv) root@jupyter-alberto-2egarciaga:~/RV26# ollama serve
Error: listen tcp 127.0.0.1:11434: bind: address already in use
sudo apt install net-tools -y
sudo netstat -tuln | grep 11434

## puedes usar lsof para identificar qué proceso está ocupando el puerto. Si aún no has instalado lsof, hazlo con:
sudo apt install lsof -y

## Después de instalarlo, usa el siguiente comando para identificar el proceso:
sudo lsof -i :11434

## para matar a los procesos
sudo kill -9 PIDs


# en caso de usar ollama activamos el servidor en otra terminal
Start Ollama
ollama serve #is used when you want to start ollama without running the desktop application.

ollama pull llama3.1
ollama pull llama3
ollama pull mistral

ollama run llama3.1:70b
ollama pull llama3.1:70b

ollama pull llama-3.1-70b-versatile

ollama run deepseek-r1:8b


