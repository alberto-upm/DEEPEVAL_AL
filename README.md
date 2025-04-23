python3 -m venv venv
source venv/bin/activate
pip install -U deepeval

ollama run deepseek-r1:1.5b
deepeval set-ollama deepseek-r1:1.5b
deepeval set-ollama deepseek-r1:1.5b \
    --base-url="http://localhost:11434"



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

