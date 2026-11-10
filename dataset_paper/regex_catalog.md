# Regular expressions for import and model-load detection

18 import regexes (one per library) and 48 model-load regexes (37 model-loading APIs). Generated from python/ppoem/rules.py.

## sklearn

Import:
```
^\s*(?:import\s+(?:sklearn)(?:\.\w+)*\b|from\s+(?:sklearn)(?:\.\w+)*\s+import\s)
```

Model load `sklearn.estimator`, qualified call name (model: (the class itself is the model)):
```
sklearn(?:\.\w+)*\.[A-Z]\w*
```

## torch

Import:
```
^\s*(?:import\s+(?:torch)(?:\.\w+)*\b|from\s+(?:torch)(?:\.\w+)*\s+import\s)
```

Model load `torch.load`, qualified call name (model: position 0 or f=):
```
torch\.load
```

Model load `torch.jit.load`, qualified call name (model: position 0 or f=):
```
torch\.jit\.load
```

Model load `torch.export.load`, qualified call name (model: position 0 or f=):
```
torch\.export\.load
```

Model load `torch.hub.load`, qualified call name (model: position 1 or model=):
```
torch\.hub\.load
```

Model load `torch.hub.load_state_dict_from_url`, qualified call name (model: position 0 or url=):
```
torch\.(?:hub\.load_state_dict_from_url|utils\.model_zoo\.load_url)
```

## tensorflow

Import:
```
^\s*(?:import\s+(?:tensorflow|keras)(?:\.\w+)*\b|from\s+(?:tensorflow|keras)(?:\.\w+)*\s+import\s)
```

Model load `tensorflow.load_model`, qualified call name (model: position 0 or filepath=; position 0 or export_dir=; export_dir=):
```
(?:tensorflow\.(?:python\.)?keras|keras)\.(?:models\.|saving\.)?load_model|tensorflow\.(?:compat\.v1\.)?saved_model\.load|tensorflow\.keras\.experimental\.load_from_saved_model
```

Model load `tensorflow.lite.Interpreter`, qualified call name (model: model_path=; position 0 or model_path=):
```
tensorflow\.lite\.(?:python\.interpreter\.)?Interpreter
```

Model load `tensorflow.keras.applications`, qualified call name (model: (the class itself is the model)):
```
(?:tensorflow\.)?keras\.applications(?:\.\w+)*\.[A-Z]\w*
```

## transformers

Import:
```
^\s*(?:import\s+(?:transformers)(?:\.\w+)*\b|from\s+(?:transformers)(?:\.\w+)*\s+import\s)
```

Model load `transformers.from_pretrained`, qualified call name (model: position 0 or pretrained_model_name_or_path=):
```
transformers(?:\.\w+)*\.\w+\.from_pretrained
```

Model load `transformers.pipeline`, qualified call name (model: model=; position 1 or model=):
```
transformers\.pipeline
```

## openai

Import:
```
^\s*(?:import\s+(?:openai)(?:\.\w+)*\b|from\s+(?:openai)(?:\.\w+)*\s+import\s)
```

Model load `openai.chat_api`, qualified call name (model: model=; engine=; deployment_id=):
```
openai(?:\.\w+(?:\(\))?)*\.(?:create|acreate|generate|parse|stream|edit|create_variation)
```

Model value from the call text (group `value`):
```
(?:client|openai)\.(?:\w+\.)+(?:create|generate)\s*\([\s\S]*?(?:model|engine|deployment_id)\s*=\s*(['"])(?P<value>[^'"]+)\1
```

Fallback, client object not identified:
```
\.(?:\w+\.)*(?:completions|embeddings|responses)\.(?:create|parse|stream)$
```

## anthropic

Import:
```
^\s*(?:import\s+(?:anthropic)(?:\.\w+)*\b|from\s+(?:anthropic)(?:\.\w+)*\s+import\s)
```

Model load `anthropic.messages`, qualified call name (model: model=):
```
anthropic\.(?:Async)?Anthropic(?:Bedrock|Vertex)?\(\)\.(?:messages\.(?:create|stream|count_tokens|batches\.create)|beta\.messages\.(?:create|stream)|completions\.create)
```

Fallback, client object not identified:
```
\.(?:beta\.)?messages\.(?:create|stream)$
```

## groq

Import:
```
^\s*(?:import\s+(?:groq)(?:\.\w+)*\b|from\s+(?:groq)(?:\.\w+)*\s+import\s)
```

Model load `groq.chat_api`, qualified call name (model: model=; engine=; deployment_id=):
```
groq(?:\.\w+(?:\(\))?)*\.(?:create|acreate|generate|parse|stream|edit|create_variation)
```

Model value from the call text (group `value`):
```
(?:client|groq)\.(?:\w+\.)+(?:create|generate)\s*\([\s\S]*?(?:model|engine|deployment_id)\s*=\s*(['"])(?P<value>[^'"]+)\1
```

Fallback, client object not identified:
```
\.(?:\w+\.)*(?:completions|embeddings|responses)\.(?:create|parse|stream)$
```

## fireworks

Import:
```
^\s*(?:import\s+(?:fireworks)(?:\.\w+)*\b|from\s+(?:fireworks)(?:\.\w+)*\s+import\s)
```

Model load `fireworks.chat_api`, qualified call name (model: model=; engine=; deployment_id=; position 0 or model=):
```
fireworks(?:\.\w+(?:\(\))?)*\.(?:create|acreate|generate|parse|stream|edit|create_variation)|fireworks(?:\.llm)?\.LLM
```

Model value from the call text (group `value`):
```
(?:client|fireworks)\.(?:\w+\.)+(?:create|generate)\s*\([\s\S]*?(?:model|engine|deployment_id)\s*=\s*(['"])(?P<value>[^'"]+)\1
```

Fallback, client object not identified:
```
\.(?:\w+\.)*(?:completions|embeddings|responses)\.(?:create|parse|stream)$
```

## llama_api_client

Import:
```
^\s*(?:import\s+(?:llama_api_client)(?:\.\w+)*\b|from\s+(?:llama_api_client)(?:\.\w+)*\s+import\s)
```

Model load `llama_api_client.chat_api`, qualified call name (model: model=; engine=; deployment_id=):
```
llama_api_client(?:\.\w+(?:\(\))?)*\.(?:create|acreate|generate|parse|stream|edit|create_variation)
```

Model value from the call text (group `value`):
```
(?:client|llama)\.(?:\w+\.)+(?:create|generate)\s*\([\s\S]*?(?:model|engine|deployment_id)\s*=\s*(['"])(?P<value>[^'"]+)\1
```

Fallback, client object not identified:
```
\.(?:\w+\.)*(?:completions|embeddings|responses)\.(?:create|parse|stream)$
```

## deepseek

Import:
```
^\s*(?:import\s+(?:deepseek)(?:\.\w+)*\b|from\s+(?:deepseek)(?:\.\w+)*\s+import\s)
```

Model load `deepseek.chat_api`, qualified call name (model: model=; engine=; deployment_id=):
```
deepseek(?:\.\w+(?:\(\))?)*\.(?:create|acreate|generate|parse|stream|edit|create_variation)
```

Model value from the call text (group `value`):
```
(?:client|deepseek)\.(?:\w+\.)+(?:create|generate)\s*\([\s\S]*?(?:model|engine|deployment_id)\s*=\s*(['"])(?P<value>[^'"]+)\1
```

Fallback, client object not identified:
```
\.(?:\w+\.)*(?:completions|embeddings|responses)\.(?:create|parse|stream)$
```

## vllm

Import:
```
^\s*(?:import\s+(?:vllm)(?:\.\w+)*\b|from\s+(?:vllm)(?:\.\w+)*\s+import\s)
```

Model load `vllm.LLM`, qualified call name (model: position 0 or model=; model=):
```
vllm\.(?:entrypoints\.llm\.)?LLM
```

Model load `vllm.EngineArgs`, qualified call name (model: model=):
```
vllm\.(?:engine\.arg_utils\.)?(?:Async)?EngineArgs
```

## llama_cpp

Import:
```
^\s*(?:import\s+(?:llama_cpp)(?:\.\w+)*\b|from\s+(?:llama_cpp)(?:\.\w+)*\s+import\s)
```

Model load `llama_cpp.Llama`, qualified call name (model: position 0 or model_path=; model_path=):
```
llama_cpp\.(?:llama\.)?Llama
```

Model load `llama_cpp.from_pretrained`, qualified call name (model: position 0 or repo_id=):
```
llama_cpp\.Llama\.from_pretrained
```

Model load `llama_cpp.load_model_from_file`, qualified call name (model: position 0 or path_model=):
```
llama_cpp\.llama_(?:load_model_from_file|model_load_from_file)
```

## xgboost

Import:
```
^\s*(?:import\s+(?:xgboost)(?:\.\w+)*\b|from\s+(?:xgboost)(?:\.\w+)*\s+import\s)
```

Model load `xgboost.estimator`, qualified call name (model: (the class itself is the model)):
```
xgboost\.(?:sklearn\.)?XGB(?:RF)?(?:Classifier|Regressor)|xgboost\.(?:sklearn\.)?XGBRanker|xgboost\.dask\.DaskXGB(?:Classifier|Regressor)
```

Model load `xgboost.train`, qualified call name (model: (the class itself is the model)):
```
xgboost\.(?:training\.|dask\.)?train
```

Model load `xgboost.Booster`, qualified call name (model: model_file=):
```
xgboost\.Booster
```

Model load `xgboost.load_model`, qualified call name (model: position 0 or fname=):
```
xgboost(?:\.\w+)*\(\)\.load_model
```

## ollama

Import:
```
^\s*(?:import\s+(?:ollama)(?:\.\w+)*\b|from\s+(?:ollama)(?:\.\w+)*\s+import\s)
```

Model load `ollama.api`, qualified call name (model: position 0 or model=; model=):
```
ollama\.(?:(?:Async)?Client\(\)\.)?(?:chat|generate|embeddings|embed|pull|show)
```

## vertexai

Import:
```
^\s*(?:import\s+(?:vertexai)(?:\.\w+)*\b|from\s+(?:vertexai)(?:\.\w+)*\s+import\s)
```

Model load `vertexai.GenerativeModel`, qualified call name (model: position 0 or model_name=):
```
vertexai\.(?:preview\.)?(?:generative_models\.)?GenerativeModel
```

Model load `vertexai.from_pretrained`, qualified call name (model: position 0 or model_name=):
```
vertexai\.(?:preview\.)?(?:language|vision)_models(?:\.\w+)*\.\w+\.from_pretrained
```

## genai

Import:
```
^\s*(?:import\s+google\.(?:genai|generativeai)\b|from\s+google(?:\.(?:genai|generativeai)\S*\s+import\s|\s+import\s+(?:genai|generativeai)\b))
```

Model load `genai.GenerativeModel`, qualified call name (model: position 0 or model_name=):
```
google\.generativeai\.GenerativeModel
```

Model load `genai.client`, qualified call name (model: model=):
```
google\.genai\.Client\(\)\.(?:aio\.)?(?:models\.(?:generate_content(?:_stream)?|embed_content|generate_images|generate_videos|count_tokens)|chats\.create)
```

## diffusers

Import:
```
^\s*(?:import\s+(?:diffusers)(?:\.\w+)*\b|from\s+(?:diffusers)(?:\.\w+)*\s+import\s)
```

Model load `diffusers.from_pretrained`, qualified call name (model: position 0 or pretrained_model_name_or_path=):
```
diffusers(?:\.\w+)*\.\w+\.from_pretrained
```

Model load `diffusers.from_single_file`, qualified call name (model: position 0 or pretrained_model_link_or_path=; position 0 or pretrained_model_link_or_path_or_dict=):
```
diffusers(?:\.\w+)*\.\w+\.from_single_file
```

Model load `diffusers.load_lora_weights`, qualified call name (model: position 0 or pretrained_model_name_or_path_or_dict=):
```
diffusers(?:\.\w+)*\(\)\.load_lora_weights
```

## huggingface_hub

Import:
```
^\s*(?:import\s+(?:huggingface_hub)(?:\.\w+)*\b|from\s+(?:huggingface_hub)(?:\.\w+)*\s+import\s)
```

Model load `huggingface_hub.download`, qualified call name (model: position 0 or repo_id=):
```
huggingface_hub\.(?:HfApi\(\)\.|file_download\.|_snapshot_download\.)?(?:hf_hub_download|snapshot_download)
```

Model load `huggingface_hub.from_pretrained_keras`, qualified call name (model: position 0 or repo_id=):
```
huggingface_hub\.from_pretrained_keras
```

Model load `huggingface_hub.InferenceClient`, qualified call name (model: model=; position 0 or model=):
```
huggingface_hub\.(?:Async)?InferenceClient(?:\(\)\.\w+)?
```
