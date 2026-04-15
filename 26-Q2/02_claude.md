# Claude Code Startup

```
curl -LsSf https://astral.sh/uv/install.sh | sh
uv tool install 'litellm[proxy]'

curl -fsSL https://claude.ai/install.sh | bash

# write config.yaml
model_list:
- model_name: DeepSeek-V3.2
  litellm_params:
    model: openai/DeepSeek-V3.2
    api_key: key
    api_base: "url"
- model_name: GLM-5-Turbo
  litellm_params:
    model: openai/GLM-5-Turbo
    api_key: key
    api_base: "url"

export LITELLM_MASTER_KEY="sk-1234567890"

litellm --config config.yaml

curl -X POST http://0.0.0.0:4000/v1/chat/completions \
-H "Authorization: Bearer $LITELLM_MASTER_KEY" \
-H "Content-Type: application/json" \
-d '{
    "model": "DeepSeek-V3.2",
    "max_tokens": 1000,
    "messages": [{"role": "user", "content": "What is the capital of France?"}]
}'

export ANTHROPIC_BASE_URL="http://0.0.0.0:4000"
export ANTHROPIC_AUTH_TOKEN="$LITELLM_MASTER_KEY"

claude --model DeepSeek-V3.2
```
